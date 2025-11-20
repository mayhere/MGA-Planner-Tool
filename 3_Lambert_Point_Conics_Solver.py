# Lambert_Point_Conics_Solver.py
"""
Staged 2-leg seeded Lambert Stage-A harvester.
- Input: CSV with 'chain_str' column (e.g., Tisserand_Fast_Survivors.csv)
- Output: lambert_Point_Conics_Results.csv with vector JSONs and metrics
- Features: dedupe, staged expansion (2-leg seeds -> iterative), autosave, checkpoint/resume (Windows-safe)
"""

import os
import time
import json
import tempfile
import stat
import traceback
import itertools
import math
from collections import deque

import numpy as np
import pandas as pd
from tqdm import tqdm

from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from poliastro.iod.izzo import lambert
from poliastro.bodies import Sun
from poliastro.util import norm

# ---------------- USER CONFIG ----------------
INPUT_TISSERAND_CSV = r"C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/0_Tisserand_Chain_Finder/2026-27_Tisserand_Chains_.csv"
OUT_DIR = r"C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/A_Lambert_Point_Conics_Solver"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "2026-27_StageA_Lambert_Point_Conics_Results.csv")
CHECKPOINT_JSON = os.path.join(OUT_DIR, "2026-27_StageA_Checkpoint_Per_Chain_Index.json")
DEDUP_MAP_CSV = os.path.join(OUT_DIR, "2026-27_StageA_Dedup_Map.csv")

# Campaign windows (seed sweep)
CAMPAIGN_WINDOWS = {
    "2026-27": (Time("2026-01-01", scale="tdb"), Time("2027-12-31", scale="tdb")),
    #"2033-34": (Time("2033-01-01", scale="tdb"), Time("2034-12-31", scale="tdb")),
}
SEED_STEP_DAYS = 15  # days between seeds

# TOF sampling (can set per-leg if you like)
# Option A: explicit sample list (coarse)
GENERIC_TOF_CHOICES_DAYS = [20.0, 60.0, 120.0, 200.0, 300.0, 400.0]
# Option B: if you want uniform linspace per leg, set N_TOF_PER_LEG > 0 and GENERIC_TOF_CHOICES_DAYS will be ignored.
N_TOF_PER_LEG = 5  # set >0 to use linspace per leg

# Staging & branching control (key to speed)
MAX_BRANCHES_PER_LEG = 15     # after expanding a leg, keep only this many best partials (prune)
BRANCH_SORT_KEY = "first_C3"   # key to sort partials when pruning: first_C3 or first_vinf or total_tof etc.

# Safety limits
MAX_PARTIALS_TOTAL = 5000      # absolute cap on partials to avoid memory blowup (safety)
MAX_LEG_COMBO_TRY = 200000     # fallback cap for naive combos (shouldn't hit with staged method)

# Autosave & checkpoint cadence
AUTOSAVE_EVERY_ROWS = 500
SAVE_EVERY_SECONDS = 180

# Deduplicate chains before harvesting?
DEDUP_INPUT = True

# Ephemeris setup either JPL (preferred) or builtin
try:
    solar_system_ephemeris.set("C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/kernels/de440.bsp")
except Exception:
    solar_system_ephemeris.set("builtin")
    print("Warning: JPL ephemeris unavailable, using builtin (lower precision).")

LABELS = {"Ea": "earth", "Ve": "venus", "Me": "mercury", "Ma": "mars"}

# ------------------------------------Functions------------------------------------------------
def save_checkpoint_safe(path, ck):
    d = os.path.dirname(path) or "."
    fd_temp, tmp_path = tempfile.mkstemp(prefix="tmp_ckpt_", dir=d, text=True)
    try:
        with os.fdopen(fd_temp, "w", encoding="utf-8") as f:
            json.dump(ck, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        try:
            os.replace(tmp_path, path)
            return True
        except PermissionError:
            try:
                if os.path.exists(path):
                    os.chmod(path, stat.S_IWRITE)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(ck, f, indent=2)
                    f.flush(); os.fsync(f.fileno())
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return True
            except Exception as e:
                print("Checkpoint fallback write failed:", e)
                traceback.print_exc()
                return False
    except Exception as e:
        print("Failed writing tmp checkpoint:", e)
        traceback.print_exc()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(ck, f, indent=2)
            return True
        except Exception as e2:
            print("Final direct write failed:", e2); traceback.print_exc(); return False
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def load_checkpoint(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def ensure_out_csv_header(path):
    header_cols = [
        "campaign","chain_str","seed_epoch_iso","tof_combo_days","total_tof_days",
        "first_leg_vinf_kms","last_leg_vinf_kms","first_leg_C3_km2s2",
        "leg_v1_vectors_json","leg_v2_vectors_json","notes"
    ]
    if not os.path.exists(path):
        pd.DataFrame([], columns=header_cols).to_csv(path, index=False)

def vectors_to_json(vecs):
    return json.dumps([[float(x) for x in v] for v in vecs])

def make_tof_grid_for_leg(dep, arr, n_points):
    # heuristic per-pair bounds (days)
    if n_points <= 0:
        return np.array(GENERIC_TOF_CHOICES_DAYS) * u.day
    lo, hi = 20.0, 900.0
    if dep == "Ea" and arr == "Me":
        lo, hi = 30.0, 400.0
    elif arr == "Me":
        lo, hi = 20.0, 400.0
    elif arr == "Ve":
        lo, hi = 40.0, 300.0
    elif arr == "Ma":
        lo, hi = 200.0, 900.0
    return (np.linspace(lo, hi, n_points) * u.day)

def qnorm_kms(q):
    return float(norm(q).to(u.km/u.s).value)

def get_body_rv_heliocentric(body_name, epoch):
    """
    Returns:
      r_helio_q  -- astropy Quantity array shape (3,) in u.km
      v_helio_q  -- astropy Quantity array shape (3,) in u.km/u.s
    """
    r_body_b, v_body_b = get_body_barycentric_posvel(body_name, epoch)
    r_sun_b, v_sun_b   = get_body_barycentric_posvel("sun", epoch)
    r_helio_q = (r_body_b.xyz - r_sun_b.xyz).to(u.km)
    v_helio_q = (v_body_b.xyz - v_sun_b.xyz).to(u.km / u.s)
    # r_helio_q, v_helio_q are astropy Quantity / CartesianRepresentation objects with .xyz
    return r_helio_q, v_helio_q

# partial solution structure:
# {
#   "nodes": [...],
#   "leg_index": idx next to solve,
#   "epoch": astropy.Time at current node (i.e., arrival epoch of last solved leg),
#   "r_prev": r vector (Quantity km) at current epoch,
#   "v_prev": v vector (Quantity km/s) at current epoch,
#   "tofs": [tof days...],  # matched to solved legs
#   "leg_v1": [numpy arrays],
#   "leg_v2": [numpy arrays],
#   "first_C3": float,
#   "first_vinf": float,
#   "total_tof": float (days)
# }

def seeded_expand_two_leg(partials, nodes, start_leg, per_leg_tofs):
    """
    Expand partials by solving up to two legs ahead (staged 2-leg seeding).
    Returns list of new partials (pruned externally).
    """
    new_partials = []
    # For each partial, try combinations for next 1 or 2 legs
    for p in partials:
        leg_idx = p["leg_index"]
        # if already complete, carry forward
        if leg_idx >= len(nodes)-1:
            new_partials.append(p)
            continue
        # choose how many legs to attempt this stage (1 or 2)
        legs_to_attempt = 2 if (leg_idx <= len(nodes)-3) else 1
        # build per-leg grids for these legs
        grids = [per_leg_tofs[leg_idx + i] for i in range(legs_to_attempt)]
        # iterate combos
        for idx_tuple in itertools.product(*[range(len(g)) for g in grids]):
            try:
                epoch = p["epoch"]
                r_prev = p["r_prev"]
                v_prev = p["v_prev"]
                tofs_new = list(p["tofs"])
                leg_v1_new = list(p["leg_v1"])
                leg_v2_new = list(p["leg_v2"])
                success = True
                # iterate legs
                for local_i, idx_choice in enumerate(idx_tuple):
                    real_leg_i = leg_idx + local_i
                    dep = nodes[real_leg_i]; arr = nodes[real_leg_i+1]
                    tof = grids[local_i][idx_choice]
                    # fetch target state at epoch+tof
                    r_arr, v_arr = get_body_rv_heliocentric(LABELS[arr], epoch + tof)
                    r1 = r_prev.to(u.km) #if hasattr(r_prev, "xyz") else r_prev
                    # r_prev may be Quantity already (we maintain as Quantity)
                    # lambert: Sun.k, r1, r2, tof_seconds
                    r2 = r_arr.to(u.km)
                    v1, v2 = lambert(Sun.k, r1, r2, tof.to(u.s))
                    v1_kms = np.array(v1.to(u.km/u.s).value)
                    v2_kms = np.array(v2.to(u.km/u.s).value)
                    # append
                    leg_v1_new.append(v1_kms)
                    leg_v2_new.append(v2_kms)
                    tofs_new.append(float(tof.to(u.day).value))
                    # update epoch & r_prev, v_prev for next leg
                    epoch = epoch + tof
                    r_prev = r_arr.to(u.km)
                    v_prev = v_arr.to(u.km/u.s)
                # compute first_C3 & first_vinf if not present
                if len(p["leg_v1"])==0 and len(leg_v1_new)>0:
                    # compute planet velocity at initial departure epoch (seed)
                    dep0 = nodes[0]
                    r_dep0, v_dep0 = get_body_rv_heliocentric(LABELS[dep0], p["seed_epoch"])
                    v_dep0_kms = np.array(v_dep0.to(u.km/u.s).value)
                    v1_first = np.array(leg_v1_new[0])
                    vinf_dep_vec = v1_first - v_dep0_kms
                    first_vinf = float(np.linalg.norm(vinf_dep_vec))
                    first_C3 = first_vinf**2
                else:
                    first_vinf = p.get("first_vinf", None)
                    first_C3 = p.get("first_C3", None)
                total_tof = sum(tofs_new)
                newp = {
                    "nodes": nodes,
                    "leg_index": leg_idx + len(idx_tuple),  # advanced
                    "epoch": epoch,
                    "r_prev": r_prev,
                    "v_prev": v_prev,
                    "tofs": tofs_new,
                    "leg_v1": leg_v1_new,
                    "leg_v2": leg_v2_new,
                    "first_C3": first_C3,
                    "first_vinf": first_vinf,
                    "total_tof": total_tof,
                    "seed_epoch": p["seed_epoch"]
                }
                new_partials.append(newp)
                # safety cap
                if len(new_partials) > MAX_PARTIALS_TOTAL:
                    return new_partials, False
            except Exception:
                # ignore failing combos (lambert errors), continue
                continue
    return new_partials, True

def prune_partials(partials, maxkeep, sort_key):
    if len(partials) <= maxkeep:
        return partials
    # define key function
    if sort_key == "first_C3":
        keyfunc = lambda x: (x.get("first_C3") if x.get("first_C3") is not None else 1e9)
    elif sort_key == "first_vinf":
        keyfunc = lambda x: (x.get("first_vinf") if x.get("first_vinf") is not None else 1e9)
    elif sort_key == "total_tof":
        keyfunc = lambda x: x.get("total_tof", 1e9)
    else:
        keyfunc = lambda x: x.get("first_C3") if x.get("first_C3") is not None else 1e9
    partials_sorted = sorted(partials, key=keyfunc)
    return partials_sorted[:maxkeep]

# ---------------- Main run ----------------
def run():
    # read input
    if not os.path.exists(INPUT_TISSERAND_CSV):
        raise FileNotFoundError(f"Input CSV missing: {INPUT_TISSERAND_CSV}")
    df_in = pd.read_csv(INPUT_TISSERAND_CSV)
    if "chain_str" not in df_in.columns:
        raise ValueError("Input CSV must contain 'chain_str' column")

    # dedupe optionally
    if DEDUP_INPUT:
        grouped = df_in.groupby("chain_str", sort=False)
        dedup_df = grouped.first().reset_index()
        mapping = []
        for name, g in grouped:
            seeds = g.get("seed_epoch", g.index.astype(str)).astype(str).tolist() if "seed_epoch" in g.columns else g.index.astype(str).tolist()
            mapping.append({"chain_str": name, "count": len(g), "seeds": ";".join(seeds)})
        pd.DataFrame(mapping).to_csv(DEDUP_MAP_CSV, index=False)
        chains = dedup_df["chain_str"].astype(str).unique().tolist()
        print(f"Deduplicated: {len(chains)} unique chains (map saved to {DEDUP_MAP_CSV})")
    else:
        chains = df_in["chain_str"].astype(str).unique().tolist()
        print(f"Chains count: {len(chains)}")

    # checkpoint load
    ck = load_checkpoint(CHECKPOINT_JSON)
    if "per_chain_last_seed" not in ck:
        ck["per_chain_last_seed"] = {}

    ensure_out_csv_header(OUT_CSV)
    rows_buffer = []
    last_save_time = time.time()
    written_rows = 0

    # build seeds per campaign
    campaign_seeds = {}
    for camp_name, (s,e) in CAMPAIGN_WINDOWS.items():
        seeds = []
        cur = s
        step = TimeDelta(SEED_STEP_DAYS, format="jd")
        while cur <= e:
            seeds.append(cur)
            cur = cur + step
        campaign_seeds[camp_name] = seeds
        print(f"Campaign {camp_name}: seeds {len(seeds)} ({seeds[0].utc.iso} -> {seeds[-1].utc.iso})")

    try:
        # iterate campaigns
        for camp_name, seeds in campaign_seeds.items():
            pbar_chains = tqdm(chains, desc=f"{camp_name} chains", unit="chain")
            for chain in pbar_chains:
                last_seed_idx = ck["per_chain_last_seed"].get(chain, -1)
                # start seeds loop
                for seed_idx, seed_epoch in enumerate(seeds):
                    if seed_idx <= last_seed_idx:
                        continue  # already done
                    nodes = [n.strip() for n in chain.split("--") if n.strip()]
                    if len(nodes) < 2:
                        ck["per_chain_last_seed"][chain] = seed_idx
                        continue

                    # prepare per-leg grids
                    per_leg_tofs = []
                    for i in range(len(nodes)-1):
                        if N_TOF_PER_LEG > 0:
                            grid = make_tof_grid_for_leg(nodes[i], nodes[i+1], N_TOF_PER_LEG)
                        else:
                            grid = np.array(GENERIC_TOF_CHOICES_DAYS) * u.day
                        per_leg_tofs.append(grid)

                    # initialize partials with seed state
                    r0, v0 = get_body_rv_heliocentric(LABELS[nodes[0]], seed_epoch)
                    initial_partial = {
                        "nodes": nodes,
                        "leg_index": 0,
                        "epoch": seed_epoch,
                        "r_prev": r0.to(u.km),
                        "v_prev": v0.to(u.km/u.s),
                        "tofs": [],
                        "leg_v1": [],
                        "leg_v2": [],
                        "first_C3": None,
                        "first_vinf": None,
                        "total_tof": 0.0,
                        "seed_epoch": seed_epoch
                    }
                    partials = [initial_partial]

                    # staged expansion leg-by-leg (attempt 2 legs per expand where possible)
                    full_solutions = []
                    leg_num = 0
                    while partials:
                        # expand each partial by 1-2 legs using seeded_expand_two_leg
                        new_partials, okflag = seeded_expand_two_leg(partials, nodes, leg_num, per_leg_tofs)
                        if not okflag:
                            # safety cap hit; abort expansion for this seed+chain
                            print(f"[WARN] safety cap hit for chain {chain} seed {seed_epoch.utc.iso}")
                            break
                        # prune partials to MAX_BRANCHES_PER_LEG (sorting by chosen key)
                        partials = prune_partials(new_partials, MAX_BRANCHES_PER_LEG, BRANCH_SORT_KEY)
                        # check for completed partials
                        still_work = []
                        for p in partials:
                            if p["leg_index"] >= len(nodes)-1:
                                # complete solution
                                full_solutions.append(p)
                            else:
                                still_work.append(p)
                        partials = still_work
                        leg_num += 2  # we attempted up to 2 legs
                        # safety overall cap
                        if len(partials) > MAX_PARTIALS_TOTAL:
                            print(f"[WARN] partials exceed MAX_PARTIALS_TOTAL ({len(partials)}). Pruning.")
                            partials = prune_partials(partials, MAX_PARTIALS_TOTAL, BRANCH_SORT_KEY)

                    # For all full_solutions, prepare CSV rows
                    for sol in full_solutions:
                        try:
                            # compute first_vinf & first_C3 if missing
                            if sol["first_vinf"] is None:
                                # get planet vel at seed
                                r_dep0, v_dep0 = get_body_rv_heliocentric(LABELS[nodes[0]], sol["seed_epoch"])
                                v_dep0_kms = np.array(v_dep0.to(u.km/u.s).value)
                                v1_first = np.array(sol["leg_v1"][0])
                                vinf_dep_vec = v1_first - v_dep0_kms
                                first_vinf = float(np.linalg.norm(vinf_dep_vec))
                                sol["first_vinf"] = first_vinf
                                sol["first_C3"] = first_vinf**2
                            # get final arrival vinf
                            # compute arrival epoch (seed + tofs)
                            epochf = sol["seed_epoch"]
                            for t in sol["tofs"]:
                                epochf = epochf + TimeDelta(t, format="jd") if False else epochf + (t*u.day)
                            # actually simpler: convert t list days to TimeDelta days
                            epochf = sol["seed_epoch"]
                            for t in sol["tofs"]:
                                epochf = epochf + (t * u.day)
                            last_body = nodes[-1]
                            r_last, v_last = get_body_rv_heliocentric(LABELS[last_body], epochf)
                            v_last_planet_kms = np.array(v_last.to(u.km/u.s).value)
                            v2_last = np.array(sol["leg_v2"][-1])
                            vinf_arr_vec = v2_last - v_last_planet_kms
                            last_vinf = float(np.linalg.norm(vinf_arr_vec))

                            row = {
                                "campaign": camp_name,
                                "chain_str": chain,
                                "seed_epoch_iso": sol["seed_epoch"].utc.strftime("%Y-%m-%dT%H:%M:%S"),
                                "tof_combo_days": ";".join([f"{t:.2f}" for t in sol["tofs"]]),
                                "total_tof_days": sol["total_tof"],
                                "first_leg_vinf_kms": sol["first_vinf"],
                                "last_leg_vinf_kms": last_vinf,
                                "first_leg_C3_km2s2": sol["first_C3"],
                                "leg_v1_vectors_json": vectors_to_json(sol["leg_v1"]),
                                "leg_v2_vectors_json": vectors_to_json(sol["leg_v2"]),
                                "notes": ""
                            }
                            rows_buffer.append(row)
                        except Exception as e:
                            # continue; don't crash whole run
                            print("Row creation failed:", e)
                            traceback.print_exc()
                            continue

                        # flush buffer occasionally
                        if len(rows_buffer) >= AUTOSAVE_EVERY_ROWS or (time.time() - last_save_time) > SAVE_EVERY_SECONDS:
                            pd.DataFrame(rows_buffer).to_csv(OUT_CSV, mode="a", header=False, index=False)
                            written_rows += len(rows_buffer)
                            rows_buffer = []
                            ck["per_chain_last_seed"][chain] = seed_idx
                            save_checkpoint_safe(CHECKPOINT_JSON, ck)
                            last_save_time = time.time()
                            print(f"Autosaved {written_rows} rows (chain {chain} seed {seed_idx})")

                    # mark this seed processed for this chain
                    ck["per_chain_last_seed"][chain] = seed_idx
                    # periodic checkpoint
                    if (time.time() - last_save_time) > SAVE_EVERY_SECONDS:
                        save_checkpoint_safe(CHECKPOINT_JSON, ck)
                        last_save_time = time.time()

                # update progress bar
                pbar_chains.update(1)
            pbar_chains.close()

        # final flush
        if rows_buffer:
            pd.DataFrame(rows_buffer).to_csv(OUT_CSV, mode="a", header=False, index=False)
            written_rows += len(rows_buffer)
            rows_buffer = []
        save_checkpoint_safe(CHECKPOINT_JSON, ck)
        print(f"Finished. total_written_rows: {written_rows}. Outputs:\n  {OUT_CSV}\n  {CHECKPOINT_JSON}")

    except KeyboardInterrupt:
        print("Interrupted by user: flushing & saving checkpoint...")
        if rows_buffer:
            pd.DataFrame(rows_buffer).to_csv(OUT_CSV, mode="a", header=False, index=False)
        save_checkpoint_safe(CHECKPOINT_JSON, ck)
    except Exception as e:
        print("Exception in run():", e)
        traceback.print_exc()
        if rows_buffer:
            pd.DataFrame(rows_buffer).to_csv(OUT_CSV, mode="a", header=False, index=False)
        save_checkpoint_safe(CHECKPOINT_JSON, ck)

if __name__ == "__main__":
    ensure_out_csv_header(OUT_CSV)
    run()
