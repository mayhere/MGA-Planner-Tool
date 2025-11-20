# Tisserand_Tree_Search.py
# Rigorous Tisserand Tree Search (fast) with autosave & resume checkpointing.
# Based on your original Tisserand_Tree_Fast.py with added checkpoint + autosave logic.

import os, math, itertools, time, json, traceback
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from poliastro.bodies import Sun
from poliastro.twobody import Orbit
from tqdm import tqdm

# ---------------- USER CONFIG (tune these) ----------------
OUTDIR = r"C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/0_Tisserand_Chain_Finder"
os.makedirs(OUTDIR, exist_ok=True)
OUT_CSV = os.path.join(OUTDIR, "2026-27_Tisserand_Chains_.csv")
LOG_CSV = os.path.join(OUTDIR, "2026-27_Tisserand_Chains_Log.csv")
CHECKPOINT_JSON = os.path.join(OUTDIR, "2026-27_Tisserand_Tree_Checkpoint.json")

# chain generation limits
MIN_FLYBYS = 4
MAX_FLYBYS = 5

# resonance caps
MAX_RES_OTHER = 3
MAX_RES_MERCURY = 4

# Tisserand search parameters (fast defaults — change to explore)
T_TOL = 0.03
N_VINF_DIRECTIONS = 72
MAX_BRANCHES_PER_LEG = 3
SKIP_DIRECT_EA_ME = True
REQUIRE_FIRST = None  # or None

# caching & autosave
AUTOSAVE_EVERY = 300       # autosave after this many survivors collected
SAVE_EVERY_SECONDS = 210  # autosave every X seconds
MAX_CHAINS_TO_TRY = None  # None => try all generated chains; set small int for quick tests

# Campaign windows
CAMPAIGN_WINDOWS = {
    "2026-27": (Time("2026-01-01", scale="tdb"), Time("2027-12-31", scale="tdb")),
    #"2033-34": (Time("2033-01-01", scale="tdb"), Time("2034-12-31", scale="tdb")),
}

# Seed step for sweeping across a campaign window (days)
SEED_STEP_DAYS = 30  # days

# planets & constants
AU = 149597870.7  # km
SMAS_AU = {"earth":1.00000011, "venus":0.723321, "mercury":0.387099, "mars":1.523679}
SMAS_KM = {k: SMAS_AU[k]*AU for k in SMAS_AU}
MU_SUN = Sun.k.to(u.km**3/u.s**2).value

# Ephemeris setup either JPL (preferred) or builtin
try:
    solar_system_ephemeris.set("C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/kernels/de440.bsp")
except Exception:
    solar_system_ephemeris.set("builtin")
    print("Warning: JPL ephemeris unavailable, using builtin (lower precision).")

# bodies & labels
FLYBY_BODIES = ["Ea","Ve","Me","Ma"]
LABELS = {"Ea":"earth","Ve":"venus","Me":"mercury","Ma":"mars"}

# ---------------- helper definitions (copied & slightly adjusted) ----------------
def check_resonance_cap(chain_nodes):
    counts = {b:0 for b in FLYBY_BODIES}
    for n in chain_nodes[1:]:
        counts[n] = counts.get(n,0)+1
    for b in ["Ve","Ea","Ma"]:
        if counts.get(b,0) > MAX_RES_OTHER:
            return False
    if counts.get("Me",0) > MAX_RES_MERCURY:
        return False
    return True

def generate_chains(min_fb, max_fb):
    chains=[]
    intermediates = ["Ve","Me","Ma","Ea"]
    for fb in range(min_fb, max_fb+1):
        for seq in itertools.product(intermediates, repeat=fb):
            nodes = ["Ea"] + list(seq)
            if nodes[-1] != "Me":
                continue
            if not check_resonance_cap(nodes):
                continue
            chains.append(nodes)
    return chains

def hohmann_vinf_kms(from_code, to_code):
    r1 = SMAS_KM[LABELS[from_code]]
    r2 = SMAS_KM[LABELS[to_code]]
    a_t = 0.5*(r1+r2)
    v_tr = math.sqrt(MU_SUN*(2.0/r1 - 1.0/a_t))
    v_c = math.sqrt(MU_SUN/r1)
    return abs(v_tr - v_c)

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

# caches
_planet_state_cache = {}
_orbit_cache = {}

def get_planet_state_cached(body_label, epoch):
    key = (body_label, epoch.utc.iso)
    if key in _planet_state_cache:
        return _planet_state_cache[key]
    r,v = get_body_rv_heliocentric(body_label, epoch)
    r_q, v_q = r.to(u.km), v.to(u.km/u.s)
    _planet_state_cache[key] = (r_q, v_q)
    return r_q, v_q

def orbit_cache_key(body_label, epoch_iso, theta_idx, vinf_mag):
    return f"{body_label}|{epoch_iso}|{theta_idx}|{round(float(vinf_mag),6)}"

def orbit_from_body_plus_vinf_cached(body_label, epoch, vinf_vec_kms, theta_idx):
    vinf_mag = float(np.linalg.norm(vinf_vec_kms))
    key = orbit_cache_key(body_label, epoch.utc.iso, theta_idx, vinf_mag)
    if key in _orbit_cache:
        return _orbit_cache[key]
    r_planet, v_planet = get_planet_state_cached(body_label, epoch)
    v_sc = v_planet + (vinf_vec_kms * u.km/u.s)
    try:
        orb = Orbit.from_vectors(Sun, r_planet, v_sc, epoch=epoch)
        _orbit_cache[key] = orb
        return orb
    except Exception:
        _orbit_cache[key] = None
        return None

def compute_tisserand(orbit, planet_sma_km):
    try:
        a = orbit.a.to(u.km).value
        e = float(orbit.ecc.value)
        inc = float(orbit.inc.to(u.rad).value)
        ap = float(planet_sma_km)
        return ap/a + 2.0*math.cos(inc) * math.sqrt((a*(1-e*e))/ap)
    except Exception:
        return float("nan")

def sample_unit_dirs(n):
    thetas = np.linspace(0, 2*math.pi, n, endpoint=False)
    return [(math.cos(t), math.sin(t), 0.0) for t in thetas]

def expand_leg(incoming_parents, dep_code, arr_code, epoch, n_dirs, max_branches, T_tol):
    out_candidates = []
    sma_arr = SMAS_KM[LABELS[arr_code]]
    dirs = sample_unit_dirs(n_dirs)
    for parent in incoming_parents:
        T_before = parent.get("T_for_next", None)
        if T_before is None:
            parent_orb = parent["orbit"]
            if parent_orb is None:
                T_before = 3.0
            else:
                T_before = compute_tisserand(parent_orb, sma_arr)
        vinf_mag = hohmann_vinf_kms(dep_code, arr_code)
        for t_idx, d in enumerate(dirs):
            vinf_vec = np.array(d) * vinf_mag
            orb_after = orbit_from_body_plus_vinf_cached(LABELS[arr_code], epoch, vinf_vec, t_idx)
            if orb_after is None:
                continue
            T_after = compute_tisserand(orb_after, sma_arr)
            deltaT = abs(T_after - T_before) if (not math.isnan(T_after) and not math.isnan(T_before)) else float("inf")
            out_candidates.append({
                "parent_id": parent["id"],
                "orbit": orb_after,
                "vinf_vec": vinf_vec,
                "vinf_mag": vinf_mag,
                "T_after": T_after,
                "T_before": T_before,
                "deltaT": deltaT,
                "theta_idx": t_idx
            })
    if not out_candidates:
        return []
    out_candidates.sort(key=lambda x: x["deltaT"])
    grouped = {}
    for c in out_candidates:
        grouped.setdefault(c["parent_id"], []).append(c)
    selected = []
    for pid, lst in grouped.items():
        selected.extend(lst[:max_branches])
    selected.sort(key=lambda x: x["deltaT"])
    selected = selected[: max_branches * len(incoming_parents) ]
    next_parents = []
    for i, s in enumerate(selected):
        nb = {
            "id": f"b_{int(time.time()*1000)}_{i}",
            "orbit": s["orbit"],
            "vinf_mag": s["vinf_mag"],
            "T_for_next": s["T_after"],
            "deltaT": s["deltaT"]
        }
        next_parents.append(nb)
    return next_parents

# ---------------- seed generator ----------------
def generate_seeds_for_window(start_time: Time, end_time: Time, step_days: float):
    step = TimeDelta(step_days, format='jd')
    seeds=[]
    cur = start_time
    while cur <= end_time:
        seeds.append(cur)
        cur = cur + step
    return seeds

# ---------------- checkpoint helpers ----------------
def load_checkpoint(path):
    if os.path.exists(path):
        try:
            with open(path,"r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_checkpoint(path, ck):
    tmp = path + ".tmp"
    with open(tmp,"w") as f:
        json.dump(ck, f, indent=2)
    os.replace(tmp, path)

# ---------------- main campaign runner (with autosave/resume) ----------------
def run_campaign_sweep_with_resume(campaign_name, window_tuple, chains, seed_step_days=SEED_STEP_DAYS):
    ck = load_checkpoint(CHECKPOINT_JSON)
    # checkpoint format: { campaign_name: { chain_str: last_seed_index_processed (int) }, ... , "last_saved": timestamp }
    if campaign_name not in ck:
        ck[campaign_name] = {}

    start, end = window_tuple
    seeds = generate_seeds_for_window(start, end, seed_step_days)
    print(f"Campaign {campaign_name}: seeds count = {len(seeds)} ({seeds[0].iso} -> {seeds[-1].iso})")

    survivors = []
    logs = []
    last_save_time = time.time()
    save_counter = 0

    # outer loop: seeds
    for seed_idx, seed_epoch in enumerate(seeds):
        # progress bar across chains for this seed (but skip chains already processed via checkpoint)
        with tqdm(total=len(chains), desc=f"{campaign_name} seed {seed_idx+1}/{len(seeds)} {seed_epoch.utc.iso}", unit="chain") as pbar:
            for ch_idx, nodes in enumerate(chains):
                chain_label = "--".join(nodes)
                # resume check for this chain in this campaign
                last_done_idx = ck[campaign_name].get(chain_label, -1)  # last seed index processed
                if seed_idx <= last_done_idx:
                    pbar.update(1)
                    continue

                # chain filtering
                if SKIP_DIRECT_EA_ME and len(nodes) > 1 and nodes[1] == "Me":
                    ck[campaign_name][chain_label] = seed_idx
                    pbar.update(1); continue
                if REQUIRE_FIRST and nodes[1] != REQUIRE_FIRST:
                    ck[campaign_name][chain_label] = seed_idx
                    pbar.update(1); continue

                # initial parents (sampleed directions)
                initial_vinf = hohmann_vinf_kms(nodes[0], nodes[1])
                unit_dirs = sample_unit_dirs(N_VINF_DIRECTIONS)
                parents = []
                for idx_dir, d in enumerate(unit_dirs[:MAX_BRANCHES_PER_LEG]):
                    vinf_vec = np.array(d) * initial_vinf
                    orb = orbit_from_body_plus_vinf_cached(LABELS[nodes[1]], seed_epoch, vinf_vec, idx_dir)
                    if orb is None:
                        continue
                    parents.append({"id": f"p_{ch_idx}_{idx_dir}", "orbit": orb, "vinf_mag": initial_vinf, "T_for_next": None})
                if not parents:
                    logs.append({"campaign": campaign_name, "seed": seed_epoch.utc.iso, "chain": chain_label, "status":"NO_INIT"})
                    # mark this seed as done for chain (so we don't try again)
                    ck[campaign_name][chain_label] = seed_idx
                    pbar.update(1); continue

                failed = False
                for leg_i in range(len(nodes)-1):
                    dep = nodes[leg_i]; arr = nodes[leg_i+1]
                    nxt = expand_leg(parents, dep, arr, seed_epoch, N_VINF_DIRECTIONS, MAX_BRANCHES_PER_LEG, T_TOL)
                    if not nxt:
                        failed = True
                        break
                    parents = nxt

                if not failed and parents:
                    for br in parents[:MAX_BRANCHES_PER_LEG]:
                        survivors.append({
                            "campaign": campaign_name,
                            "seed_epoch": seed_epoch.utc.iso,
                            "chain_str": chain_label,
                            "first_vinf_kms": float(hohmann_vinf_kms(nodes[0], nodes[1])),
                            "first_leg_C3_km2s2": float(hohmann_vinf_kms(nodes[0], nodes[1])**2),
                            "sum_deltaT": float(sum([br.get("deltaT",0.0)])),
                            "legs_summary": f"final_branches={len(parents)}"
                        })
                    logs.append({"campaign": campaign_name, "seed": seed_epoch.utc.iso, "chain": chain_label, "status":"SURVIVED", "count":len(parents)})
                else:
                    logs.append({"campaign": campaign_name, "seed": seed_epoch.utc.iso, "chain": chain_label, "status":"FAILED"})

                # mark chain as processed for this seed index
                ck[campaign_name][chain_label] = seed_idx

                pbar.update(1)

                # autosave triggers (count or time)
                save_counter += 1
                if (len(survivors) >= AUTOSAVE_EVERY) or (time.time() - last_save_time) > SAVE_EVERY_SECONDS:
                    try:
                        if survivors:
                            pd.DataFrame(survivors).to_csv(OUT_CSV, index=False)
                        if logs:
                            pd.DataFrame(logs).to_csv(LOG_CSV, index=False)
                        ck["last_saved"] = time.time()
                        save_checkpoint(CHECKPOINT_JSON, ck)
                        print(f"Autosaved: survivors={len(survivors)} logs={len(logs)} checkpoint updated")
                    except Exception as e:
                        print("Autosave failed:", e)
                        traceback.print_exc()
                    last_save_time = time.time()
                    save_counter = 0

            # end for chains in seed
        # end progress bar for seed

    # final save
    try:
        if survivors:
            df_surv = pd.DataFrame(survivors)
            df_surv.sort_values(by=["first_leg_C3_km2s2","sum_deltaT"], inplace=True)
            df_surv.to_csv(OUT_CSV, index=False)
            print("Saved survivors:", OUT_CSV)
        if logs:
            pd.DataFrame(logs).to_csv(LOG_CSV, index=False)
            print("Saved logs:", LOG_CSV)
        ck["last_saved"] = time.time()
        save_checkpoint(CHECKPOINT_JSON, ck)
        print("Final checkpoint saved.")
    except Exception as e:
        print("Final save failed:", e)
        traceback.print_exc()

    return survivors, logs

# ---------------- main ----------------
def main():
    print("Generating candidate chains...")
    chains = generate_chains(MIN_FLYBYS, MAX_FLYBYS)
    print(f"Total chains generated: {len(chains)}")
    if MAX_CHAINS_TO_TRY is not None:
        chains = chains[:MAX_CHAINS_TO_TRY]
        print(f"Limiting to first {len(chains)} chains for quick test")

    all_survivors = []
    all_logs = []

    try:
        for camp_name, window in CAMPAIGN_WINDOWS.items():
            surv, logs = run_campaign_sweep_with_resume(camp_name, window, chains, seed_step_days=SEED_STEP_DAYS)
            all_survivors.extend(surv)
            all_logs.extend(logs)
    except KeyboardInterrupt:
        print("Interrupted by user — saving what we have...")
    except Exception as e:
        print("Main exception:", e)
        traceback.print_exc()
    finally:
        # Ensure final save if any results exist
        try:
            if all_survivors:
                df = pd.DataFrame(all_survivors)
                df.sort_values(by=["first_leg_C3_km2s2","sum_deltaT"], inplace=True)
                df.to_csv(OUT_CSV, index=False)
            if all_logs:
                pd.DataFrame(all_logs).to_csv(LOG_CSV, index=False)
            # save checkpoint
            ck = load_checkpoint(CHECKPOINT_JSON)
            ck["last_saved"] = time.time()
            save_checkpoint(CHECKPOINT_JSON, ck)
            print("Final files + checkpoint saved.")
        except Exception as e:
            print("Final save failed:", e)
            traceback.print_exc()

if __name__ == "__main__":
    main()
