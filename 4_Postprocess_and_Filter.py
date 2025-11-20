# Postprocess_and_Filter.py  (PATCHED)
"""
Stage-B postprocessing & filtering for Lambert harvest output.
Drop-in replacement: robust CSV reading, safer autosave, fallback logic,
dedup helper, diagnostic prints, and optional C3/filters.
"""

import os
import time
import json
import tempfile
import traceback
from math import acos, degrees
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from poliastro.bodies import Sun, Earth, Mercury, Venus, Mars

# Ephemeris setup either JPL (preferred) or builtin
try:
    solar_system_ephemeris.set("C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/kernels/de440.bsp")
except Exception:
    solar_system_ephemeris.set("builtin")
    print("Warning: JPL ephemeris unavailable, using builtin (lower precision).")

# ---------------- USER CONFIG ----------------
STAGEA_CSV = r"C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/A_Lambert_Point_Conics_Solver/2026-27_StageA_Lambert_Point_Conics_Results.csv"
OUT_DIR = r"C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/B_Postprocess_and_Filter"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_ENRICHED = os.path.join(OUT_DIR, "2026-27_StageB_Lambert_Enriched_Full.csv")  # full enriched output (all rows)
OUT_FILTERED = os.path.join(OUT_DIR, "2026-27_StageB_Filtered_Ranked.csv")        # filtered & ranked output
OUT_TOPK = os.path.join(OUT_DIR, "2026-27_StageB_TopK_per_Campaign.csv")          # top-K per campaign (if APPLY_PARETO True)
CHECKPOINT_JSON = os.path.join(OUT_DIR, "2026-27_StageB_StageB_Checkpoint.json")  # checkpoint file

PREVIEW = False
PREVIEW_N = 2000

AUTOSAVE_EVERY_ROWS = 500
SAVE_EVERY_SECONDS = 180

# toggles & thresholds (edit as needed)
APPLY_C3_CAP = True          # If True, we enforce C3 cap (C3_CAP_km2s2)
C3_CAP_km2s2 = 70.0

APPLY_VINF_MATCHING = True
VINF_MATCH_TOL_kms = 1.0

APPLY_BENDING_CHECK = True
MIN_FLYBY_ALT_km = {"earth":200.0, "venus":300.0, "mercury":100.0, "mars":150.0}

APPLY_PERIAPSE_SWEEP = True
APPLY_MOI_CHECK = True
MOI_DV_MAX_kms = 4.0
APPLY_DSM_CAP = True
DSM_MAX_kms = 4.0

APPLY_PARETO = False   # if False, use scalar cost ranking
TOP_K_PER_CAMPAIGN = 50

RE_EVAL_ON_MISSING_VECTORS = True

# --- User tuning: final numeric C3 range used by FILTER stage (only applied if APPLY_C3_FILTER True)
# Set APPLY_C3_FILTER to False to skip absolute C3-range filter and keep more rows.
APPLY_C3_FILTER = True
C3_MIN = 9.0      # km^2/s^2 (exclude suspiciously low C3) - used only if APPLY_C3_FILTER True
C3_MAX = 100.0    # km^2/s^2 - used only if APPLY_C3_FILTER True

# cost weights (after normalization)
W_C3 = 0.1
W_VINF = 0.6
W_DSM = 0.3
BENDING_PENALTY = 200.0    # large penalty for bending_feasible==False

# Parking orbits and constants
EPO_perigee_alt_km = 250.0
EPO_apogee_alt_km  = 23000.0
MPO_periapsis_alt_km = 500.0
MPO_apoapsis_alt_km  = 50000.0

R_EARTH_KM = Earth.R.to(u.km).value
R_MERCURY_KM = Mercury.R.to(u.km).value
R_VENUS_KM = Venus.R.to(u.km).value
R_MARS_KM = Mars.R.to(u.km).value

mu_earth = Earth.k.to(u.km**3/u.s**2).value
mu_mercury = Mercury.k.to(u.km**3/u.s**2).value
mu_venus = Venus.k.to(u.km**3/u.s**2).value
mu_sun = Sun.k.to(u.km**3/u.s**2).value

LABELS = {"Ea":"earth","Ve":"venus","Me":"mercury","Ma":"mars"}

# Stage-A expected columns (adjust if your Stage-A CSV has different names)
COL_CAMPAIGN = "campaign"
COL_CHAIN = "chain_str"
COL_SEED_ISO = "seed_epoch_iso"
COL_TOF_COMBO = "tof_combo_days"
COL_TOTAL_TOF = "total_tof_days"
COL_FIRST_VINF = "first_leg_vinf_kms"
COL_LAST_VINF = "last_leg_vinf_kms"
COL_FIRST_C3 = "first_leg_C3_km2s2"
COL_LEG_V1_JSON = "leg_v1_vectors_json"
COL_LEG_V2_JSON = "leg_v2_vectors_json"

# ---------------- helpers ----------------
def save_json_safe(path, data):
    d = os.path.dirname(path) or "."
    fd_temp, tmp_path = tempfile.mkstemp(prefix="tmp_stageB_", dir=d, text=True)
    try:
        with os.fdopen(fd_temp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        try:
            os.replace(tmp_path, path)
        except Exception:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
    except Exception as e:
        print("Checkpoint write failed:", e)
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

def angle_between_vectors(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    val = np.clip(np.dot(a,b) / denom, -1.0, 1.0)
    return degrees(acos(val))

def vec_norm_kms(vec):
    return float(np.linalg.norm(vec))

def required_periapsis_for_turn(vinf_kms, delta_rad, mu_planet_km3s2):
    s = np.sin(delta_rad/2.0)
    if s <= 0: return np.nan
    e = 1.0 / s
    rp = (e - 1.0) * mu_planet_km3s2 / (vinf_kms**2)
    return rp

def compute_EPO_escape_dv(vinf_dep_kms):
    r_perigee = R_EARTH_KM + EPO_perigee_alt_km
    r_apogee = R_EARTH_KM + EPO_apogee_alt_km
    a_park = (r_perigee + r_apogee) / 2.0
    v_perigee = np.sqrt(mu_earth * (2.0 / r_perigee - 1.0 / a_park))
    v_hyper = np.sqrt(vinf_dep_kms**2 + 2.0 * mu_earth / r_perigee)
    return v_hyper - v_perigee

def compute_MPO_capture_dv(vinf_arr_kms):
    r_peri = R_MERCURY_KM + MPO_periapsis_alt_km
    v_peri_hyp = np.sqrt(vinf_arr_kms**2 + 2.0 * mu_mercury / r_peri)
    r_a = R_MERCURY_KM + MPO_apoapsis_alt_km
    a_parking = (r_peri + r_a) / 2.0
    v_pari = np.sqrt(mu_mercury * (2.0 / r_peri - 1.0 / a_parking))
    return abs(v_peri_hyp - v_pari)

def parse_vectors_json(jstr):
    try:
        if pd.isna(jstr) or jstr is None or str(jstr).strip()=="":
            return None
        data = json.loads(jstr)
        return [np.array(v, dtype=float) for v in data]
    except Exception:
        return None

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

# Full header — include every column we will write
FULL_HEADER = [
    "campaign","chain_str","seed_epoch_iso","tof_combo_days","total_tof_days",
    "first_leg_C3_km2s2","first_leg_vinf_kms","last_leg_vinf_kms",
    "leg_v1_vectors_json","leg_v2_vectors_json",
    "vinf_in_list_kms","vinf_out_list_kms","vinf_mismatch_list_kms",
    "vinf_mismatch_max_kms","vinf_mismatch_mean_kms",
    "DSM_est_total_kms","DSM_est_list_kms",
    "bending_feasible","bending_details_str","bending_req_rp_km_list",
    "min_required_rp_km","min_required_rp_body","min_required_rp_alt_km",
    "dv_escape_EPO_kms","dv_moi_kms","sum_deltaV_kms",
    "DLA_list_deg","RLA_list_deg","notes",
    "pass_c3","pass_vinf_match","pass_bending","pass_dsm","pass_moi",
    "overall_pass"
]

# ---------------- Dedup helper ----------------
def deduplicate_keep_best(df, groupby_cols=("campaign","chain_str","seed_epoch_iso")):
    """
    Deduplicate stage-A rows by group and keep the row with minimum first_leg_C3_km2s2.
    Useful if many identical chains exist with identical seeds or small variations.
    """
    if "first_leg_C3_km2s2" not in df.columns:
        return df
    df2 = df.copy()
    df2["first_leg_C3_km2s2_num"] = pd.to_numeric(df2["first_leg_C3_km2s2"], errors="coerce").fillna(np.inf)
    idx = df2.groupby(list(groupby_cols))["first_leg_C3_km2s2_num"].idxmin().dropna().astype(int)
    return df2.loc[idx].reset_index(drop=True)

# ---------------- main ----------------
def main():
    if not os.path.exists(STAGEA_CSV):
        raise FileNotFoundError("Stage-A CSV not found: "+STAGEA_CSV)

    # read Stge-A robustly
    print("Loading Stage-A CSV (robust read)...")
    try:
        dfA = pd.read_csv(STAGEA_CSV, engine="c", quoting=csv.QUOTE_MINIMAL, on_bad_lines="warn", low_memory=False)
    except TypeError:
        # fallback for older pandas versions (no on_bad_lines)
        dfA = pd.read_csv(STAGEA_CSV, engine="c", quoting=csv.QUOTE_MINIMAL, error_bad_lines=False, warn_bad_lines=True, low_memory=False)

    if PREVIEW:
        dfA = dfA.head(PREVIEW_N)
    total_rows = len(dfA)
    print("Rows to process:", total_rows)

    # prepare enriched file with full header if missing
    if not os.path.exists(OUT_ENRICHED):
        pd.DataFrame(columns=FULL_HEADER).to_csv(OUT_ENRICHED, index=False, quoting=csv.QUOTE_ALL)
        print("Created enriched CSV with full header:", OUT_ENRICHED)

    ck = load_checkpoint(CHECKPOINT_JSON)
    last_idx = ck.get("last_index_processed", -1)

    rows_buffer = []
    last_save_time = time.time()
    written = 0

    pbar = tqdm(range(total_rows), desc="StageB rows", unit="row")
    for idx in pbar:
        pbar.set_postfix({"last_idx": last_idx})
        if idx <= last_idx:
            pbar.update(1)
            continue

        try:
            row = dfA.iloc[idx]
            campaign = row.get(COL_CAMPAIGN,"")
            chain = row.get(COL_CHAIN,"")
            seed_iso = str(row.get(COL_SEED_ISO,""))
            tof_combo_str = str(row.get(COL_TOF_COMBO,""))
            total_tof = row.get(COL_TOTAL_TOF, np.nan)
            try:
                total_tof = float(total_tof)
            except Exception:
                total_tof = np.nan
            try:
                first_vinf = float(row.get(COL_FIRST_VINF, np.nan))
            except Exception:
                first_vinf = np.nan
            try:
                last_vinf = float(row.get(COL_LAST_VINF, np.nan))
            except Exception:
                last_vinf = np.nan
            try:
                first_c3 = float(row.get(COL_FIRST_C3, np.nan))
            except Exception:
                first_c3 = np.nan
            v1_json = row.get(COL_LEG_V1_JSON,"")
            v2_json = row.get(COL_LEG_V2_JSON,"")
            leg_v1 = parse_vectors_json(v1_json)
            leg_v2 = parse_vectors_json(v2_json)
            notes = ""

            # seed time handling robustly
            try:
                seed_time = Time(seed_iso, scale="tdb")
            except Exception:
                try:
                    seed_time = Time(seed_iso, scale="utc")
                except Exception:
                    # if still failing, set placeholder and note it
                    seed_time = Time("2000-01-01T00:00:00", scale="utc")
                    notes += " seed_parse_fail;"

            tof_list = [float(x) for x in tof_combo_str.split(";") if x.strip()!=""]
            nodes = chain.split("--")
            nlegs = max(0, len(nodes)-1)
            if len(tof_list) != nlegs:
                if nlegs>0 and not np.isnan(total_tof):
                    avg = total_tof / nlegs
                    tof_list = [avg]*nlegs
                else:
                    tof_list = [30.0]*nlegs

            vinf_in_list=[]; vinf_out_list=[]
            vinf_mismatch_list=[]
            DLA_list=[]; RLA_list=[]
            bending_req_rp_list=[]; bending_flags=[]
            epoch_leg = seed_time

            for leg_i in range(nlegs):
                dep = nodes[leg_i]; arr = nodes[leg_i+1]
                r_dep, v_dep = get_body_rv_heliocentric(LABELS[dep], epoch_leg)
                epoch_arr = epoch_leg + (tof_list[leg_i]*u.day)
                r_arr, v_arr = get_body_rv_heliocentric(LABELS[arr], epoch_arr)

                if leg_v1 is None or leg_v2 is None or leg_i >= len(leg_v1) or leg_i >= len(leg_v2):
                    v1_vec = np.zeros(3); v2_vec = np.zeros(3)
                else:
                    v1_vec = np.array(leg_v1[leg_i])
                    v2_vec = np.array(leg_v2[leg_i])

                v_planet_dep = np.array(v_dep.to(u.km/u.s).value)
                v_planet_arr = np.array(v_arr.to(u.km/u.s).value)
                vinf_in = vec_norm_kms(v1_vec - v_planet_dep)
                vinf_out = vec_norm_kms(v2_vec - v_planet_arr)
                vinf_in_list.append(vinf_in); vinf_out_list.append(vinf_out)

                r_dep_km = np.array(r_dep.to(u.km).value)
                r_arr_km = np.array(r_arr.to(u.km).value)
                dla = angle_between_vectors(r_dep_km, v1_vec)
                rla = angle_between_vectors(r_arr_km, v2_vec)
                DLA_list.append(dla); RLA_list.append(rla)

                epoch_leg = epoch_arr

            # mismatches & bending (for inter-leg nodes)
            for leg_i in range(max(0,nlegs-1)):
                # compute planet epoch at that arrival
                epoch_arrival = seed_time
                for t in tof_list[:leg_i+1]:
                    epoch_arrival = epoch_arrival + (t*u.day)
                r_p, v_p = get_body_rv_heliocentric(LABELS[nodes[leg_i+1]], epoch_arrival)
                v_planet_kms = np.array(v_p.to(u.km/u.s).value)

                vinf_out_vec = (np.array(leg_v2[leg_i]) - v_planet_kms) if (leg_v2 is not None and leg_i < len(leg_v2)) else np.zeros(3)
                vinf_next_in_vec = (np.array(leg_v1[leg_i+1]) - v_planet_kms) if (leg_v1 is not None and (leg_i+1) < len(leg_v1)) else np.zeros(3)
                mismatch_vec = vinf_next_in_vec - vinf_out_vec
                mismatch_mag = vec_norm_kms(mismatch_vec)
                vinf_mismatch_list.append(mismatch_mag)

                turn_deg = angle_between_vectors(vinf_out_vec, vinf_next_in_vec)
                delta_rad = np.deg2rad(turn_deg)
                vinf_ref = vinf_out_list[leg_i] if leg_i < len(vinf_out_list) else max(vinf_out_list) if len(vinf_out_list)>0 else 0.0
                pl = nodes[leg_i+1]
                if pl=="Ea": mu_pl=mu_earth; Rpl=R_EARTH_KM; min_alt=MIN_FLYBY_ALT_km["earth"]
                elif pl=="Ve": mu_pl=mu_venus; Rpl=R_VENUS_KM; min_alt=MIN_FLYBY_ALT_km["venus"]
                elif pl=="Me": mu_pl=mu_mercury; Rpl=R_MERCURY_KM; min_alt=MIN_FLYBY_ALT_km["mercury"]
                elif pl=="Ma": mu_pl=42828.0; Rpl=R_MARS_KM; min_alt=MIN_FLYBY_ALT_km["mars"]
                else: mu_pl=mu_earth; Rpl=R_EARTH_KM; min_alt=200.0

                rp_req = required_periapsis_for_turn(vinf_ref, delta_rad, mu_pl)
                bending_req_rp_list.append(rp_req)
                if np.isfinite(rp_req):
                    alt_req = rp_req - Rpl
                    feasible = (alt_req >= min_alt)
                else:
                    alt_req = np.nan; feasible=False
                bending_flags.append(feasible)

            # pad last entries to keep lengths consistent
            if nlegs>0:
                vinf_mismatch_list.append(np.nan)
                bending_req_rp_list.append(np.nan)
                bending_flags.append(True)

            DSM_list = [0.0 if (isinstance(x,float) and np.isnan(x)) else float(x) for x in vinf_mismatch_list]
            DSM_total = sum(DSM_list)
            dv_escape = compute_EPO_escape_dv(first_vinf if not np.isnan(first_vinf) else 0.0)
            dv_moi = compute_MPO_capture_dv(last_vinf if not np.isnan(last_vinf) else 0.0)
            bending_feasible = all(bending_flags) if len(bending_flags)>0 else True

            pass_c3 = (not APPLY_C3_CAP) or (not np.isnan(first_c3) and first_c3 <= C3_CAP_km2s2)
            if APPLY_VINF_MATCHING:
                mm = [x for x in vinf_mismatch_list if not (isinstance(x,float) and np.isnan(x))]
                pass_vinf = (len(mm)>0) and all([m <= VINF_MATCH_TOL_kms+1e-8 for m in mm])
            else:
                pass_vinf = True
            pass_bending = (not APPLY_BENDING_CHECK) or bending_feasible
            pass_dsm = (not APPLY_DSM_CAP) or (DSM_total <= DSM_MAX_kms)
            pass_moi = (not APPLY_MOI_CHECK) or (dv_moi <= MOI_DV_MAX_kms)
            overall_pass = pass_c3 and pass_vinf and pass_bending and pass_dsm and pass_moi

            enriched_row = {
                "campaign": campaign, "chain_str": chain,
                "seed_epoch_iso": seed_time.utc.strftime("%Y-%m-%dT%H:%M:%S"),
                "tof_combo_days": ";".join([f"{t:.2f}" for t in tof_list]),
                "total_tof_days": total_tof,
                "first_leg_C3_km2s2": first_c3, "first_leg_vinf_kms": first_vinf, "last_leg_vinf_kms": last_vinf,
                "leg_v1_vectors_json": v1_json, "leg_v2_vectors_json": v2_json,
                "vinf_in_list_kms": ";".join([f"{v:.4f}" for v in vinf_in_list]),
                "vinf_out_list_kms": ";".join([f"{v:.4f}" for v in vinf_out_list]),
                "vinf_mismatch_list_kms": ";".join([("" if (isinstance(v,float) and np.isnan(v)) else f"{v:.4f}") for v in vinf_mismatch_list]),
                "vinf_mismatch_max_kms": float(np.nanmax([v for v in vinf_mismatch_list if not (isinstance(v,float) and np.isnan(v))])) if any([not (isinstance(v,float) and np.isnan(v)) for v in vinf_mismatch_list]) else np.nan,
                "vinf_mismatch_mean_kms": float(np.nanmean([v for v in vinf_mismatch_list if not (isinstance(v,float) and np.isnan(v))])) if any([not (isinstance(v,float) and np.isnan(v)) for v in vinf_mismatch_list]) else np.nan,
                "DSM_est_total_kms": DSM_total, "DSM_est_list_kms": ";".join([f"{v:.4f}" for v in DSM_list]),
                "bending_feasible": bending_feasible,
                "bending_details_str": ";".join([f"{nodes[i+1]}:rp_req={('nan' if np.isnan(r) else f'{r:.1f}')}" for i,r in enumerate(bending_req_rp_list)]),
                "bending_req_rp_km_list": ";".join([("" if (isinstance(r,float) and np.isnan(r)) else f"{r:.3f}") for r in bending_req_rp_list]),
                "min_required_rp_km": float(np.nanmin([r for r in bending_req_rp_list if not (isinstance(r,float) and np.isnan(r))])) if any([not (isinstance(r,float) and np.isnan(r)) for r in bending_req_rp_list]) else np.nan,
                "min_required_rp_body": (nodes[1 + int(np.nanargmin([r if not (isinstance(r,float) and np.isnan(r)) else np.inf for r in bending_req_rp_list]))] if any([not (isinstance(r,float) and np.isnan(r)) for r in bending_req_rp_list]) else ""),
                "min_required_rp_alt_km": ((float(np.nanmin([r for r in bending_req_rp_list if not (isinstance(r,float) and np.isnan(r))])) - R_MERCURY_KM) if any([not (isinstance(r,float) and np.isnan(r)) for r in bending_req_rp_list]) else np.nan),
                "dv_escape_EPO_kms": dv_escape, "dv_moi_kms": dv_moi, "sum_deltaV_kms": (DSM_total + dv_escape + dv_moi),
                "DLA_list_deg": ";".join([f"{d:.3f}" for d in DLA_list]),
                "RLA_list_deg": ";".join([f"{r:.3f}" for r in RLA_list]),
                "notes": notes,
                "pass_c3": pass_c3, "pass_vinf_match": pass_vinf, "pass_bending": pass_bending,
                "pass_dsm": pass_dsm, "pass_moi": pass_moi, "overall_pass": overall_pass
            }

            rows_buffer.append(enriched_row)

            # Autosave buffer to enriched CSV (append mode). Use safe tempfile replace.
            if len(rows_buffer) >= AUTOSAVE_EVERY_ROWS or (time.time() - last_save_time) > SAVE_EVERY_SECONDS:
                df_buf = pd.DataFrame(rows_buffer, columns=FULL_HEADER)
                # append safely
                tmp = OUT_ENRICHED + ".tmp"
                try:
                    # append to file (we want to preserve existing header)
                    df_buf.to_csv(tmp, mode="w", header=False, index=False, quoting=csv.QUOTE_ALL)
                    # now append tmp content to file using binary-safe append
                    with open(tmp, "rb") as fr, open(OUT_ENRICHED, "ab") as fw:
                        fw.write(fr.read())
                    os.remove(tmp)
                except Exception:
                    # fallback direct append (less safe)
                    df_buf.to_csv(OUT_ENRICHED, mode="a", header=False, index=False, quoting=csv.QUOTE_ALL)
                written += len(rows_buffer)
                rows_buffer = []
                ck["last_index_processed"] = idx
                save_json_safe(CHECKPOINT_JSON, ck)
                last_save_time = time.time()
                print(f"Autosaved {written} enriched rows; last idx {idx}")

            ck["last_index_processed"] = idx

        except Exception as e:
            print(f"Error at idx {idx}: {e}")
            traceback.print_exc()
            errrow = {k: "" for k in FULL_HEADER}
            errrow.update({
                "campaign": row.get(COL_CAMPAIGN,"") if 'row' in locals() else "",
                "chain_str": row.get(COL_CHAIN,"") if 'row' in locals() else "",
                "seed_epoch_iso": row.get(COL_SEED_ISO,"") if 'row' in locals() else "",
                "tof_combo_days": row.get(COL_TOF_COMBO,"") if 'row' in locals() else "",
                "total_tof_days": row.get(COL_TOTAL_TOF,"") if 'row' in locals() else "",
                "first_leg_C3_km2s2": row.get(COL_FIRST_C3, np.nan) if 'row' in locals() else np.nan,
                "first_leg_vinf_kms": row.get(COL_FIRST_VINF, np.nan) if 'row' in locals() else np.nan,
                "notes": f"ERROR: {e}", "overall_pass": False
            })
            rows_buffer.append(errrow)
            if len(rows_buffer) >= AUTOSAVE_EVERY_ROWS:
                pd.DataFrame(rows_buffer, columns=FULL_HEADER).to_csv(
                    OUT_ENRICHED, mode="a", header=False, index=False, quoting=csv.QUOTE_ALL)
                written += len(rows_buffer)
                rows_buffer=[]
                ck["last_index_processed"] = idx
                save_json_safe(CHECKPOINT_JSON, ck)
                last_save_time = time.time()

        pbar.update(1)

    # flush buffer
    if rows_buffer:
        pd.DataFrame(rows_buffer, columns=FULL_HEADER).to_csv(
                    OUT_ENRICHED, mode="a", header=False, index=False, quoting=csv.QUOTE_ALL)
        written += len(rows_buffer)
        rows_buffer = []
    save_json_safe(CHECKPOINT_JSON, ck)
    pbar.close()
    print(f"StageB enrichment done. Rows written: {written}. Output: {OUT_ENRICHED}")

    # ---------------- Postprocessing: load enriched and apply fallback if needed ----------------
    print("Loading enriched CSV for filtering & ranking...")
    # use robust read for enriched (handle weird rows)
    try:
        dfE = pd.read_csv(OUT_ENRICHED, engine="c", on_bad_lines="warn", low_memory=False)
    except TypeError:
        dfE = pd.read_csv(OUT_ENRICHED, engine="c", error_bad_lines=False, warn_bad_lines=True, low_memory=False)

    # If overall_pass missing, compute from pass_* columns (robust conversion)
    if "overall_pass" not in dfE.columns:
        pass_cols = [c for c in dfE.columns if c.startswith("pass_")]
        if len(pass_cols)==0:
            dfE["overall_pass"] = True
        else:
            for c in pass_cols:
                # convert many textual variants to boolean
                dfE[c] = dfE[c].astype(str).str.strip().str.lower().isin(["true","1","t","yes","y"])
            dfE["overall_pass"] = dfE[pass_cols].all(axis=1)
        print("Computed overall_pass fallback from pass_* columns.")

    # Ensure overall_pass is boolean
    dfE["overall_pass_bool"] = dfE["overall_pass"].astype(str).str.strip().str.lower().isin(["true","1","t","yes","y"])

    # By default, filter only those rows that explicitly passed the coarse checks
    df_filtered = dfE[dfE["overall_pass_bool"]==True].copy()
    print("Filtered survivors count (overall_pass==True):", len(df_filtered))

    # If nothing survived, we will relax and collect candidates (diagnostic mode)
    if len(df_filtered) == 0:
        print("WARNING: No rows marked overall_pass==True. Falling back to consider all rows (for diagnostics).")
        df_filtered = dfE.copy()

    # Fill missing expected numeric columns with NaN
    for col in ["first_leg_C3_km2s2", "last_leg_vinf_kms", "DSM_est_total_kms", "total_tof_days"]:
        if col not in df_filtered.columns:
            df_filtered[col] = np.nan

    # Normalize/clean bending_feasible into boolean
    if "bending_feasible" in df_filtered.columns:
        df_filtered["bending_feasible"] = df_filtered["bending_feasible"].apply(
            lambda x: bool(x) if not pd.isna(x) and (not isinstance(x, str) or x.strip().lower()!="false") else False
        )
    else:
        df_filtered["bending_feasible"] = False

    # Convert numeric columns safely
    df_filtered["first_leg_C3_km2s2"] = pd.to_numeric(df_filtered["first_leg_C3_km2s2"], errors="coerce")
    df_filtered["last_leg_vinf_kms"] = pd.to_numeric(df_filtered["last_leg_vinf_kms"], errors="coerce")
    df_filtered["DSM_est_total_kms"] = pd.to_numeric(df_filtered.get("DSM_est_total_kms", pd.Series(np.nan)), errors="coerce")

    # --- Optional C3 range filter (applied only if user requested) ---
    if APPLY_C3_FILTER:
        before_count = len(df_filtered)
        mask_c3 = df_filtered["first_leg_C3_km2s2"].between(C3_MIN, C3_MAX, inclusive="both")
        df_filtered = df_filtered[mask_c3].copy()
        after_count = len(df_filtered)
        print(f"C3 filter: kept {after_count} / {before_count} rows (range {C3_MIN}-{C3_MAX})")
    else:
        print("C3 range filter skipped (APPLY_C3_FILTER=False).")

    # If after filtering we are empty, produce diagnostics and keep the best N by simple proxy
    if len(df_filtered) == 0:
        print("No rows survived C3/overall filters. Producing diagnostic outputs and selecting top candidates by first-leg C3.")
        # save a repaired numeric CSV for examination
        repaired_path = os.path.join(OUT_DIR, "Lambert_Enriched_Full_REPAIRED.csv")
        # create numeric columns if missing and write
        tmpdf = dfE.copy()
        tmpdf["first_leg_C3_km2s2_num"] = pd.to_numeric(tmpdf["first_leg_C3_km2s2"], errors="coerce")
        tmpdf.to_csv(repaired_path, index=False, quoting=csv.QUOTE_ALL)
        print("Wrote repaired CSV (all rows, with parsed numeric columns) to:", repaired_path)
        # Attempt to salvage some rows by picking top 100 rows with smallest numeric C3 (non-NaN)
        tmpdf2 = tmpdf[tmpdf["first_leg_C3_km2s2_num"].notna()].sort_values(by="first_leg_C3_km2s2_num").head(100)
        if not tmpdf2.empty:
            df_filtered = tmpdf2.copy()
            print("Recovered", len(df_filtered), "rows by picking smallest numeric first_leg_C3.")
        else:
            print("No numeric C3 values available; exiting with empty filtered set.")
            # create empty Filtered file and exit gracefully
            pd.DataFrame(columns=FULL_HEADER).to_csv(OUT_FILTERED, index=False, quoting=csv.QUOTE_ALL)
            print("Wrote empty filtered file:", OUT_FILTERED)
            return

    # --- Compute ranking: either Pareto or scalar cost ---
    def safe_percentile(series, q=75):
        s = pd.to_numeric(series, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
        if len(s) < 10:
            if series.name == "first_leg_C3_km2s2": return 100.0
            if series.name == "last_leg_vinf_kms": return 10.0
            if series.name == "DSM_est_total_kms": return 50.0
            return 1.0
        return np.nanpercentile(s, q)

    scale_C3 = safe_percentile(df_filtered["first_leg_C3_km2s2"])
    scale_vinf = safe_percentile(df_filtered["last_leg_vinf_kms"])
    scale_DSM = safe_percentile(df_filtered["DSM_est_total_kms"])
    scale_C3 = max(scale_C3, 1.0)
    scale_vinf = max(scale_vinf, 0.1)
    scale_DSM = max(scale_DSM, 1.0)
    print(f"Auto scales: C3={scale_C3:.3f}, vinf={scale_vinf:.3f}, DSM={scale_DSM:.3f}")

    def scalar_cost_row(row):
        c3 = row.get("first_leg_C3_km2s2", np.nan)
        vinf = row.get("last_leg_vinf_kms", np.nan)
        dsm = row.get("DSM_est_total_kms", np.nan)
        bend_ok = row.get("bending_feasible", True)
        c3 = float(c3) if pd.notna(c3) else np.nan
        vinf = float(vinf) if pd.notna(vinf) else np.nan
        dsm = float(dsm) if pd.notna(dsm) else np.nan
        term_c3 = (c3/scale_C3) if np.isfinite(c3) else 10.0
        term_vinf = (vinf/scale_vinf) if np.isfinite(vinf) else 10.0
        term_dsm = (dsm/scale_DSM) if np.isfinite(dsm) else 10.0
        base = W_C3*term_c3 + W_VINF*term_vinf + W_DSM*term_dsm
        if not bool(bend_ok):
            base += BENDING_PENALTY
        return base

    # ---------- Pareto filtering helper ----------
    def pareto_mask(array):
        N = array.shape[0]
        is_pareto = np.ones(N, dtype=bool)
        for i in range(N):
            if not is_pareto[i]:
                continue
            better = np.all(array <= array[i], axis=1) & np.any(array < array[i], axis=1)
            is_pareto[better] = False
        return is_pareto

    # ---------- Apply Pareto or scalar ranking ----------
    if APPLY_PARETO:
        objs = df_filtered[["first_leg_C3_km2s2","last_leg_vinf_kms","vinf_mismatch_max_kms","DSM_est_total_kms"]].copy()
        objs = objs.replace([np.inf, -np.inf], np.nan).fillna(1e12).values
        try:
            is_pareto = pareto_mask(objs)
            df_filtered["is_pareto"] = is_pareto
            filtered_for_rank = df_filtered[df_filtered["is_pareto"]==True].copy()
            print(f"[Pareto] Kept {len(filtered_for_rank)} pareto rows out of {len(df_filtered)}")
        except Exception as e:
            print("Pareto failed:", e)
            filtered_for_rank = df_filtered.copy()
    else:
        filtered_for_rank = df_filtered.copy()

    filtered_for_rank["cost"] = filtered_for_rank.apply(scalar_cost_row, axis=1)

    # ---------- Save filtered & ranked (sorted by campaign then cost) ----------
    sorted_df = filtered_for_rank.sort_values(by=["campaign","cost","first_leg_C3_km2s2"])
    sorted_df.to_csv(OUT_FILTERED, index=False, quoting=csv.QUOTE_ALL)
    print("Wrote filtered & ranked:", OUT_FILTERED)

    # ---------- Diversified Top-K per campaign ----------
    def select_diverse_topk(df_in, k=5):
        if df_in.empty:
            return df_in.copy()
        picks = []
        used_idx = set()
        best_idx = df_in["cost"].idxmin()
        picks.append(best_idx); used_idx.add(best_idx)
        objectives = ["first_leg_C3_km2s2","last_leg_vinf_kms","DSM_est_total_kms","total_tof_days"]
        for obj in objectives:
            if len(picks) >= k:
                break
            if obj not in df_in.columns:
                continue
            cand = df_in[~df_in.index.isin(used_idx)].sort_values(by=obj, na_position="last")
            if not cand.empty:
                picks.append(cand.index[0]); used_idx.add(cand.index[0])
        if len(picks) < k:
            for idx in df_in.sort_values(by="cost").index:
                if idx in used_idx: continue
                picks.append(idx); used_idx.add(idx)
                if len(picks) >= k: break
        return df_in.loc[picks].copy()

    topk_rows = []
    for camp_name, group in sorted_df.groupby("campaign"):
        topk_df = select_diverse_topk(group, k=TOP_K_PER_CAMPAIGN)
        if not topk_df.empty:
            topk_df = topk_df.copy()
            topk_df["selected_rank_campaign"] = range(1, len(topk_df)+1)
            topk_rows.append(topk_df)

    if len(topk_rows) > 0:
        df_topk = pd.concat(topk_rows, axis=0, ignore_index=False)
        df_topk.to_csv(OUT_TOPK, index=False, quoting=csv.QUOTE_ALL)
        print(f"Wrote Top-{TOP_K_PER_CAMPAIGN} diversified per campaign to:", OUT_TOPK)
    else:
        print("No Top-K rows to write (empty).")

    # ---------- Print summary ----------
    # n_total = len(df_filtered)
    # n_ranked = len(sorted_df)
    # n_topk = len(df_topk) if ('df_topk' in locals()) else 0
    # print(f"Summary: total_rows={n_total}, after_rank={n_ranked}, topk_saved={n_topk}")

    # # some diagnostics
    # print("df_filtered rows:", len(df_filtered))
    # df_filtered["first_leg_C3_km2s2"] = pd.to_numeric(df_filtered["first_leg_C3_km2s2"], errors="coerce")
    # df_filtered["last_leg_vinf_kms"] = pd.to_numeric(df_filtered["last_leg_vinf_kms"], errors="coerce")
    # c3 = df_filtered["first_leg_C3_km2s2"]
    # print("C3: count non-null:", c3.count())
    # try:
    #     print("C3 min, 1pct, 5pct, 25, 50, 75, 95,99, max:")
    #     print(np.nanpercentile(c3.fillna(np.nan).values, [0,1,5,25,50,75,95,99,100]))
    # except Exception:
    #     pass
    # print("C3 NaNs:", c3.isna().sum())

    # # some samples
    # print("\n--- Rows with C3 <= 0.1 (sample 10) ---")
    # print(df_filtered[df_filtered["first_leg_C3_km2s2"] <= 0.1].head(10)[["chain_str","seed_epoch_iso","first_leg_C3_km2s2","first_leg_vinf_kms"]])

    # print("\n--- Rows with 0.1 < C3 < 9 (sample 10) ---")
    # print(df_filtered[(df_filtered["first_leg_C3_km2s2"]>0.1) & (df_filtered["first_leg_C3_km2s2"]<9)].head(10)[["chain_str","seed_epoch_iso","first_leg_C3_km2s2","first_leg_vinf_kms"]])

    # print("\n--- Rows with 9 <= C3 <= 100 (sample 10) ---")
    # ok_mask = (df_filtered["first_leg_C3_km2s2"]>=9) & (df_filtered["first_leg_C3_km2s2"]<=100)
    # print(df_filtered[ok_mask].head(10)[["chain_str","seed_epoch_iso","first_leg_C3_km2s2","first_leg_vinf_kms"]])

    # print("\n--- Rows with C3 > 100 (sample 10) ---")
    # print(df_filtered[df_filtered["first_leg_C3_km2s2"]>100].head(10)[["chain_str","seed_epoch_iso","first_leg_C3_km2s2","first_leg_vinf_kms"]])

    # bins = [ -1, 0, 0.1, 9, 100, 1e9 ]
    # labels = ["<0","0-0.1","0.1-9","9-100",">100"]
    # bcat = pd.cut(df_filtered["first_leg_C3_km2s2"].fillna(-99999), bins=bins, labels=labels)
    # print("\nBucket counts:")
    # print(bcat.value_counts().reindex(labels))

if __name__ == "__main__":
    main()
