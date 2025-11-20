# StageC_Optimization_Adv.py (patched with resilient DE fallback)
# Global optimization (Differential Evolution) for MGA chains (Earth->...->Mercury)
# Produces leg-by-leg Lambert solutions + periapsis sweep, DSM estimates, MOI, EPO Δv, etc.
# Autosave + resume + tqdm progress.
# WARNING: This script can be computationally heavy. Start with small subset (PREVIEW mode).
# Robust stage-C global optimizer (DE) with resilient fallback (parallel -> single-thread -> tiny-DE).
# Robust Stage-C optimizer: lambert point-conics + periapsis sweep + DSM + MOI + autosave/resume + top-K
# Updated to include strict v∞ matching toggle and leg-wise v1/v2 JSON fields.

import os
import sys
import json
import time
import math
import traceback
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel

from poliastro.iod.izzo import lambert
from poliastro.util import norm

from scipy.optimize import differential_evolution, minimize

# ----------------------------
# User-configurable section
# ----------------------------
INPUT_CHAINS_CSV = r"C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/C_Verification_and_Optimization/2026-27_StageC_Top5_per_campaign.csv"
OUT_DIR = r"C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/C_Verification_and_Optimization"
OUT_CSV = os.path.join(OUT_DIR, "2026-27_StageC_Top5_Optimized_Results.csv")
#TOPK_OUT = os.path.join(OUT_DIR, "2026-27_StageC_Top5_per_campaign.csv")
CHECKPOINT_JSON = os.path.join(OUT_DIR, "2026-27_StageC_Top5_Optimization_checkpoint.json")

# Global planetary constants / lookup (km, km^3/s^2)
R_EARTH = 6378.1363  # km (approx)
MU_SUN = 1.32712440018e11  # km^3 / s^2
# Minimal planetary mu table (km^3/s^2)
MU = {
    "earth": 398600.4418,
    "venus": 324858.592,
    "mercury": 22032.080,
    "mars": 42828.375214
}
PLANET_RAD = {
    "earth": 6378.1363,
    "venus": 6052.0,
    "mercury": 2439.7,
    "mars": 3396.2
}

# Tunable config for periapsis sweep and vinf matching
cfg = {
    # campaign name (for output labeling)
    "CAMPAIGN_NAME": "2033-34",

    # lambert seeding & search control
    "GENERIC_TOF_CHOICES": [700.0, 700.0, 700.0, 700.0, 700.0],  # days per-leg seeds as fallback
    "MAX_LEG_TOF_DAYS": 1000.0,

    # per-chain optimization control
    "APPLY_PERIAPSE_SWEEP": True,
    "PERIAPSE_SWEEP_POINTS": 100,
    "PERIAPSE_SWEEP_RP_MAX_km": 5e5,  # absolute max rp in km
    "PERIAPSE_SWEEP_FACTOR": 1.0,  # max rp = factor * rp_required
    "MIN_FLYBY_ALT_km": {"earth":200.0,"venus":300.0,"mercury":100.0,"mars":150.0},

    # v-infinity matching / DSM caps
    "APPLY_STRICT_VINF_MATCH": True,   # NEW: enable strict v∞ matching
    "STRICT_VINF_TOL_kms": 0.3,         # tolerance for strict check
    "VINF_MATCH_TOL_kms": 0.5,
    "VINF_DSM_CAP_kms": 0.7,

    # C3 / launch caps
    "APPLY_C3_CAP": True,
    "C3_CAP_km2s2": 60.0,

    # parking orbits (used to compute escape & MOI delta-v)
    "EPO_perigee_km_from_surface": 250.0,
    "EPO_apogee_km_from_surface": 23000.0,
    "MPO_periapsis_km_from_surface": 500.0,
    "MPO_apoapsis_km_from_surface": 50000.0,

    # thresholds used in filtering / pass/fail
    "DSM_MAX_kms": 1.0,
    "MOI_DV_MAX_kms": 4.0,

    # performance / optimizer
    "DE_POPSIZE": 30,
    "DE_MAXITER": 500,
    "DE_WORKERS": 8,  # use 1 to avoid pickling issues; set >1 with care

    "LOCAL_POLISH": True,

    # autosave
    "AUTOSAVE_EVERY": 1,  # save every N chains

    # debug / preview
    "PREVIEW": False,
    "PREVIEW_N": 1,

    # ------------------ VINFINITY OPTIMIZATION CONFIG (NEW) ------------------
    # (Option A) Toggle to include vinf mismatch directly in the DE objective 
    "APPLY_VINF_IN_OBJECTIVE" : False,       # If True, include vinf mismatch term in objective
    "VINF_WEIGHT" : 1000.0,                  # Weight applied to vinf mismatch term (tune as needed)
    "VINF_POWER" : 3.0,                      # Power applied to vinf mismatch (1 = linear, 2 = squared)

    # (Option B) Toggle to apply strict local polishing that minimizes vinf mismatch subject to DSM/C3 constraints
    "APPLY_VINF_LOCAL_POLISH" : True,        # If True, run the local constrained vinf polish on Top-K candidates
    "VINF_LOCAL_POLISH_TOPK" : 5,            # Run the constrained polish on Top-K candidates selected from DE results
    "VINF_LOCAL_POLISH_MAXITER" : 500,       # Max iterations for local SLSQP polish
    "VINF_LOCAL_POLISH_DSM_CAP_kms" : 0.5,   # DSM limit to preserve during polish (kms)
    "VINF_LOCAL_POLISH_C3_CAP_km2s2" : 60.0,  # Set to large if you don't want to cap C3 (or set reasonable cap)
    # ----------------------------------------------------------------------
}

# ----------------------------
# Helpers
# ----------------------------
def safe_float(x, default=np.nan):
    try:
        return float(x)
    except:
        return default

def normalize_label_to_astropy_name(lbl):
    if lbl is None: return None
    s = str(lbl).strip().lower()
    if s in ("ea","earth","earthbary","terra"): return "earth"
    if s in ("ve","venus"): return "venus"
    if s in ("me","mercury"): return "mercury"
    if s in ("ma","mars"): return "mars"
    return s

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

def get_body_rv(label, epoch):
    """
    Return barycentric position (km) and velocity (km/s) numpy arrays for the given body label at epoch (astropy Time).
    label can be short 'Ea' or 'earth' etc.
    """
    name = normalize_label_to_astropy_name(label)
    if name is None:
        raise ValueError("Bad body label")
    # Use jpl if available for better accuracy; 'builtin' can be used if jpl not present
    try:
        with solar_system_ephemeris.set("C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/kernels/de440.bsp"):
            r, v = get_body_rv_heliocentric(name, epoch)
    except Exception:
        with solar_system_ephemeris.set("builtin"):
            r, v = get_body_rv_heliocentric(name, epoch)
    r_km = np.array(r.to(u.km).value).flatten()
    v_kms = np.array(v.to(u.km/u.s).value).flatten()
    return r_km, v_kms

def solve_lambert(r1_km, r2_km, tof_s, mu=MU_SUN):
    """
    Solve Lambert heliocentric with poliastro. Accept numpy arrays.
    Returns v1q, v2q (astropy Quantities).
    """
    r1 = u.Quantity(r1_km, u.km)
    r2 = u.Quantity(r2_km, u.km)
    tof = u.Quantity(tof_s, u.s)
    v1q, v2q = lambert(mu * u.km**3 / u.s**2, r1, r2, tof)
    return v1q, v2q

def compute_escape_delta_v_perigee(vinf_kms, r_perigee_km, r_apogee_km):
    mu = MU["earth"]
    a_park = 0.5*(r_perigee_km + r_apogee_km)
    v_perigee = math.sqrt(mu * (2.0 / r_perigee_km - 1.0 / a_park))
    v_hyper = math.sqrt(vinf_kms**2 + 2.0 * mu / r_perigee_km)
    return float(v_hyper - v_perigee)

def compute_moi_delta_v(vinf_arr_kms, rp_peri_km, rp_apo_km, mu_planet):
    v_peri_hyp = math.sqrt(vinf_arr_kms**2 + 2.0 * mu_planet / rp_peri_km)
    v_circ = math.sqrt(mu_planet / rp_peri_km)
    return float(max(0.0, v_peri_hyp - v_circ))

# ----------------------------
# Main evaluator with periapsis sweep and strict v∞ option
# ----------------------------
def evaluate_chain_candidate_with_rp_sweep(chain_parts,
                                          seed_epoch_time_or_iso,
                                          x_args_or_array,
                                          first_leg_launch_body=None,
                                          cfg_local=None,
                                          rp_sweep_points=None,
                                          **kwargs):
    """
    Evaluate a chain candidate.
    Returns dict with standard StageC fields (see bottom for keys).
    """
    import json, traceback
    try:
        if cfg_local is None:
            cfg_local = cfg

        # config
        APPLY_PERIAPSE_SWEEP = bool(cfg_local.get("APPLY_PERIAPSE_SWEEP", True))
        PERIAPSE_SWEEP_POINTS = int(cfg_local.get("PERIAPSE_SWEEP_POINTS", 50)) if rp_sweep_points is None else int(rp_sweep_points)
        PERIAPSE_SWEEP_RP_MAX_km = float(cfg_local.get("PERIAPSE_SWEEP_RP_MAX_km", 5e5))
        PERIAPSE_SWEEP_FACTOR = float(cfg_local.get("PERIAPSE_SWEEP_FACTOR", 0.5))
        MIN_FLYBY_ALT_km = cfg_local.get("MIN_FLYBY_ALT_km", {"earth":200.0,"venus":300.0,"mercury":100.0,"mars":150.0})
        VINF_MATCH_TOL_kms = float(cfg_local.get("VINF_MATCH_TOL_kms", 1.0))
        VINF_DSM_CAP_kms = float(cfg_local.get("VINF_DSM_CAP_kms", 1.5))
        APPLY_C3_CAP = bool(cfg_local.get("APPLY_C3_CAP", True))
        C3_CAP_km2s2 = float(cfg_local.get("C3_CAP_km2s2", 70.0))
        APPLY_STRICT_VINF_MATCH = bool(cfg_local.get("APPLY_STRICT_VINF_MATCH", True))
        STRICT_VINF_TOL_kms = float(cfg_local.get("STRICT_VINF_TOL_kms", 0.7))

        # normalize chain parts
        if isinstance(chain_parts, str):
            parts = [p.strip() for p in chain_parts.split("--") if p.strip()]
        else:
            parts = list(chain_parts)

        # seed time
        if isinstance(seed_epoch_time_or_iso, Time):
            seed_time = seed_epoch_time_or_iso
        else:
            try:
                seed_time = Time(str(seed_epoch_time_or_iso), scale="tdb")
            except Exception:
                seed_time = Time(str(seed_epoch_time_or_iso))

        # parse x
        launch_offset_days = 0.0
        tof_list_days = []
        if hasattr(x_args_or_array, "__len__") and not isinstance(x_args_or_array, dict):
            try:
                arr = np.asarray(x_args_or_array, dtype=float)
                if arr.size >= 1:
                    launch_offset_days = float(arr[0])
                    tof_list_days = [float(v) for v in arr[1:]]
            except:
                launch_offset_days = 0.0
        elif isinstance(x_args_or_array, dict):
            launch_offset_days = float(x_args_or_array.get("launch_offset_days", 0.0))
            tof_list_days = [float(v) for v in x_args_or_array.get("tof_list_days", [])]
        else:
            launch_offset_days = 0.0

        nlegs = max(len(parts)-1, 0)
        if len(tof_list_days) < nlegs:
            choices = cfg_local.get("GENERIC_TOF_CHOICES", [20.0,60.0,120.0,200.0])
            while len(tof_list_days) < nlegs:
                tof_list_days.append(float(choices[min(len(tof_list_days), len(choices)-1)]))

        # epochs
        dep0 = seed_time + launch_offset_days * u.day
        leg_dep_epochs = []
        leg_arr_epochs = []
        epoch = dep0
        for tof in tof_list_days:
            leg_dep_epochs.append(epoch)
            epoch = epoch + tof * u.day
            leg_arr_epochs.append(epoch)

        # attempt to reuse leg vectors from kwargs if present
        vinf_out_vecs = kwargs.get("vinf_out_vecs", None)
        vinf_in_vecs  = kwargs.get("vinf_in_vecs", None)
        vinf_out_list = []
        vinf_in_list  = []

        v1_vecs = [None]*nlegs
        v2_vecs = [None]*nlegs

        # if vectors not supplied, compute lambert leg-by-leg
        for i in range(nlegs):
            dep_label = parts[i]; arr_label = parts[i+1]
            try:
                r_dep, v_dep_planet = get_body_rv(dep_label, leg_dep_epochs[i])
                r_arr, v_arr_planet = get_body_rv(arr_label, leg_arr_epochs[i])
            except Exception as e:
                return {"error": f"ephem_failed_leg_{i}: {e}", "chain_str": "--".join(parts), "seed_epoch_iso":seed_time.iso}

            tof_s = (leg_arr_epochs[i] - leg_dep_epochs[i]).to(u.s).value
            if tof_s <= 0:
                return {"error": f"nonpositive_tof_leg_{i}", "chain_str": "--".join(parts), "seed_epoch_iso":seed_time.iso}

            try:
                v1q, v2q = solve_lambert(r_dep, r_arr, tof_s)
            except Exception as e:
                return {"error": f"lambert_failed_leg_{i}: {e}", "chain_str": "--".join(parts), "seed_epoch_iso":seed_time.iso}

            v1 = np.array(v1q.to(u.km/u.s).value).flatten()
            v2 = np.array(v2q.to(u.km/u.s).value).flatten()
            v1_vecs[i] = v1.tolist()
            v2_vecs[i] = v2.tolist()
            vinf_out_vec = v1 - v_dep_planet
            vinf_in_vec  = v2 - v_arr_planet
            vinf_out_list.append(float(np.linalg.norm(vinf_out_vec)))
            vinf_in_list.append(float(np.linalg.norm(vinf_in_vec)))

        # produce vinf vector lists (barycentric heliocentric)
        vinf_out_vecs = []
        vinf_in_vecs = []
        for i in range(nlegs):
            r_dep, v_dep_planet = get_body_rv(parts[i], leg_dep_epochs[i])
            r_arr, v_arr_planet = get_body_rv(parts[i+1], leg_arr_epochs[i])
            v1 = np.array(v1_vecs[i])
            v2 = np.array(v2_vecs[i])
            vinf_out_vecs.append((v1 - v_dep_planet).tolist())
            vinf_in_vecs.append((v2 - v_arr_planet).tolist())

        # compute vinf mismatch per flyby (between outgoing previous and incoming next)
        vinf_mismatch_list = []
        for i in range(nlegs-1):
            vec_out = np.asarray(vinf_out_vecs[i])
            vec_in_next = np.asarray(vinf_in_vecs[i+1])
            diff = float(np.linalg.norm(vec_in_next - vec_out))
            vinf_mismatch_list.append(diff)

        # PERIAPSIS SWEEP: for each flyby compute rp choice that minimizes DSM
        rp_choice_list = []
        rp_req_list = []
        dsm_list = []
        bending_ok = True
        total_dsm = 0.0

        for i in range(nlegs-1):
            body = normalize_label_to_astropy_name(parts[i+1])
            planet_R = PLANET_RAD.get(body, None)
            mu_planet = MU.get(body, None)
            if planet_R is None or mu_planet is None:
                rp_choice_list.append(None); rp_req_list.append(None); dsm_list.append(np.nan); bending_ok=False
                continue

            v_out_prev = np.asarray(vinf_out_vecs[i])
            v_in_next = np.asarray(vinf_in_vecs[i+1])
            norm_out = np.linalg.norm(v_out_prev); norm_in = np.linalg.norm(v_in_next)
            if (norm_out <= 0) or (norm_in <= 0):
                req_delta = math.radians(30.0)  # fallback
            else:
                dot = float(np.dot(v_out_prev, v_in_next) / (norm_out * norm_in))
                dot = max(-1.0, min(1.0, dot))
                req_delta = math.acos(dot)

            vinf_mag = max(norm_out, norm_in, 1e-6)

            if req_delta <= 1e-9:
                rp_req = 0.0
            else:
                sin_half = math.sin(req_delta/2.0)
                if sin_half <= 0:
                    rp_req = None
                else:
                    e_req = 1.0 / sin_half
                    rp_req = mu_planet * (e_req - 1.0) / (vinf_mag**2)

            rp_req_list.append(None if rp_req is None else float(rp_req))

            min_allowed_rp = planet_R + float(MIN_FLYBY_ALT_km.get(body, 100.0))
            if (rp_req is None) or (rp_req is not None and math.isinf(rp_req)):
                rp_max = min(min_allowed_rp * 1000.0, PERIAPSE_SWEEP_RP_MAX_km)
            else:
                rp_max = min(max(rp_req * PERIAPSE_SWEEP_FACTOR, min_allowed_rp*2.0), PERIAPSE_SWEEP_RP_MAX_km)

            if rp_max <= min_allowed_rp:
                rp_choice_list.append(None); dsm_list.append(np.nan); bending_ok=False
                continue

            # rp grid
            if PERIAPSE_SWEEP_POINTS <= 4:
                rp_grid = np.linspace(min_allowed_rp, rp_max, max(3, PERIAPSE_SWEEP_POINTS))
            else:
                rp_grid = np.concatenate([
                    np.linspace(min_allowed_rp, min_allowed_rp + 0.1*(rp_max-min_allowed_rp), int(0.6*PERIAPSE_SWEEP_POINTS)),
                    np.linspace(min_allowed_rp + 0.1*(rp_max-min_allowed_rp), rp_max, int(0.4*PERIAPSE_SWEEP_POINTS))
                ])
                rp_grid = np.unique(rp_grid)

            best = {"rp": None, "alt": None, "dsm": float("inf"), "ach_delta": 0.0}
            for rp in rp_grid:
                # approximate eccentricity at rp from energy/angular momentum balance: simplified formula
                e = 1.0 + (rp * vinf_mag**2) / mu_planet
                if e <= 1.0:
                    ach_delta = 0.0
                else:
                    val = 1.0 / e
                    val = max(-1.0, min(1.0, val))
                    ach_delta = 2.0 * math.asin(val)
                if ach_delta + 1e-9 >= req_delta:
                    dsm_required = 0.0
                else:
                    missing = max(0.0, req_delta - ach_delta)
                    dsm_required = 2.0 * vinf_mag * math.sin(missing/2.0)

                # vector-based DSM as alternative
                try:
                    vec_dsm = float(np.linalg.norm(v_in_next - v_out_prev))
                    dsm_est = min(dsm_required, vec_dsm)
                except Exception:
                    dsm_est = dsm_required

                if dsm_est < best["dsm"]:
                    best.update({"rp": float(rp), "alt": float(rp-planet_R), "dsm": float(dsm_est), "ach_delta": ach_delta})
                    if dsm_est == 0.0:
                        break

            if best["rp"] is None:
                rp_choice_list.append(None); dsm_list.append(np.nan); bending_ok=False
            else:
                rp_choice_list.append(best)
                dsm_list.append(best["dsm"])
                total_dsm += best["dsm"]
                if best["alt"] < MIN_FLYBY_ALT_km.get(body, 0.0) - 1e-6:
                    bending_ok = False
        
            # Build arrays of required/achieved deflection in degrees per flyby
            bending_req_turns_deg = []   # required turn (deg) for each flyby (based on vectors)
            bending_ach_turns_deg = []   # achieved turn (deg) for chosen rp (from ach_delta)
            bending_rp_alt_km_list = []  # chosen periapsis altitude above surface (km) for each flyby (or None)

            for i in range(nlegs-1):
                # compute requested turn based on outgoing vs next incoming vectors
                try:
                    v_out = np.asarray(vinf_out_vecs[i])
                    v_in_next = np.asarray(vinf_in_vecs[i+1])
                    n_out = np.linalg.norm(v_out)
                    n_in = np.linalg.norm(v_in_next)
                    if n_out > 0 and n_in > 0:
                        cosang = np.dot(v_out, v_in_next) / (n_out * n_in)
                        cosang = float(max(-1.0, min(1.0, cosang)))
                        req_turn_rad = math.acos(cosang)
                        req_turn_deg = float(np.degrees(req_turn_rad))
                    else:
                        req_turn_deg = float("nan")
                except Exception:
                    req_turn_deg = float("nan")

                bending_req_turns_deg.append(req_turn_deg)

                # achieved from rp_choice_list (if available)
                chosen = rp_choice_list[i] if i < len(rp_choice_list) else None
                if chosen and isinstance(chosen, dict) and ("ach_delta" in chosen):
                    ach_deg = float(np.degrees(chosen.get("ach_delta", 0.0)))
                    alt_km = None
                    try:
                        body = normalize_label_to_astropy_name(parts[i+1])
                        planet_R = PLANET_RAD.get(body, None)
                        if planet_R is not None and chosen.get("rp") is not None:
                            alt_km = float(chosen["rp"] - planet_R)
                    except Exception:
                        alt_km = None
                else:
                    ach_deg = float("nan")
                    alt_km = None

                bending_ach_turns_deg.append(ach_deg)
                bending_rp_alt_km_list.append(alt_km)

            # Now compute the global minimum required rp (if any), and which body caused it
            min_rp = None
            min_rp_body = None
            min_rp_alt_km = None
            for i, rp_req in enumerate(rp_req_list):
                if rp_req is None or (isinstance(rp_req, float) and (math.isnan(rp_req) or math.isinf(rp_req))):
                    continue
                if (min_rp is None) or (rp_req < min_rp):
                    min_rp = float(rp_req)
                    min_rp_body = normalize_label_to_astropy_name(parts[i+1])
                    # altitude above surface
                    rad = PLANET_RAD.get(min_rp_body, None)
                    if rad is not None:
                        min_rp_alt_km = float(min_rp - rad)
                    else:
                        min_rp_alt_km = None

            # Guarantee JSON leg vectors exist and are strings
            try:
                leg_v1_vectors_json = json.dumps(v1_vecs)    # list of lists
            except Exception:
                leg_v1_vectors_json = "[]"
            try:
                leg_v2_vectors_json = json.dumps(v2_vecs)
            except Exception:
                leg_v2_vectors_json = "[]"

        # summarise
        vinf_mismatch_vals = [v for v in vinf_mismatch_list if not np.isnan(v)]
        vinf_mismatch_max = float(max(vinf_mismatch_vals)) if vinf_mismatch_vals else 0.0
        vinf_mismatch_mean = float(np.mean(vinf_mismatch_vals)) if vinf_mismatch_vals else 0.0
        last_leg_vinf = float(vinf_out_list[-1]) if len(vinf_out_list)>0 else np.nan
        first_leg_vinf = float(vinf_out_list[0]) if len(vinf_out_list)>0 else np.nan
        first_leg_C3 = float(first_leg_vinf**2) if (first_leg_vinf is not None and not math.isnan(first_leg_vinf)) else np.nan

        # EPO escape & MOI
        epo_perigee = PLANET_RAD["earth"] + float(cfg_local.get("EPO_perigee_km_from_surface",250.0))
        epo_apogee = PLANET_RAD["earth"] + float(cfg_local.get("EPO_apogee_km_from_surface",23000.0))
        dv_escape = compute_escape_delta_v_perigee(first_leg_vinf, epo_perigee, epo_apogee) if (not math.isnan(first_leg_vinf)) else 0.0

        final_body = normalize_label_to_astropy_name(parts[-1])
        dv_moi = 0.0
        if final_body == "mercury" and (not math.isnan(last_leg_vinf)):
            mpo_peri = PLANET_RAD["mercury"] + float(cfg_local.get("MPO_periapsis_km_from_surface",500.0))
            mpo_apo = PLANET_RAD["mercury"] + float(cfg_local.get("MPO_apoapsis_km_from_surface",50000.0))
            dv_moi = compute_moi_delta_v(last_leg_vinf, mpo_peri, mpo_apo, MU["mercury"])

        sum_deltaV = total_dsm + dv_escape + dv_moi

        # v∞ matching judgment
        if APPLY_STRICT_VINF_MATCH:
            pass_vinf = (vinf_mismatch_max <= STRICT_VINF_TOL_kms)
        else:
            # relaxed: allow if vinf mismatch within VINF_MATCH_TOL or total DSM <= cap
            if vinf_mismatch_max <= VINF_MATCH_TOL_kms:
                pass_vinf = True
            else:
                pass_vinf = (total_dsm <= VINF_DSM_CAP_kms)

        pass_c3 = True if (not APPLY_C3_CAP) else (first_leg_C3 <= C3_CAP_km2s2)
        pass_bending = bool(bending_ok)
        pass_dsm = (total_dsm <= float(cfg_local.get("DSM_MAX_kms", 1e9)))
        pass_moi = (dv_moi <= float(cfg_local.get("MOI_DV_MAX_kms", 1e9)))
        overall_pass = pass_c3 and pass_vinf and pass_bending and pass_dsm and pass_moi

        # Prepare JSON-safe v1/v2 lists
        leg_v1_vectors_json = json.dumps([v.tolist() if isinstance(v,np.ndarray) else v for v in v1_vecs])
        leg_v2_vectors_json = json.dumps([v.tolist() if isinstance(v,np.ndarray) else v for v in v2_vecs])

        out = {
            "chain_str": "--".join(parts),
            "seed_epoch_iso": seed_time.iso,
            "campaign": cfg_local.get("CAMPAIGN_NAME", ""),
            "tof_combo_days": ";".join([f"{t:.2f}" for t in tof_list_days]),
            "total_tof_days": float(sum(tof_list_days)),
            "first_leg_C3_km2s2": float(first_leg_C3),
            "first_leg_vinf_kms": float(first_leg_vinf),
            "last_leg_vinf_kms": float(last_leg_vinf),
            "vinf_in_list_kms": ";".join([f"{x:.6f}" for x in vinf_in_list]),
            "vinf_out_list_kms": ";".join([f"{x:.6f}" for x in vinf_out_list]),
            "vinf_mismatch_list_kms": ";".join([f"{x:.6f}" for x in vinf_mismatch_list]),
            "vinf_mismatch_max_kms": float(vinf_mismatch_max),
            "vinf_mismatch_mean_kms": float(vinf_mismatch_mean),
            "DSM_est_total_kms": float(total_dsm),
            "DSM_est_list_kms": ";".join([ (str(d) if (d is not None and not np.isnan(d)) else "") for d in dsm_list ]),
            # flyby details
            "bending_feasible": bool(pass_bending),
            "bending_req_rp_km_list": ";".join([ (str(r) if (r is not None) else "") for r in rp_req_list ]),
            "bending_req_turns_deg": ";".join([ (f"{x:.6f}" if (x is not None and not np.isnan(x)) else "") for x in bending_req_turns_deg ]),
            "bending_ach_turns_deg": ";".join([ (f"{x:.6f}" if (x is not None and not np.isnan(x)) else "") for x in bending_ach_turns_deg ]),
            "bending_rp_alt_km_list": ";".join([ (f"{x:.3f}" if (x is not None and not np.isnan(x)) else "") for x in bending_rp_alt_km_list ]),
            "bending_rp_choice_details": json.dumps(rp_choice_list),
            "bending_rp_choice_selected_json": json.dumps(rp_choice_list),
            "min_required_rp_km": float(min([r for r in rp_req_list if (r is not None and not math.isinf(r))])) if any([r for r in rp_req_list if (r is not None and not math.isinf(r))]) else np.nan,
            "min_required_rp_body": min_rp_body if min_rp_body is not None else "",
            "min_required_rp_alt_km": min_rp_body if min_rp_body is not None else "",
            #deltaV
            "dv_escape_EPO_kms": float(dv_escape),
            "dv_moi_kms": float(dv_moi),
            "sum_deltaV_kms": float(sum_deltaV),
            "notes": "",
            # passes and checsk
            "pass_c3": bool(pass_c3),
            "pass_vinf_match": bool(pass_vinf),
            "pass_bending": bool(pass_bending),
            "pass_dsm": bool(pass_dsm),
            "pass_moi": bool(pass_moi),
            "overall_pass": bool(overall_pass),
            # vectors for debugging
            "leg_v1_vectors_json": leg_v1_vectors_json,
            "leg_v2_vectors_json": leg_v2_vectors_json
        }

        return out

    except Exception as e:
        tb = traceback.format_exc()
        return {"error": f"eval_exc: {e}", "traceback": tb, "chain_str": (chain_parts if isinstance(chain_parts,str) else "--".join(chain_parts))}

# ----------------------------
# Optimizer glue
# ----------------------------
def try_optimize_chain(chain_row, cfg_local):
    """
    Attempt to optimize chain using evaluate_chain_candidate_with_rp_sweep.
    """
    chain_str = chain_row["chain_str"]
    seed_iso = chain_row.get("seed_epoch_iso", chain_row.get("seed_epoch", chain_row.get("seed_epoch_time", None)))
    try:
        seed_time = Time(seed_iso)
    except Exception:
        seed_time = Time(str(seed_iso))

    parts = [p.strip() for p in chain_str.split("--") if p.strip()]

    # prepare x0 from chain_row if available
    try:
        tofs_s = chain_row.get("tof_combo_days", "")
        tof_list = []
        if isinstance(tofs_s, str):
            for s in tofs_s.split(";"):
                try: tof_list.append(float(s))
                except: pass
        elif isinstance(tofs_s, (list,tuple)):
            tof_list = [float(x) for x in tofs_s]
        if len(tof_list) == 0:
            tof_list = cfg_local.get("GENERIC_TOF_CHOICES", [20.0] * (len(parts)-1))
    except Exception:
        tof_list = cfg_local.get("GENERIC_TOF_CHOICES", [20.0]*(len(parts)-1))

    nlegs = max(len(parts)-1, 0)
    if len(tof_list) < nlegs:
        tof_list = tof_list + [120.0] * (nlegs - len(tof_list))

    x0 = np.array([0.0] + tof_list[:nlegs], dtype=float)

    bounds = [(-30.0, 30.0)] + [(5.0, float(cfg_local.get("MAX_LEG_TOF_DAYS", 900.0)))] * nlegs

    # object function at top-level (for pickling safety)
    # ---------------------- OBJECTIVE FUNCTION (modified to optionally include v_inf) ----------------------
    def obj_fn(x):
        """
        x is the decision vector (launch offsets + tof choices etc).
        This wrapper evaluates the candidate (using your evaluate_chain_candidate_with_rp_sweep)
        and composes the cost as the original cost plus an optional v_inf mismatch penalty.
        """
        # Evaluate candidate using your existing heavy function - unchanged
        out = evaluate_chain_candidate_with_rp_sweep(parts, seed_time, x, cfg_local=cfg_local)
        if out is None or "error" in out:
            # return a large penalty for bad/invalid solutions
            return 1e9

        # existing cost terms (unchanged semantics)
        c3 = out.get("first_leg_C3_km2s2")
        if c3 is None:
            c3 = 1e6
        dsm = out.get("DSM_est_total_kms", 0.0)
        dv_moi = out.get("dv_moi_kms", 0.0)

        # existing weights already defined in the file - reuse them if present, else fall back
        w_c3 = cfg_local.get("W_C3", 1.0) if "W_C3" in cfg_local else 1.0
        w_dsm = cfg_local.get("W_DSM", 50.0) if "W_DSM" in cfg_local else 50.0
        w_moi = cfg_local.get("W_MOI", 10.0) if "W_MOI" in cfg_local else 10.0

        # Compose base cost (existing behavior)
        cost = w_c3 * float(c3) + w_dsm * float(dsm) + w_moi * float(dv_moi)

        # Add a large penalty if bending/geometry infeasible (preserve existing logic if present)
        if not out.get("bending_feasible", True):
            cost += cfg_local.get("BENDING_PENALTY", 1000.0)

        # Optionally add v_inf mismatch penalty (Option A)
        if cfg_local.get("APPLY_VINF_IN_OBJECTIVE", False):
            vinf_mismatch = out.get("vinf_mismatch_max_kms", None)
            # If the evaluation didn't compute vinf mismatch, penalize heavily.
            if vinf_mismatch is None:
                cost += cfg_local.get("VINF_WEIGHT", 500.0) * 1e3
            else:
                v_w = float(cfg_local.get("VINF_WEIGHT", 500.0))
                v_pow = float(cfg_local.get("VINF_POWER", 2.0))
                cost += v_w * (float(vinf_mismatch) ** v_pow)

        # Keep additional penalties (if your code had them), preserve original return
        return float(cost)
    # --------------------------------------------------------------------------------------------------------

    # Differential evolution with fallback
    try:
        res_de = differential_evolution(obj_fn, bounds, popsize=int(cfg_local.get("DE_POPSIZE",12)),
                                       maxiter=int(cfg_local.get("DE_MAXITER",200)),
                                       workers=int(cfg_local.get("DE_WORKERS",1)),
                                       updating="deferred")
    except Exception as e:
        try:
            res_de = differential_evolution(obj_fn, bounds, popsize=int(cfg_local.get("DE_POPSIZE",12)),
                                           maxiter=int(cfg_local.get("DE_MAXITER",200)),
                                           workers=1, updating="deferred")
        except Exception as e2:
            coarse = evaluate_chain_candidate_with_rp_sweep(parts, seed_time, x0, cfg_local=cfg_local)
            if coarse and ("error" not in coarse):
                return coarse
            return {"error": f"DE_failed: {e2}", "chain_str": chain_str, "seed_epoch_iso": seed_time.iso}

    best_x = res_de.x
    final_eval = evaluate_chain_candidate_with_rp_sweep(parts, seed_time, best_x, cfg_local=cfg_local)

    
    # ---------------------- STRICT VINFINITY LOCAL POLISH (Option B) ----------------------
    if cfg_local.get("APPLY_VINF_LOCAL_POLISH", False):

        def _local_vinf_objective(xx):
            val = evaluate_chain_candidate_with_rp_sweep(parts, seed_time, xx, cfg_local=cfg_local)
            if val is None or "error" in val:
                return 1e6
            vm = val.get("vinf_mismatch_max_kms", None)
            return 1e6 if vm is None else float(vm) ** 2

        def _constraint_dsm(xx):
            val = evaluate_chain_candidate_with_rp_sweep(parts, seed_time, xx, cfg_local=cfg_local)
            if val is None or "error" in val:
                return -1e6
            dsm_val = float(val.get("DSM_est_total_kms", 1e6))
            cap = float(cfg_local.get("VINF_LOCAL_POLISH_DSM_CAP_kms", 1.0))
            # positive means constraint satisfied
            return cap - dsm_val

        def _constraint_c3(xx):
            val = evaluate_chain_candidate_with_rp_sweep(parts, seed_time, xx, cfg_local=cfg_local)
            if val is None or "error" in val:
                return -1e6
            c3_val = float(val.get("first_leg_C3_km2s2", 1e6))
            cap = float(cfg_local.get("VINF_LOCAL_POLISH_C3_CAP_km2s2", 1e9))
            return cap - c3_val

        try:
            print(f"[VINFINITY POLISH] Starting strict v∞ minimization for {chain_str}…")
            res_vinf = minimize(
                _local_vinf_objective,
                best_x,
                method="SLSQP",
                bounds=bounds,
                constraints=[
                    {"type": "ineq", "fun": _constraint_dsm},
                    {"type": "ineq", "fun": _constraint_c3},
                ],
                options={
                    "maxiter": int(cfg_local.get("VINF_LOCAL_POLISH_MAXITER", 200)),
                    "ftol": 1e-3,
                },
            )

            if res_vinf.success:
                improved_eval = evaluate_chain_candidate_with_rp_sweep(parts, seed_time, res_vinf.x, cfg_local=cfg_local)
                if improved_eval and "error" not in improved_eval:
                    final_eval = improved_eval
                    final_eval["notes"] = (final_eval.get("notes", "") + " | vinf_local_polished").strip()
                    print(f"[VINFINITY POLISH] Chain {chain_str}: vinf mismatch after polish = {final_eval.get('vinf_mismatch_max_kms', 'NA'):.4f} km/s")
        except Exception as e:
            print(f"[VINFINITY POLISH] failed for {chain_str}: ", e)
    # --------------------------------------------------------------------------------------

    return final_eval

# ----------------------------
# Runner
# ----------------------------
def main():
    print("StageC_Optimization_Adv starting...")
    print("Loading chains from:", INPUT_CHAINS_CSV)
    df_chains = pd.read_csv(INPUT_CHAINS_CSV)
    if "chain_str" not in df_chains.columns:
        if "chain" in df_chains.columns:
            df_chains = df_chains.rename(columns={"chain":"chain_str"})
        else:
            raise RuntimeError("Input chains CSV must contain 'chain_str' column")

    # deduplicate by chain+seed
    df_chains_unique = df_chains.drop_duplicates(subset=["chain_str","seed_epoch_iso"]).reset_index(drop=True)
    if cfg.get("PREVIEW", False):
        n = min(cfg.get("PREVIEW_N", 2000), len(df_chains_unique))
        df_chains_unique = df_chains_unique.iloc[:n].copy()

    chains = df_chains_unique.to_dict(orient="records")
    print("Total unique chains to process:", len(chains))

    # load checkpoint
    start_idx = 0
    checkpoint = {}
    if os.path.exists(CHECKPOINT_JSON):
        try:
            checkpoint = json.load(open(CHECKPOINT_JSON,"r"))
            start_idx = checkpoint.get("last_idx", 0) + 1
            print("Resuming from checkpoint idx:", start_idx)
        except Exception:
            print("Failed to read checkpoint; starting from 0")

    out_rows = []
    autosave_every = cfg.get("AUTOSAVE_EVERY", 50)
    for idx in tqdm(range(start_idx, len(chains)), desc="Chains", unit="chain"):
        row = chains[idx]
        try:
            evald = try_optimize_chain(row, cfg)
            if isinstance(evald, dict):
                # ensure required fields exist
                evald.setdefault("chain_str", row["chain_str"])
                evald.setdefault("seed_epoch_iso", row.get("seed_epoch_iso",""))
                evald.setdefault("campaign", cfg.get("CAMPAIGN_NAME",""))
                out_rows.append(evald)
            else:
                out_rows.append({"chain_str": row["chain_str"], "seed_epoch_iso": row.get("seed_epoch_iso",""), "error":"no_eval"})
        except Exception as e:
            out_rows.append({"chain_str": row["chain_str"], "seed_epoch_iso": row.get("seed_epoch_iso",""), "error": str(e)})

        # autosave
        if (idx+1) % autosave_every == 0:
            tmp_df = pd.DataFrame(out_rows)
            tmp_df.to_csv(OUT_CSV, index=False)
            checkpoint = {"last_idx": idx}
            try:
                with open(CHECKPOINT_JSON + ".tmp","w") as f:
                    json.dump(checkpoint, f)
                os.replace(CHECKPOINT_JSON + ".tmp", CHECKPOINT_JSON)
            except Exception:
                json.dump(checkpoint, open(CHECKPOINT_JSON,"w"))
            print(f"Autosaved at idx {idx}; wrote {len(out_rows)} rows so far.")

    # final save
    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(OUT_CSV, index=False)
    print("Done. Wrote:", OUT_CSV)

    # # create Top-K per campaign
    # try:
    #     df_out["first_leg_C3_km2s2"] = pd.to_numeric(df_out.get("first_leg_C3_km2s2", np.nan), errors="coerce").fillna(np.inf)
    #     df_out["DSM_est_total_kms"] = pd.to_numeric(df_out.get("DSM_est_total_kms", np.nan), errors="coerce").fillna(np.inf)
    #     df_out["bending_feasible"] = df_out.get("bending_feasible", False).astype(bool)
    #     groups = []
    #     if "campaign" in df_out.columns:
    #         for camp, g in df_out.groupby("campaign"):
    #             g_sorted = g.sort_values(by=["bending_feasible","first_leg_C3_km2s2","DSM_est_total_kms"],
    #                                      ascending=[False, True, True])
    #             groups.append(g_sorted.head(5))
    #         if groups:
    #             pd.concat(groups).to_csv(TOPK_OUT, index=False)
    #             print("Wrote Top K per campaign:", TOPK_OUT)
    # except Exception as e:
    #     print("Top-K generation failed:", e)

if __name__ == "__main__":
    main()
