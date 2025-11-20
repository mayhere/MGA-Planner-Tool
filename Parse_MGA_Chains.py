# Parse_MGA_Chains.py
import pandas as pd, os, math, datetime, re
from pathlib import Path

# === Adjust these paths to match your environment if needed ===
CSV_PATH = r"C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/C_Verification_and_Optimization/2026-27_StageC_Top5_Optimized_Results.csv"
OUT_DIR = Path(r"C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/D_Plotting_and_Validation/Parse_MGA_Chains")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

def parse_tof_list(s):
    if pd.isna(s): return []
    parts = re.split(r'[;,]\s*', str(s).strip())
    vals = []
    for p in parts:
        p = p.strip()
        if p == '': continue
        try:
            vals.append(float(p))
        except:
            p2 = re.sub(r'[^\d\.\-eE+]','',p)
            try:
                vals.append(float(p2))
            except:
                vals.append(float('nan'))
    return vals

def parse_list_col(s):
    if pd.isna(s): return []
    s = str(s).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            import ast
            return list(ast.literal_eval(s))
        except:
            s = s[1:-1]
    return [float(x) for x in re.split(r'[;,]\s*', s) if x!='']

# Mapping for perigee altitude (per your EPO / MPO definitions)
# note: keys are expected short codes used in chain_str (e.g., 'Ea','Me','Ve','Ma' etc.)
PERIGEE_ALT_MAP = {
    'Ea': 250.0,   # EPO perigee altitude in km (EPO = 250 x 23000 km -> perigee = 250 km)
    'Earth': 250.0,
    'Me': 500.0,   # MPO perigee altitude in km (MPO = 500 x 50000 km -> perigee = 500 km)
    'Mercury': 500.0
    # other bodies left blank unless you want to add mappings
}

reports = []
summary_rows = []

for idx, row in df.iterrows():
    chain = row.get('chain_str', f'chain_{idx}')
    seed_iso = row.get('seed_epoch_iso', None)
    try:
        seed_dt = pd.to_datetime(seed_iso)
    except:
        seed_dt = None
    tof_list = parse_tof_list(row.get('tof_combo_days',''))
    total_tof = None
    try:
        total_tof = float(row.get('total_tof_days', math.nan))
    except:
        total_tof = math.nan
    arrival_dt = seed_dt + pd.to_timedelta(total_tof, unit='D') if (seed_dt is not None and not math.isnan(total_tof)) else None
    sum_dv = row.get('sum_deltaV_kms', row.get('sum_deltaV_km', row.get('sum_deltaV', None)))
    vinf_in = parse_list_col(row.get('vinf_in_list_kms',''))
    vinf_out = parse_list_col(row.get('vinf_out_list_kms',''))
    rp_alts = parse_list_col(row.get('bending_rp_alt_km_list',''))
    turn_deg = parse_list_col(row.get('bending_ach_turns_deg',''))
    
    # Determine departure Vinf:
    departure_vinf = None
    # prefer explicit scalar column 'first_leg_vinf_kms' if present and valid
    try:
        vtmp = row.get('first_leg_vinf_kms', None)
        if vtmp is not None and str(vtmp).strip() != '':
            departure_vinf = float(vtmp)
    except Exception:
        departure_vinf = None
    # fallback: use first element of vinf_out or vinf_in
    if departure_vinf is None:
        if len(vinf_out) > 0:
            try:
                if not math.isnan(vinf_out[0]):
                    departure_vinf = vinf_out[0]
            except:
                departure_vinf = vinf_out[0]
        if departure_vinf is None and len(vinf_in) > 0:
            try:
                if not math.isnan(vinf_in[0]):
                    departure_vinf = vinf_in[0]
            except:
                departure_vinf = vinf_in[0]

    # Build flyby dates (cumulative TOF from seed)
    flyby_dates = []
    cum = 0.0
    for t in tof_list:
        cum += t
        if seed_dt is not None:
            flyby_dates.append(seed_dt + pd.to_timedelta(cum, unit='D'))
        else:
            flyby_dates.append(cum)

    n_flybys = max(len(vinf_in), len(vinf_out), len(rp_alts), len(turn_deg), len(tof_list))
    flybys = []
    for i in range(n_flybys):
        date = flyby_dates[i] if i < len(flyby_dates) else (flyby_dates[-1] if flyby_dates else None)
        alt = rp_alts[i] if i < len(rp_alts) else None
        vin_i = vinf_in[i] if i < len(vinf_in) else None
        vout_i = vinf_out[i] if i < len(vinf_out) else None
        turn_i = turn_deg[i] if i < len(turn_deg) else None
        flybys.append({
            'flyby_index': i+1,
            'date': date,
            'altitude_km': alt,
            'vinf_in_kms': vin_i,
            'vinf_out_kms': vout_i,
            'turn_deg': turn_i
        })

    # Determine body codes from chain_str
    bodies = chain.split('--') if isinstance(chain, str) else []
    departure_planet_code = bodies[0] if len(bodies) > 0 else ''
    arrival_planet_code = bodies[-1] if len(bodies) > 0 else ''

    # Determine perigee altitudes for departure & arrival using the mapping
    dep_perigee_alt = PERIGEE_ALT_MAP.get(departure_planet_code, PERIGEE_ALT_MAP.get(departure_planet_code.capitalize(), None))
    arr_perigee_alt = PERIGEE_ALT_MAP.get(arrival_planet_code, PERIGEE_ALT_MAP.get(arrival_planet_code.capitalize(), None))

    # --- SUMMARY CSV: Insert departure row before flyby rows (flyby_index = 0) ---
    dep_date_str = seed_dt.strftime("%Y-%m-%d %H:%M:%S") if seed_dt is not None else (seed_iso if seed_iso is not None else "")
    summary_rows.append({
        'chain_index': idx+1,
        'chain_str': chain,
        'flyby_index': 0,
        'flyby_date': dep_date_str,
        'altitude_km': dep_perigee_alt,
        'turn_deg': '',
        'vinf_in_kms': '',
        'vinf_out_kms': departure_vinf
    })
    # Now append each flyby as a row (flyby_index 1..N)
    for i, fb in enumerate(flybys):
        date_str = fb['date'].strftime("%Y-%m-%d %H:%M:%S") if isinstance(fb['date'], pd.Timestamp) else (f"{fb['date']} days after departure" if fb['date'] is not None else "")
        summary_rows.append({
            'chain_index': idx+1,
            'chain_str': chain,
            'flyby_index': fb['flyby_index'],
            'flyby_date': date_str,
            'altitude_km': fb['altitude_km'],
            'turn_deg': fb['turn_deg'],
            'vinf_in_kms': fb['vinf_in_kms'],
            'vinf_out_kms': fb['vinf_out_kms']
        })

    safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', chain)[:80]
    out_path = OUT_DIR / f"Chain_{idx+1}_{safe_name}.txt"

    # Write report with UTF-8 encoding (to avoid Windows encoding issues)
    with open(out_path, 'w', encoding='utf-8') as f:
        # Header block (kept in exact order you requested)
        f.write(f"Sequence: {chain}\n")
        f.write(f"Departure (seed_epoch_iso): {seed_dt.strftime('%Y-%m-%d %H:%M:%S') if seed_dt is not None else seed_iso}\n")
        f.write(f"Departure Vinf (km/s): {departure_vinf}\n")
        f.write(f"Arrival (seed + total_tof_days): {arrival_dt.strftime('%Y-%m-%d %H:%M:%S') if arrival_dt is not None else 'unknown'}\n")
        f.write(f"Total TOF (days): {total_tof}\n")
        f.write(f"Total ΔV (sum_deltaV_kms): {sum_dv}\n\n")

        # Now the departure line as you requested
        # Use perigee altitude mapping if available, else blank
        dep_alt_display = f"{dep_perigee_alt}" if dep_perigee_alt is not None else ""
        f.write(f"Departure: {dep_date_str}  Altitude (km): {dep_alt_display}  Vinf_out (km/s): {departure_vinf}\n\n")

        # Followed by flyby lines
        for fb in flybys:
            date_str = fb['date'].strftime("%Y-%m-%d %H:%M:%S") if isinstance(fb['date'], pd.Timestamp) else (f"{fb['date']} days after departure" if fb['date'] is not None else 'unknown')
            f.write(f"Flyby {fb['flyby_index']}: {date_str}, Altitude (km): {fb['altitude_km']}, Vinf_in (km/s): {fb['vinf_in_kms']}, Vinf_out (km/s): {fb['vinf_out_kms']}, Turn angle (deg): {fb['turn_deg']}\n")

        # Nicely formatted table with fixed column widths, including departure row as first row
        f.write("\n\nTable format:\n")
        col_planet_w = 8
        col_date_w = 19
        col_alt_w = 12
        col_turn_w = 12
        col_vinf_in_w = 12
        col_vinf_out_w = 12

        hdr = f"{'Planet':<{col_planet_w}} {'Flyby_Date':<{col_date_w}} {'Altitude_km':>{col_alt_w}} {'Turn_deg':>{col_turn_w}} {'Vinf_in_kms':>{col_vinf_in_w}} {'Vinf_out_kms':>{col_vinf_out_w}}\n"
        f.write(hdr)
        sep_len = col_planet_w + 1 + col_date_w + 1 + col_alt_w + 1 + col_turn_w + 1 + col_vinf_in_w + 1 + col_vinf_out_w
        f.write("-" * sep_len + "\n")

        # Departure row
        f.write(f"{departure_planet_code:<{col_planet_w}} {dep_date_str:<{col_date_w}} {dep_alt_display:>{col_alt_w}} {'':>{col_turn_w}} {'':>{col_vinf_in_w}} {str(departure_vinf):>{col_vinf_out_w}}\n")

        # Flyby rows
        for i, fb in enumerate(flybys):
            planet = bodies[i+1] if i+1 < len(bodies) else f"Body_{i+1}"
            date_str = fb['date'].strftime("%Y-%m-%d %H:%M:%S") if isinstance(fb['date'], pd.Timestamp) else (f"{fb['date']} days after departure" if fb['date'] is not None else 'unknown')
            alt_str = f"{fb['altitude_km']}" if fb['altitude_km'] is not None else ""
            turn_str = f"{fb['turn_deg']}" if fb['turn_deg'] is not None else ""
            vinin_str = f"{fb['vinf_in_kms']}" if fb['vinf_in_kms'] is not None else ""
            vinout_str = f"{fb['vinf_out_kms']}" if fb['vinf_out_kms'] is not None else ""
            f.write(f"{planet:<{col_planet_w}} {date_str:<{col_date_w}} {alt_str:>{col_alt_w}} {turn_str:>{col_turn_w}} {vinin_str:>{col_vinf_in_w}} {vinout_str:>{col_vinf_out_w}}\n")

    reports.append(str(out_path))

# Save summary CSV (no departure_date or departure_vinf_kms columns)
summary_df = pd.DataFrame(summary_rows)
summary_csv = OUT_DIR / "Summary_Table.csv"
summary_df.to_csv(summary_csv, index=False)

print("Generated report files:", len(reports))
print("Folder:", OUT_DIR)
print("Summary CSV:", summary_csv)
