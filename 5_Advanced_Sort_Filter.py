# Advanced_Sort_Filter.py
# Use: edit CONFIG block then run
import pandas as pd
import numpy as np
import sys
import os

# ================= CONFIG =================
INPUT_CSV = "C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/B_Postprocess_and_Filter/2026-27_StageB_Filtered_Ranked.csv"
OUTPUT_CSV = "C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/C_Verification_and_Optimization/2026-27_StageC_Adv_Sorted_Filtered.csv"

# primary filter ranges (inclusive)
C3_MIN, C3_MAX = 4.0, 95.0
LASTV_MIN, LASTV_MAX = 0.0, 100.0

# required boolean check
REQUIRE_BENDING_FEASIBLE = True   # if True, keep only bending_feasible == True

# Additional sort toggles
SORT_DV_ESCAPE = False     # include dv_escape_EPO_kms after primary keys
SORT_DV_MOI = False        # include dv_moi_kms after previous keys (if present)
SORT_SUM_DELTAV = False   # include sum_deltaV_kms after previous keys (if present)
SORT_ALL_ADDITIONAL = False  # if True, set all three to True

# Behavior toggles
DROP_ROWS_MISSING_PRIMARY = True  # drop rows missing any of the three primary numeric keys
N_TOP_SHOW = 5                   # number of rows to print in summary

# Column name mapping (edit if your CSV uses slightly different names)
COL_C3 = "first_leg_C3_km2s2"
COL_LASTV = "last_leg_vinf_kms"
COL_VINFMM = "vinf_mismatch_max_kms"
COL_DV_ESCAPE = "dv_escape_EPO_kms"
COL_DV_MOI = "dv_moi_kms"
COL_SUM_DELTAV = "sum_deltaV_kms"
# ==========================================

# Apply SORT_ALL_ADDITIONAL
if SORT_ALL_ADDITIONAL:
    SORT_DV_ESCAPE = SORT_DV_MOI = SORT_SUM_DELTAV = True

def ensure_cols_exist(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"Warning: missing columns in CSV (they will be created as NaN): {missing}")
        for c in missing:
            df[c] = np.nan
    return df

def to_numeric_fill(df, col, fill_large=True):
    num = pd.to_numeric(df[col], errors="coerce")
    if fill_large:
        # fill NaN with very large number so they sort to bottom
        num_filled = num.fillna(1e12)
        return num, num_filled
    else:
        return num, num

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Input CSV not found: {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, dtype=str)  # read as strings for robustness
    print(f"Loaded {len(df):,} rows from {INPUT_CSV}")

    # Ensure necessary columns exist
    df = ensure_cols_exist(df, [COL_C3, COL_LASTV, COL_VINFMM, COL_DV_ESCAPE, COL_DV_MOI, COL_SUM_DELTAV, "bending_feasible"])

    # Coerce primary keys to numeric (keep original strings too)
    df[COL_C3 + "_num"], df[COL_C3 + "_num_sort"] = to_numeric_fill(df, COL_C3, fill_large=True)
    df[COL_LASTV + "_num"], df[COL_LASTV + "_num_sort"] = to_numeric_fill(df, COL_LASTV, fill_large=True)
    df[COL_VINFMM + "_num"], df[COL_VINFMM + "_num_sort"] = to_numeric_fill(df, COL_VINFMM, fill_large=True)

    # Additional numeric conversions (if requested)
    if SORT_DV_ESCAPE:
        df[COL_DV_ESCAPE + "_num"], df[COL_DV_ESCAPE + "_num_sort"] = to_numeric_fill(df, COL_DV_ESCAPE, fill_large=True)
    if SORT_DV_MOI:
        df[COL_DV_MOI + "_num"], df[COL_DV_MOI + "_num_sort"] = to_numeric_fill(df, COL_DV_MOI, fill_large=True)
    if SORT_SUM_DELTAV:
        df[COL_SUM_DELTAV + "_num"], df[COL_SUM_DELTAV + "_num_sort"] = to_numeric_fill(df, COL_SUM_DELTAV, fill_large=True)

    # Apply range filters
    mask_range = (
        (df[COL_C3 + "_num"].between(C3_MIN, C3_MAX, inclusive="both"))
        & (df[COL_LASTV + "_num"].between(LASTV_MIN, LASTV_MAX, inclusive="both"))
    )
    df_filtered = df[mask_range].copy()
    print(f"After C3 & last_leg_vinf range filter -> {len(df_filtered):,} rows")

    # Enforce bending_feasible if requested
    if REQUIRE_BENDING_FEASIBLE:
        # Accept common truthy string representations + boolean-like values
        def parse_bool_like(x):
            if pd.isna(x): return False
            s = str(x).strip().upper()
            if s in {"TRUE","T","1","YES","Y"}: return True
            if s in {"FALSE","F","0","NO","N"}: return False
            # fallback: non-empty string -> True
            return bool(s)
        df_filtered["bending_feasible_bool"] = df_filtered["bending_feasible"].apply(parse_bool_like)
        df_filtered = df_filtered[df_filtered["bending_feasible_bool"] == True].copy()
        print(f"After bending_feasible filter -> {len(df_filtered):,} rows")

    # Optionally drop rows missing any of the three primary numeric keys
    if DROP_ROWS_MISSING_PRIMARY:
        before = len(df_filtered)
        df_filtered = df_filtered.dropna(subset=[COL_C3 + "_num", COL_LASTV + "_num", COL_VINFMM + "_num"])
        after = len(df_filtered)
        print(f"Dropped {before-after:,} rows with missing primary numeric keys -> {after:,} rows remain")

    if len(df_filtered) == 0:
        print("No rows left after filtering. Exiting.")
        df_filtered.to_csv(OUTPUT_CSV, index=False)
        print("Wrote empty CSV:", OUTPUT_CSV)
        return

    # Prepare deterministic sort order: primary keys first
    sort_order = [
        (COL_C3 + "_num_sort", True),
        (COL_LASTV + "_num_sort", True),
        (COL_VINFMM + "_num_sort", True),
    ]

    # Append optional fields if toggled (they will be sorted ascending)
    if SORT_DV_ESCAPE:
        sort_order.append((COL_DV_ESCAPE + "_num_sort", True))
    if SORT_DV_MOI:
        sort_order.append((COL_DV_MOI + "_num_sort", True))
    if SORT_SUM_DELTAV:
        sort_order.append((COL_SUM_DELTAV + "_num_sort", True))

    # Build args for sort_values
    sort_cols = [c for c,asc in sort_order]
    asc_flags = [asc for c,asc in sort_order]

    # Do the sort (NaNs already mapped to large fill value via _num_sort)
    df_sorted = df_filtered.sort_values(by=sort_cols, ascending=asc_flags, na_position="last").reset_index(drop=True)

    # Save cleaned/sorted CSV
    df_sorted.to_csv(OUTPUT_CSV, index=False)
    print(f"Sorted output saved to: {OUTPUT_CSV}  (rows: {len(df_sorted):,})")

    # Print summary top rows (showing main columns)
    display_cols = [COL_C3, COL_LASTV, COL_VINFMM]
    if SORT_DV_ESCAPE: display_cols.append(COL_DV_ESCAPE)
    if SORT_DV_MOI: display_cols.append(COL_DV_MOI)
    if SORT_SUM_DELTAV: display_cols.append(COL_SUM_DELTAV)
    display_cols += ["chain_str", "seed_epoch_iso", "tof_combo_days", "bending_feasible"]
    display_cols = [c for c in display_cols if c in df_sorted.columns]

    print("\nTop rows after sort:")
    print(df_sorted[display_cols].head(N_TOP_SHOW).to_string(index=False))

if __name__ == "__main__":
    main()
