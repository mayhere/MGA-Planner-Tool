# This script computes the porkchop plot for a ballistic transfer from Earth to Mercury.
# It uses SPICE kernels for planetary ephemerides and poliastro for trajectory calculations.
# The results are saved as CSV files and visualized with contour plots.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
import spiceypy as spice
from poliastro.iod import izzo

# =========================================================
# CONFIG
# =========================================================
KERNELS = [
    "C:/Users/Mayank/Desktop/VSSC-ISRO/Codes/Porkchop Plots/kernels/naif0012.tls",  # Leap seconds
    "C:/Users/Mayank/Desktop/VSSC-ISRO/Codes/Porkchop Plots/kernels/pck00010.tpc",  # Planetary constants
    "C:/Users/Mayank/Desktop/VSSC-ISRO/Codes/Porkchop Plots/kernels/de440.bsp",     # Planet ephemerides
]

FRAME_ECL = "ECLIPJ2000"   # Heliocentric ecliptic of J2000
FRAME_EQ  = "J2000"        # Heliocentric equatorial J2000
CENTER    = "SUN"

DEP = "VENUS"
ARR = "MERCURY"

DEBUG = False

# Sun GM (km^3/s^2)
MU_SUN = 1.32712440018e11 * u.km**3 / u.s**2

# Nominal dates (same as your MATLAB inputs)
LAUNCH_NOMINAL = Time("2026-10-01", scale="utc")
ARRIVE_NOMINAL = Time("2027-04-30", scale="utc")

SPAN_DAYS_DEPARTURE = 50  # ± span around each nominal date
SPAN_DAYS_ARRIVAL = 30   # ± span around each nominal date
STEP_DAYS = 0.5      # grid step
TOF_MIN_D = 50    # Mercury sanity filter
TOF_MAX_D = 300   # Mercury sanity filter

# MATLAB default contour levels (you can tweak)
C3_LEVELS   = list(np.arange(10, 151, 5))  # C3 from 10 to 150 km²/s² in steps of 5
VINF_LEVELS = list(np.arange(4, 18, 1))  # 4 to 17 km/s in steps of 1
DLA_LEVELS  = list(range(-90, 91, 10))  # DLA from -90 to +90 degrees in steps of 10
RLA_LEVELS  = list(range(0, 361, 10)) # RLA from 0 to 360 degrees in steps of 10
TOF_LEVELS  = list(range(50, 251, 10))  # TOF from 50 to 250 days in steps of 10
DVT_LEVELS  = list(np.arange(0,30,1)) # Total ΔV from 0 to 30 km/s in steps of 1

# Output folder
OUTDIR = "C:/Users/Mayank/Desktop/VSSC-ISRO/Codes/Porkchop Plots/Results_Venus_Mercury"
os.makedirs(OUTDIR, exist_ok=True)

# =========================================================
# UTILITIES
# =========================================================
def furnish_kernels(KERNELS):
    for k in KERNELS:
        if not os.path.exists(k):
            raise FileNotFoundError(f"Missing SPICE kernel: {k}")
        spice.furnsh(k)

def unload_kernels():
    spice.kclear()

def state_wrt_sun(target: str, et: float, frame: str = FRAME_ECL):
    """Return heliocentric state (r[km], v[km/s]) in given frame."""
    state, _ = spice.spkezr(target, et, frame, "NONE", CENTER)
    r = np.array(state[:3])
    v = np.array(state[3:])
    return r, v

def to_equatorial(vec_ecl: np.ndarray, et: float):
    """Transform vector from ECLIPJ2000 to J2000 (equatorial) at epoch et."""
    xform = spice.pxform(FRAME_ECL, FRAME_EQ, et)
    return xform @ vec_ecl

def radec_deg(vec: np.ndarray):
    """Right ascension and declination (deg) from equatorial vector (J2000)."""
    v = np.array(vec, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return np.nan, np.nan
    x, y, z = v / n
    dec = np.degrees(np.arcsin(z))                        # declination
    ra  = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0  # right ascension
    return ra, dec

def save_grid_csv(path, grid, launch_offsets, arrival_offsets):
    """Save 2D grid as CSV with labelled axes (rows=launch offsets, cols=arrival offsets)."""
    df = pd.DataFrame(grid, index=launch_offsets, columns=arrival_offsets)
    df.index.name = "launch_offset_days"
    df.columns.name = "arrival_offset_days"
    df.to_csv(path)

def contour_plot(
    ax, X, Y, Z1, levels1, color1, label1,
    Z2, levels2, color2, label2,
    title, xlab, ylab
):
    """
    Draw two contour sets on ax.
    - Ensures contour levels are increasing (required by matplotlib).
    - If either contour label contains 'TOF', rotate that Z-grid by 180°
      (flip both axes) before plotting so TOF isolines visually increase
      from bottom-right -> top-left while keeping axes/ticks unchanged.
    - When TOF is present, increase clabel inline spacing for readability.
    """

    # Detect TOF contours by label text
    is_tof_1 = "TOF" in (label1 or "").upper()
    is_tof_2 = "TOF" in (label2 or "").upper()

    # Ensure level arrays are numpy and strictly increasing (matplotlib needs this)
    levels1 = np.array(levels1, dtype=float)
    levels2 = np.array(levels2, dtype=float)
    if not np.all(np.diff(levels1) > 0):
        levels1 = np.sort(levels1)
    if not np.all(np.diff(levels2) > 0):
        levels2 = np.sort(levels2)

    # For TOF-like quantities, rotate the Z grid by 180 degrees (flip both axes)
    # so that visually the isolines ascend from bottom-right -> top-left.
    Z1_plot = np.rot90(Z1, 2) if (is_tof_1 and (Z1 is not None)) else Z1
    Z2_plot = np.rot90(Z2, 2) if (is_tof_2 and (Z2 is not None)) else Z2

    # First contour set
    c1 = ax.contour(X, Y, Z1_plot, levels=levels1, colors=color1, linewidths=0.6)
    if is_tof_1:
        ax.clabel(c1, fmt="%g", fontsize=10, inline_spacing=12)  # larger spacing for TOF
    else:
        ax.clabel(c1, fmt="%g", fontsize=8)

    # Second contour set
    c2 = ax.contour(X, Y, Z2_plot, levels=levels2, colors=color2, linewidths=0.6)
    if is_tof_2:
        ax.clabel(c2, fmt="%g", fontsize=10, inline_spacing=12)
    else:
        ax.clabel(c2, fmt="%g", fontsize=8)

    # Title & axis labels
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(xlab, fontsize=7, fontweight="bold")
    ax.set_ylabel(ylab, fontsize=7, fontweight="bold")

    # Keep axis limits exactly as mesh extents
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    # Crosshairs & grid
    ax.axhline(0, color="k", linestyle="--", linewidth=0.3)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.3)
    ax.grid(True, linestyle=":", linewidth=0.7)

    # Legend (best-effort)
    try:
        ax.legend(
            [c1.legend_elements()[0][0], c2.legend_elements()[0][0]],
            [label1, label2],
            loc="upper right",
            fontsize=9
        )
    except Exception:
        pass

# =========================================================
# MAIN
# =========================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    furnish_kernels(KERNELS)

    et_dep0 = spice.utc2et(LAUNCH_NOMINAL.utc.iso)
    et_arr0 = spice.utc2et(ARRIVE_NOMINAL.utc.iso)
    state_dep0, _ = spice.spkezr(DEP, et_dep0, FRAME_ECL, "NONE", CENTER)
    state_arr0, _ = spice.spkezr(ARR, et_arr0, FRAME_ECL, "NONE", CENTER)
    r1_0 = np.array(state_dep0[:3]); r2_0 = np.array(state_arr0[:3])
    tof0 = (et_arr0 - et_dep0)/86400.0
    print(f"[sanity] TOF nominal days = {tof0:.1f}")
    vd0, va0 = izzo.lambert(MU_SUN, r1_0*u.km, r2_0*u.km, tof0*u.day)
    print("[sanity] Lambert nominal OK")

    try:
        # Build date vectors
        launch_offsets = np.arange(-SPAN_DAYS_DEPARTURE, SPAN_DAYS_DEPARTURE + 1, STEP_DAYS, dtype=int)
        arrival_offsets = np.arange(-SPAN_DAYS_ARRIVAL, SPAN_DAYS_ARRIVAL + 1, STEP_DAYS, dtype=int)

        launch_dates = LAUNCH_NOMINAL + launch_offsets * u.day
        arrival_dates = ARRIVE_NOMINAL + arrival_offsets * u.day

        # Allocate grids (rows=launch, cols=arrival)
        shape = (len(launch_dates), len(arrival_dates))
        C3_grid       = np.full(shape, np.nan, dtype=float)
        VINF_arr_grid = np.full(shape, np.nan, dtype=float)
        TOF_grid      = np.full(shape, np.nan, dtype=float)
        DVT_grid      = np.full(shape, np.nan, dtype=float)
        DLA_dep_grid  = np.full(shape, np.nan, dtype=float)
        RLA_dep_grid  = np.full(shape, np.nan, dtype=float)
        DLA_arr_grid  = np.full(shape, np.nan, dtype=float)
        RLA_arr_grid  = np.full(shape, np.nan, dtype=float)

        # Auto-scale ΔV levels from min→max
        finite_vals = DVT_grid[np.isfinite(DVT_grid)]
        if finite_vals.size > 0:
            DVT_LEVELS = np.linspace(finite_vals.min(), finite_vals.max(), 30)
        else:
            DVT_LEVELS  = list(np.arange(0,30,1))  # fallback

        # Precompute ET arrays
        et_launch = np.array([spice.utc2et(t.utc.iso) for t in launch_dates])
        et_arrive = np.array([spice.utc2et(t.utc.iso) for t in arrival_dates])

        ok = 0
        fail = 0
        skipped = 0

        for i, et_dep in enumerate(et_launch):
            # one call only (faster + consistent)
            state_dep, _ = spice.spkezr(DEP, et_dep, FRAME_ECL, "NONE", CENTER)
            r1 = np.array(state_dep[:3])
            v1 = np.array(state_dep[3:])

            for j, et_arr in enumerate(et_arrive):
                tof_days = (et_arr - et_dep)/86400.0
                if tof_days < TOF_MIN_D or tof_days > TOF_MAX_D:
                    skipped += 1
                    continue

                state_arr, _ = spice.spkezr(ARR, et_arr, FRAME_ECL, "NONE", CENTER)
                r2 = np.array(state_arr[:3])
                v2 = np.array(state_arr[3:])

                try:
                    v_dep, v_arr = izzo.lambert(
                        MU_SUN,
                        r1 * u.km,
                        r2 * u.km,
                        tof_days * u.day
                    )
                except Exception as e:
                    fail += 1
                    # If debugging the first few failures helps:
                    if DEBUG and fail < 5:
                        print(f"LAMFAIL i={i}, j={j}, tof_days={tof_days:.1f} -> {e}")
                    continue

                dv1_vec = (v_dep - v1*(u.km/u.s)).to(u.km/u.s).value
                dv2_vec = (v_arr - v2*(u.km/u.s)).to(u.km/u.s).value

                C3_grid[i, j]       = float(np.dot(dv1_vec, dv1_vec))
                VINF_arr_grid[i, j] = float(np.linalg.norm(dv2_vec))
                TOF_grid[i, j]      = float(tof_days)
                DVT_grid[i, j]      = float(np.linalg.norm(dv1_vec) + np.linalg.norm(dv2_vec))

                dv1_eq = to_equatorial(dv1_vec, et_dep)
                rla_dep, dla_dep = radec_deg(dv1_eq)
                RLA_dep_grid[i, j] = rla_dep
                DLA_dep_grid[i, j] = dla_dep

                dv2_eq = to_equatorial(-dv2_vec, et_arr)
                rla_arr, dla_arr = radec_deg(dv2_eq)
                RLA_arr_grid[i, j] = rla_arr
                DLA_arr_grid[i, j] = dla_arr

                ok += 1

        print(f"[diag] ok={ok}, fail={fail}, skipped={skipped}")

        # Save CSVs (rows=launch offsets, cols=arrival offsets)
        save_grid_csv(os.path.join(OUTDIR, "C3_km2s2.csv"),       C3_grid,       launch_offsets, arrival_offsets)
        save_grid_csv(os.path.join(OUTDIR, "Vinf_arr_kms.csv"),   VINF_arr_grid, launch_offsets, arrival_offsets)
        save_grid_csv(os.path.join(OUTDIR, "TOF_days.csv"),       TOF_grid,      launch_offsets, arrival_offsets)
        save_grid_csv(os.path.join(OUTDIR, "DVT_kms.csv"),        DVT_grid,      launch_offsets, arrival_offsets)
        save_grid_csv(os.path.join(OUTDIR, "DLA_dep_deg.csv"),    DLA_dep_grid,  launch_offsets, arrival_offsets)
        save_grid_csv(os.path.join(OUTDIR, "RLA_dep_deg.csv"),    RLA_dep_grid,  launch_offsets, arrival_offsets)
        save_grid_csv(os.path.join(OUTDIR, "DLA_arr_deg.csv"),    DLA_arr_grid,  launch_offsets, arrival_offsets)
        save_grid_csv(os.path.join(OUTDIR, "RLA_arr_deg.csv"),    RLA_arr_grid,  launch_offsets, arrival_offsets)

        # Build plotting mesh (X=arrival offset, Y=launch offset)
        X, Y = np.meshgrid(arrival_offsets, launch_offsets)

        # === Auto-center the contour region (without changing scale or axis range) ===
        # We'll compute the centroid of valid (non-NaN) C3 values and shift so it's centered
        valid_mask = np.isfinite(C3_grid)
        if np.any(valid_mask):
            # Get weighted centroid (using C3 inverse as a weight so low-C3 region dominates)
            weights = 1.0 / np.clip(C3_grid, 1e-3, np.nanmax(C3_grid))
            weights[~valid_mask] = 0.0
            cx = np.nansum(X * weights) / np.nansum(weights)
            cy = np.nansum(Y * weights) / np.nansum(weights)

            # Current grid center
            x_mid = 0.5 * (X.max() + X.min())
            y_mid = 0.5 * (Y.max() + Y.min())

            # Shift so that low-C3 centroid aligns with geometric plot center
            X_centered = X - (cx - x_mid)
            Y_centered = Y - (cy - y_mid)
        else:
            # Fallback if no valid data
            X_centered, Y_centered = X, Y
        
        # 1) C3 + v∞_arr
        fig, ax = plt.subplots(figsize=(7, 7))
        contour_plot(
            ax, X_centered, Y_centered, C3_grid, C3_LEVELS, 'r', 'C3L (km²/s²)',
            VINF_arr_grid, VINF_LEVELS, 'b', 'Arrival v∞ (km/s)',
            "Ballistic Earth→Mercury\nLaunch C3 and Arrival v∞",
            f"Days relative to arrival date {ARRIVE_NOMINAL.utc.iso}",
            f"Days relative to launch date {LAUNCH_NOMINAL.utc.iso}"
        )
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(OUTDIR, "plot_c3l_vinf.png"), dpi=300)

        # 2) C3 + DLA (launch)
        fig, ax = plt.subplots(figsize=(7, 7))
        contour_plot(
            ax, X_centered, Y_centered, DLA_dep_grid, DLA_LEVELS, 'b', 'DLA (deg)',
            C3_grid, C3_LEVELS, 'r', 'C3L (km²/s²)',
            "Ballistic Earth→Mercury\nLaunch C3 and DLA",
            f"Days relative to arrival date {ARRIVE_NOMINAL.utc.iso}",
            f"Days relative to launch date {LAUNCH_NOMINAL.utc.iso}"
        )
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(OUTDIR, "plot_c3l_dla.png"), dpi=300)

        # 3) C3 + RLA (launch)
        fig, ax = plt.subplots(figsize=(7, 7))
        contour_plot(
            ax, X_centered, Y_centered, RLA_dep_grid, RLA_LEVELS, 'b', 'RLA (deg)',
            C3_grid, C3_LEVELS, 'r', 'C3L (km²/s²)',
            "Ballistic Earth→Mercury\nLaunch C3 and RLA",
            f"Days relative to arrival date {ARRIVE_NOMINAL.utc.iso}",
            f"Days relative to launch date {LAUNCH_NOMINAL.utc.iso}"
        )
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(OUTDIR, "plot_c3l_rla.png"), dpi=300)

        # 4) C3 + TOF  (your main porkchop)
        fig, ax = plt.subplots(figsize=(7, 7))
        contour_plot(
            ax, X_centered, Y_centered, TOF_grid, TOF_LEVELS, 'b', 'TOF (days)',
            C3_grid, C3_LEVELS, 'r', 'C3L (km²/s²)',
            "Ballistic Earth→Mercury\nLaunch C3 and Flight Time",
            f"Days relative to arrival date {ARRIVE_NOMINAL.utc.iso}",
            f"Days relative to launch date {LAUNCH_NOMINAL.utc.iso}"
        )
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(OUTDIR, "plot_c3l_tof.png"), dpi=300)

        # 5) v∞_arr + DLA (arrival)
        fig, ax = plt.subplots(figsize=(7, 7))
        contour_plot(
            ax, X_centered, Y_centered, VINF_arr_grid, VINF_LEVELS, 'b', 'Arrival v∞ (km/s)',
            DLA_arr_grid, DLA_LEVELS, 'r', 'DLA (deg)',
            "Ballistic Earth→Mercury\nArrival v∞ and DLA",
            f"Days relative to arrival date {ARRIVE_NOMINAL.utc.iso}",
            f"Days relative to launch date {LAUNCH_NOMINAL.utc.iso}"
        )
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(OUTDIR, "plot_vinf_dla.png"), dpi=300)

        # 6) v∞_arr + RLA (arrival)
        fig, ax = plt.subplots(figsize=(7, 7))
        contour_plot(
            ax, X_centered, Y_centered, VINF_arr_grid, VINF_LEVELS, 'b', 'Arrival v∞ (km/s)',
            RLA_arr_grid, RLA_LEVELS, 'r', 'RLA (deg)',
            "Ballistic Earth→Mercury\nArrival v∞ and RLA",
            f"Days relative to arrival date {ARRIVE_NOMINAL.utc.iso}",
            f"Days relative to launch date {LAUNCH_NOMINAL.utc.iso}"
        )
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(OUTDIR, "plot_vinf_rla.png"), dpi=300)

        # 7) TOF + total ΔV
        fig, ax = plt.subplots(figsize=(7, 7))
        contour_plot(
            ax, X_centered, Y_centered, TOF_grid, TOF_LEVELS, 'b', 'TOF (days)',
            DVT_grid, DVT_LEVELS, 'r', 'DVT (km/s)',
            "Ballistic Earth→Mercury\nTotal ΔV and Flight Time",
            f"Days relative to arrival date {ARRIVE_NOMINAL.utc.iso}",
            f"Days relative to launch date {LAUNCH_NOMINAL.utc.iso}"
        )
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(OUTDIR, "plot_dvt_tof.png"), dpi=300)

        print(f"Done. CSVs and plots saved in: {OUTDIR}")

    finally:
        unload_kernels()

if __name__ == "__main__":
    main()
