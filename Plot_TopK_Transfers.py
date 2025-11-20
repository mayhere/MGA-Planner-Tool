# Plot_TopK_Transfers.py
"""
Reads Top5_per_campaign.csv (user-specified path), and for each chain row:
 - build heliocentric, physically-correct transfer arcs using poliastro Orbit.from_vectors + propagation
 - plot planet orbits (heliocentric) and transfer arcs with Plotly (interactive HTML)
 - optionally export sampled ephemerides (CSV) per chain for STK import
 - optionally create PNG thumbnails (requires kaleido installed)
 - create gallery.html linking all generated HTMLs (and embedding thumbnails if available)

Configure paths & options in the CONFIG block below.
"""

import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel

from poliastro.bodies import Sun
from poliastro.iod.izzo import lambert
from poliastro.twobody import Orbit

import plotly.graph_objects as go
import plotly.io as pio

# --------------------- CONFIG  ---------------------
# Input CSV produced by StageC (Top-5 per campaign)
INPUT_CSV = r"C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/C_Verification_and_Optimization/2033-34_StageC_Top5_Optimized_Results.csv"

# Where to store HTMLs, ephemeris CSVs, thumbnails and gallery
OUT_DIR = r"C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/D_Plotting_and_Validation/2033-34/Ephemeris"
os.makedirs(OUT_DIR, exist_ok=True)

# Ephemeris kernel (jpl or builtin)
EPHEMERIS = "C:/Users/Mayank/Desktop/Earth-Mercury MGA_Tool/kernels/de440.bsp"

# Sampling densities
PLANET_ORBIT_SAMPLES = 3000    # how many points to draw full planet orbit lines
TRANSFER_SAMPLES = 3000        # how many points per transfer arc (increase for publication images)

# Transfer sampling export (CSV) toggle
EXPORT_EPHEMERIS_CSV = True    # writes per-chain CSV with epoch_iso,x,y,z,vx,vy,vz

# Thumbnail settings (requires kaleido)
MAKE_THUMBNAILS = True
THUMB_WIDTH = 520
THUMB_HEIGHT = 360

# Plot styling
LEG_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
PLANET_COLORS = {"Ea":"blue","Ve":"gold","Me":"grey","Ma":"red","Sun":"yellow"}

# Map short labels to astropy body strings accepted by get_body_barycentric_posvel
LABEL_TO_BODY = {"Ea": "earth", "Ve":"venus", "Me":"mercury", "Ma":"mars"}

# gallery file
GALLERY_FILE = os.path.join(OUT_DIR, "Gallery.html")
# ----------------------------------------------------

# helper utilities -----------------------------------------------------------
def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def parse_tof_combo(field):
    """
    parse string '20.00;30.00;146.67;...' OR list OR already numeric list
    returns list of floats (days)
    """
    if field is None:
        return []
    if isinstance(field, (list, tuple, np.ndarray)):
        return [safe_float(x) for x in field]
    s = str(field).strip()
    if s == "":
        return []
    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            # try comma-separated
            for q in p.replace(",", " ").split():
                try:
                    out.append(float(q))
                except Exception:
                    pass
    return out

def ensure_time(t):
    if isinstance(t, Time):
        return t
    try:
        return Time(str(t))
    except Exception:
        return Time(t, scale="tdb")

# ----------------- heliocentric helpers -------------------------------------
def get_body_heliocentric_rv(label, epoch):
    """
    Returns (r_vec, v_vec) heliocentric (Sun-centered) for a given body label
    r_vec, v_vec are astropy Quantity arrays (3,) in km and km/s respectively.
    """
    if label not in LABEL_TO_BODY:
        raise ValueError(f"Unknown body label {label}")
    bodyname = LABEL_TO_BODY[label]

    with solar_system_ephemeris.set(EPHEMERIS):
        r_body_b, v_body_b = get_body_barycentric_posvel(bodyname, epoch)
        r_sun_b, v_sun_b   = get_body_barycentric_posvel("sun", epoch)

    r_helio = (r_body_b.xyz - r_sun_b.xyz).to(u.km)
    v_helio = (v_body_b.xyz - v_sun_b.xyz).to(u.km/u.s)
    # r_helio and v_helio are 3x1 Quantity arrays; convert to 1D arrays
    rq = np.array(r_helio).reshape(3,) * u.km
    vq = np.array(v_helio).reshape(3,) * (u.km/u.s)
    return rq, vq

def sample_transfer_arc_by_propagation(r_dep, v_dep, dep_epoch, arr_epoch, samples=400):
    """
    Build a heliocentric Orbit from r_dep/v_dep (quantities) and propagate between dep_epoch and arr_epoch.
    Returns tuple (times, r_km_array (N,3), v_kms_array (N,3))
    times are astropy.Time object array of length N.
    """
    dep_epoch = ensure_time(dep_epoch)
    arr_epoch = ensure_time(arr_epoch)
    tof = (arr_epoch - dep_epoch).to(u.s).value
    if tof <= 0:
        return np.array([]), np.zeros((0,3)), np.zeros((0,3))

    try:
        orb = Orbit.from_vectors(Sun, r_dep, v_dep, epoch=dep_epoch)
    except Exception as e:
        # failed to build orbit (bad vectors)
        # print("Orbit.from_vectors failed:", e)
        return np.array([]), np.zeros((0,3)), np.zeros((0,3))

    secs = np.linspace(0.0, tof, samples)
    times = dep_epoch + TimeDelta(secs, format='sec')
    r_arr = np.zeros((samples, 3))
    v_arr = np.zeros((samples, 3))
    for i, s in enumerate(secs):
        try:
            st = orb.propagate(TimeDelta(s, format="sec"))
            r_arr[i, :] = st.r.to(u.km).value
            v_arr[i, :] = st.v.to(u.km/u.s).value
        except Exception:
            r_arr[i, :] = np.nan
            v_arr[i, :] = np.nan
    return times, r_arr, v_arr

# ----------------- plotting helpers ----------------------------------------
def build_fig_layout(title):
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X (km)"),
            yaxis=dict(title="Y (km)"),
            zaxis=dict(title="Z (km)"),
            aspectmode="data"
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=50, b=0),
        template="plotly_white",
    )
    return layout

def try_write_thumbnail(fig, out_png):
    """
    Try to export Plotly fig to PNG using kaleido. Return True if succeeded.
    """
    try:
        pio.write_image(fig, out_png, format="png", width=THUMB_WIDTH, height=THUMB_HEIGHT, scale=1)
        return True
    except Exception as e:
        # likely no kaleido; print once
        print("  thumbnail generation failed (kaleido missing?), continuing without PNG:", e)
        return False

# ----------------- export helpers ------------------------------------------
def export_ephemeris_csv(times, r_array, v_array, out_csv_path):
    """
    Write CSV with columns: epoch_iso, x_km,y_km,z_km,vx_kms,vy_kms,vz_kms
    times: astropy.Time array or list; r_array: Nx3; v_array: Nx3
    """
    rows = []
    for i, t in enumerate(times):
        iso = t.iso
        rx, ry, rz = r_array[i,:]
        vx, vy, vz = v_array[i,:]
        rows.append([iso, rx, ry, rz, vx, vy, vz])
    df = pd.DataFrame(rows, columns=["epoch_iso","x_km","y_km","z_km","vx_kms","vy_kms","vz_kms"])
    df.to_csv(out_csv_path, index=False)

def export_stk_sample_csv(times, r_array, v_array, out_csv_path):
    """
    STK import sample format: epoch_iso, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms
    Same as export_ephemeris_csv; provided for naming clarity.
    """
    export_ephemeris_csv(times, r_array, v_array, out_csv_path)

# ----------------- main chain plotting function ----------------------------
def plot_chain_row(row, out_dir=OUT_DIR, make_thumbnail=MAKE_THUMBNAILS, export_ephem=EXPORT_EPHEMERIS_CSV):
    """
    row: pandas Series with expected fields:
      - chain_str (e.g., Ea--Ma--Ve--Me)
      - seed_epoch_iso (ISO string)
      - tof_combo_days (semi-colon separated string)
      - optionally leg_v1_vectors_json, leg_v2_vectors_json (json lists of vectors)
      - optionally first_leg_C3_km2s2 and first_leg_vinf_kms (for annotation)
    """
    chain_str = str(row.get("chain_str", "")).strip()
    seed_iso = row.get("seed_epoch_iso", row.get("seed_epoch", ""))
    tof_combo_days = parse_tof_combo(row.get("tof_combo_days", ""))

    if chain_str == "":
        print("Skipping empty chain row.")
        return None

    try:
        seed_time = ensure_time(seed_iso)
    except Exception:
        seed_time = Time(seed_iso)

    parts = [p.strip() for p in chain_str.split("--") if p.strip() != ""]
    if len(parts) < 2:
        print("Chain too short:", chain_str)
        return None

    # load stored v1/v2 JSON if present
    v1_vecs = []
    v2_vecs = []
    try:
        raw1 = row.get("leg_v1_vectors_json", "[]")
        v1_vecs = json.loads(raw1) if raw1 and raw1 != "nan" else []
    except Exception:
        v1_vecs = []
    try:
        raw2 = row.get("leg_v2_vectors_json", "[]")
        v2_vecs = json.loads(raw2) if raw2 and raw2 != "nan" else []
    except Exception:
        v2_vecs = []

    traces = []
    # plot planet orbits for each unique body in parts (sample +/- 0.5 year around seed_time)
    unique_bodies = sorted(set(parts))
    for label in unique_bodies:
        if label not in LABEL_TO_BODY:
            continue
        # sample 1 year window centered at seed_time to show orbit ring
        start = seed_time - 0.5 * u.year
        stop  = seed_time + 0.5 * u.year
        times = start + (stop - start) * np.linspace(0, 1, PLANET_ORBIT_SAMPLES)
        coords = np.zeros((len(times), 3))
        with solar_system_ephemeris.set(EPHEMERIS):
            for i, t in enumerate(times):
                r_b, v_b = get_body_barycentric_posvel(LABEL_TO_BODY[label], t)
                r_sun_b, _ = get_body_barycentric_posvel("sun", t)
                r_helio = (r_b.xyz - r_sun_b.xyz).to(u.km)
                coords[i, :] = np.array(r_helio).reshape(3,)
        traces.append(go.Scatter3d(
            x=coords[:,0], y=coords[:,1], z=coords[:,2],
            mode="lines", name=f"{label} orbit",
            line=dict(width=2, dash="dash", color=PLANET_COLORS.get(label,"black")),
            opacity=0.6
        ))

    # Sun marker
    traces.append(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers+text',
                               name="Sun", marker=dict(size=8, color="yellow"), text=["Sun"],
                               textposition="bottom center"))

    # Walk each leg and build transfer arc using heliocentric lambert -> orbit propagation
    current_epoch = seed_time
    nlegs = len(parts) - 1
    # pad tof_combo to nlegs with 120d default
    if len(tof_combo_days) < nlegs:
        tof_combo_days = tof_combo_days + [120.0] * (nlegs - len(tof_combo_days))

    leg_color_idx = 0
    all_sampled_times = []
    all_sampled_r = []
    all_sampled_v = []

    for i in range(nlegs):
        dep_label = parts[i]; arr_label = parts[i+1]
        tof_days = tof_combo_days[i]
        tof = float(tof_days) * u.day
        arr_epoch = current_epoch + tof

        # get heliocentric r/v for bodies
        try:
            r_dep, v_dep_planet = get_body_heliocentric_rv(dep_label, current_epoch)
            r_arr, v_arr_planet = get_body_heliocentric_rv(arr_label, arr_epoch)
        except Exception as e:
            print("  Failed to get heliocentric r/v:", e)
            current_epoch = arr_epoch
            continue

        # if saved v1/v2 exist use them, else compute lambert
        v1_q = None; v2_q = None
        if i < len(v1_vecs) and v1_vecs[i] not in (None, [], "null"):
            try:
                v1_q = (np.array(v1_vecs[i]) * u.km / u.s)
            except Exception:
                v1_q = None
        if i < len(v2_vecs) and v2_vecs[i] not in (None, [], "null"):
            try:
                v2_q = (np.array(v2_vecs[i]) * u.km / u.s)
            except Exception:
                v2_q = None

        if v1_q is None or v2_q is None:
            try:
                v1_out, v2_out = lambert(Sun.k, r_dep, r_arr, tof)
                v1_q = v1_out
                v2_q = v2_out
            except Exception as e:
                print(f"  Lambert failed for leg {dep_label}->{arr_label} at {current_epoch.iso} -> {arr_epoch.iso}: {e}")
                current_epoch = arr_epoch
                continue

        # sample the true conic via propagation
        times_s, r_samps, v_samps = sample_transfer_arc_by_propagation(r_dep, v1_q, current_epoch, arr_epoch, samples=TRANSFER_SAMPLES)
        if times_s.size == 0 or r_samps.shape[0] == 0:
            current_epoch = arr_epoch
            continue

        color = LEG_COLORS[leg_color_idx % len(LEG_COLORS)]
        leg_label = f"{dep_label}->{arr_label} ({i+1})"
        traces.append(go.Scatter3d(x=r_samps[:,0], y=r_samps[:,1], z=r_samps[:,2],
                                   mode="lines", name=f"Transfer {leg_label}", line=dict(width=4, color=color)))

        # markers for dep/arr
        traces.append(go.Scatter3d(mode="markers+text", name=f"{dep_label} dep {i+1}",
                                   marker=dict(size=4, color=color)))
        traces.append(go.Scatter3d(x=[r_arr[0].value], y=[r_arr[1].value], z=[r_arr[2].value],
                                   mode="markers+text", name=f"{arr_label} arr {i+1}",
                                   marker=dict(size=4, color=color, symbol="diamond")))
        if parts == "Mercury":
            traces.append(go.Scatter3d(x=[r_arr[0].value], y=[r_arr[1].value], z=[r_arr[2].value],
                                   mode="markers+text", name=f"{arr_label} arr {i+1}",
                                   marker=dict(size=4, color=color, symbol="diamond"),
                                   text=[f"{arr_label} arr\n{arr_epoch.iso}"], textposition="top center"))

        # collect ephemeris for export
        all_sampled_times.append(times_s)           # Time array length M
        all_sampled_r.append(r_samps)               # Mx3
        all_sampled_v.append(v_samps)               # Mx3

        # annotate first-leg C3/vinf
        if i == 0:
            try:
                c3 = float(row.get("first_leg_C3_km2s2", np.nan))
            except Exception:
                c3 = np.nan
            try:
                vinf = float(row.get("first_leg_vinf_kms", np.nan))
            except Exception:
                vinf = np.nan
            ann_text = f"C3={c3:.3f} km²/s²<br>v∞={vinf:.3f} km/s"
            traces.append(go.Scatter3d(x=[r_dep[0].value], y=[r_dep[1].value], z=[r_dep[2].value],
                                       mode="text", text=[ann_text], showlegend=False))

        current_epoch = arr_epoch
        leg_color_idx += 1

    # Build figure
    title = f"{chain_str}  seed={seed_time.iso}"
    fig = go.Figure(data=traces, layout=build_fig_layout(title))
    fig.update_layout(scene_camera=dict(eye=dict(x=1.25, y=1.25, z=0.6)))

    safe_name = chain_str.replace("--", "_").replace(" ", "_")
    seed_tag = seed_time.strftime("%Y%m%dT%H%M%S")
    html_name = f"{safe_name}__{seed_tag}.html"
    html_path = os.path.join(out_dir, html_name)

    # Save interactive HTML
    try:
        pio.write_html(fig, file=html_path, full_html=True, include_plotlyjs="cdn")
        print("Saved HTML:", html_path)
    except Exception as e:
        print("Failed to save HTML:", e)
        # fallback to JSON if HTML fails
        json_path = html_path + ".json"
        with open(json_path, "w") as fh:
            fh.write(fig.to_json())
        print("Saved fallback JSON:", json_path)

    # Make thumbnail (PNG) if requested/possible
    thumb_path = os.path.join(out_dir, html_name.replace(".html", ".png"))
    thumb_ok = False
    if make_thumbnail:
        thumb_ok = try_write_thumbnail(fig, thumb_path)

    # Export sampled ephemeris to CSV (concatenate legs with increasing times)
    if export_ephem and len(all_sampled_times) > 0:
        # flatten arrays
        times_cat = np.concatenate(all_sampled_times)
        r_cat = np.vstack(all_sampled_r)
        v_cat = np.vstack(all_sampled_v)
        ephem_csv = os.path.join(out_dir, html_name.replace(".html", "_ephemeris.csv"))
        try:
            export_ephemeris_csv(times_cat, r_cat, v_cat, ephem_csv)
            # also produce STK sample file with a different name
            stk_csv = os.path.join(out_dir, html_name.replace(".html", "_STK_sample.csv"))
            export_stk_sample_csv(times_cat, r_cat, v_cat, stk_csv)
            print("  Exported ephemeris CSVs:", ephem_csv, stk_csv)
        except Exception as e:
            print("  Failed to export ephemeris CSV:", e)

    # return summary (for gallery building)
    return {"html": html_path, "thumb": thumb_path if thumb_ok else None, "chain": chain_str, "seed": seed_time.iso}

# ----------------- gallery builder -----------------------------------------
def build_gallery(entries, out_html=GALLERY_FILE):
    """
    entries: list of dicts returned by plot_chain_row: keys html, thumb, chain, seed
    """
    lines = []
    lines.append("<html><head><meta charset='utf-8'><title>Chain Gallery</title></head><body>")
    lines.append("<h1>MGA Top-K Chains Gallery</h1>")
    lines.append("<p>Generated: %s</p>" % datetime.now().isoformat())
    lines.append("<ul>")
    for e in entries:
        html_fn = os.path.basename(e["html"])
        thumb = e.get("thumb")
        lines.append("<li style='margin-bottom:12px;'>")
        if thumb and os.path.exists(thumb):
            thumb_fn = os.path.basename(thumb)
            lines.append(f"<a href='{html_fn}' target='_blank'><img src='{thumb_fn}' width='300' style='vertical-align:middle;margin-right:10px;'/></a>")
        else:
            lines.append(f"<a href='{html_fn}' target='_blank'>[Open {html_fn}]</a> &nbsp; ")
        lines.append(f"<b>{e['chain']}</b><br/>seed: {e['seed']}")
        lines.append("</li>")
    lines.append("</ul>")
    lines.append("</body></html>")
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("Gallery written:", out_html)

# ----------------- main entrypoint -----------------------------------------
def main():
    print("Plot_TopK_Transfers.py starting...")
    print("Input CSV:", INPUT_CSV)
    print("Output folder:", OUT_DIR)
    if not os.path.exists(INPUT_CSV):
        print("ERROR: input CSV not found:", INPUT_CSV)
        return

    try:
        df = pd.read_csv(INPUT_CSV, dtype=str)
    except Exception as e:
        print("Failed to read input CSV:", e)
        return

    # If it's already a chain-level StageC CSV (has seed_epoch_iso), great — proceed.
    if "chain_str" in df.columns and ("seed_epoch_iso" in df.columns or "seed_epoch" in df.columns):
        print("Using chain-level StageC CSV for plotting.")
        chain_df = df.copy()
    else:
        # Possibly a per-state heliocentric CSV produced by the converter.
        # Detect per-state CSV by presence of Time_TDB_ISO/Time_TDB or 'row_index' and X_km/Y_km/Z_km columns.
        per_state_detect = ("Time_TDB_ISO" in df.columns or "epoch_iso" in df.columns or "time_tdb_iso" in df.columns) and \
                           (("X_km" in df.columns and "Y_km" in df.columns and "Z_km" in df.columns) or ("x_km" in df.columns and "y_km" in df.columns and "z_km" in df.columns))
        if not per_state_detect:
            print("Input CSV missing necessary chain-level or per-state columns (chain_str/seed_epoch or Time_TDB_ISO + X_km/Y_km/Z_km).")
            return
        # require we have row_index to link back to original StageC
        if "row_index" not in df.columns:
            print("Per-state CSV detected but it lacks 'row_index'. Convert script must write 'row_index' to link to chain-level rows.")
            return

    entries = []
    # iterate rows (tqdm)
    for idx, row in tqdm(chain_df.iterrows(), total=len(chain_df), desc="Plotting chains"):
        try:
            res = plot_chain_row(row, out_dir=OUT_DIR, make_thumbnail=MAKE_THUMBNAILS, export_ephem=EXPORT_EPHEMERIS_CSV)
            if res:
                entries.append(res)
        except Exception as e:
            print("Error plotting row idx", idx, "chain:", row.get("chain_str"), ":", e)

    # copy thumbnail files into OUT_DIR (they are already created there)
    # produce gallery.html
    build_gallery(entries, out_html=GALLERY_FILE)
    print("Done. All HTMLs in:", OUT_DIR)
    print("If you need STK import examples, see one of the *_STK_sample.csv files next to each HTML (if exported).")

if __name__ == "__main__":
    main()
