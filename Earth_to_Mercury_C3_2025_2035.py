from astropy import units as u
from astropy.constants import G, M_earth, R_earth
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from poliastro.bodies import Sun
from poliastro.iod.izzo import lambert
from poliastro.util import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define Parking Orbits (elliptical) altitudes (from surface)
EPO_perigee = 250 * u.km
EPO_apogee = 23000 * u.km
MPO_periapsis = 500 * u.km
MPO_apoapsis = 50000 * u.km

# Planets radii
R_Earth = R_earth.to(u.km)
R_Mercury = 2439.7 * u.km

# Gravitational parameter for Earth
mu_earth = (G * M_earth).to(u.km**3 / u.s**2)
# Gravitational parameter for Mercury
mu_mercury = 22032 * u.km**3 / u.s**2

# Launch window
launch_start = Time("2025-01-01", scale="tdb")
launch_end = Time("2035-12-31", scale="tdb")
# Arrival window
arrival_start = Time("2025-03-01", scale="tdb")
arrival_end = Time("2036-03-01", scale="tdb")

num_launch = 200
num_arrival = 200

launch_dates = launch_start + (launch_end - launch_start) * np.linspace(0, 1, num_launch)
arrival_dates = arrival_start + (arrival_end - arrival_start) * np.linspace(0, 1, num_arrival)

results = []

# Threshold for V∞ Matching
threshold = 1.5 * u.km / u.s

matching_count = 0
total_valid = 0

for ld in launch_dates:
    for ad in arrival_dates:
        if ad <= ld:
            continue
        try:
            with solar_system_ephemeris.set("jpl"):
                r_Earth, v_Earth = get_body_barycentric_posvel("earth", ld)
                r_Mercury, v_Mercury = get_body_barycentric_posvel("mercury", ad)

            r1 = r_Earth.xyz.to(u.km)
            r2 = r_Mercury.xyz.to(u.km)

            tof = (ad - ld).to(u.s)
            v_depart, v_arrive = lambert(Sun.k, r1, r2, tof)

            v_inf_dep = norm(v_depart - v_Earth.to_cartesian().xyz.to(u.km/u.s))
            v_inf_arr = norm(v_arrive - v_Mercury.to_cartesian().xyz.to(u.km/u.s))
            C3L = v_inf_dep**2

            v_inf_match = abs(v_inf_dep - v_inf_arr) < threshold
            if v_inf_match:
                matching_count += 1
            total_valid += 1

            # RLA & DLA
            rla = np.arccos(np.clip(np.dot(r2.value, v_arrive.value) / (norm(r2).value * norm(v_arrive).value), -1, 1)) * u.rad
            dla = np.arccos(np.clip(np.dot(r1.value, v_depart.value) / (norm(r1).value * norm(v_depart).value), -1, 1)) * u.rad

            # Earth escape Δv at perigee
            r_perigee = R_Earth + EPO_perigee
            r_apogee = R_Earth + EPO_apogee
            a_park = (r_perigee + r_apogee) / 2

            v_perigee = np.sqrt(mu_earth.value * (2 / r_perigee.value - 1 / a_park.value))  # km/s
            v_hyper = np.sqrt(v_inf_dep.value**2 + 2 * mu_earth.value / r_perigee.value) # km/s
             # Departure delta-V (ΔV₁)
            delta_v_escape = v_hyper - v_perigee  # km/s

            # Mercury Capture Δv at perigee
            r_mercury_periapsis = R_Mercury + MPO_periapsis
            r_mercury_apoapsis = R_Mercury + MPO_apoapsis
            a_mercury_orbit = (r_mercury_periapsis + r_mercury_apoapsis) / 2

            # Velocity at periapsis of Mercury orbit (after capture)
            v_circ_peri_mercury = np.sqrt(mu_mercury.value * (2 / r_mercury_periapsis.value - 1 / a_mercury_orbit.value))
            v_hyper_arrival = np.sqrt(v_inf_arr.value**2 + 2 * mu_mercury.value / r_mercury_periapsis.value)

            # Capture delta-V (ΔV₂)
            delta_v_capture = v_hyper_arrival - v_circ_peri_mercury  # km/s

            # Total mission delta-V (ΔV₁ + ΔV₂)
            delta_v_total = delta_v_escape + delta_v_capture

            results.append({
                "Launch Date": ld.utc.iso,
                "Arrival Date": ad.utc.iso,
                "Time of Flight (days)": (ad - ld).to(u.day).value,
                "C3L (km²/s²)": C3L.value,
                "V∞ Departure (km/s)": v_inf_dep.value,
                "V∞ Arrival (km/s)": v_inf_arr.value,
                "DLA (deg)": dla.to(u.deg).value,
                "RLA (deg)": rla.to(u.deg).value,
                "ΔV_Escape_Perigee (km/s)": delta_v_escape,
                "ΔV_Capture_Peripasis (km/s)": delta_v_capture,
                "ΔV_Total (km/s)": delta_v_total,
                "V∞ Matching": v_inf_match,
                "Inclination (deg)": 20.7,
                "AOP (deg)": 179
            })

        except Exception as e:
            results.append({
                "Launch Date": ld.utc.iso,
                "Arrival Date": ad.utc.iso,
                "Time of Flight (days)": None,
                "C3L (km²/s²)": None,
                "V∞ Departure (km/s)": None,
                "V∞ Arrival (km/s)": None,
                "DLA (deg)": None,
                "RLA (deg)": None,
                "ΔV_Escape_Perigee (km/s)": None,
                "ΔV_Capture_Peripasis (km/s)": None,
                "ΔV_Total (km/s)": None,
                "V∞ Matching": None,
                "Inclination (deg)": None,
                "AOP (deg)": None,
                "Error": str(e)
            })

# === Save to DataFrame & CSV ===
df = pd.DataFrame(results)

# Clean & convert columns for matching logic
df = df[df["V∞ Departure (km/s)"].notna() & df["V∞ Arrival (km/s)"].notna()]
df["V∞ Departure (km/s)"] = df["V∞ Departure (km/s)"].astype(float)
df["V∞ Arrival (km/s)"] = df["V∞ Arrival (km/s)"].astype(float)

# Format Dates
df["Launch Date"] = pd.to_datetime(df["Launch Date"]).dt.strftime("%d-%m-%Y")
df["Arrival Date"] = pd.to_datetime(df["Arrival Date"]).dt.strftime("%d-%m-%Y")

# Count valid solutions and matching V∞
print(f"Total valid Lambert solutions: {total_valid}")
print(f"V∞ matching (|v∞ dep - v∞ arr| < {threshold} km/s): {matching_count}")

# Get minimum Δv value
min_delta_v = df["ΔV_Escape_Perigee (km/s)"].min()
print(f"Minimum ΔV_Escape_Perigee: {min_delta_v:.3f} km/s")

# Define tolerance for float comparison
tolerance = 1e-3  # km/s

# Filter only V∞ Matching = True cases
matching_df = df[df["V∞ Matching"] == True]

# Sort these cases by Δv_escape_perigee in ascending order
sorted_matching = matching_df.sort_values(by="ΔV_Escape_Perigee (km/s)", ascending=True)

# Get top 10 best cases
top_matches = sorted_matching.head(10)

# Print Top 10 V∞ matching cases with lowest Earth escape ΔV
if not top_matches.empty:
    print("🔍 Top 10 best V∞ matching cases with lowest Earth escape ΔV:")
    print(
        top_matches[
            ["Launch Date", "Arrival Date", "ΔV_Escape_Perigee (km/s)",
             "V∞ Departure (km/s)", "V∞ Arrival (km/s)", "C3L (km²/s²)", "Time of Flight (days)"]
        ].to_string(index=False)
    )
else:
    print("❌ No V∞ matching cases found.")

# Print Top 10 by Lowest Total ΔV Cases
df_valid_total = df[df["ΔV_Total (km/s)"].notna()]
top_total_dv = df_valid_total.sort_values(by="ΔV_Total (km/s)").head(10)

if not top_total_dv.empty:
    print("\n🚀 Top 10 lowest total ΔV (Escape + Capture):")
    print(
        top_total_dv[
            ["Launch Date", "Arrival Date", "ΔV_Total (km/s)", "ΔV_Escape_Perigee (km/s)",
             "ΔV_Capture_Peripasis (km/s)", "C3L (km²/s²)", "Time of Flight (days)"]
        ].to_string(index=False)
    )

# Print Top 10 Best C3L Cases (Lowest Launch Energy) with V∞ matching
sorted_c3 = matching_df.sort_values(by="C3L (km²/s²)", ascending=True)
top_c3 = sorted_c3.head(10)

if not top_c3.empty:
    print("\n🚀 Top 10 best cases with lowest Launch Energy (C3L) V∞ matching:")
    print(
        top_c3[
            ["Launch Date", "Arrival Date", "C3L (km²/s²)", "ΔV_Escape_Perigee (km/s)",
             "V∞ Departure (km/s)", "V∞ Arrival (km/s)", "Time of Flight (days)"]
        ].to_string(index=False)
    )
else:
    print("❌ No valid C3L cases found.")
    
# Save DataFrame
output_path = "C:/Users/Mayank/Desktop/VSSC-ISRO/Codes/C3_2025-2035/Earth_to_Mercury_C3_RLA_DLA_2025_2035.csv"
df.to_csv(output_path, index=False)
print(f"✅ Done! {len(df)} entries saved to Earth_to_Mercury_C3_RLA_DLA_2025_2035.csv")