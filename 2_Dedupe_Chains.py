# Dedupe_Chains.py
import pandas as pd
import os
INPUT = r"C:/Users/Mayank/Desktop/VSSC-ISRO/Codes/Earth-Mercury MGA/Tisserand_Chain_Finder/2026-27_Tisserand_Fast_Survivors.csv"
OUT = os.path.splitext(INPUT)[0] + "_Unique_Chains.csv"

df = pd.read_csv(INPUT)
# find chain column heuristically
if "chain_str" in df.columns:
    chain_col = "chain_str"
elif "chain" in df.columns:
    chain_col = "chain"
else:
    cand = None
    for c in df.columns:
        if df[c].astype(str).str.contains("--").any():
            cand = c; break
    if not cand:
        raise SystemExit("No chain column found; open CSV and check")
    chain_col = cand

grp = df.groupby(chain_col).size().reset_index(name="count").sort_values("count", ascending=False)
print("Unique chains:", len(grp))
print(grp.head(20).to_string(index=False))
grp.to_csv(OUT, index=False)
print("Wrote unique chains to:", OUT)
