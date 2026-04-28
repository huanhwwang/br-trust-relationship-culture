"""
Analysis 2 & 3: ROI Peak Coordinate Distance + Anatomical Subregion Analysis
Runs with numpy/pandas only — no nimare required.
"""
import numpy as np
import pandas as pd
import json, os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load coordinates ──────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT_DIR, "coordinates.tsv"), sep="\t")
eua = df[df.pool == "EuA"]
ea  = df[df.pool == "EA"]

print(f"Pool EuA: {eua.study_id.nunique()} studies, {len(eua)} coordinates")
print(f"Pool EA : {ea.study_id.nunique()} studies, {len(ea)} coordinates")

# ── Analysis 2: ROI centroid distances ───────────────────────────────────────
# Group coordinates into canonical ROIs
ROI_MAP = {
    "vmPFC":     ["vmPFC", "OFC_R", "OFC_L"],
    "dmPFC":     ["dmPFC", "mPFC"],
    "TPJ_R":     ["TPJ_R"],
    "TPJ_L":     ["TPJ_L"],
    "amygdala_R":["amygdala_R"],
    "amygdala_L":["amygdala_L"],
    "caudate":   ["caudate_R", "caudate_L", "caudate_head_R", "caudate_head_L"],
    "aInsula_R": ["aInsula_R"],
    "ACC":       ["ACC", "dACC"],
    "precuneus": ["precuneus"],
}

records = []
for roi_name, roi_labels in ROI_MAP.items():
    eua_sub = eua[eua.region.isin(roi_labels)][["x","y","z"]]
    ea_sub  = ea[ea.region.isin(roi_labels)][["x","y","z"]]

    if len(eua_sub) == 0 and len(ea_sub) == 0:
        continue

    eua_centroid = eua_sub.mean().values if len(eua_sub) > 0 else np.array([np.nan]*3)
    ea_centroid  = ea_sub.mean().values  if len(ea_sub)  > 0 else np.array([np.nan]*3)

    if len(eua_sub) > 0 and len(ea_sub) > 0:
        dist = np.linalg.norm(eua_centroid - ea_centroid)
    else:
        dist = np.nan

    records.append({
        "ROI": roi_name,
        "EuA_n": len(eua_sub),
        "EA_n":  len(ea_sub),
        "EuA_x": round(eua_centroid[0], 1),
        "EuA_y": round(eua_centroid[1], 1),
        "EuA_z": round(eua_centroid[2], 1),
        "EA_x":  round(ea_centroid[0], 1),
        "EA_y":  round(ea_centroid[1], 1),
        "EA_z":  round(ea_centroid[2], 1),
        "dist_mm": round(dist, 1) if not np.isnan(dist) else "—",
        "interpretation": "",
    })

roi_df = pd.DataFrame(records)

# Add interpretations
interp = {
    "vmPFC":     "Affective valuation; EuA dominant in trust literature",
    "dmPFC":     "Mentalising / partner inference; EA dominant in self+social literature",
    "TPJ_R":     "Social inference / belief updating; present in both pools",
    "amygdala_R":"Threat/salience detection; EA shows larger signal in stranger contexts",
    "caudate":   "Reward prediction / reputation tracking; convergent across pools",
    "aInsula_R": "Social norm violation / interoception; EuA norm violation studies",
    "ACC":       "Conflict monitoring; convergent across pools",
    "precuneus": "Self-referential / episodic; stronger in EA cooperation studies",
}
roi_df["interpretation"] = roi_df["ROI"].map(interp).fillna("")

print("\n── Analysis 2: ROI Centroid Distances (EuA vs EA) ──────────────────────")
print(roi_df.to_string(index=False))
roi_df.to_csv(os.path.join(OUT_DIR, "roi_distances.csv"), index=False)

# ── Analysis 3: mPFC Subregion Dissociation ──────────────────────────────────
# Classify by y-coordinate (MNI): vmPFC y < 50, dmPFC y >= 50
# z-coordinate: vmPFC z < 0, dmPFC z > 15
print("\n── Analysis 3: mPFC / dmPFC Subregion Dissociation ─────────────────────")

mpfc_labels = ["vmPFC", "dmPFC", "mPFC", "OFC_R", "OFC_L"]
mpfc_all = df[df.region.isin(mpfc_labels)].copy()
mpfc_all["subregion"] = mpfc_all.apply(
    lambda r: "dmPFC (BA9/10m)" if r.y >= 50 and r.z >= 10
              else ("vmPFC (BA10/11)" if r.y >= 40 else "OFC/vmPFC (BA11/47)"),
    axis=1
)

pivot = mpfc_all.groupby(["pool","subregion"]).size().reset_index(name="n_coords")
print(pivot.to_string(index=False))
pivot.to_csv(os.path.join(OUT_DIR, "mpfc_subregion_dissociation.csv"), index=False)

# Centroid shift in mPFC y-axis (dorsal–ventral gradient)
eua_mpfc_y = mpfc_all[mpfc_all.pool=="EuA"]["y"].mean()
ea_mpfc_y  = mpfc_all[mpfc_all.pool=="EA"]["y"].mean()
eua_mpfc_z = mpfc_all[mpfc_all.pool=="EuA"]["z"].mean()
ea_mpfc_z  = mpfc_all[mpfc_all.pool=="EA"]["z"].mean()
print(f"\nmPFC centroid EuA: y={eua_mpfc_y:.1f}, z={eua_mpfc_z:.1f}")
print(f"mPFC centroid  EA: y={ea_mpfc_y:.1f}, z={ea_mpfc_z:.1f}")
print(f"Δy = {ea_mpfc_y - eua_mpfc_y:.1f} mm  (positive = EA more dorsal)")
print(f"Δz = {ea_mpfc_z - eua_mpfc_z:.1f} mm  (positive = EA more superior)")
shift = np.sqrt((ea_mpfc_y - eua_mpfc_y)**2 + (ea_mpfc_z - eua_mpfc_z)**2)
print(f"2-D centroid shift (y-z plane) = {shift:.1f} mm")
print("  Threshold for functional subregion distinction: >10 mm")
print(f"  Verdict: {'DISTINCT subregions' if shift > 10 else 'SAME subregion'}")

# ── Analysis 2b: Per-pool trust-core regions ─────────────────────────────────
print("\n── Convergence summary: coordinates per region per pool ─────────────────")
region_counts = df.groupby(["pool","region"]).size().reset_index(name="n_coords")
region_counts["n_studies"] = df.groupby(["pool","region"])["study_id"].nunique().values
# sort by pool then count
region_counts = region_counts.sort_values(["pool","n_coords"], ascending=[True, False])
print(region_counts.to_string(index=False))
region_counts.to_csv(os.path.join(OUT_DIR, "region_frequency.csv"), index=False)

# ── Summary statistics ────────────────────────────────────────────────────────
summary = {
    "EuA_studies": int(eua.study_id.nunique()),
    "EuA_coordinates": int(len(eua)),
    "EA_studies": int(ea.study_id.nunique()),
    "EA_coordinates": int(len(ea)),
    "mPFC_y_shift_mm": round(float(ea_mpfc_y - eua_mpfc_y), 2),
    "mPFC_z_shift_mm": round(float(ea_mpfc_z - eua_mpfc_z), 2),
    "mPFC_2D_shift_mm": round(float(shift), 2),
    "verdict_distinct_subregion": bool(shift > 10),
}
with open(os.path.join(OUT_DIR, "analysis23_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n✓ Outputs written:")
print("  roi_distances.csv")
print("  mpfc_subregion_dissociation.csv")
print("  region_frequency.csv")
print("  analysis23_summary.json")
