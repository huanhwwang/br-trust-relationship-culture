"""
Analysis 5: Meta-Analytic Connectivity Modelling (MACM) Inference
Uses our coordinate pools to estimate which regions co-activate with
vmPFC (EuA seed) vs. dmPFC (EA seed), by counting how many studies
in each pool report activation near each ROI.

This is a simplified MACM: for each pool, compute the fraction of studies
that report any coordinate within 15 mm of each brain ROI centroid.
The result approximates which network the seed is embedded in within each
cultural literature.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
EUA_COL, EUA_LIGHT = "#2563A8", "#92B8D8"
EA_COL,  EA_LIGHT  = "#C0392B", "#E89080"
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT_DIR, "coordinates.tsv"), sep="\t")

# Seeds: vmPFC centroid (EuA) and dmPFC centroid (EA) from Analysis 2
SEEDS = {
    "vmPFC_EuA": np.array([5.3, 39.6, -9.5]),   # EuA vmPFC centroid
    "dmPFC_EA":  np.array([-0.3, 52.9, 14.0]),  # EA dmPFC centroid
}

# ROI centroids to test co-activation with (from Analysis 2 results)
ROI_CENTROIDS = {
    "vmPFC":       np.array([5.3,  39.6, -9.5]),
    "dmPFC":       np.array([-0.3, 52.9, 14.0]),
    "TPJ_R":       np.array([54.0,-52.7, 18.0]),
    "TPJ_L":       np.array([-52.0,-56.0, 20.0]),
    "amygdala_R":  np.array([23.0, -3.0,-18.0]),
    "amygdala_L":  np.array([-21.0,-3.0,-16.0]),
    "caudate":     np.array([4.2,  12.2,  2.6]),
    "aInsula_R":   np.array([39.3, 21.3,  1.7]),
    "ACC":         np.array([2.0,  30.0, 21.0]),
    "precuneus":   np.array([0.7, -60.0, 30.0]),
    "OFC":         np.array([16.0, 26.0,-15.5]),
}

RADIUS_MM = 15.0  # co-activation radius

# ── MACM: for each seed × pool, compute co-activation frequency ───────────────
records = []
for seed_label, seed_xyz in SEEDS.items():
    for pool_label in ["EuA", "EA"]:
        pool_df = df[df.pool == pool_label]
        n_studies = pool_df.study_id.nunique()

        for roi_label, roi_xyz in ROI_CENTROIDS.items():
            # Find studies with any coord within RADIUS_MM of this ROI
            studies_near_roi = set()
            for sid, grp in pool_df.groupby("study_id"):
                coords = grp[["x","y","z"]].values
                dists = np.linalg.norm(coords - roi_xyz, axis=1)
                if np.any(dists <= RADIUS_MM):
                    studies_near_roi.add(sid)

            # Seed relevance: is this seed within RADIUS_MM of the ROI?
            seed_dist_to_roi = np.linalg.norm(seed_xyz - roi_xyz)
            seed_relevant = seed_dist_to_roi <= RADIUS_MM

            records.append({
                "seed":       seed_label,
                "pool":       pool_label,
                "roi":        roi_label,
                "n_studies":  n_studies,
                "n_coact":    len(studies_near_roi),
                "freq":       len(studies_near_roi) / n_studies,
                "seed_dist_mm": round(seed_dist_to_roi, 1),
            })

macm_df = pd.DataFrame(records)

# ── Pivot: co-activation frequencies ─────────────────────────────────────────
pivot_eua = macm_df[macm_df.pool=="EuA"].pivot(index="roi", columns="seed", values="freq")
pivot_ea  = macm_df[macm_df.pool=="EA"].pivot(index="roi", columns="seed", values="freq")

print("── MACM: Co-activation frequency (proportion of studies in pool) ─────────")
print("\nPool: EuA")
print(pivot_eua.to_string())
print("\nPool: EA")
print(pivot_ea.to_string())

# ── Key inference: differential network membership ────────────────────────────
# vmPFC network (EuA) vs dmPFC network (EA)
vmPFC_network_EuA = macm_df[(macm_df.pool=="EuA") &
                             (macm_df["seed"]=="vmPFC_EuA")].set_index("roi")["freq"]
dmPFC_network_EA  = macm_df[(macm_df.pool=="EA")  &
                             (macm_df["seed"]=="dmPFC_EA")].set_index("roi")["freq"]

network_df = pd.DataFrame({
    "vmPFC_network (EuA)": vmPFC_network_EuA,
    "dmPFC_network (EA)":  dmPFC_network_EA,
})
network_df["differential (EA-EuA)"] = (network_df["dmPFC_network (EA)"] -
                                        network_df["vmPFC_network (EuA)"])
network_df = network_df.sort_values("differential (EA-EuA)", ascending=False)

print("\n── vmPFC (EuA) vs dmPFC (EA) network co-activation ──────────────────────")
print(network_df.round(2).to_string())

# ── Figure: MACM network comparison ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("white")
fig.suptitle("MACM Network Co-activation  ·  EuA vmPFC network (blue) vs EA dmPFC network (red)",
             fontsize=11, fontweight="bold", color="#1A1A2E")

rois = list(ROI_CENTROIDS.keys())

# Panel A: vmPFC network (EuA) — blue
ax = axes[0]
vals_eua = vmPFC_network_EuA.reindex(rois).fillna(0)
colors = [EUA_COL if v > 0.3 else EUA_LIGHT for v in vals_eua]
ax.barh(rois, vals_eua, color=colors, alpha=0.88)
ax.set_xlabel("Co-activation frequency", fontsize=10)
ax.set_title("vmPFC Network\n(EuA)", fontweight="bold", color=EUA_COL, fontsize=11)
ax.set_xlim(0, 1)
ax.axvline(0.3, color="#AAAAAA", linewidth=0.8, linestyle="--")
ax.tick_params(labelsize=9)
ax.invert_yaxis()

# Panel B: dmPFC network (EA) — red
ax = axes[1]
vals_ea = dmPFC_network_EA.reindex(rois).fillna(0)
colors = [EA_COL if v > 0.3 else EA_LIGHT for v in vals_ea]
ax.barh(rois, vals_ea, color=colors, alpha=0.88)
ax.set_xlabel("Co-activation frequency", fontsize=10)
ax.set_title("dmPFC Network\n(EA)", fontweight="bold", color=EA_COL, fontsize=11)
ax.set_xlim(0, 1)
ax.axvline(0.3, color="#AAAAAA", linewidth=0.8, linestyle="--")
ax.tick_params(labelsize=9)
ax.invert_yaxis()

# Panel C: Differential — red = EA-enriched, blue = EuA-enriched
ax = axes[2]
diff_vals = network_df["differential (EA-EuA)"].reindex(rois).fillna(0)
colors = [EA_COL if v >= 0 else EUA_COL for v in diff_vals]
ax.barh(rois, diff_vals, color=colors, alpha=0.88)
ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
ax.set_xlabel("EA − EuA co-activation", fontsize=10)
ax.set_title("Differential\nRed = EA-enriched  ·  Blue = EuA-enriched",
             fontweight="bold", fontsize=10, color="#1A1A2E")
ax.tick_params(labelsize=9)
ax.invert_yaxis()

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_analysis5_macm_networks.pdf"),
            dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ MACM figure saved: fig_analysis5_macm_networks.pdf")

macm_df.to_csv(os.path.join(OUT_DIR, "analysis5_macm_results.csv"), index=False)
network_df.to_csv(os.path.join(OUT_DIR, "analysis5_network_comparison.csv"))
