"""
Analysis 1: Dual-Pool ALE (Activation Likelihood Estimation)
NiMARE 0.16 compatible.
"""
import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import nimare
from nimare.dataset import Dataset
from nimare.meta.cbma.ale import ALE
from nimare.correct import FWECorrector
from nilearn.image import threshold_img, math_img, load_img
from nilearn.reporting import get_clusters_table
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
# Canonical palette: EuA = blue, EA = red
EUA_COL, EA_COL = "#2563A8", "#C0392B"
CMAP_EUA, CMAP_EA, CMAP_CONJ = "Blues", "Reds", "Greens"
from nilearn import plotting

print(f"NiMARE version: {nimare.__version__}")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(OUT_DIR, "ale_maps")
os.makedirs(MAPS_DIR, exist_ok=True)

# ── 1. Load coordinates ───────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT_DIR, "coordinates.tsv"), sep="\t")

def build_nimare_dataset(pool_df):
    studies = {}
    for sid, grp in pool_df.groupby("study_id"):
        row = grp.iloc[0]
        coords = grp[["x","y","z"]].values.tolist()
        studies[sid] = {
            "contrasts": {"1": {
                "coords": {
                    "space": "MNI",
                    "x": [c[0] for c in coords],
                    "y": [c[1] for c in coords],
                    "z": [c[2] for c in coords],
                },
                "metadata": {"sample_sizes": [int(row["n"])]}
            }}
        }
    return Dataset(studies)

eua_df = df[df.pool == "EuA"]
ea_df  = df[df.pool == "EA"]
print(f"EuA: {eua_df.study_id.nunique()} studies, {len(eua_df)} coordinates")
print(f"EA : {ea_df.study_id.nunique()} studies, {len(ea_df)} coordinates")

eua_dset = build_nimare_dataset(eua_df)
ea_dset  = build_nimare_dataset(ea_df)

# ── 2. ALE + FWE correction ───────────────────────────────────────────────────
corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=1000, n_cores=4)

print("\nRunning EuA ALE + FWE (1000 permutations)...")
ale_eua = ALE(kernel__fwhm=None)
eua_res  = ale_eua.fit(eua_dset)
eua_cres = corr.transform(eua_res)

print("Running EA ALE + FWE (1000 permutations)...")
ale_ea  = ALE(kernel__fwhm=None)
ea_res  = ale_ea.fit(ea_dset)
ea_cres = corr.transform(ea_res)

# ── 3. Save maps ──────────────────────────────────────────────────────────────
STAT_MAP      = "stat"
LOGP_CLUST    = "logp_desc-size_level-cluster_corr-FWE_method-montecarlo"
Z_CLUST       = "z_desc-size_level-cluster_corr-FWE_method-montecarlo"
LOGP_VOXEL    = "logp_level-voxel_corr-FWE_method-montecarlo"

eua_res.get_map(STAT_MAP).to_filename(os.path.join(MAPS_DIR, "ale_eua_stat.nii.gz"))
ea_res.get_map(STAT_MAP).to_filename(os.path.join(MAPS_DIR, "ale_ea_stat.nii.gz"))
eua_cres.get_map(LOGP_CLUST).to_filename(os.path.join(MAPS_DIR, "ale_eua_FWE_logp_clust.nii.gz"))
ea_cres.get_map(LOGP_CLUST).to_filename(os.path.join(MAPS_DIR, "ale_ea_FWE_logp_clust.nii.gz"))
eua_cres.get_map(Z_CLUST).to_filename(os.path.join(MAPS_DIR, "ale_eua_FWE_z_clust.nii.gz"))
ea_cres.get_map(Z_CLUST).to_filename(os.path.join(MAPS_DIR, "ale_ea_FWE_z_clust.nii.gz"))

# ── 4. Conjunction & subtraction ─────────────────────────────────────────────
eua_stat = eua_res.get_map(STAT_MAP)
ea_stat  = ea_res.get_map(STAT_MAP)

conj_img        = math_img("np.minimum(img1, img2)", img1=eua_stat, img2=ea_stat)
diff_eua_gt_ea  = math_img("np.maximum(img1 - img2, 0)", img1=eua_stat, img2=ea_stat)
diff_ea_gt_eua  = math_img("np.maximum(img2 - img1, 0)", img1=eua_stat, img2=ea_stat)

conj_img.to_filename(os.path.join(MAPS_DIR, "ale_conjunction.nii.gz"))
diff_eua_gt_ea.to_filename(os.path.join(MAPS_DIR, "ale_EuA_minus_EA.nii.gz"))
diff_ea_gt_eua.to_filename(os.path.join(MAPS_DIR, "ale_EA_minus_EuA.nii.gz"))
print("\n✓ NIfTI maps saved.")

# ── 5. Extract cluster peaks ──────────────────────────────────────────────────
def report_peaks(cres, label, logp_key=LOGP_CLUST):
    logp_img = cres.get_map(logp_key)
    try:
        tbl = get_clusters_table(logp_img, stat_threshold=1.3,   # logp > 1.3 → p < 0.05
                                  cluster_threshold=10)
        if len(tbl) == 0:
            print(f"{label}: no suprathreshold clusters at p<0.05 FWE.")
        else:
            print(f"\n{label} — Significant clusters (p<0.05 FWE, k≥10 voxels):")
            cols = [c for c in ["Cluster ID","X","Y","Z","Peak Stat","Cluster Size (mm3)"]
                    if c in tbl.columns]
            print(tbl[cols].to_string(index=False))
            tbl.to_csv(os.path.join(OUT_DIR, f"ale_peaks_{label}.csv"), index=False)
    except Exception as e:
        print(f"  Cluster table error ({label}): {e}")
        # Fallback: report max voxel
        data = logp_img.get_fdata()
        idx = np.unravel_index(np.argmax(data), data.shape)
        affine = logp_img.affine
        xyz = nib.affines.apply_affine(affine, idx)
        print(f"  Peak voxel MNI: [{xyz[0]:.0f}, {xyz[1]:.0f}, {xyz[2]:.0f}]  "
              f"logp={data[idx]:.2f}")

report_peaks(eua_cres, "EuA")
report_peaks(ea_cres,  "EA")

# ── 6. Glass brain visualisations ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.patch.set_facecolor("white")
fig.suptitle("ALE: Cultural Trust Networks  ·  EuA (blue) vs EA (red)",
             fontsize=12, fontweight="bold", color="#1A1A2E")

# Row 0: EuA — blue
eua_logp = eua_cres.get_map(LOGP_CLUST)
for i, (disp, ax) in enumerate(zip(["x","y","z"], axes[0])):
    plotting.plot_glass_brain(eua_logp, threshold=1.3,
                               display_mode=disp, colorbar=(i==2),
                               cmap=CMAP_EUA, vmax=3.0,
                               title="EuA Trust Network" if i==1 else "",
                               axes=ax)

# Row 1: EA — red
ea_logp = ea_cres.get_map(LOGP_CLUST)
for i, (disp, ax) in enumerate(zip(["x","y","z"], axes[1])):
    plotting.plot_glass_brain(ea_logp, threshold=1.3,
                               display_mode=disp, colorbar=(i==2),
                               cmap=CMAP_EA, vmax=3.0,
                               title="EA Social Cognition Network" if i==1 else "",
                               axes=ax)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_analysis1_ale_glassbrain.pdf"),
            dpi=150, bbox_inches="tight")
plt.close()

# Side-by-side stat maps
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.patch.set_facecolor("white")
fig2.suptitle("ALE Stat Maps  ·  EuA (blue)  |  EA (red)  |  Conjunction (green)",
              fontsize=11, fontweight="bold", color="#1A1A2E")

# EuA ALE stat — blue
d = eua_stat.get_fdata(); vmax = np.percentile(d[d>0], 99) if (d>0).any() else 0.02
plotting.plot_stat_map(eua_stat, display_mode="z", cut_coords=[-18,-4,8,20,32],
                        threshold=vmax*0.1, vmax=vmax, cmap=CMAP_EUA,
                        title="EuA Trust Network", axes=axes2[0])

# EA ALE stat — red
d = ea_stat.get_fdata(); vmax = np.percentile(d[d>0], 99) if (d>0).any() else 0.02
plotting.plot_stat_map(ea_stat, display_mode="z", cut_coords=[-18,-4,8,20,32],
                        threshold=vmax*0.1, vmax=vmax, cmap=CMAP_EA,
                        title="EA Social Cognition Network", axes=axes2[1])

# Conjunction — green
d = conj_img.get_fdata(); vmax = np.percentile(d[d>0], 99) if (d>0).any() else 0.02
plotting.plot_stat_map(conj_img, display_mode="z", cut_coords=[-18,-4,8,20,32],
                        threshold=vmax*0.1, vmax=vmax, cmap=CMAP_CONJ,
                        title="Conjunction (Both Cultures)", axes=axes2[2])

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "fig_analysis1_ale_statmaps.pdf"),
             dpi=150, bbox_inches="tight")
plt.close()

print("\n✓ Figures saved:")
print("  fig_analysis1_ale_glassbrain.pdf")
print("  fig_analysis1_ale_statmaps.pdf")
print(f"\nAll maps in: {MAPS_DIR}")
for f in sorted(os.listdir(MAPS_DIR)):
    print(f"  {f}  ({os.path.getsize(os.path.join(MAPS_DIR,f))//1024} KB)")
