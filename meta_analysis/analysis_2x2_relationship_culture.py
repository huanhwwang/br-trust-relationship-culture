"""
2×2 Factorial Analysis: Social Relationship (Stranger / Close Other) × Culture (EuA / EA)

Four pools:
  EuA-Stranger  — anonymous partner trust/cooperation games (Western)
  EuA-Close     — known/friend partner, reputation-based, in-group (Western)
  EA-Stranger   — anonymous economic games (East Asian)
  EA-Close      — close-other, in-group, self-referential social (East Asian)

Analyses:
  A. Pool composition & coordinate counts
  B. ROI centroid distances (2×2 table)
  C. mPFC subregion dissociation (4-cell breakdown)
  D. Functional decoding profiles (4-cell + interaction contrasts)
  E. MACM connectivity inference (seed × pool)
  F. Interaction figure: main effects + interaction decomposition
  G. ALE (marginal pools only — EuA-Stranger k=8; all others underpowered, exploratory)
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT_DIR, "ale_maps_2x2"), exist_ok=True)

# ── 0. Load & classify ────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT_DIR, "coordinates.tsv"), sep="\t")
df["paper"] = df["study_id"].str.replace(r"_\d+$", "", regex=True)

# Relationship classification by study
STRANGER = {
    # EuA — anonymous economic game
    "rilling2002","king_casas2008","krueger2007","vanden_bos2009",
    "baumgartner2008","harle2012","zak2011","sripada2009",
    # EA — anonymous economic game
    "takahashi2008","chang2010","zheng2014","sun2016",
    "haruno2014","takahashi2016",
}
CLOSE = {
    # EuA — known/friend partner or reputation-based
    "king_casas2005","fouragnan2013","delgado2005","dewall2012",
    "bellucci2017","fareri2012",
    # EA — close other, ingroup, self-referential social
    "chiao2009","chiao2010","zhu2007","han2012","ma2014","wang2011",
}

df["relationship"] = df["paper"].apply(
    lambda p: "stranger" if p in STRANGER else ("close" if p in CLOSE else "other"))
df = df[df["relationship"].isin(["stranger","close"])].copy()
df["cell"] = df["pool"] + "_" + df["relationship"].str.capitalize()

# ── A. Pool composition ───────────────────────────────────────────────────────
print("=" * 65)
print("ANALYSIS: 2×2 Social Relationship × Culture")
print("=" * 65)
for cell in ["EuA_Stranger","EuA_Close","EA_Stranger","EA_Close"]:
    sub = df[df.cell == cell]
    k   = sub["paper"].nunique()
    n   = sub.groupby("paper")["n"].first().sum()
    print(f"\n{cell}: k={k} papers, {len(sub)} coords, total N={n}")
    papers = sub.groupby("paper").agg(task=("task","first"), coords=("x","count")).reset_index()
    print(papers.to_string(index=False))

# ── B. ROI centroid distances (2×2) ──────────────────────────────────────────
ROI_MAP = {
    "vmPFC":      ["vmPFC","OFC_R","OFC_L"],
    "dmPFC":      ["dmPFC","mPFC","DLPFC_L"],
    "aInsula_R":  ["aInsula_R"],
    "aInsula_L":  ["aInsula_L"],
    "amygdala_R": ["amygdala_R"],
    "amygdala_L": ["amygdala_L"],
    "caudate":    ["caudate_R","caudate_L","caudate_head_R","caudate_head_L","NAcc_R"],
    "ACC":        ["ACC","dACC"],
    "TPJ_R":      ["TPJ_R"],
    "precuneus":  ["precuneus"],
}

def centroid(pool_df, roi_labels):
    sub = pool_df[pool_df.region.isin(roi_labels)][["x","y","z"]]
    return sub.mean().values if len(sub) > 0 else np.array([np.nan]*3), len(sub)

cells = ["EuA_Stranger","EuA_Close","EA_Stranger","EA_Close"]
roi_records = []
for roi, labels in ROI_MAP.items():
    row = {"ROI": roi}
    for cell in cells:
        c, n = centroid(df[df.cell==cell], labels)
        row[f"{cell}_xyz"] = f"[{c[0]:.0f},{c[1]:.0f},{c[2]:.0f}]" if n>0 else "—"
        row[f"{cell}_n"]   = n
    roi_records.append(row)
roi_df = pd.DataFrame(roi_records)

print("\n\n── B. ROI Centroids (2×2 cells) ─────────────────────────────────────────")
print(roi_df[["ROI"]+[f"{c}_xyz" for c in cells]+[f"{c}_n" for c in cells]].to_string(index=False))
roi_df.to_csv(os.path.join(OUT_DIR, "2x2_roi_centroids.csv"), index=False)

# Euclidean distances for the 4 contrasts of interest
print("\n── Interaction distances ──")
for roi, labels in ROI_MAP.items():
    cents = {}
    ns    = {}
    for cell in cells:
        c, n = centroid(df[df.cell==cell], labels)
        cents[cell] = c; ns[cell] = n
    # Main effect of relationship (pooled)
    eua_s, eua_c = cents["EuA_Stranger"], cents["EuA_Close"]
    ea_s,  ea_c  = cents["EA_Stranger"],  cents["EA_Close"]
    if all(not np.any(np.isnan(cents[c])) for c in cells):
        d_rel_eua = np.linalg.norm(eua_s - eua_c)
        d_rel_ea  = np.linalg.norm(ea_s  - ea_c)
        d_cul_str = np.linalg.norm(eua_s - ea_s)
        d_cul_cls = np.linalg.norm(eua_c - ea_c)
        print(f"  {roi:12s}  ΔRel(EuA)={d_rel_eua:.1f}  ΔRel(EA)={d_rel_ea:.1f}  "
              f"ΔCul(Str)={d_cul_str:.1f}  ΔCul(Cls)={d_cul_cls:.1f}")

# ── C. mPFC subregion dissociation (all 4 cells) ─────────────────────────────
mpfc_labels = ["vmPFC","dmPFC","mPFC","OFC_R","OFC_L","DLPFC_L"]
mpfc = df[df.region.isin(mpfc_labels)].copy()
mpfc["subregion"] = mpfc.apply(
    lambda r: "dmPFC (BA9/10m)" if r.y >= 50 and r.z >= 10
              else ("vmPFC (BA10/11)" if r.y >= 40 else "OFC/vmPFC (BA11/47)"), axis=1)

print("\n── C. mPFC Subregion Distribution by Cell ───────────────────────────────")
sub_pivot = mpfc.groupby(["cell","subregion"]).size().reset_index(name="n")
print(sub_pivot.to_string(index=False))

# Centroid y,z per cell
print()
for cell in cells:
    sub = mpfc[mpfc.cell == cell]
    if len(sub):
        y, z = sub["y"].mean(), sub["z"].mean()
        print(f"  {cell:16s}: mPFC centroid  y={y:.1f}  z={z:.1f}  n={len(sub)}")

# Key 2×2 mPFC centroid shifts
for rel in ["Stranger","Close"]:
    c_eua = mpfc[mpfc.cell==f"EuA_{rel}"]
    c_ea  = mpfc[mpfc.cell==f"EA_{rel}"]
    if len(c_eua) and len(c_ea):
        shift = np.sqrt((c_ea["y"].mean()-c_eua["y"].mean())**2 +
                        (c_ea["z"].mean()-c_eua["z"].mean())**2)
        print(f"  Culture shift ({rel}): {shift:.1f} mm")
for pool in ["EuA","EA"]:
    c_str = mpfc[mpfc.cell==f"{pool}_Stranger"]
    c_cls = mpfc[mpfc.cell==f"{pool}_Close"]
    if len(c_str) and len(c_cls):
        shift = np.sqrt((c_cls["y"].mean()-c_str["y"].mean())**2 +
                        (c_cls["z"].mean()-c_str["z"].mean())**2)
        print(f"  Relationship shift ({pool}): {shift:.1f} mm")

# ── D. Cognitive decoding (4-cell profiles) ───────────────────────────────────
TERM_ANNOTATIONS = {
    # EuA-Stranger
    "rilling2002":    ["reward","cooperation","reciprocity","caudate"],
    "king_casas2008": ["norm_violation","insula","affective","clinical"],
    "krueger2007":    ["mentalising","TPJ","belief_updating","stranger"],
    "vanden_bos2009": ["reward","affective","vmPFC","stranger"],
    "baumgartner2008":["affective","threat","amygdala","oxytocin"],
    "harle2012":      ["affective","norm_violation","insula","emotion"],
    "zak2011":        ["reward","affective","vmPFC","oxytocin"],
    "sripada2009":    ["affective","threat","amygdala","oxytocin"],
    # EuA-Close
    "king_casas2005": ["reward","reciprocity","reputation","partner_identity"],
    "fouragnan2013":  ["reputation","reward","caudate","partner_identity"],
    "delgado2005":    ["moral","social","reward","partner_identity"],
    "dewall2012":     ["social","mentalising","TPJ","partner_identity"],
    "bellucci2017":   ["social","mentalising","TPJ","reciprocity"],
    "fareri2012":     ["reward","social","friend","striatum"],
    # EA-Stranger
    "takahashi2008":  ["norm_violation","affective","insula","unfairness"],
    "chang2010":      ["cooperation","dmPFC","social","stranger"],
    "zheng2014":      ["norm_violation","unfairness","insula","affective"],
    "sun2016":        ["cooperation","dmPFC","self_referential","social"],
    "haruno2014":     ["affective","norm_violation","amygdala","SVO"],
    "takahashi2016":  ["affective","reward","vmPFC","coupling"],
    # EA-Close
    "chiao2009":      ["ingroup","amygdala","threat","social_identity"],
    "chiao2010":      ["self_referential","cultural","dmPFC","ingroup"],
    "zhu2007":        ["self_referential","cultural","dmPFC","close_other"],
    "han2012":        ["self_referential","cultural","dmPFC","collective"],
    "ma2014":         ["mentalising","social","dmPFC","TPJ"],
    "wang2011":       ["close_other","vmPFC","ACC","self_referential"],
}

TERMS = sorted({t for tags in TERM_ANNOTATIONS.values() for t in tags})
papers = df["paper"].unique()
term_mat = pd.DataFrame(0, index=papers, columns=TERMS)
for p in papers:
    for t in TERM_ANNOTATIONS.get(p, []):
        if t in TERMS:
            term_mat.loc[p, t] = 1

freq = {}
for cell in cells:
    pps = df[df.cell==cell]["paper"].unique()
    freq[cell] = term_mat.loc[pps].mean()

print("\n── D. Top discriminating terms per cell ─────────────────────────────────")
for cell in cells:
    top = freq[cell].sort_values(ascending=False).head(6)
    print(f"  {cell}: {dict(top.round(2))}")

# Interaction terms: (EuA_Close - EuA_Stranger) vs (EA_Close - EA_Stranger)
rel_effect_eua = freq["EuA_Close"] - freq["EuA_Stranger"]
rel_effect_ea  = freq["EA_Close"]  - freq["EA_Stranger"]
interaction    = rel_effect_ea - rel_effect_eua   # pure Culture×Relationship interaction

decoding_df = pd.DataFrame({
    "EuA_Stranger": freq["EuA_Stranger"],
    "EuA_Close":    freq["EuA_Close"],
    "EA_Stranger":  freq["EA_Stranger"],
    "EA_Close":     freq["EA_Close"],
    "RelEffect_EuA (Close-Stranger)": rel_effect_eua,
    "RelEffect_EA  (Close-Stranger)": rel_effect_ea,
    "Interaction  (EAxRel)": interaction,
})
decoding_df.to_csv(os.path.join(OUT_DIR, "2x2_decoding.csv"))

print("\n── Interaction terms (EA×Rel enriched = positive, EuA×Rel = negative) ──")
print(interaction.sort_values(ascending=False).round(2).to_string())

# ── E. MACM: vmPFC-EuA seed vs dmPFC-EA seed, across 4 pools ─────────────────
SEEDS = {
    "vmPFC_seed": np.array([5.3, 39.6, -9.5]),
    "dmPFC_seed": np.array([-0.3, 52.9, 14.0]),
}
ROI_CENTROIDS = {
    "vmPFC":     np.array([5.3,  39.6, -9.5]),
    "dmPFC":     np.array([-0.3, 52.9, 14.0]),
    "aInsula_R": np.array([38.0, 18.0, 2.0]),
    "caudate":   np.array([8.0,  12.0, 4.0]),
    "amygdala_R":np.array([21.0, -3.0,-17.0]),
    "ACC":       np.array([2.0,  28.0, 22.0]),
    "TPJ_R":     np.array([54.0,-55.0, 20.0]),
    "precuneus": np.array([0.0, -60.0, 30.0]),
}
RADIUS_MM = 15.0

macm_records = []
for cell in cells:
    cell_df = df[df.cell==cell]
    n_papers = cell_df["paper"].nunique()
    for seed_lbl, seed_xyz in SEEDS.items():
        for roi_lbl, roi_xyz in ROI_CENTROIDS.items():
            near = set()
            for paper, grp in cell_df.groupby("paper"):
                dists = np.linalg.norm(grp[["x","y","z"]].values - roi_xyz, axis=1)
                if np.any(dists <= RADIUS_MM):
                    near.add(paper)
            macm_records.append({"cell":cell,"seed":seed_lbl,"roi":roi_lbl,
                "n_papers":n_papers,"n_coact":len(near),
                "freq":len(near)/n_papers if n_papers else 0})

macm_df = pd.DataFrame(macm_records)
macm_df.to_csv(os.path.join(OUT_DIR, "2x2_macm.csv"), index=False)

# vmPFC-seeded network across 4 cells
print("\n── E. MACM vmPFC-seed co-activation frequency per cell ─────────────────")
vmPFC_macm = macm_df[macm_df.seed=="vmPFC_seed"].pivot(index="roi",columns="cell",values="freq")
print(vmPFC_macm.round(2).to_string())

# ── F. MAIN FIGURE: 2×2 interaction panel ────────────────────────────────────
fig = plt.figure(figsize=(20, 18))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

# Canonical palette: EuA = blue, EA = red
EUA_COL, EUA_LIGHT = "#2563A8", "#92B8D8"
EA_COL,  EA_LIGHT  = "#C0392B", "#E89080"
COL = {"EuA_Stranger": EUA_COL,   "EuA_Close": EUA_LIGHT,
       "EA_Stranger":  EA_COL,    "EA_Close":  EA_LIGHT}
LABEL = {"EuA_Stranger":"EuA\nStranger","EuA_Close":"EuA\nClose",
         "EA_Stranger":"EA\nStranger","EA_Close":"EA\nClose"}

# ── Panel row 0: mPFC scatter (4 cells) ───────────────────────────────────────
mpfc_labels_all = ["vmPFC","dmPFC","mPFC","OFC_R","OFC_L","DLPFC_L"]
ax_mpfc = fig.add_subplot(gs[0, :2])
for cell in cells:
    sub = df[(df.cell==cell) & (df.region.isin(mpfc_labels_all))]
    if len(sub):
        ax_mpfc.scatter(sub["y"], sub["z"], c=COL[cell], s=55, alpha=0.75,
                        label=LABEL[cell], zorder=3)
        cy, cz = sub["y"].mean(), sub["z"].mean()
        ax_mpfc.scatter([cy],[cz], c=COL[cell], s=200, marker="*",
                        edgecolors="k", linewidths=0.7, zorder=5)
# Boundary lines
ax_mpfc.axvline(50, color="gray", lw=0.8, ls="--", alpha=0.6)
ax_mpfc.axhline(10, color="gray", lw=0.8, ls="--", alpha=0.6)
ax_mpfc.set_xlabel("MNI y", fontsize=10); ax_mpfc.set_ylabel("MNI z", fontsize=10)
ax_mpfc.set_title("mPFC Scatter: 2×2 Cells\n(★=centroid, dashes=BA9/10m boundary)",
                  fontweight="bold", fontsize=10)
ax_mpfc.legend(fontsize=8, ncol=2); ax_mpfc.spines[["top","right"]].set_visible(False)

# ── Panel row 0: mPFC subregion bar chart ─────────────────────────────────────
ax_sub = fig.add_subplot(gs[0, 2:])
sub_data = sub_pivot.copy()
subregions = ["dmPFC (BA9/10m)","vmPFC (BA10/11)","OFC/vmPFC (BA11/47)"]
x  = np.arange(len(cells))
w  = 0.25
hatches = ["///","","..."]
for i, sr in enumerate(subregions):
    vals = [sub_data[(sub_data.cell==c)&(sub_data.subregion==sr)]["n"].sum()
            for c in cells]
    ax_sub.bar(x + i*w, vals, w, label=sr, hatch=hatches[i],
               color=[COL[c] for c in cells], alpha=0.75, edgecolor="k", lw=0.5)
ax_sub.set_xticks(x + w); ax_sub.set_xticklabels([LABEL[c] for c in cells], fontsize=8)
ax_sub.set_ylabel("n coordinates"); ax_sub.set_title("mPFC Subregion Distribution",
                                                       fontweight="bold", fontsize=10)
ax_sub.legend(fontsize=7); ax_sub.spines[["top","right"]].set_visible(False)

# ── Panel row 1: Region frequency heatmap ────────────────────────────────────
ax_heat = fig.add_subplot(gs[1, :2])
rois_plot = ["vmPFC","dmPFC","aInsula_R","amygdala_R","caudate","ACC","TPJ_R","precuneus"]
heat_data = np.zeros((len(rois_plot), len(cells)))
for j, cell in enumerate(cells):
    cell_df2 = df[df.cell==cell]
    n_papers = cell_df2["paper"].nunique()
    for i, roi in enumerate(rois_plot):
        labels = ROI_MAP[roi]
        near = 0
        for paper, grp in cell_df2.groupby("paper"):
            if grp["region"].isin(labels).any():
                near += 1
        heat_data[i, j] = near / n_papers if n_papers else 0

im = ax_heat.imshow(heat_data, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
ax_heat.set_xticks(range(len(cells))); ax_heat.set_xticklabels([LABEL[c] for c in cells], fontsize=8)
ax_heat.set_yticks(range(len(rois_plot))); ax_heat.set_yticklabels(rois_plot, fontsize=8)
ax_heat.set_title("Region Convergence\n(prop. studies per cell)", fontweight="bold", fontsize=10)
plt.colorbar(im, ax=ax_heat, fraction=0.03)
for i in range(len(rois_plot)):
    for j in range(len(cells)):
        ax_heat.text(j, i, f"{heat_data[i,j]:.2f}", ha="center", va="center",
                     fontsize=7, color="black")

# ── Panel row 1: Interaction terms (decoding) ────────────────────────────────
ax_int = fig.add_subplot(gs[1, 2:])
int_sorted = interaction.sort_values()
colors_int = [EA_COL if v > 0 else EUA_COL for v in int_sorted]
ax_int.barh(int_sorted.index, int_sorted.values, color=colors_int, alpha=0.85)
ax_int.axvline(0, color="k", lw=0.8, ls="--")
ax_int.set_xlabel("Interaction effect (EA×Close) − (EuA×Close)", fontsize=9)
ax_int.set_title("Cognitive Term Interaction\n(Blue=EA more enriched by closeness;\nRed=EuA more enriched by closeness)",
                 fontweight="bold", fontsize=9)
ax_int.spines[["top","right"]].set_visible(False)

# ── Panel row 2: MACM vmPFC-seed across cells ─────────────────────────────────
ax_macm_v = fig.add_subplot(gs[2, :2])
rois_m = list(ROI_CENTROIDS.keys())
x_pos = np.arange(len(rois_m)); w2 = 0.22
for j, cell in enumerate(cells):
    vals = [macm_df[(macm_df.cell==cell)&(macm_df.seed=="vmPFC_seed")&
                    (macm_df.roi==r)]["freq"].values[0] for r in rois_m]
    ax_macm_v.bar(x_pos + j*w2, vals, w2, label=LABEL[cell], color=COL[cell],
                  alpha=0.8, edgecolor="k", lw=0.3)
ax_macm_v.set_xticks(x_pos + 1.5*w2)
ax_macm_v.set_xticklabels(rois_m, rotation=35, ha="right", fontsize=8)
ax_macm_v.set_ylabel("Co-activation freq.")
ax_macm_v.set_title("MACM: vmPFC-seed Co-activation\n(by cell)", fontweight="bold", fontsize=10)
ax_macm_v.legend(fontsize=7, ncol=2); ax_macm_v.spines[["top","right"]].set_visible(False)

# ── Panel row 2: Relationship effect per culture (vmPFC-seed MACM) ───────────
ax_rel = fig.add_subplot(gs[2, 2:])
rois_r = rois_m
for i, pool in enumerate(["EuA","EA"]):
    str_vals = [macm_df[(macm_df.cell==f"{pool}_Stranger")&(macm_df.seed=="vmPFC_seed")&
                        (macm_df.roi==r)]["freq"].values[0] for r in rois_r]
    cls_vals = [macm_df[(macm_df.cell==f"{pool}_Close")   &(macm_df.seed=="vmPFC_seed")&
                        (macm_df.roi==r)]["freq"].values[0] for r in rois_r]
    delta = np.array(cls_vals) - np.array(str_vals)
    col = EUA_COL if pool=="EuA" else EA_COL
    x_off = x_pos + i*0.4
    ax_rel.bar(x_off, delta, 0.38, label=f"{pool} (Close−Str)", color=col, alpha=0.8,
               edgecolor="k", lw=0.3)
ax_rel.axhline(0, color="k", lw=0.8)
ax_rel.set_xticks(x_pos + 0.2)
ax_rel.set_xticklabels(rois_r, rotation=35, ha="right", fontsize=8)
ax_rel.set_ylabel("Δ co-activation (Close − Stranger)")
ax_rel.set_title("Relationship Effect on MACM\n(per culture, vmPFC seed)",
                 fontweight="bold", fontsize=10)
ax_rel.legend(fontsize=8); ax_rel.spines[["top","right"]].set_visible(False)

fig.suptitle("2×2 Analysis: Social Relationship (Stranger/Close) × Culture (EuA/EA)\n"
             "Neural Mechanisms of Trust — Interaction Effects",
             fontsize=13, fontweight="bold", y=0.98)

out_fig = os.path.join(OUT_DIR, "fig_2x2_interaction.pdf")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Main figure saved: {out_fig}")

# ── G. ALE on EuA-Stranger (best-powered cell, k=8, 500 perms) ───────────────
print("\n── G. ALE on EuA-Stranger (k=8, exploratory) ───────────────────────────")
try:
    from nimare.dataset import Dataset
    from nimare.meta.cbma.ale import ALE
    from nimare.correct import FWECorrector
    from nilearn.reporting import get_clusters_table

    def build_dset(pool_df):
        studies = {}
        for paper, grp in pool_df.groupby("paper"):
            row = grp.iloc[0]
            studies[paper] = {"contrasts": {"1": {
                "coords": {"space":"MNI","x":grp["x"].tolist(),
                           "y":grp["y"].tolist(),"z":grp["z"].tolist()},
                "metadata":{"sample_sizes":[int(row["n"])]}}}}
        return Dataset(studies)

    for cell in ["EuA_Stranger","EuA_Close","EA_Stranger","EA_Close"]:
        sub = df[df.cell==cell]
        k   = sub["paper"].nunique()
        print(f"\n  {cell} (k={k}): ", end="")
        if k < 5:
            print("skip — k<5"); continue
        dset = build_dset(sub)
        ale  = ALE(kernel__fwhm=None)
        res  = ale.fit(dset)
        res.get_map("stat").to_filename(
            os.path.join(OUT_DIR, f"ale_maps_2x2/ale_{cell}_stat.nii.gz"))
        if k >= 6:
            corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=500, n_cores=4)
            cres = corr.transform(res)
            LOGP = "logp_desc-size_level-cluster_corr-FWE_method-montecarlo"
            cres.get_map(LOGP).to_filename(
                os.path.join(OUT_DIR, f"ale_maps_2x2/ale_{cell}_FWE_logp.nii.gz"))
            try:
                tbl = get_clusters_table(cres.get_map(LOGP), stat_threshold=1.3, cluster_threshold=10)
                if len(tbl):
                    print(f"FWE clusters:")
                    print(tbl[["X","Y","Z","Peak Stat","Cluster Size (mm3)"]].to_string(index=False))
                    tbl.to_csv(os.path.join(OUT_DIR, f"ale_peaks_2x2_{cell}.csv"), index=False)
                else:
                    print("no FWE clusters at p<0.05")
            except Exception as e:
                print(f"cluster table error: {e}")
        else:
            print("stat map saved (k<6, no FWE run)")
except ImportError:
    print("  NiMARE not available — skipping ALE")

# ── Summary JSON ──────────────────────────────────────────────────────────────
summary = {}
for cell in cells:
    sub = df[df.cell==cell]
    m   = mpfc[mpfc.cell==cell]
    summary[cell] = {
        "k": int(sub["paper"].nunique()),
        "n_coords": int(len(sub)),
        "mPFC_centroid_y": round(float(m["y"].mean()), 1) if len(m) else None,
        "mPFC_centroid_z": round(float(m["z"].mean()), 1) if len(m) else None,
    }
with open(os.path.join(OUT_DIR, "2x2_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n✓ All 2×2 outputs saved.")
print("  2x2_roi_centroids.csv")
print("  2x2_decoding.csv")
print("  2x2_macm.csv")
print("  2x2_summary.json")
print("  fig_2x2_interaction.pdf")
print("  ale_maps_2x2/*.nii.gz")
