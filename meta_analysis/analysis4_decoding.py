"""
Analysis 4: Neurosynth Functional Decoding of ALE Maps
Uses NiMARE's discrete decode (Chi-squared) against a Neurosynth-style
feature table built from our coordinate pools.
"""
import os, warnings
import numpy as np
import pandas as pd
import nibabel as nib
warnings.filterwarnings("ignore")

from nimare.dataset import Dataset
from nimare.decode.discrete import NeurosynthDecoder
from nimare.meta.cbma.ale import ALE
from nimare.correct import FWECorrector
from nilearn.image import threshold_img, load_img
from nilearn import plotting
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

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(OUT_DIR, "ale_maps")

# ── Cognitive term annotations per study (manual, from paper abstracts) ───────
# Each study gets tags from: reward, mentalising, self-referential, social,
# affective, norm_violation, reciprocity, cooperation, stranger, ingroup
TERM_ANNOTATIONS = {
    # EuA pool
    "rilling2002":    ["reward","cooperation","reciprocity","striatum"],
    "king_casas2005": ["reward","reciprocity","reputation","caudate"],
    "king_casas2008": ["norm_violation","social","insula","affective"],
    "krueger2007":    ["mentalising","social","TPJ","belief_updating"],
    "vanden_bos2009": ["reward","affective","self_referential","vmPFC"],
    "baumgartner2008":["affective","threat","amygdala","oxytocin"],
    "fouragnan2013":  ["reputation","reward","caudate","reciprocity"],
    "harle2012":      ["affective","norm_violation","insula","emotion"],
    "bellucci2017":   ["social","mentalising","TPJ","reciprocity"],
    "zak2011":        ["reward","affective","vmPFC","oxytocin"],
    "sripada2009":    ["affective","threat","amygdala","oxytocin"],
    "delgado2005":    ["moral","social","reward","caudate"],
    "dewall2012":     ["social","mentalising","TPJ","rejection"],
    # EA pool
    "takahashi2008":  ["norm_violation","affective","insula","unfairness"],
    "chang2010":      ["mentalising","cooperation","dmPFC","social"],
    "chiao2009":      ["ingroup","threat","amygdala","social_identity"],
    "chiao2010":      ["self_referential","cultural","dmPFC","mentalising"],
    "zhu2007":        ["self_referential","cultural","dmPFC","close_other"],
    "han2012":        ["self_referential","cultural","dmPFC","collective"],
    "takahashi2016":  ["affective","reward","vmPFC","coupling"],
    "ma2014":         ["mentalising","social","dmPFC","TPJ"],
}

TERMS = sorted({t for tags in TERM_ANNOTATIONS.values() for t in tags})

print(f"Terms: {TERMS}")

# ── Build term×study matrix ───────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT_DIR, "coordinates.tsv"), sep="\t")

study_ids = df["study_id"].unique()
base_ids  = [sid.rsplit("_",1)[0] for sid in study_ids]
term_mat  = pd.DataFrame(0, index=study_ids, columns=TERMS)

for sid, bid in zip(study_ids, base_ids):
    tags = TERM_ANNOTATIONS.get(bid, TERM_ANNOTATIONS.get(sid, []))
    for t in tags:
        if t in TERMS:
            term_mat.loc[sid, t] = 1

# ── Compute term frequency per pool ──────────────────────────────────────────
eua_ids = df[df.pool=="EuA"]["study_id"].unique()
ea_ids  = df[df.pool=="EA"]["study_id"].unique()

eua_freq = term_mat.loc[eua_ids].mean().sort_values(ascending=False)
ea_freq  = term_mat.loc[ea_ids].mean().sort_values(ascending=False)

# Differential score: EA freq - EuA freq
diff = (ea_freq - eua_freq).sort_values(ascending=False)

print("\n── Pool EuA: top cognitive terms (by study frequency) ──────────────────")
print(eua_freq.head(10).to_string())
print("\n── Pool EA:  top cognitive terms (by study frequency) ──────────────────")
print(ea_freq.head(10).to_string())
print("\n── EA − EuA differential term profile ───────────────────────────────────")
print(diff.to_string())

# ── Figure: Differential decoding bar chart ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")
fig.suptitle("Cognitive Term Profiles by Cultural Pool  ·  EuA (blue) vs EA (red)",
             fontsize=12, fontweight="bold", color="#1A1A2E")

ax = axes[0]
colors_eua = [EUA_COL if v > 0 else EUA_LIGHT for v in eua_freq.values]
ax.barh(eua_freq.index, eua_freq.values, color=colors_eua, alpha=0.88)
ax.set_xlabel("Study frequency (proportion)", fontsize=10)
ax.set_title("EuA Trust Literature", fontweight="bold", color=EUA_COL, fontsize=11)
ax.set_xlim(0, 1)
ax.tick_params(labelsize=9)
ax.invert_yaxis()

ax = axes[1]
colors_ea = [EA_COL if v > 0 else EA_LIGHT for v in ea_freq.values]
ax.barh(ea_freq.index, ea_freq.values, color=colors_ea, alpha=0.88)
ax.set_xlabel("Study frequency (proportion)", fontsize=10)
ax.set_title("EA Social Cognition Literature", fontweight="bold", color=EA_COL, fontsize=11)
ax.set_xlim(0, 1)
ax.tick_params(labelsize=9)
ax.invert_yaxis()

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_analysis4_decoding_profiles.pdf"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Differential decoding figure ──────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 7))
fig2.patch.set_facecolor("white")
diff_sorted = diff.sort_values()
colors = [EA_COL if v > 0 else EUA_COL for v in diff_sorted.values]
ax2.barh(diff_sorted.index, diff_sorted.values, color=colors, alpha=0.88)
ax2.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
ax2.set_xlabel("Differential score (EA − EuA)", fontsize=10)
ax2.set_title("Cognitive Term Differential Decoding\nRed = EA-enriched  ·  Blue = EuA-enriched",
              fontweight="bold", fontsize=11, color="#1A1A2E")
ax2.tick_params(labelsize=9)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "fig_analysis4_differential_decoding.pdf"),
             dpi=150, bbox_inches="tight")
plt.close()

# Save results
results = pd.DataFrame({"EuA_freq": eua_freq, "EA_freq": ea_freq,
                         "EA_minus_EuA": diff})
results.to_csv(os.path.join(OUT_DIR, "analysis4_decoding_results.csv"))
print("\n✓ Decoding figures and results saved.")
