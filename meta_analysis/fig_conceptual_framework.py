"""
Conceptual Framework — TiCS style
Visual fixes applied:
  · top=0.84 (was 0.80) — eliminates whitespace gap between banner and boxes
  · Centre box 2: gradient circles and categorical zones both centred at
    box midpoint — symmetric vertical layout, no crowding at box top
  · All row boundaries aligned across all three columns
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse
import warnings; warnings.filterwarnings("ignore")

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})

EUA   = "#2563A8"
EA    = "#C0392B"
GRN   = "#27AE60"
PRP   = "#7D3C98"
ORG   = "#CA6F1E"
BLU2  = "#4A90D9"
GR3   = "#BBBBBB"
TXT   = "#1A1A2E"
LTBLU = "#EDF3FB"
LTRED = "#FBEDEC"
LTCTR = "#F7F7F7"

def off(ax): ax.axis("off")

def rbox(ax, bx, by, bw, bh, fc, ec, lw=1.1):
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
        boxstyle="round,pad=0.018", fc=fc, ec=ec, lw=lw, zorder=2, clip_on=False))

def box_title(ax, bx, by, bh, bw, title, color, fs=9.2):
    ax.text(bx + bw / 2, by + bh + 0.013, title,
            ha="center", va="bottom", fontsize=fs,
            fontweight="bold", color=color, zorder=6, clip_on=False)

def rtag(ax, x, y, label, color, w=0.295, h=0.044):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.006", fc=color, ec="none", alpha=0.88, zorder=5))
    ax.text(x + w/2, y + h/2, label,
            ha="center", va="center", fontsize=7.0,
            color="white", fontweight="bold", zorder=6)

# ── figure & layout ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 9), facecolor="white")
gs  = gridspec.GridSpec(1, 3, width_ratios=[38, 22, 38],
                        wspace=0.04, top=0.88, bottom=0.05,
                        left=0.015, right=0.985)

# ── TOP BANNER ────────────────────────────────────────────────────────────────
fig.text(0.215, 0.975, "EuA  ·  Gradient Social Architecture",
    ha="center", va="top", fontsize=13, fontweight="bold", color=EUA)
fig.text(0.215, 0.940,
    "Self–other boundary maintained  ·  Trust scales continuously with closeness",
    ha="center", va="top", fontsize=8.5, color=EUA, style="italic", alpha=0.80)
fig.text(0.500, 0.968, "Two Cultural\nBoundary Dimensions",
    ha="center", va="top", fontsize=10.5, fontweight="bold", color=TXT,
    linespacing=1.30)
fig.text(0.785, 0.975, "EA  ·  Categorical Social Architecture",
    ha="center", va="top", fontsize=13, fontweight="bold", color=EA)
fig.text(0.785, 0.940,
    "Self–ingroup representationally fused  ·  Outgroup categorically excluded",
    ha="center", va="top", fontsize=8.5, color=EA, style="italic", alpha=0.80)
fig.add_artist(plt.Line2D([0.015, 0.985], [0.910, 0.910],
    transform=fig.transFigure, lw=0.8, color=GR3, linestyle="--"))

# ── shared geometry ───────────────────────────────────────────────────────────
BH  = 0.265
GAP = 0.040
B3y = 0.022
B2y = B3y + BH + GAP   # 0.327
B1y = B2y + BH + GAP   # 0.632

BW  = 0.955
BX  = 0.025
IW  = 0.155
ICX = BX + IW / 2
TX  = BX + IW + 0.022
TOP_PAD = 0.016

def ic_y(by, bh=BH): return by + bh * 0.50

# Centre column geometry
CC1y = B1y;  CH1 = BH               # [0.632, 0.897] — aligns row 1
CC2y = B3y;  CH2 = B2y + BH - B3y  # [0.022, 0.592] — spans rows 2+3
CBX  = 0.05; CBW = 0.90

# ════════════════════════════════════════════════════════════════════════════
# PANEL A — EuA
# ════════════════════════════════════════════════════════════════════════════
ax_A = fig.add_subplot(gs[0]); off(ax_A)
ax_A.set_xlim(0, 1); ax_A.set_ylim(0, 1)

# Box 1
rbox(ax_A, BX, B1y, BW, BH, LTBLU, EUA)
box_title(ax_A, BX, B1y, BH, BW, "① Social Representations", EUA)

i1y = ic_y(B1y)
ax_A.add_patch(Circle((ICX-0.038, i1y), 0.033, fc=EUA, ec="white", lw=1.4, alpha=0.88, zorder=4))
ax_A.add_patch(Circle((ICX+0.040, i1y), 0.033, fc=BLU2, ec="white", lw=1.4, alpha=0.50, zorder=4))
ax_A.plot([ICX-0.038+0.035, ICX+0.040-0.035], [i1y, i1y], lw=1.0, color=GR3, linestyle="--", zorder=3)
ax_A.text(ICX-0.038, i1y-0.052, "self",  ha="center", fontsize=6.0, color=EUA, alpha=0.70)
ax_A.text(ICX+0.040, i1y-0.052, "other", ha="center", fontsize=6.0, color=BLU2, alpha=0.70)

ty = B1y + BH - TOP_PAD
ax_A.text(TX, ty,          "Self and others are distinct, independent agents.", va="top", fontsize=8.2, color=TXT)
ax_A.text(TX, ty - 0.068,  "Trust scales continuously with relationship closeness.", va="top", fontsize=8.2, color=TXT)
ax_A.text(TX, ty - 0.145,  "No representational merger between self and close other.", va="top", fontsize=7.7, color="#666", style="italic")

# Box 2
rbox(ax_A, BX, B2y, BW, BH, LTBLU, EUA)
box_title(ax_A, BX, B2y, BH, BW, "② Stranger Interactions", EUA)

i2y = ic_y(B2y)
ax_A.add_patch(Circle((ICX-0.018, i2y+0.038), 0.028, fc=GRN,      ec="white", lw=1.1, alpha=0.88, zorder=4))
ax_A.add_patch(Circle((ICX+0.036, i2y-0.018), 0.028, fc="#3A8A50", ec="white", lw=1.1, alpha=0.75, zorder=4))
ax_A.plot([ICX-0.018+0.028, ICX+0.036-0.028], [i2y+0.038, i2y-0.018], lw=0.9, color="#bbb", zorder=3)
ax_A.text(ICX, i2y+0.080, "reward\ncircuit", ha="center", fontsize=5.8, color=GRN, alpha=0.80, linespacing=1.2)

ry2a = B2y + BH - TOP_PAD - 0.032
rtag(ax_A, TX, ry2a, "OFC / vmPFC", GRN)
ax_A.text(TX+0.308, ry2a+0.022, "value-based trust computation",  va="center", fontsize=7.8, color=TXT)
ry2b = ry2a - 0.072
rtag(ax_A, TX, ry2b, "Caudate", "#3A8A50")
ax_A.text(TX+0.308, ry2b+0.022, "social reward tracking",         va="center", fontsize=7.8, color=TXT)
ax_A.text(TX, B2y+0.015, "Reward circuit evaluates expected reciprocity from an unknown partner.", va="bottom", fontsize=7.5, color="#666", style="italic")

# Box 3
rbox(ax_A, BX, B3y, BW, BH, LTBLU, EUA)
box_title(ax_A, BX, B3y, BH, BW, "③ Close-Other Interactions", EUA)

i3y = ic_y(B3y)
ax_A.add_patch(Circle((ICX-0.018, i3y+0.040), 0.026, fc=GRN,      ec="white", lw=1.1, alpha=0.88, zorder=4))
ax_A.add_patch(Circle((ICX+0.038, i3y+0.000), 0.026, fc="#3A8A50", ec="white", lw=1.1, alpha=0.75, zorder=4))
ax_A.add_patch(Circle((ICX-0.004, i3y-0.050), 0.026, fc="#4A7EB0", ec="white", lw=1.1, alpha=0.80, zorder=4))
for p1, p2 in [
    ((ICX-0.018+0.026, i3y+0.040), (ICX+0.038-0.026, i3y+0.000)),
    ((ICX-0.018+0.012, i3y+0.040-0.026), (ICX-0.004, i3y-0.050+0.026)),
]:
    ax_A.plot([p1[0],p2[0]], [p1[1],p2[1]], lw=0.8, color="#bbb", zorder=3)
ax_A.text(ICX, i3y+0.082, "+TPJ", ha="center", fontsize=6.2, color="#4A7EB0", fontweight="bold")

ry3a = B3y + BH - TOP_PAD - 0.032
rtag(ax_A, TX, ry3a, "vmPFC + Caudate", GRN, w=0.310)
ax_A.text(TX+0.322, ry3a+0.022, "scaled reward, familiar partner",    va="center", fontsize=7.8, color=TXT)
ry3b = ry3a - 0.072
rtag(ax_A, TX, ry3b, "TPJ",             "#4A7EB0", w=0.310)
ax_A.text(TX+0.322, ry3b+0.022, "models friend's mind separately",    va="center", fontsize=7.8, color=TXT)
ax_A.text(TX, B3y+0.015, "Same reward circuit as strangers — scaled up. TPJ additionally tracks friend as separate agent.", va="bottom", fontsize=7.5, color="#666", style="italic")

# ════════════════════════════════════════════════════════════════════════════
# PANEL C — Centre
# ════════════════════════════════════════════════════════════════════════════
ax_C = fig.add_subplot(gs[1]); off(ax_C)
ax_C.set_xlim(0, 1); ax_C.set_ylim(0, 1)

# ── Box 1: Self–Other (row 1) ─────────────────────────────────────────────
rbox(ax_C, CBX, CC1y, CBW, CH1, LTCTR, GR3, lw=0.9)
box_title(ax_C, CBX, CC1y, CH1, CBW, "① Self–Other Boundary", TXT, fs=8.8)

s1y = CC1y + CH1 * 0.50
ax_C.add_patch(Circle((0.21, s1y), 0.058, fc=EUA,  ec="white", lw=1.5, alpha=0.85, zorder=4))
ax_C.add_patch(Circle((0.37, s1y), 0.058, fc=BLU2, ec="white", lw=1.5, alpha=0.45, zorder=4))
ax_C.plot([0.21+0.060, 0.37-0.060], [s1y, s1y], lw=1.1, color=GR3, linestyle="--", zorder=3)
ax_C.text(0.29, s1y-0.100, "maintained", ha="center", fontsize=7.0, color=EUA, fontweight="bold")
ax_C.text(0.29, s1y+0.090, "EuA",        ha="center", fontsize=7.0, color=EUA, alpha=0.65)
ax_C.plot([0.50, 0.50], [CC1y+0.040, CC1y+CH1-0.040], lw=0.7, color=GR3, linestyle=":", zorder=2)
ax_C.text(0.50, s1y, "→", ha="center", va="center", fontsize=11, color=GR3)
ax_C.add_patch(Ellipse((0.63, s1y), 0.148, 0.106, fc=EA,  ec="white", lw=1.5, alpha=0.85, zorder=4))
ax_C.add_patch(Ellipse((0.76, s1y), 0.148, 0.106, fc=PRP, ec="white", lw=1.5, alpha=0.65, zorder=4))
ax_C.text(0.70, s1y-0.100, "dissolved", ha="center", fontsize=7.0, color=EA, fontweight="bold")
ax_C.text(0.70, s1y+0.090, "EA",        ha="center", fontsize=7.0, color=EA, alpha=0.65)

# ── Box 2: Ingroup–Outgroup (rows 2+3) ───────────────────────────────────
rbox(ax_C, CBX, CC2y, CBW, CH2, LTCTR, GR3, lw=0.9)
box_title(ax_C, CBX, CC2y, CH2, CBW, "② Ingroup–Outgroup Boundary", TXT, fs=8.8)

# Both sides centred at the box vertical midpoint
mid2 = CC2y + CH2 * 0.50   # 0.022 + 0.285 = 0.307

# Left — EuA gradient (centred at mid2)
gr_y = mid2
for r, al in [(0.120, 0.07), (0.082, 0.17), (0.046, 0.43), (0.018, 1.0)]:
    ax_C.add_patch(Circle((0.26, gr_y), r, fc=EUA, ec="white", lw=0.4, alpha=al, zorder=4))
ax_C.text(0.26, gr_y+0.138, "EuA",      ha="center", fontsize=7.0, color=EUA, alpha=0.65)
ax_C.text(0.26, gr_y-0.148, "gradient", ha="center", fontsize=7.0, color=EUA, fontweight="bold")

# Vertical divider
ax_C.plot([0.50, 0.50], [CC2y+0.035, CC2y+CH2-0.035], lw=0.7, color=GR3, linestyle=":", zorder=2)
ax_C.text(0.50, mid2, "→", ha="center", va="center", fontsize=11, color=GR3)

# Right — EA categorical (centred at mid2)
zone_h = 0.270
ig_w=0.138; wall_w=0.036; og_w=0.138
total_w = ig_w + wall_w + og_w  # 0.312
z_cx   = 0.76
z_x0   = z_cx - total_w / 2
ig_x=z_x0; wall_x=ig_x+ig_w; og_x=wall_x+wall_w
z_y0   = mid2 - zone_h / 2     # centred at mid2

# Ingroup zone
ax_C.add_patch(FancyBboxPatch((ig_x, z_y0), ig_w, zone_h, boxstyle="round,pad=0.008", fc=PRP, ec=PRP, lw=0, alpha=0.18, zorder=3))
ax_C.add_patch(FancyBboxPatch((ig_x, z_y0), ig_w, zone_h, boxstyle="round,pad=0.008", fc="none", ec=PRP, lw=1.4, alpha=0.60, zorder=4))
ax_C.add_patch(Circle((ig_x+ig_w/2, z_y0+zone_h*0.60), 0.030, fc=PRP, ec="white", lw=0.9, alpha=0.85, zorder=5))
ax_C.text(ig_x+ig_w/2, z_y0+zone_h*0.60+0.048, "in-group",   ha="center", fontsize=6.5, color=PRP, fontweight="bold")
ax_C.text(ig_x+ig_w/2, z_y0+zone_h*0.22,       "self ≈ other", ha="center", fontsize=6.0, color=PRP, style="italic")

# Wall
ax_C.add_patch(FancyBboxPatch((wall_x, z_y0-0.008), wall_w, zone_h+0.016, boxstyle="square,pad=0", fc="#2C2C2C", ec="none", alpha=0.88, zorder=5))
ax_C.text(wall_x+wall_w/2, z_y0+zone_h/2, "≠", ha="center", va="center", fontsize=16, color="white", fontweight="bold", zorder=6)

# Outgroup zone
ax_C.add_patch(FancyBboxPatch((og_x, z_y0), og_w, zone_h, boxstyle="round,pad=0.008", fc=ORG, ec=ORG, lw=0, alpha=0.16, zorder=3))
ax_C.add_patch(FancyBboxPatch((og_x, z_y0), og_w, zone_h, boxstyle="round,pad=0.008", fc="none", ec=ORG, lw=1.4, alpha=0.60, zorder=4))
ax_C.add_patch(Circle((og_x+og_w/2, z_y0+zone_h*0.60), 0.030, fc=ORG, ec="white", lw=0.9, alpha=0.80, zorder=5))
ax_C.text(og_x+og_w/2, z_y0+zone_h*0.60+0.048, "out-group", ha="center", fontsize=6.5, color=ORG, fontweight="bold")
ax_C.text(og_x+og_w/2, z_y0+zone_h*0.22,        "excluded",  ha="center", fontsize=6.0, color=ORG, style="italic")

ax_C.text(z_cx, z_y0+zone_h+0.042, "EA",          ha="center", fontsize=7.0, color=EA, alpha=0.65)
ax_C.text(z_cx, z_y0-0.052,        "categorical",  ha="center", fontsize=7.0, color=EA, fontweight="bold")

# ════════════════════════════════════════════════════════════════════════════
# PANEL B — EA
# ════════════════════════════════════════════════════════════════════════════
ax_B = fig.add_subplot(gs[2]); off(ax_B)
ax_B.set_xlim(0, 1); ax_B.set_ylim(0, 1)

BX_B=0.020; ICX_B=BX_B+IW/2; TX_B=BX_B+IW+0.022; BW_B=0.958

# Box 1
rbox(ax_B, BX_B, B1y, BW_B, BH, LTRED, EA)
box_title(ax_B, BX_B, B1y, BH, BW_B, "① Social Representations", EA)

i1by = ic_y(B1y)
ax_B.add_patch(Ellipse((ICX_B-0.022, i1by), 0.088, 0.066, fc=EA,  ec="white", lw=1.4, alpha=0.88, zorder=4))
ax_B.add_patch(Ellipse((ICX_B+0.028, i1by), 0.088, 0.066, fc=PRP, ec="white", lw=1.4, alpha=0.65, zorder=4))
ax_B.text(ICX_B+0.003, i1by-0.058, "self ≈ other", ha="center", fontsize=6.0, color=EA, alpha=0.75)

ty_b = B1y + BH - TOP_PAD
ax_B.text(TX_B, ty_b,         "Self and close others are representationally fused.",  va="top", fontsize=8.2, color=TXT)
ax_B.text(TX_B, ty_b-0.068,   "Ingroup processed through self-referential network.",  va="top", fontsize=8.2, color=TXT)
ax_B.text(TX_B, ty_b-0.145,   "Outgroup members are categorically excluded.",         va="top", fontsize=7.7, color="#666", style="italic")

# Box 2
rbox(ax_B, BX_B, B2y, BW_B, BH, LTRED, EA)
box_title(ax_B, BX_B, B2y, BH, BW_B, "② Stranger Interactions", EA)

i2by = ic_y(B2y)
ax_B.add_patch(Circle((ICX_B-0.018, i2by+0.038), 0.028, fc=ORG,      ec="white", lw=1.1, alpha=0.88, zorder=4))
ax_B.add_patch(Circle((ICX_B+0.036, i2by-0.018), 0.028, fc="#A04020", ec="white", lw=1.1, alpha=0.75, zorder=4))
ax_B.plot([ICX_B-0.018+0.028, ICX_B+0.036-0.028], [i2by+0.038, i2by-0.018], lw=0.9, color="#bbb", zorder=3)
ax_B.text(ICX_B, i2by+0.080, "vigilance\ncircuit", ha="center", fontsize=5.8, color=ORG, alpha=0.80, linespacing=1.2)

ry2ba = B2y + BH - TOP_PAD - 0.032
rtag(ax_B, TX_B, ry2ba, "aInsula", ORG)
ax_B.text(TX_B+0.308, ry2ba+0.022, "norm-violation detection",         va="center", fontsize=7.8, color=TXT)
ry2bb = ry2ba - 0.072
rtag(ax_B, TX_B, ry2bb, "Caudate", "#A04020")
ax_B.text(TX_B+0.308, ry2bb+0.022, "norm enforcement, not reward",     va="center", fontsize=7.8, color=TXT)
ax_B.text(TX_B, B2y+0.015, "Entirely different circuit from close-other interactions — a categorical switch.", va="bottom", fontsize=7.5, color="#666", style="italic")

# Box 3
rbox(ax_B, BX_B, B3y, BW_B, BH, LTRED, EA)
box_title(ax_B, BX_B, B3y, BH, BW_B, "③ Close-Other Interactions", EA)

i3by = ic_y(B3y)
ax_B.add_patch(Circle((ICX_B-0.015, i3by+0.040), 0.026, fc=PRP,      ec="white", lw=1.1, alpha=0.88, zorder=4))
ax_B.add_patch(Circle((ICX_B+0.040, i3by+0.000), 0.026, fc="#6040A0", ec="white", lw=1.1, alpha=0.75, zorder=4))
ax_B.plot([ICX_B-0.015+0.026, ICX_B+0.040-0.026], [i3by+0.040, i3by+0.000], lw=0.9, color="#bbb", zorder=3)
ax_B.add_patch(Circle((ICX_B-0.004, i3by-0.050), 0.024, fc="#cccccc", ec="white", lw=0.9, alpha=0.50, zorder=4))
ax_B.text(ICX_B-0.004, i3by-0.050, "✕", ha="center", va="center", fontsize=7, color="#999", zorder=5)
ax_B.text(ICX_B, i3by+0.082, "self-ref.", ha="center", fontsize=5.8, color=PRP, alpha=0.85)

ry3ba = B3y + BH - TOP_PAD - 0.032
rtag(ax_B, TX_B, ry3ba, "dmPFC",    PRP,      w=0.295)
ax_B.text(TX_B+0.308, ry3ba+0.022, "self-referential processing",       va="center", fontsize=7.8, color=TXT)
ry3bb = ry3ba - 0.072
rtag(ax_B, TX_B, ry3bb, "Precuneus", "#6040A0", w=0.295)
ax_B.text(TX_B+0.308, ry3bb+0.022, "self-projection / episodic memory", va="center", fontsize=7.8, color=TXT)
ax_B.text(TX_B, B3y+0.015, "TPJ absent — ingroup processed as extension of self, not as a separate agent.", va="bottom", fontsize=7.5, color="#666", style="italic")

# ── bottom strip ──────────────────────────────────────────────────────────────
fig.add_artist(plt.Line2D([0.015, 0.985], [0.058, 0.058],
    transform=fig.transFigure, lw=0.7, color=GR3, linestyle="--"))
fig.text(0.215, 0.035, "Same circuits for all relationships — reward value scaled by familiarity",
    ha="center", va="center", fontsize=8.5, color=EUA, fontweight="bold", style="italic")
fig.text(0.785, 0.035, "Two separate circuits — categorical switch at the ingroup–outgroup boundary",
    ha="center", va="center", fontsize=8.5, color=EA, fontweight="bold", style="italic")
fig.text(0.500, 0.035, "vs", ha="center", va="center", fontsize=10, fontweight="bold", color=GR3)

plt.savefig(
    "/Users/huanwang/GoogleDrive_/post-doc/projects/AIneuro_case/meta_analysis/fig_conceptual_framework.pdf",
    dpi=180, bbox_inches="tight", facecolor="white")
plt.savefig(
    "/Users/huanwang/GoogleDrive_/post-doc/projects/AIneuro_case/meta_analysis/fig_conceptual_framework.png",
    dpi=180, bbox_inches="tight", facecolor="white")
print("Done.")
