"""
Canonical colour palette — consistent across all figures.
EuA = blue, EA = red (matches conceptual framework figure A/B).
Nature Neuroscience / Trends in Cognitive Sciences style.
"""
# Culture colours
EUA       = "#2563A8"   # medium blue       — EuA (Western)
EUA_LIGHT = "#92B8D8"   # light blue        — EuA faded / Stranger cell
EA        = "#C0392B"   # brick red         — EA (East Asian)
EA_LIGHT  = "#E89080"   # light red/salmon  — EA faded / Stranger cell

# Circuit colours
GRN  = "#27AE60"        # green   — reward / value circuit
PRP  = "#7D3C98"        # purple  — self-referential / dmPFC
ORG  = "#CA6F1E"        # amber   — norm vigilance / aInsula

# Neutrals
GR1  = "#F4F6F7"        # panel background
GR2  = "#ECF0F1"        # secondary fill
GR3  = "#BDC3C7"        # borders / axes
TXT  = "#1A1A2E"        # near-black text

# Colormaps (for nilearn / imshow)
CMAP_EUA  = "Blues"
CMAP_EA   = "Reds"
CMAP_CONJ = "Greens"
CMAP_DIFF = "RdBu_r"    # EA>EuA = red, EuA>EA = blue

# 2x2 cell colours
CELL_COL = {
    "EuA_Stranger": EUA,
    "EuA_Close":    EUA_LIGHT,
    "EA_Stranger":  EA,
    "EA_Close":     EA_LIGHT,
}
