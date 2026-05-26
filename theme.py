"""
Visual theme and matplotlib styling constants.

Call ``apply_dark_style()`` once before creating any Figure objects.
"""

# ── Colour palette ──────────────────────────────────────────────────
COLORS = {
    "bg_dark":      "#1a1a2e",
    "bg_mid":       "#16213e",
    "bg_light":     "#0f3460",
    "panel":        "#16213e",
    "text":         "#e0e0e0",
    "text_dim":     "#9e9e9e",
    "accent":       "#4ECDC4",
    "vanilla":      "#FF6B6B",
    "momentum":     "#4ECDC4",
    "gold":         "#FFD700",
    "surface_cmap": "viridis",
}

# ── Matplotlib colour constants ─────────────────────────────────────
MPL_BG      = "#1a1a2e"
MPL_AXES_BG = "#16213e"
MPL_TEXT     = "#e0e0e0"
MPL_SPINE   = "#333355"
MPL_GRID     = "#2a2a4a"


def apply_dark_style():
    """Apply the dark-theme rcParams to *matplotlib.pyplot*."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor":  MPL_BG,
        "axes.facecolor":    MPL_AXES_BG,
        "axes.edgecolor":    MPL_SPINE,
        "axes.labelcolor":   MPL_TEXT,
        "xtick.color":       MPL_TEXT,
        "ytick.color":       MPL_TEXT,
        "text.color":        MPL_TEXT,
        "grid.color":        MPL_GRID,
        "grid.alpha":        0.6,
        "legend.facecolor":  "#1f1f3a",
        "legend.edgecolor":  MPL_SPINE,
        "font.family":       "sans-serif",
        "font.size":         9,
    })
