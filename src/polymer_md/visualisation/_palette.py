from __future__ import annotations

import matplotlib.pyplot as plt

REGION_HEX = {
    "left": "#4C72B0",
    "central": "#55A868",
    "right": "#C44E52",
    "cap": "#BFBFBF",
}

REGION_RGB = {
    name: tuple(int(h[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    for name, h in REGION_HEX.items()
}

REGION_FACE = {
    "left": "#E8EFF8",
    "central": "#E6F4E8",
    "right": "#F8EAEA",
    "cap": "#F4F4F4",
}

CHARGE_CMAP = "coolwarm"
COMPARISON_PALETTE = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B2",
    "#CCB974", "#64B5CD", "#DD8452", "#937860",
]


def clean_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_facecolor("white")
