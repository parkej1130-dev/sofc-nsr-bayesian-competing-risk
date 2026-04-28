"""
make_figure5.py - JMSE/IJHE-standard Figure 5 (140mm wide).

Maintenance intervals: N_maint (cycles) and calendar period (months) per scenario,
with dominant failure mode marker (Crack: blue square / ASR: red circle).

Source: outputs/tables/Table_6_maintenance_v3.csv
Layout: 1 row x 2 cols (cycles | months)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.lines import Line2D

PROJ = Path(r"C:\Users\parke\sofc_revision")
FIG_DIR = PROJ / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCEN_ORDER = ["S1a", "S1b", "S2a", "S2b", "S3a", "S3b"]
SCEN_COLORS = {
    "S1a": "#0072B2", "S1b": "#56B4E9",
    "S2a": "#009E73", "S2b": "#F0E442",
    "S3a": "#E69F00", "S3b": "#D55E00",
}
# Dominant-mode markers (independent of scenario color)
MODE_MARKERS = {
    "Crack": dict(marker="s", color="#1F4E89",  label="Crack-dominant"),
    "ASR":   dict(marker="o", color="#B83A4B",  label="ASR-dominant"),
}

MM = 1.0 / 25.4
WIDTH_MM = 140.0
HEIGHT_MM = 78.0


def configure_jmse_style() -> None:
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":        8,
        "axes.titlesize":   9,
        "axes.labelsize":   8.5,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  7.5,
        "axes.linewidth":   0.8,
        "axes.spines.top":     True,
        "axes.spines.right":   True,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.top":        True,
        "ytick.right":      True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "legend.frameon":     True,
        "legend.edgecolor":   "0.7",
        "legend.framealpha":  1.0,
        "legend.fancybox":    False,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def panel(ax, x, heights, modes, *, ylabel: str,
          fmt_value, ylim_max: float, panel_tag: str,
          y_minor: float, y_major: float):
    bars = []
    for xi, h, scen, mode in zip(x, heights, SCEN_ORDER, modes):
        col = SCEN_COLORS[scen]
        b = ax.bar(xi, h, width=0.66, color=col,
                   edgecolor="black", linewidth=0.6, zorder=2)
        bars.append(b)

        # numeric label above bar
        ax.text(xi, h + ylim_max * 0.018,
                fmt_value(h),
                ha="center", va="bottom", fontsize=7.0,
                color="black", zorder=4)

        # dominant-mode marker inside top of bar (safe for short bars too)
        m = MODE_MARKERS[mode]
        marker_y = max(h * 0.85, h - ylim_max * 0.06)
        ax.plot([xi], [marker_y],
                marker=m["marker"], ms=7.0,
                mfc=m["color"], mec="white", mew=1.0,
                ls="none", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(SCEN_ORDER)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, ylim_max)
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.yaxis.grid(True, which="major", lw=0.4, color="0.85", zorder=0)
    ax.set_axisbelow(True)

    ax.text(-0.13, 1.02, panel_tag, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def main() -> None:
    configure_jmse_style()
    df = pd.read_csv(PROJ / "outputs" / "tables" / "Table_6_maintenance_v3.csv")
    df = df.set_index("scenario").reindex(SCEN_ORDER).reset_index()
    print(df[["scenario", "N_maint", "period_months",
              "pfail_at_maint", "dominant_cause"]].to_string(index=False))

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(WIDTH_MM * MM, HEIGHT_MM * MM),
        gridspec_kw=dict(wspace=0.30),
    )

    x = np.arange(len(SCEN_ORDER))
    modes = df["dominant_cause"].tolist()

    # Left: cycles
    cyc = df["N_maint"].values.astype(float)
    panel(ax_l, x, cyc, modes,
          ylabel=r"Maintenance interval, $N_{\mathrm{maint}}$ (cycles)",
          fmt_value=lambda v: f"{int(v)}",
          ylim_max=720.0,
          panel_tag="(a)",
          y_minor=20, y_major=100)

    # Right: months
    mon = df["period_months"].values.astype(float)
    panel(ax_r, x, mon, modes,
          ylabel="Calendar maintenance period (months)",
          fmt_value=lambda v: f"{v:.1f}",
          ylim_max=190.0,
          panel_tag="(b)",
          y_minor=10, y_major=40)

    # Shared legend (mode markers) outside, below
    handles = [
        Line2D([0], [0], marker=MODE_MARKERS["Crack"]["marker"],
               color="white", mfc=MODE_MARKERS["Crack"]["color"],
               mec="white", mew=1.0, ms=7.0,
               label=MODE_MARKERS["Crack"]["label"]),
        Line2D([0], [0], marker=MODE_MARKERS["ASR"]["marker"],
               color="white", mfc=MODE_MARKERS["ASR"]["color"],
               mec="white", mew=1.0, ms=7.0,
               label=MODE_MARKERS["ASR"]["label"]),
    ]
    leg = fig.legend(handles=handles, loc="upper center",
                     ncol=2, bbox_to_anchor=(0.5, 1.005),
                     handlelength=1.2, handletextpad=0.4,
                     borderpad=0.3, columnspacing=1.6,
                     frameon=True)
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_edgecolor("0.7")

    out_tiff = FIG_DIR / "Figure5_MaintenanceIntervals.tiff"
    out_pdf  = FIG_DIR / "Figure5_MaintenanceIntervals.pdf"
    out_png  = FIG_DIR / "Figure5_MaintenanceIntervals.png"

    fig.savefig(out_tiff, dpi=600,
                pil_kwargs=dict(compression="tiff_lzw"))
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    for p in (out_tiff, out_pdf, out_png):
        sz_kb = p.stat().st_size / 1024
        print(f"wrote  {p.name:36s}  {sz_kb:8.1f} KB")


if __name__ == "__main__":
    main()
