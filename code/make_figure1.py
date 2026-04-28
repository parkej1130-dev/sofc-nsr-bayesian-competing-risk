"""
make_figure1.py - JMSE/IJHE-standard Figure 1 (2-panel, 140mm wide).

  1a: ASR degradation trajectories  (90 cells, 6 scenarios, ASR* dash-dot at 0.40)
  1b: Failure-mode distribution     (stacked bar: ASR / Crack / Censored)

Output:
  outputs/figures/Figure1_ASR_FailureMode.tiff   (600 dpi, LZW compressed)
  outputs/figures/Figure1_ASR_FailureMode.pdf    (vector)
  outputs/figures/Figure1_ASR_FailureMode.png    (preview, 200 dpi)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJ = Path(r"C:\Users\parke\sofc_revision")
DATA = Path(r"C:\Users\parke\Downloads\sofc_v3_output (1)")
FIG_DIR = PROJ / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ASR_FAIL = 0.40
SCEN_ORDER = ["S1a", "S1b", "S2a", "S2b", "S3a", "S3b"]

MM = 1.0 / 25.4
WIDTH_MM = 140.0
HEIGHT_MM = 75.0

# Colorblind-safe palette (Wong 2011) for the 6 scenarios.
SCEN_COLORS = {
    "S1a": "#0072B2",  # blue
    "S1b": "#56B4E9",  # sky blue
    "S2a": "#009E73",  # bluish green
    "S2b": "#F0E442",  # yellow
    "S3a": "#E69F00",  # orange
    "S3b": "#D55E00",  # vermilion
}
# Failure-mode colors (red/blue/gray theme, colorblind-safe variants)
MODE_COLORS = {
    "ASR":      "#D55E00",  # red-orange
    "Crack":    "#0072B2",  # blue
    "Censored": "#9A9A9A",  # gray
}


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


def panel_1a(ax: plt.Axes, df: pd.DataFrame) -> None:
    """ASR degradation trajectories: per-cell light + per-scenario mean bold."""
    for scen in SCEN_ORDER:
        sub = df[df["scenario"] == scen]
        col = SCEN_COLORS[scen]
        for cid, g in sub.groupby("cell_id"):
            g = g.sort_values("N_cycle")
            ax.plot(g["N_cycle"].values, g["ASR_obs"].values,
                    color=col, lw=0.4, alpha=0.35, zorder=1)
        # scenario-level mean trajectory (binned by N_cycle deciles)
        # Use simple groupby on N_cycle since cycles are evenly stepped per scenario.
        mean_curve = (sub.groupby("N_cycle")["ASR_obs"].mean().sort_index())
        ax.plot(mean_curve.index.values, mean_curve.values,
                color=col, lw=1.4, alpha=0.95, zorder=3, label=scen)

    ax.axhline(ASR_FAIL, ls=(0, (5, 1.5, 1, 1.5)), lw=0.9,
               color="black", zorder=2)
    ax.text(0.985, ASR_FAIL + 0.012, r"$\mathrm{ASR}^{*}=0.40$",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7.5, color="black",
            zorder=4)

    ax.set_xlabel("Thermal cycles, $N$")
    ax.set_ylabel(r"ASR  ($\Omega\cdot\mathrm{cm}^{2}$)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    leg = ax.legend(loc="lower right", ncol=2, columnspacing=1.0,
                    handlelength=1.6, handletextpad=0.5,
                    borderpad=0.4, labelspacing=0.25)
    leg.get_frame().set_linewidth(0.6)

    ax.text(-0.14, 1.02, "(a)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def panel_1b(ax: plt.Axes, t4: pd.DataFrame) -> None:
    """Stacked bar: observed_ASR / observed_Crack / censored_cells per scenario."""
    t4 = t4.set_index("scenario").reindex(SCEN_ORDER)
    asr  = t4["observed_ASR"].values.astype(float)
    crk  = t4["observed_Crack"].values.astype(float)
    cen  = t4["censored_cells"].values.astype(float)

    x = np.arange(len(SCEN_ORDER))
    width = 0.66

    b1 = ax.bar(x, asr, width, color=MODE_COLORS["ASR"],
                edgecolor="black", linewidth=0.5, label="ASR")
    b2 = ax.bar(x, crk, width, bottom=asr,
                color=MODE_COLORS["Crack"], edgecolor="black",
                linewidth=0.5, label="Crack")
    b3 = ax.bar(x, cen, width, bottom=asr + crk,
                color=MODE_COLORS["Censored"], edgecolor="black",
                linewidth=0.5, label="Censored")

    # numeric annotations inside each non-zero segment (hide if 0)
    for xi, (a, c, ce) in enumerate(zip(asr, crk, cen)):
        if a > 0:
            ax.text(xi, a/2, f"{int(a)}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
        if c > 0:
            ax.text(xi, a + c/2, f"{int(c)}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
        if ce > 0:
            ax.text(xi, a + c + ce/2, f"{int(ce)}", ha="center",
                    va="center", fontsize=7, color="black",
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(SCEN_ORDER)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Cell count")
    ax.set_ylim(0, 18.0)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    # x is categorical → no minor x ticks
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    leg = ax.legend(loc="upper left", ncol=3, columnspacing=0.9,
                    handlelength=1.2, handletextpad=0.4,
                    borderpad=0.4, labelspacing=0.25)
    leg.get_frame().set_linewidth(0.6)

    ax.text(-0.14, 1.02, "(b)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def main() -> None:
    configure_jmse_style()

    df = pd.read_csv(DATA / "data_01_ASR_observations_v3.csv")
    t4 = pd.read_csv(PROJ / "outputs" / "tables" / "Table_4_failure_mode_v3.csv")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(WIDTH_MM * MM, HEIGHT_MM * MM),
        gridspec_kw=dict(width_ratios=[1.55, 1.0], wspace=0.30),
    )
    panel_1a(ax_a, df)
    panel_1b(ax_b, t4)

    out_tiff = FIG_DIR / "Figure1_ASR_FailureMode.tiff"
    out_pdf  = FIG_DIR / "Figure1_ASR_FailureMode.pdf"
    out_png  = FIG_DIR / "Figure1_ASR_FailureMode.png"

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
