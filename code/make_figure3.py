"""
make_figure3.py - JMSE/IJHE-standard Figure 3 (140mm wide).

Sobol total-order indices (S_T) for 5 inputs across 6 scenarios.
  - Inputs: ASR0, k_cyc, k_time, alpha_cyc, alpha_time
  - Grouped bar chart (5 bars per scenario)
  - 95% bootstrap CI as error bars (from ST_conf_*)
  - Dominant input within each scenario flagged with hatching
  - Horizontal dashed reference at S_T = 0.10 (significance heuristic)
  - Input colors chosen distinct from scenario palette in Figs 1/2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

PROJ = Path(r"C:\Users\parke\sofc_revision")
FIG_DIR = PROJ / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCEN_ORDER = ["S1a", "S1b", "S2a", "S2b", "S3a", "S3b"]
INPUT_ORDER = ["ASR0", "k_cyc", "k_time", "alpha_cyc", "alpha_time"]
INPUT_LABELS = {
    "ASR0":        r"$\mathrm{ASR}_{0}$",
    "k_cyc":       r"$k_{\mathrm{cyc}}$",
    "k_time":      r"$k_{\mathrm{time}}$",
    "alpha_cyc":   r"$\alpha_{\mathrm{cyc}}$",
    "alpha_time":  r"$\alpha_{\mathrm{time}}$",
}
# Distinct from Wong scenario palette used in Figs 1, 2.
INPUT_COLORS = {
    "ASR0":        "#4D4D4D",  # dark gray (neutral baseline term)
    "k_cyc":       "#B83A4B",  # deep red    (cycle rate)
    "k_time":      "#2E5894",  # deep blue   (time rate)
    "alpha_cyc":   "#E27D60",  # warm coral  (cycle exponent)
    "alpha_time":  "#5A9367",  # sage green  (time exponent)
}

MM = 1.0 / 25.4
WIDTH_MM = 140.0
HEIGHT_MM = 80.0
THRESH = 0.10


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
        "hatch.linewidth":    0.6,
    })


def load_sobol() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ST and ST_conf wide tables indexed by scenario (rows) x input (cols)."""
    df = pd.read_csv(PROJ / "outputs" / "tables" / "Table_5_sobol_v3.csv")
    df = df.set_index("scenario").reindex(SCEN_ORDER)
    ST = pd.DataFrame({inp: df[f"ST_{inp}"] for inp in INPUT_ORDER})
    CI = pd.DataFrame({inp: df[f"ST_conf_{inp}"] for inp in INPUT_ORDER})
    return ST, CI


def main() -> None:
    configure_jmse_style()
    ST, CI = load_sobol()
    print("S_T values:")
    print(ST.round(4).to_string())
    print()
    print("Dominant input per scenario:")
    for s in SCEN_ORDER:
        d = ST.loc[s].idxmax()
        print(f"  {s}:  {d:11s}  S_T = {ST.loc[s, d]:.4f}")

    fig, ax = plt.subplots(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))

    n_scen = len(SCEN_ORDER)
    n_inp = len(INPUT_ORDER)
    group_width = 0.84
    bar_w = group_width / n_inp
    x = np.arange(n_scen)

    for j, inp in enumerate(INPUT_ORDER):
        # bar offset within group, centered
        offs = (j - (n_inp - 1) / 2.0) * bar_w
        heights = ST[inp].values
        errs = CI[inp].values
        col = INPUT_COLORS[inp]

        # detect dominant per scenario for hatching
        dominant_mask = (ST.idxmax(axis=1).values == inp)
        hatches = ["///" if d else "" for d in dominant_mask]

        for k in range(n_scen):
            ax.bar(
                x[k] + offs, heights[k], bar_w,
                color=col, edgecolor="black", linewidth=0.5,
                hatch=hatches[k], zorder=2,
                yerr=errs[k], error_kw=dict(
                    ecolor="black", elinewidth=0.6,
                    capsize=1.6, capthick=0.5,
                ),
                label=INPUT_LABELS[inp] if k == 0 else None,
            )

    # Significance threshold reference
    ax.axhline(THRESH, ls=(0, (4, 2)), lw=0.7, color="0.4", zorder=1)
    ax.text(0.005, THRESH + 0.005,
            f"$S_{{T}} = {THRESH:.2f}$",
            transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=7.0, color="0.30",
            zorder=3)

    # Hatched-bar legend entry (extra) for dominant marking
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="black",
                         hatch="///", linewidth=0.5,
                         label="dominant within scenario"))
    leg = ax.legend(
        handles=handles, labels=labels + ["dominant within scenario"],
        loc="upper left", ncol=3, columnspacing=1.0,
        handlelength=1.6, handletextpad=0.5,
        borderpad=0.4, labelspacing=0.25,
    )
    leg.get_frame().set_linewidth(0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(SCEN_ORDER)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(r"Sobol total-order index, $S_{T}$")

    ax.set_ylim(0.0, 0.66)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    ax.set_xlim(-0.5, n_scen - 0.5)

    out_tiff = FIG_DIR / "Figure3_SobolIndices.tiff"
    out_pdf  = FIG_DIR / "Figure3_SobolIndices.pdf"
    out_png  = FIG_DIR / "Figure3_SobolIndices.png"

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
