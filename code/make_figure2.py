"""
make_figure2.py - JMSE/IJHE-standard Figure 2 (90mm wide).

Weibull crack-hazard scale lambda_cr (cycles) vs cycle frequency f_cyc (cycles/month).
  - 6 scenario points with 94% HDI error bars
  - log-log axes, 1 <= f_cyc <= 30
  - Wong colorblind-safe palette (consistent with Figure 1)
  - Each point annotated with scenario label (S1a..S3b)
  - Power-law guide line fit through scenario posterior means
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

PROJ = Path(r"C:\Users\parke\sofc_revision")
DATA = Path(r"C:\Users\parke\Downloads\sofc_v3_output (1)")
FIG_DIR = PROJ / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCEN_ORDER = ["S1a", "S1b", "S2a", "S2b", "S3a", "S3b"]

MM = 1.0 / 25.4
WIDTH_MM = 90.0
HEIGHT_MM = 80.0

SCEN_COLORS = {
    "S1a": "#0072B2", "S1b": "#56B4E9",
    "S2a": "#009E73", "S2b": "#F0E442",
    "S3a": "#E69F00", "S3b": "#D55E00",
}

# Annotation offsets (in points) per scenario to avoid overlap with markers/error bars.
LABEL_OFFSETS = {
    "S1a": ( 7,  6),
    "S1b": ( 7, -10),
    "S2a": ( 7,  6),
    "S2b": ( 7, -10),
    "S3a": ( 7,  6),
    "S3b": (-26,  6),
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


def scenario_lambda_posterior(idata: az.InferenceData,
                              df_events: pd.DataFrame) -> pd.DataFrame:
    """For each scenario, compute per-draw mean of lam_s over cells in
    that scenario, then collapse to mean / 94% HDI across draws."""
    # cell_id_int 0..89 -> scenario string
    cell2scen = (df_events
                 .drop_duplicates("cell_id_int")
                 .set_index("cell_id_int")["scenario"]
                 .to_dict())

    lam_s = idata.posterior["lam_s"]              # (chain, draw, lam_s_dim_0)
    lam_flat = lam_s.stack(sample=("chain", "draw")).values   # (90, n_draws)

    rows = []
    for scen in SCEN_ORDER:
        cells_in_scen = [c for c, s in cell2scen.items() if s == scen]
        scen_mean_per_draw = lam_flat[cells_in_scen, :].mean(axis=0)
        m = float(np.mean(scen_mean_per_draw))
        hdi = az.hdi(scen_mean_per_draw, hdi_prob=0.94)
        rows.append({
            "scenario": scen,
            "f_cyc":    df_events.loc[df_events["scenario"] == scen,
                                      "f_cyc"].iloc[0],
            "lam_mean": m,
            "lam_lo":   float(hdi[0]),
            "lam_hi":   float(hdi[1]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    configure_jmse_style()

    df_events = pd.read_csv(DATA / "data_02_failure_events_v3.csv")
    idata = az.from_netcdf(DATA / "idata_surv_v3.nc")
    summ = scenario_lambda_posterior(idata, df_events)
    print(summ.to_string(index=False))

    fig, ax = plt.subplots(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))

    # power-law fit on log-log: log(lam_mean) ~ a + b * log(f_cyc)
    lf = np.log(summ["f_cyc"].values)
    lL = np.log(summ["lam_mean"].values)
    b, a = np.polyfit(lf, lL, 1)
    xfit = np.logspace(np.log10(0.9), np.log10(33), 100)
    yfit = np.exp(a) * xfit ** b
    ax.plot(xfit, yfit, color="0.45", lw=0.9,
            ls=(0, (4, 2)), zorder=1,
            label=fr"power-law fit  $\lambda \propto f_{{\mathrm{{cyc}}}}^{{{b:.2f}}}$")

    # scatter + asymmetric error bars per scenario
    for _, row in summ.iterrows():
        s = row["scenario"]
        c = SCEN_COLORS[s]
        yerr_lo = row["lam_mean"] - row["lam_lo"]
        yerr_hi = row["lam_hi"]   - row["lam_mean"]
        ax.errorbar(
            row["f_cyc"], row["lam_mean"],
            yerr=[[yerr_lo], [yerr_hi]],
            fmt="o", color=c, ecolor=c, mec="black", mew=0.5,
            ms=6, elinewidth=1.0, capsize=2.5, capthick=1.0,
            zorder=4,
        )
        dx, dy = LABEL_OFFSETS[s]
        ax.annotate(s,
                    xy=(row["f_cyc"], row["lam_mean"]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=7.5, fontweight="bold", color=c,
                    zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.85, 33)
    # auto y-limits with some headroom
    y_lo = summ["lam_lo"].min() * 0.7
    y_hi = summ["lam_hi"].max() * 1.4
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel(r"Cycle frequency, $f_{\mathrm{cyc}}$ (cycles/month)")
    ax.set_ylabel(r"Weibull scale, $\lambda_{\mathrm{cr}}$ (cycles)")

    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax.xaxis.set_minor_locator(LogLocator(base=10,
                                          subs=np.arange(2, 10) * 0.1,
                                          numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax.yaxis.set_minor_locator(LogLocator(base=10,
                                          subs=np.arange(2, 10) * 0.1,
                                          numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())

    leg = ax.legend(loc="upper right", handlelength=2.4,
                    handletextpad=0.5, borderpad=0.4)
    leg.get_frame().set_linewidth(0.6)

    out_tiff = FIG_DIR / "Figure2_WeibullScale.tiff"
    out_pdf  = FIG_DIR / "Figure2_WeibullScale.pdf"
    out_png  = FIG_DIR / "Figure2_WeibullScale.png"

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
