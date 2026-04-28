"""
figures_all.py - Consolidated, standalone reproducer for Figures 1-6.

This script merges scripts/make_figure1.py .. make_figure6.py.  Run any
figure on demand without depending on the per-figure scripts:

    python scripts/figures_all.py --figure 3
    python scripts/figures_all.py --all
    python scripts/figures_all.py --all --output outputs/figures_v2/

JMSE/IJHE-standard layout for all panels:
  Arial 8-9 pt, top/right spines on, ticks inward, minor ticks, legend
  frame edge 0.7, Wong colorblind-safe scenario palette.

Outputs per figure: <basename>.tiff (600 dpi LZW) + .pdf (vector) + .png
(200 dpi preview).  Existing files in the target directory are
overwritten only for the figures requested.

Data dependencies (absolute paths, not modified):
  Fig 1 : data_01_ASR_observations_v3.csv  +  Table_4_failure_mode_v3.csv
  Fig 2 : idata_surv_v3.nc                 +  data_02_failure_events_v3.csv
  Fig 3 : Table_5_sobol_v3.csv
  Fig 4 : idata_asr_v3.nc + idata_surv_v3.nc + data_02_failure_events_v3.csv
  Fig 5 : Table_6_maintenance_v3.csv
  Fig 6 : idata_asr_v3.nc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (LogLocator, NullFormatter,
                               AutoMinorLocator, MultipleLocator)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------- paths
PROJ = Path(r"C:\Users\parke\sofc_revision")
DATA = Path(r"C:\Users\parke\Downloads\sofc_v3_output (1)")
TBL_DIR = PROJ / "outputs" / "tables"
DEFAULT_FIG_DIR = PROJ / "outputs" / "figures"

# ---------------------------------------------------- constants & palettes
SCEN_ORDER = ["S1a", "S1b", "S2a", "S2b", "S3a", "S3b"]

# Wong colorblind-safe scenario palette (consistent across all figures).
SCEN_COLORS = {
    "S1a": "#0072B2", "S1b": "#56B4E9",
    "S2a": "#009E73", "S2b": "#F0E442",
    "S3a": "#E69F00", "S3b": "#D55E00",
}
SCEN_TAU = {"S1a": 9, "S1b": 15, "S2a": 36,
            "S2b": 72, "S3a": 120, "S3b": 240}

# Failure-mode palette (Fig 1b)
MODE_COLORS = {"ASR":      "#D55E00",
               "Crack":    "#0072B2",
               "Censored": "#9A9A9A"}

# Sobol input palette (Fig 3) - distinct from scenario palette
INPUT_ORDER = ["ASR0", "k_cyc", "k_time", "alpha_cyc", "alpha_time"]
INPUT_LABELS = {
    "ASR0":        r"$\mathrm{ASR}_{0}$",
    "k_cyc":       r"$k_{\mathrm{cyc}}$",
    "k_time":      r"$k_{\mathrm{time}}$",
    "alpha_cyc":   r"$\alpha_{\mathrm{cyc}}$",
    "alpha_time":  r"$\alpha_{\mathrm{time}}$",
}
INPUT_COLORS = {
    "ASR0":        "#4D4D4D",
    "k_cyc":       "#B83A4B",
    "k_time":      "#2E5894",
    "alpha_cyc":   "#E27D60",
    "alpha_time":  "#5A9367",
}

# Maintenance dominant-mode markers (Fig 5)
MODE_MARKERS = {
    "Crack": dict(marker="s", color="#1F4E89", label="Crack-dominant"),
    "ASR":   dict(marker="o", color="#B83A4B", label="ASR-dominant"),
}

# Fig 6 overlay colors
MODEL_COLOR = "#214487"
FZJ_COLOR   = "#A23535"

# Physical constants & thresholds (NOT tuned; from common.py and reviewer spec)
ASR_FAIL    = 0.40
BETA_W      = 2.8
FZJ_LIMIT_H = 93_000
FZJ_DECEL   = 5.7
N_MAX_CYC   = 600

MM = 1.0 / 25.4


# --------------------------------------------------------------- jmse style
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


# ---------------------------------------------------------- save helper
def _save_fig(fig: plt.Figure, base: str, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_tiff = fig_dir / f"{base}.tiff"
    out_pdf  = fig_dir / f"{base}.pdf"
    out_png  = fig_dir / f"{base}.png"
    fig.savefig(out_tiff, dpi=600,
                pil_kwargs=dict(compression="tiff_lzw"))
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    for p in (out_tiff, out_pdf, out_png):
        sz_kb = p.stat().st_size / 1024
        print(f"  wrote  {p.name:36s}  {sz_kb:8.1f} KB")


# ===========================================================================
# Figure 1 - ASR trajectories + failure mode distribution
# ===========================================================================
def make_figure_1(fig_dir: Path) -> None:
    print("[Figure 1] ASR trajectories + failure modes")
    df = pd.read_csv(DATA / "data_01_ASR_observations_v3.csv")
    t4 = pd.read_csv(TBL_DIR / "Table_4_failure_mode_v3.csv")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(140 * MM, 75 * MM),
        gridspec_kw=dict(width_ratios=[1.55, 1.0], wspace=0.30),
    )

    # 1a: trajectories ---------------------------------------------------
    for scen in SCEN_ORDER:
        sub = df[df["scenario"] == scen]
        col = SCEN_COLORS[scen]
        for cid, g in sub.groupby("cell_id"):
            g = g.sort_values("N_cycle")
            ax_a.plot(g["N_cycle"].values, g["ASR_obs"].values,
                      color=col, lw=0.4, alpha=0.35, zorder=1)
        mean_curve = sub.groupby("N_cycle")["ASR_obs"].mean().sort_index()
        ax_a.plot(mean_curve.index.values, mean_curve.values,
                  color=col, lw=1.4, alpha=0.95, zorder=3, label=scen)
    ax_a.axhline(ASR_FAIL, ls=(0, (5, 1.5, 1, 1.5)), lw=0.9,
                 color="black", zorder=2)
    ax_a.text(0.985, ASR_FAIL + 0.012, r"$\mathrm{ASR}^{*}=0.40$",
              transform=ax_a.get_yaxis_transform(),
              ha="right", va="bottom", fontsize=7.5, color="black", zorder=4)
    ax_a.set_xlabel("Thermal cycles, $N$")
    ax_a.set_ylabel(r"ASR  ($\Omega\cdot\mathrm{cm}^{2}$)")
    ax_a.set_xlim(left=0)
    ax_a.set_ylim(bottom=0)
    ax_a.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_a.yaxis.set_minor_locator(AutoMinorLocator(2))
    leg = ax_a.legend(loc="lower right", ncol=2, columnspacing=1.0,
                      handlelength=1.6, handletextpad=0.5,
                      borderpad=0.4, labelspacing=0.25)
    leg.get_frame().set_linewidth(0.6)
    ax_a.text(-0.14, 1.02, "(a)", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold", va="bottom", ha="left")

    # 1b: failure-mode stacked bar --------------------------------------
    t4 = t4.set_index("scenario").reindex(SCEN_ORDER)
    asr = t4["observed_ASR"].values.astype(float)
    crk = t4["observed_Crack"].values.astype(float)
    cen = t4["censored_cells"].values.astype(float)

    x = np.arange(len(SCEN_ORDER))
    width = 0.66
    ax_b.bar(x, asr, width, color=MODE_COLORS["ASR"],
             edgecolor="black", linewidth=0.5, label="ASR")
    ax_b.bar(x, crk, width, bottom=asr, color=MODE_COLORS["Crack"],
             edgecolor="black", linewidth=0.5, label="Crack")
    ax_b.bar(x, cen, width, bottom=asr + crk, color=MODE_COLORS["Censored"],
             edgecolor="black", linewidth=0.5, label="Censored")

    for xi, (a, c, ce) in enumerate(zip(asr, crk, cen)):
        if a > 0:
            ax_b.text(xi, a / 2, f"{int(a)}", ha="center", va="center",
                      fontsize=7, color="white", fontweight="bold")
        if c > 0:
            ax_b.text(xi, a + c / 2, f"{int(c)}", ha="center", va="center",
                      fontsize=7, color="white", fontweight="bold")
        if ce > 0:
            ax_b.text(xi, a + c + ce / 2, f"{int(ce)}", ha="center",
                      va="center", fontsize=7, color="black",
                      fontweight="bold")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(SCEN_ORDER)
    ax_b.set_xlabel("Scenario")
    ax_b.set_ylabel("Cell count")
    ax_b.set_ylim(0, 18.0)
    ax_b.yaxis.set_major_locator(MultipleLocator(5))
    ax_b.yaxis.set_minor_locator(MultipleLocator(1))
    ax_b.tick_params(axis="x", which="minor", bottom=False, top=False)
    leg = ax_b.legend(loc="upper left", ncol=3, columnspacing=0.9,
                      handlelength=1.2, handletextpad=0.4,
                      borderpad=0.4, labelspacing=0.25)
    leg.get_frame().set_linewidth(0.6)
    ax_b.text(-0.14, 1.02, "(b)", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold", va="bottom", ha="left")

    _save_fig(fig, "Figure1_ASR_FailureMode", fig_dir)


# ===========================================================================
# Figure 2 - Weibull lambda_cr vs cycle frequency
# ===========================================================================
def _scenario_lambda_posterior(idata: az.InferenceData,
                               df_events: pd.DataFrame) -> pd.DataFrame:
    cell2scen = (df_events.drop_duplicates("cell_id_int")
                 .set_index("cell_id_int")["scenario"].to_dict())
    lam_s = idata.posterior["lam_s"]
    lam_flat = lam_s.stack(sample=("chain", "draw")).values  # (90, n_draws)
    rows = []
    for scen in SCEN_ORDER:
        cells = [c for c, s in cell2scen.items() if s == scen]
        per_draw = lam_flat[cells, :].mean(axis=0)
        m = float(np.mean(per_draw))
        hdi = az.hdi(per_draw, hdi_prob=0.94)
        rows.append({
            "scenario": scen,
            "f_cyc":    df_events.loc[df_events["scenario"] == scen,
                                      "f_cyc"].iloc[0],
            "lam_mean": m,
            "lam_lo":   float(hdi[0]),
            "lam_hi":   float(hdi[1]),
        })
    return pd.DataFrame(rows)


def make_figure_2(fig_dir: Path) -> None:
    print("[Figure 2] Weibull scale vs cycle frequency")
    df_ev = pd.read_csv(DATA / "data_02_failure_events_v3.csv")
    idata = az.from_netcdf(DATA / "idata_surv_v3.nc")
    summ = _scenario_lambda_posterior(idata, df_ev)
    print(summ.to_string(index=False))

    fig, ax = plt.subplots(figsize=(90 * MM, 80 * MM))

    lf = np.log(summ["f_cyc"].values)
    lL = np.log(summ["lam_mean"].values)
    b, a = np.polyfit(lf, lL, 1)
    xfit = np.logspace(np.log10(0.9), np.log10(33), 100)
    yfit = np.exp(a) * xfit ** b
    ax.plot(xfit, yfit, color="0.45", lw=0.9,
            ls=(0, (4, 2)), zorder=1,
            label=fr"power-law fit  $\lambda \propto f_{{\mathrm{{cyc}}}}^{{{b:.2f}}}$")

    LABEL_OFF = {"S1a": (7, 6), "S1b": (7, -10),
                 "S2a": (7, 6), "S2b": (7, -10),
                 "S3a": (7, 6), "S3b": (-26, 6)}
    for _, row in summ.iterrows():
        s = row["scenario"]
        c = SCEN_COLORS[s]
        yerr_lo = row["lam_mean"] - row["lam_lo"]
        yerr_hi = row["lam_hi"]   - row["lam_mean"]
        ax.errorbar(row["f_cyc"], row["lam_mean"],
                    yerr=[[yerr_lo], [yerr_hi]],
                    fmt="o", color=c, ecolor=c, mec="black", mew=0.5,
                    ms=6, elinewidth=1.0, capsize=2.5, capthick=1.0,
                    zorder=4)
        dx, dy = LABEL_OFF[s]
        ax.annotate(s, xy=(row["f_cyc"], row["lam_mean"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, fontweight="bold", color=c, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.85, 33)
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

    _save_fig(fig, "Figure2_WeibullScale", fig_dir)


# ===========================================================================
# Figure 3 - Sobol total-order indices (grouped bar)
# ===========================================================================
def make_figure_3(fig_dir: Path) -> None:
    print("[Figure 3] Sobol total-order indices")
    df = pd.read_csv(TBL_DIR / "Table_5_sobol_v3.csv")
    df = df.set_index("scenario").reindex(SCEN_ORDER)
    ST = pd.DataFrame({inp: df[f"ST_{inp}"] for inp in INPUT_ORDER})
    CI = pd.DataFrame({inp: df[f"ST_conf_{inp}"] for inp in INPUT_ORDER})
    print(ST.round(4).to_string())

    fig, ax = plt.subplots(figsize=(140 * MM, 80 * MM))
    n_scen, n_inp = len(SCEN_ORDER), len(INPUT_ORDER)
    group_w = 0.84
    bar_w = group_w / n_inp
    x = np.arange(n_scen)

    for j, inp in enumerate(INPUT_ORDER):
        offs = (j - (n_inp - 1) / 2.0) * bar_w
        heights = ST[inp].values
        errs = CI[inp].values
        col = INPUT_COLORS[inp]
        dom_mask = (ST.idxmax(axis=1).values == inp)
        hatches = ["///" if d else "" for d in dom_mask]
        for k in range(n_scen):
            ax.bar(x[k] + offs, heights[k], bar_w,
                   color=col, edgecolor="black", linewidth=0.5,
                   hatch=hatches[k], zorder=2,
                   yerr=errs[k], error_kw=dict(
                       ecolor="black", elinewidth=0.6,
                       capsize=1.6, capthick=0.5),
                   label=INPUT_LABELS[inp] if k == 0 else None)

    THRESH = 0.10
    ax.axhline(THRESH, ls=(0, (4, 2)), lw=0.7, color="0.4", zorder=1)
    ax.text(0.005, THRESH + 0.005, f"$S_{{T}} = {THRESH:.2f}$",
            transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=7.0, color="0.30", zorder=3)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="black",
                         hatch="///", linewidth=0.5,
                         label="dominant within scenario"))
    leg = ax.legend(handles=handles,
                    labels=labels + ["dominant within scenario"],
                    loc="upper left", ncol=3, columnspacing=1.0,
                    handlelength=1.6, handletextpad=0.5,
                    borderpad=0.4, labelspacing=0.25)
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

    _save_fig(fig, "Figure3_SobolIndices", fig_dir)


# ===========================================================================
# Figure 4 - Pfail(N) with 95% PPI per scenario
# ===========================================================================
def _stack(da):
    return da.stack(sample=("chain", "draw")).transpose("sample", ...)


def make_figure_4(fig_dir: Path,
                  n_draws: int = 5000,
                  rng_seed: int = 2024) -> None:
    print("[Figure 4] Pfail(N) with 95% PPI per scenario")
    rng = np.random.default_rng(rng_seed)
    df_ev = pd.read_csv(DATA / "data_02_failure_events_v3.csv")
    cell2scen = (df_ev.drop_duplicates("cell_id_int")
                 .set_index("cell_id_int")["scenario"].to_dict())
    scen_tau = df_ev.groupby("scenario")["tau_op"].first().to_dict()

    idata_a = az.from_netcdf(DATA / "idata_asr_v3.nc")
    idata_s = az.from_netcdf(DATA / "idata_surv_v3.nc")

    ASR0 = _stack(idata_a.posterior["ASR0_i"]).values
    KCYC = _stack(idata_a.posterior["k_cyc_i"]).values
    KTIM = _stack(idata_a.posterior["k_time_i"]).values
    ACYC = _stack(idata_a.posterior["alpha_cyc"]).values
    ATIM = _stack(idata_a.posterior["alpha_time"]).values
    LAM  = _stack(idata_s.posterior["lam_s"]).values

    iA = rng.integers(0, ASR0.shape[0], n_draws)
    iS = rng.integers(0, LAM.shape[0],  n_draws)
    ASR0, KCYC, KTIM = ASR0[iA], KCYC[iA], KTIM[iA]
    ACYC, ATIM       = ACYC[iA], ATIM[iA]
    LAM = LAM[iS]

    Ngrid = np.linspace(1.0, 600.0, 81)
    P_TH = 0.10

    pfail_med, pfail_lo, pfail_hi, n_at_p10 = {}, {}, {}, {}
    for scen in SCEN_ORDER:
        cells = sorted([c for c, s in cell2scen.items() if s == scen])
        tau = float(scen_tau[scen])
        N3 = Ngrid[None, None, :]
        ACYC3 = ACYC[:, None, None]; ATIM3 = ATIM[:, None, None]
        ASR0_3 = ASR0[:, cells, None]
        KCYC_3 = KCYC[:, cells, None]
        KTIM_3 = KTIM[:, cells, None]
        LAM_3  = LAM[:, cells, None]
        asr_t = (ASR0_3 + KCYC_3 * N3 ** ACYC3
                 + KTIM_3 * (N3 * tau) ** ATIM3)
        S_asr = (asr_t < ASR_FAIL).astype(np.float32).mean(axis=1)
        S_crk = np.exp(-(N3 / LAM_3) ** BETA_W).mean(axis=1)
        Pfail = 1.0 - S_asr * S_crk

        med = np.median(Pfail, axis=0)
        lo  = np.quantile(Pfail, 0.025, axis=0)
        hi  = np.quantile(Pfail, 0.975, axis=0)
        pfail_med[scen], pfail_lo[scen], pfail_hi[scen] = med, lo, hi
        if med[-1] >= P_TH:
            j = int(np.argmax(med >= P_TH))
            if j == 0:
                n_at_p10[scen] = float(Ngrid[0])
            else:
                x0, x1 = Ngrid[j - 1], Ngrid[j]
                y0, y1 = med[j - 1], med[j]
                n_at_p10[scen] = float(
                    x0 + (P_TH - y0) / (y1 - y0) * (x1 - x0))
        else:
            n_at_p10[scen] = float("nan")

    fig, axes = plt.subplots(
        2, 3,
        figsize=(140 * MM, 95 * MM),
        sharex=True, sharey=True,
        gridspec_kw=dict(hspace=0.20, wspace=0.10),
    )
    for ax, scen in zip(axes.flat, SCEN_ORDER):
        col = SCEN_COLORS[scen]
        ax.fill_between(Ngrid, pfail_lo[scen], pfail_hi[scen],
                        color=col, alpha=0.22, lw=0, zorder=1)
        ax.plot(Ngrid, pfail_med[scen], color=col, lw=1.4, zorder=3)
        ax.axhline(P_TH, ls=(0, (4, 2)), lw=0.6, color="0.4", zorder=2)
        ax.text(0.99, P_TH + 0.012, f"$P_{{fail}}={P_TH:.2f}$",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.5, color="0.30")
        n10 = n_at_p10[scen]
        if np.isfinite(n10):
            ax.plot([n10], [P_TH], marker="o", ms=5.0,
                    mfc=col, mec="black", mew=0.6, zorder=5)
            if n10 > 380:
                dx, ha = -8, "right"
            else:
                dx, ha = 8, "left"
            ax.annotate(rf"$N_{{10}}={n10:.0f}$",
                        xy=(n10, P_TH), xytext=(dx, -10),
                        textcoords="offset points",
                        fontsize=7.0, fontweight="bold", color=col,
                        ha=ha, va="top", zorder=6)
        else:
            ax.text(0.97, 0.92, r"$N_{10} > 600$",
                    transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.0,
                    fontweight="bold", color=col)
        ax.set_title(scen, color=col, fontweight="bold",
                     loc="left", pad=2)
        ax.set_xlim(0, 600); ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Thermal cycles, $N$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$P_{\mathrm{fail}}(N)$")

    handles = [
        Line2D([0], [0], color="black", lw=1.4, label="median"),
        Patch(facecolor="0.5", alpha=0.30, label="95% PPI"),
    ]
    leg = axes[0, 0].legend(handles=handles, loc="upper left",
                            handlelength=1.6, handletextpad=0.5,
                            borderpad=0.4, labelspacing=0.25)
    leg.get_frame().set_linewidth(0.6)

    _save_fig(fig, "Figure4_FailureProbability", fig_dir)


# ===========================================================================
# Figure 5 - Maintenance intervals (cycles + months)
# ===========================================================================
def _maint_panel(ax, x, heights, modes, *, ylabel, fmt_value,
                 ylim_max, panel_tag, y_minor, y_major):
    for xi, h, scen, mode in zip(x, heights, SCEN_ORDER, modes):
        col = SCEN_COLORS[scen]
        ax.bar(xi, h, width=0.66, color=col,
               edgecolor="black", linewidth=0.6, zorder=2)
        ax.text(xi, h + ylim_max * 0.018, fmt_value(h),
                ha="center", va="bottom", fontsize=7.0, zorder=4)
        m = MODE_MARKERS[mode]
        marker_y = max(h * 0.85, h - ylim_max * 0.06)
        ax.plot([xi], [marker_y], marker=m["marker"], ms=7.0,
                mfc=m["color"], mec="white", mew=1.0, ls="none",
                zorder=5)
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


def make_figure_5(fig_dir: Path) -> None:
    print("[Figure 5] Maintenance intervals (cycles + months)")
    df = pd.read_csv(TBL_DIR / "Table_6_maintenance_v3.csv")
    df = df.set_index("scenario").reindex(SCEN_ORDER).reset_index()
    print(df[["scenario", "N_maint", "period_months",
              "pfail_at_maint", "dominant_cause"]].to_string(index=False))

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(140 * MM, 78 * MM),
        gridspec_kw=dict(wspace=0.30),
    )
    x = np.arange(len(SCEN_ORDER))
    modes = df["dominant_cause"].tolist()
    _maint_panel(ax_l, x, df["N_maint"].values.astype(float), modes,
                 ylabel=r"Maintenance interval, $N_{\mathrm{maint}}$ (cycles)",
                 fmt_value=lambda v: f"{int(v)}",
                 ylim_max=720.0, panel_tag="(a)",
                 y_minor=20, y_major=100)
    _maint_panel(ax_r, x, df["period_months"].values.astype(float), modes,
                 ylabel="Calendar maintenance period (months)",
                 fmt_value=lambda v: f"{v:.1f}",
                 ylim_max=190.0, panel_tag="(b)",
                 y_minor=10, y_major=40)

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
                     borderpad=0.3, columnspacing=1.6, frameon=True)
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_edgecolor("0.7")

    _save_fig(fig, "Figure5_MaintenanceIntervals", fig_dir)


# ===========================================================================
# Figure 6 - Model applicability vs FZJ literature
# ===========================================================================
def _get_model_params() -> tuple[float, float]:
    idata = az.from_netcdf(DATA / "idata_asr_v3.nc")
    p = idata.posterior
    a_t = float(p["alpha_time"].median().values)
    k_t = float(np.exp(np.median(p["mu_time"].values)))
    return k_t, a_t


def make_figure_6(fig_dir: Path) -> None:
    print("[Figure 6] Model applicability vs FZJ literature")
    k_time, alpha_time = _get_model_params()
    print(f"  k_time(pop)={k_time:.4e}  alpha_time={alpha_time:.4f}")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(140 * MM, 80 * MM),
        gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.35),
    )

    # 6a -----------------------------------------------------------------
    t = np.logspace(2, 5.05, 300)
    rate_uohm = (k_time * alpha_time * t ** (alpha_time - 1.0)) * 1e6
    ax_a.plot(t, rate_uohm, color=MODEL_COLOR, lw=1.5, zorder=4)

    intervals = [(100, 10_000), (10_000, 30_000), (30_000, 93_000)]
    for t1, t2 in intervals:
        rmean = k_time * (t2 ** alpha_time - t1 ** alpha_time) / (t2 - t1)
        rmean_u = rmean * 1e6
        tmid = np.exp(0.5 * (np.log(t1) + np.log(t2)))
        ax_a.plot([t1, t2], [rmean_u, rmean_u],
                  color=MODEL_COLOR, lw=0.8, alpha=0.45, zorder=2)
        ax_a.plot([tmid], [rmean_u], marker="s", ms=3.4,
                  mfc=MODEL_COLOR, mec=MODEL_COLOR, alpha=0.7,
                  ls="none", zorder=3)

    fzj_early = 5.0
    fzj_rates = [fzj_early, fzj_early / np.sqrt(FZJ_DECEL),
                 fzj_early / FZJ_DECEL]
    for (t1, t2), r in zip(intervals, fzj_rates):
        tmid = np.exp(0.5 * (np.log(t1) + np.log(t2)))
        ax_a.plot([t1, t2], [r, r],
                  color=FZJ_COLOR, lw=0.9, alpha=0.55, zorder=4)
        ax_a.plot([tmid], [r], marker="o", ms=6.0,
                  mfc=FZJ_COLOR, mec="black", mew=0.5,
                  ls="none", zorder=5)

    for boundary in (10_000, 30_000, 93_000):
        ax_a.axvline(boundary, color="0.75", lw=0.5, ls=":", zorder=1)

    ax_a.text(0.04, 0.06,
              (r"Model decel. (mid$\to$late) $\approx 1.5\times$" "\n"
               r"FZJ decel. (early$\to$late) $\approx 5.7\times$"),
              transform=ax_a.transAxes, ha="left", va="bottom",
              fontsize=7.0,
              bbox=dict(facecolor="white", edgecolor="0.7",
                        lw=0.5, boxstyle="round,pad=0.3"))
    ax_a.set_xscale("log"); ax_a.set_yscale("log")
    ax_a.set_xlim(150, 1.1e5); ax_a.set_ylim(0.3, 60)
    ax_a.set_xlabel(r"Operating time, $t$ (hours)")
    ax_a.set_ylabel(r"Degradation rate, $\mathrm{d}(\mathrm{ASR})/\mathrm{d}t$"
                    r"  ($\mu\Omega\cdot\mathrm{cm}^{2}/$h)")
    ax_a.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax_a.xaxis.set_minor_locator(LogLocator(base=10,
                                            subs=np.arange(2, 10) * 0.1,
                                            numticks=12))
    ax_a.xaxis.set_minor_formatter(NullFormatter())
    ax_a.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax_a.yaxis.set_minor_locator(LogLocator(base=10,
                                            subs=np.arange(2, 10) * 0.1,
                                            numticks=12))
    ax_a.yaxis.set_minor_formatter(NullFormatter())

    handles = [
        Line2D([0], [0], color=MODEL_COLOR, lw=1.5, label="Model continuous"),
        Line2D([0], [0], color=MODEL_COLOR, lw=0, marker="s", ms=4,
               mfc=MODEL_COLOR, label="Model interval mean"),
        Line2D([0], [0], color=FZJ_COLOR, lw=0, marker="o", ms=5,
               mfc=FZJ_COLOR, mec="black", mew=0.5,
               label="FZJ literature (schematic)"),
    ]
    leg = ax_a.legend(handles=handles, loc="upper right",
                      handlelength=1.6, handletextpad=0.5,
                      borderpad=0.4, labelspacing=0.25)
    leg.get_frame().set_linewidth(0.6)
    ax_a.text(-0.16, 1.02, "(a)", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold", va="bottom", ha="left")

    # 6b -----------------------------------------------------------------
    op_h = np.array([N_MAX_CYC * SCEN_TAU[s] for s in SCEN_ORDER],
                    dtype=float)
    x = np.arange(len(SCEN_ORDER))
    for xi, s, h in zip(x, SCEN_ORDER, op_h):
        col = SCEN_COLORS[s]
        ax_b.bar(xi, h / 1000.0, width=0.66, color=col,
                 edgecolor="black", linewidth=0.6, zorder=2)
    for xi, h in zip(x, op_h):
        ax_b.text(xi, h / 1000.0 + 4.0, f"{h/1000:.1f}",
                  ha="center", va="bottom", fontsize=7.0)

    ax_b.axhline(FZJ_LIMIT_H / 1000.0, color="black",
                 ls=(0, (5, 1.5, 1, 1.5)), lw=1.0, zorder=3)
    ax_b.text(0.99, FZJ_LIMIT_H / 1000.0 + 1.2,
              f"FZJ literature scope = {FZJ_LIMIT_H/1000:.0f} kh",
              transform=ax_b.get_yaxis_transform(),
              ha="right", va="bottom",
              fontsize=7.0, color="black", zorder=4)

    s3b_idx = SCEN_ORDER.index("S3b")
    s3b_h_kh = op_h[s3b_idx] / 1000.0
    ax_b.annotate("Beyond FZJ scope",
                  xy=(s3b_idx, FZJ_LIMIT_H / 1000.0),
                  xytext=(s3b_idx - 0.6, s3b_h_kh + 8),
                  textcoords="data",
                  fontsize=7.0, fontweight="bold",
                  color=SCEN_COLORS["S3b"], ha="center", va="bottom",
                  arrowprops=dict(arrowstyle="->",
                                  color=SCEN_COLORS["S3b"], lw=0.8))

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(SCEN_ORDER)
    ax_b.set_xlabel("Scenario")
    ax_b.set_ylabel(r"Operating time at $N_{\max}=600$ (kh)")
    ax_b.set_xlim(-0.5, len(x) - 0.5)
    ax_b.set_ylim(0, 168)
    ax_b.yaxis.set_major_locator(MultipleLocator(40))
    ax_b.yaxis.set_minor_locator(MultipleLocator(10))
    ax_b.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax_b.yaxis.grid(True, which="major", lw=0.4, color="0.85", zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.text(-0.13, 1.02, "(b)", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold", va="bottom", ha="left")

    _save_fig(fig, "Figure6_ModelApplicability", fig_dir)


# ===========================================================================
# Driver
# ===========================================================================
DISPATCH = {
    1: make_figure_1, 2: make_figure_2, 3: make_figure_3,
    4: make_figure_4, 5: make_figure_5, 6: make_figure_6,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Figures 1-6 from v3 posterior + tables.")
    parser.add_argument("--figure", "-f", type=int,
                        choices=sorted(DISPATCH.keys()),
                        help="Single figure number to (re)generate")
    parser.add_argument("--all", action="store_true",
                        help="Regenerate all six figures")
    parser.add_argument("--output", "-o", type=str,
                        default=str(DEFAULT_FIG_DIR),
                        help=f"Output directory  (default: {DEFAULT_FIG_DIR})")
    args = parser.parse_args()

    if not args.all and args.figure is None:
        parser.error("specify --figure N or --all")

    fig_dir = Path(args.output).resolve()
    print(f"output dir: {fig_dir}\n")
    configure_jmse_style()

    nums = sorted(DISPATCH.keys()) if args.all else [args.figure]
    for n in nums:
        DISPATCH[n](fig_dir)
        print()


if __name__ == "__main__":
    main()
