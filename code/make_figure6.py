"""
make_figure6.py - JMSE/IJHE-standard Figure 6 (140mm wide, 2-panel).

Model applicability vs FZJ literature.

  6a: Voltage/ASR degradation rate vs time (log-log)
       Model:     d(ASR)/dt = k_time * alpha_time * t^(alpha_time-1)
                  using exp(mu_time) population k_time and posterior median
                  alpha_time from idata_asr_v3.
       FZJ ref:   3 representative interval-mean rates with 5.7x deceleration
                  from 0-10kh to 30-93kh. Marked as schematic in legend.

  6b: Total operating time at N_max = 600 vs FZJ 93,000h scope
       op_time = N_max * tau_op[scenario]
       Horizontal dashed at 93,000h.  S3b exceeds; flagged "Beyond FZJ scope".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (LogLocator, NullFormatter,
                               FuncFormatter, MultipleLocator,
                               AutoMinorLocator)
from matplotlib.lines import Line2D

PROJ = Path(r"C:\Users\parke\sofc_revision")
DATA = Path(r"C:\Users\parke\Downloads\sofc_v3_output (1)")
FIG_DIR = PROJ / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCEN_ORDER = ["S1a", "S1b", "S2a", "S2b", "S3a", "S3b"]
SCEN_TAU = {"S1a": 9, "S1b": 15, "S2a": 36,
            "S2b": 72, "S3a": 120, "S3b": 240}
SCEN_COLORS = {
    "S1a": "#0072B2", "S1b": "#56B4E9",
    "S2a": "#009E73", "S2b": "#F0E442",
    "S3a": "#E69F00", "S3b": "#D55E00",
}

N_MAX_CYC      = 600
FZJ_LIMIT_H    = 93_000
FZJ_DECEL      = 5.7

# Color & style for the two data sources in panel (a)
MODEL_COLOR   = "#214487"     # deep blue, used in scenario palette already (S1a-ish)
FZJ_COLOR     = "#A23535"     # deep red

MM = 1.0 / 25.4
WIDTH_MM  = 140.0
HEIGHT_MM = 80.0


def configure_jmse_style() -> None:
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":        8,
        "axes.titlesize":   9,
        "axes.labelsize":   8.5,
        "xtick.labelsize":  7.5,
        "ytick.labelsize":  7.5,
        "legend.fontsize":  7.0,
        "axes.linewidth":   0.8,
        "axes.spines.top":     True,
        "axes.spines.right":   True,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.top":        True,
        "ytick.right":      True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
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


def get_model_params() -> tuple[float, float]:
    """Returns (k_time_pop, alpha_time) posterior medians from idata_asr_v3."""
    idata = az.from_netcdf(DATA / "idata_asr_v3.nc")
    post = idata.posterior
    a_t = float(post["alpha_time"].median().values)
    k_t = float(np.exp(np.median(post["mu_time"].values)))
    return k_t, a_t


def panel_6a(ax, k_time: float, alpha_time: float) -> None:
    """Model (continuous curve) vs FZJ literature (3 representative points)."""
    # ----- Model: instantaneous d(ASR)/dt -----
    t = np.logspace(2, 5.05, 300)                     # 100 h  ..  ~112 kh
    rate_model = k_time * alpha_time * t ** (alpha_time - 1.0)
    rate_model_uohm = rate_model * 1e6                # mu Ω cm² / h
    ax.plot(t, rate_model_uohm, color=MODEL_COLOR, lw=1.5,
            zorder=4,
            label=fr"Model:  $k_{{\mathrm{{time}}}}\,\alpha_{{\mathrm{{time}}}}\,t^{{\alpha-1}}$,"
                  fr"  $\alpha_{{\mathrm{{time}}}}={alpha_time:.3f}$")

    # ----- Model interval-mean rates (visual sanity, light markers) -----
    intervals = [(100, 10_000), (10_000, 30_000), (30_000, 93_000)]
    for t1, t2 in intervals:
        rmean = k_time * (t2 ** alpha_time - t1 ** alpha_time) \
                / (t2 - t1)
        rmean_u = rmean * 1e6
        tmid = np.exp(0.5 * (np.log(t1) + np.log(t2)))
        ax.plot([t1, t2], [rmean_u, rmean_u],
                color=MODEL_COLOR, lw=0.8, alpha=0.45, zorder=2)
        ax.plot([tmid], [rmean_u], marker="s", ms=3.4,
                mfc=MODEL_COLOR, mec=MODEL_COLOR, alpha=0.7,
                ls="none", zorder=3)

    # ----- FZJ representative interval rates -----
    # 5.7x deceleration scaled to overlap model magnitude in early window.
    fzj_early  = 5.0
    fzj_mid    = fzj_early / np.sqrt(FZJ_DECEL)       # ~2.1
    fzj_late   = fzj_early / FZJ_DECEL                # ~0.88
    fzj_rates  = [fzj_early, fzj_mid, fzj_late]
    for (t1, t2), r in zip(intervals, fzj_rates):
        tmid = np.exp(0.5 * (np.log(t1) + np.log(t2)))
        # horizontal stem to indicate interval extent
        ax.plot([t1, t2], [r, r],
                color=FZJ_COLOR, lw=0.9, alpha=0.55, zorder=4)
        ax.plot([tmid], [r], marker="o", ms=6.0,
                mfc=FZJ_COLOR, mec="black", mew=0.5,
                ls="none", zorder=5)

    # vertical interval guide lines
    for boundary in [10_000, 30_000, 93_000]:
        ax.axvline(boundary, color="0.75", lw=0.5, ls=":", zorder=1)

    # Annotations on deceleration ratios (early -> late)
    model_decel = rate_model_uohm[0] / rate_model_uohm[-1]   # rough instantaneous
    ax.text(0.04, 0.06,
            (rf"Model decel. (mid$\to$late) $\approx 1.5\times$" "\n"
             rf"FZJ decel. (early$\to$late) $\approx 5.7\times$"),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.0,
            bbox=dict(facecolor="white", edgecolor="0.7",
                      lw=0.5, boxstyle="round,pad=0.3"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(150, 1.1e5)
    ax.set_ylim(0.3, 60)
    ax.set_xlabel(r"Operating time, $t$ (hours)")
    ax.set_ylabel(r"Degradation rate, $\mathrm{d}(\mathrm{ASR})/\mathrm{d}t$"
                  r"  ($\mu\Omega\cdot\mathrm{cm}^{2}/$h)")

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

    handles = [
        Line2D([0], [0], color=MODEL_COLOR, lw=1.5, label="Model continuous"),
        Line2D([0], [0], color=MODEL_COLOR, lw=0, marker="s", ms=4,
               mfc=MODEL_COLOR, label="Model interval mean"),
        Line2D([0], [0], color=FZJ_COLOR, lw=0, marker="o", ms=5,
               mfc=FZJ_COLOR, mec="black", mew=0.5,
               label="FZJ literature (schematic)"),
    ]
    leg = ax.legend(handles=handles, loc="upper right",
                    handlelength=1.6, handletextpad=0.5,
                    borderpad=0.4, labelspacing=0.25)
    leg.get_frame().set_linewidth(0.6)

    ax.text(-0.16, 1.02, "(a)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def panel_6b(ax) -> None:
    """Per-scenario operating time (hours) at N_max=600, vs FZJ 93kh limit."""
    op_h = np.array([N_MAX_CYC * SCEN_TAU[s] for s in SCEN_ORDER],
                    dtype=float)

    x = np.arange(len(SCEN_ORDER))
    bars = []
    for xi, s, h in zip(x, SCEN_ORDER, op_h):
        col = SCEN_COLORS[s]
        b = ax.bar(xi, h / 1000.0, width=0.66,
                   color=col, edgecolor="black", linewidth=0.6,
                   zorder=2)
        bars.append(b)

    # numeric labels above each bar
    for xi, h in zip(x, op_h):
        ax.text(xi, h / 1000.0 + 4.0,
                f"{h/1000:.1f}",
                ha="center", va="bottom", fontsize=7.0)

    # FZJ 93kh threshold line
    ax.axhline(FZJ_LIMIT_H / 1000.0, color="black",
               ls=(0, (5, 1.5, 1, 1.5)), lw=1.0, zorder=3)
    ax.text(0.99, FZJ_LIMIT_H / 1000.0 + 1.2,
            f"FZJ literature scope = {FZJ_LIMIT_H/1000:.0f} kh",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom",
            fontsize=7.0, color="black", zorder=4)

    # "Beyond FZJ scope" annotation for S3b (last bar)
    s3b_idx = SCEN_ORDER.index("S3b")
    s3b_h_kh = op_h[s3b_idx] / 1000.0
    ax.annotate("Beyond FZJ scope",
                xy=(s3b_idx, FZJ_LIMIT_H / 1000.0),
                xytext=(s3b_idx - 0.6, s3b_h_kh + 8),
                textcoords="data",
                fontsize=7.0, fontweight="bold", color=SCEN_COLORS["S3b"],
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=SCEN_COLORS["S3b"],
                                lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(SCEN_ORDER)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(r"Operating time at $N_{\max}=600$ (kh)")
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, 168)
    ax.yaxis.set_major_locator(MultipleLocator(40))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.yaxis.grid(True, which="major", lw=0.4, color="0.85", zorder=0)
    ax.set_axisbelow(True)

    ax.text(-0.13, 1.02, "(b)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def main() -> None:
    configure_jmse_style()
    k_time, alpha_time = get_model_params()
    print(f"Model k_time(pop) = {k_time:.4e}  Ω·cm²")
    print(f"Model alpha_time  = {alpha_time:.4f}")
    print()
    print("Operating times at N_max=600:")
    for s in SCEN_ORDER:
        h = N_MAX_CYC * SCEN_TAU[s]
        rel = 100.0 * h / FZJ_LIMIT_H
        flag = "  <-- > FZJ scope" if h > FZJ_LIMIT_H else ""
        print(f"  {s:<5s} tau_op={SCEN_TAU[s]:>3d} h  -->  {h:>7d} h  "
              f"({rel:5.1f}% of FZJ){flag}")
    print()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(WIDTH_MM * MM, HEIGHT_MM * MM),
        gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.35),
    )
    panel_6a(ax_a, k_time, alpha_time)
    panel_6b(ax_b)

    out_tiff = FIG_DIR / "Figure6_ModelApplicability.tiff"
    out_pdf  = FIG_DIR / "Figure6_ModelApplicability.pdf"
    out_png  = FIG_DIR / "Figure6_ModelApplicability.png"

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
