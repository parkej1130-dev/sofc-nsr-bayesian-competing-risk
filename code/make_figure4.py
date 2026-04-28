"""
make_figure4.py - JMSE/IJHE-standard Figure 4 (140mm wide).

System failure probability with 95% Posterior Predictive Interval (PPI).
Reviewer 1 directly requested PPI display.

Pfail(N) = 1 - S_ASR(N) * S_crack(N)
  S_ASR_scen,d(N)   = mean over cells c-in-scen of  1{ASR_c,d(N) < ASR*}
  S_crack_scen,d(N) = mean over cells c-in-scen of  exp(-(N / lam_s_c,d)^BETA_W)

  ASR_c,d(N) = ASR0_i_c,d + k_cyc_i_c,d * N^alpha_cyc_d
                            + k_time_i_c,d * (N * tau_op_s)^alpha_time_d

5000 jointly-sampled draws (independent indices from idata_asr and idata_surv).
Layout: 2 rows x 3 cols, sharex=True, sharey=True.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

PROJ = Path(r"C:\Users\parke\sofc_revision")
DATA = Path(r"C:\Users\parke\Downloads\sofc_v3_output (1)")
FIG_DIR = PROJ / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCEN_ORDER = ["S1a", "S1b", "S2a", "S2b", "S3a", "S3b"]
SCEN_COLORS = {
    "S1a": "#0072B2", "S1b": "#56B4E9",
    "S2a": "#009E73", "S2b": "#F0E442",
    "S3a": "#E69F00", "S3b": "#D55E00",
}

ASR_THRESHOLD = 0.40
BETA_W        = 2.8
N_GRID        = np.linspace(1.0, 600.0, 81)        # cycles 1..600, 81 pts
N_DRAWS       = 5000
P_THRESHOLD   = 0.10
RNG_SEED      = 2024

MM = 1.0 / 25.4
WIDTH_MM = 140.0
HEIGHT_MM = 95.0


def configure_jmse_style() -> None:
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":        8,
        "axes.titlesize":   8.5,
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


def cell_to_scen(df_events: pd.DataFrame) -> dict[int, str]:
    return (df_events.drop_duplicates("cell_id_int")
            .set_index("cell_id_int")["scenario"].to_dict())


def scen_to_tau_op(df_events: pd.DataFrame) -> dict[str, float]:
    return (df_events.groupby("scenario")["tau_op"].first().to_dict())


def stack_chains(da):
    """Combine chain x draw into a single 'sample' axis at position 0."""
    return da.stack(sample=("chain", "draw")).transpose("sample", ...)


def main() -> None:
    configure_jmse_style()
    rng = np.random.default_rng(RNG_SEED)

    df_events = pd.read_csv(DATA / "data_02_failure_events_v3.csv")
    cell2scen = cell_to_scen(df_events)
    scen_tau  = scen_to_tau_op(df_events)

    idata_asr  = az.from_netcdf(DATA / "idata_asr_v3.nc")
    idata_surv = az.from_netcdf(DATA / "idata_surv_v3.nc")

    # extract: shape (n_total, ...) with sample as 0-th axis
    ASR0_all  = stack_chains(idata_asr.posterior["ASR0_i"]).values     # (n, 90)
    KCYC_all  = stack_chains(idata_asr.posterior["k_cyc_i"]).values    # (n, 90)
    KTIM_all  = stack_chains(idata_asr.posterior["k_time_i"]).values   # (n, 90)
    ACYC_all  = stack_chains(idata_asr.posterior["alpha_cyc"]).values  # (n,)
    ATIM_all  = stack_chains(idata_asr.posterior["alpha_time"]).values # (n,)
    LAM_all   = stack_chains(idata_surv.posterior["lam_s"]).values     # (m, 90)

    n_a = ASR0_all.shape[0]
    n_s = LAM_all.shape[0]

    idxA = rng.integers(0, n_a, N_DRAWS)
    idxS = rng.integers(0, n_s, N_DRAWS)

    ASR0 = ASR0_all[idxA]            # (N_DRAWS, 90)
    KCYC = KCYC_all[idxA]
    KTIM = KTIM_all[idxA]
    ACYC = ACYC_all[idxA]            # (N_DRAWS,)
    ATIM = ATIM_all[idxA]
    LAM  = LAM_all[idxS]             # (N_DRAWS, 90)

    Ngrid = N_GRID                   # (G,)
    G = len(Ngrid)

    # Compute per-scenario Pfail PPI
    pfail_med = {}
    pfail_lo  = {}
    pfail_hi  = {}
    n_at_p10  = {}

    for scen in SCEN_ORDER:
        cells = sorted([c for c, s in cell2scen.items() if s == scen])
        tau   = float(scen_tau[scen])

        ASR0_c = ASR0[:, cells]      # (N_DRAWS, n_c)
        KCYC_c = KCYC[:, cells]
        KTIM_c = KTIM[:, cells]
        LAM_c  = LAM[:, cells]

        # broadcast: N -> shape (1, 1, G)
        N3 = Ngrid[None, None, :]
        # alpha_cyc, alpha_time per draw -> (n,1,1)
        ACYC3 = ACYC[:, None, None]
        ATIM3 = ATIM[:, None, None]
        # cell params per draw -> (n, n_c, 1)
        ASR0_3 = ASR0_c[:, :, None]
        KCYC_3 = KCYC_c[:, :, None]
        KTIM_3 = KTIM_c[:, :, None]
        LAM_3  = LAM_c[:, :, None]

        # ASR(N) per (draw, cell, N)
        asr_t = (ASR0_3
                 + KCYC_3 * N3 ** ACYC3
                 + KTIM_3 * (N3 * tau) ** ATIM3)
        S_asr_cell = (asr_t < ASR_THRESHOLD).astype(np.float32)
        # scenario-level: mean over cells -> (n, G)
        S_asr = S_asr_cell.mean(axis=1)

        # Weibull crack survival per (draw, cell, N): exp(-(N/lam)^BETA)
        F_crk_cell = np.exp(-(N3 / LAM_3) ** BETA_W)   # (n, n_c, G)
        S_crk = F_crk_cell.mean(axis=1)                # (n, G)

        Pfail = 1.0 - S_asr * S_crk                    # (n, G) per draw

        med = np.median(Pfail, axis=0)
        lo  = np.quantile(Pfail, 0.025, axis=0)
        hi  = np.quantile(Pfail, 0.975, axis=0)

        pfail_med[scen] = med
        pfail_lo[scen]  = lo
        pfail_hi[scen]  = hi

        # N at Pfail_median = 10% (linear interp)
        if med[-1] >= P_THRESHOLD:
            j = int(np.argmax(med >= P_THRESHOLD))
            if j == 0:
                n_at_p10[scen] = float(Ngrid[0])
            else:
                x0, x1 = Ngrid[j - 1], Ngrid[j]
                y0, y1 = med[j - 1], med[j]
                n_at_p10[scen] = float(
                    x0 + (P_THRESHOLD - y0) / (y1 - y0) * (x1 - x0))
        else:
            n_at_p10[scen] = float("nan")

    # Print summary table
    print("\nN at Pfail_median = 10%:")
    print(f"  {'scenario':<8} {'N (cycles)':>12} {'PPI width @ N(10%)':>20}")
    for scen in SCEN_ORDER:
        n10 = n_at_p10[scen]
        if np.isfinite(n10):
            j = int(np.argmin(np.abs(Ngrid - n10)))
            ppi_w = pfail_hi[scen][j] - pfail_lo[scen][j]
            print(f"  {scen:<8} {n10:>12.1f} {ppi_w:>20.4f}")
        else:
            print(f"  {scen:<8} {'>600':>12s} {'-':>20s}")

    # PPI mean half-width over the 0..600 cycle range
    print("\nMean PPI half-width over N=1..600  (||hi-lo||/2):")
    for scen in SCEN_ORDER:
        hw = float(np.mean((pfail_hi[scen] - pfail_lo[scen]) / 2.0))
        print(f"  {scen:<8} {hw:.4f}")

    # ----------------------- Plot: 2x3 subplots -----------------------
    fig, axes = plt.subplots(
        2, 3,
        figsize=(WIDTH_MM * MM, HEIGHT_MM * MM),
        sharex=True, sharey=True,
        gridspec_kw=dict(hspace=0.20, wspace=0.10),
    )

    for ax, scen in zip(axes.flat, SCEN_ORDER):
        col = SCEN_COLORS[scen]
        ax.fill_between(Ngrid, pfail_lo[scen], pfail_hi[scen],
                        color=col, alpha=0.22, lw=0,
                        label="95% PPI", zorder=1)
        ax.plot(Ngrid, pfail_med[scen],
                color=col, lw=1.4, zorder=3,
                label="median")
        ax.axhline(P_THRESHOLD, ls=(0, (4, 2)), lw=0.6,
                   color="0.4", zorder=2)
        ax.text(0.99, P_THRESHOLD + 0.012,
                f"$P_{{fail}}={P_THRESHOLD:.2f}$",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom",
                fontsize=6.5, color="0.30")

        n10 = n_at_p10[scen]
        if np.isfinite(n10):
            ax.plot([n10], [P_THRESHOLD], marker="o", ms=5.0,
                    mfc=col, mec="black", mew=0.6, zorder=5)
            # place label on the side with more room (left if N_10 is in the
            # right half of the panel, right otherwise) so it stays inside.
            if n10 > 380:
                dx, ha = -8, "right"
            else:
                dx, ha = 8, "left"
            ax.annotate(rf"$N_{{10}}={n10:.0f}$",
                        xy=(n10, P_THRESHOLD),
                        xytext=(dx, -10), textcoords="offset points",
                        fontsize=7.0, fontweight="bold", color=col,
                        ha=ha, va="top", zorder=6)
        else:
            ax.text(0.97, 0.92, r"$N_{10} > 600$",
                    transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.0,
                    fontweight="bold", color=col)

        ax.set_title(scen, color=col, fontweight="bold",
                     loc="left", pad=2)
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    # Shared axis labels via figure-level
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Thermal cycles, $N$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$P_{\mathrm{fail}}(N)$")

    # Single legend at the top right of the first panel
    handles = [
        plt.Line2D([0], [0], color="black", lw=1.4, label="median"),
        mpl.patches.Patch(facecolor="0.5", alpha=0.30, label="95% PPI"),
    ]
    leg = axes[0, 0].legend(handles=handles, loc="upper left",
                            handlelength=1.6, handletextpad=0.5,
                            borderpad=0.4, labelspacing=0.25)
    leg.get_frame().set_linewidth(0.6)

    out_tiff = FIG_DIR / "Figure4_FailureProbability.tiff"
    out_pdf  = FIG_DIR / "Figure4_FailureProbability.pdf"
    out_png  = FIG_DIR / "Figure4_FailureProbability.png"

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
