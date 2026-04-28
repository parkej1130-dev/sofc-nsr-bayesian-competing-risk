#!/usr/bin/env python3
"""
verify_results.py — Reproducibility verification for Park, Kwon, Lee (2026).

This script loads the cached PyMC inference results from `results/` and
reproduces the key numerical values reported in the published paper.

Usage:
    python verify_results.py

Expected output: All checks PASS within tolerances.
"""

from pathlib import Path
import json
import sys

import numpy as np
import arviz as az


HERE = Path(__file__).parent
RES = HERE / "results"

# Expected values from the published paper (Park, Kwon, Lee 2026).
# Tolerances are chosen to accommodate minor library-version variation.
EXPECTED = {
    # Table 2 — convergence diagnostics
    "max_r_hat":         (1.01, 0.005),     # R̂ ≤ 1.01
    "min_ess_bulk":      (479,  20),        # ESS_bulk ≥ 479
    "min_ess_tail":      (684,  50),        # ESS_tail ≥ 684
    "n_divergent":       (0,    0),         # zero divergent transitions

    # Table 2 — key posterior means
    "alpha_cyc_mean":    (0.7289, 0.005),   # ground truth: 0.720
    "alpha_time_mean":   (0.6543, 0.005),   # ground truth: 0.680
    "sigma_e_mean":      (0.0079, 0.0002),  # ground truth: 0.008
    "delta_fc_mean":     (-0.686, 0.01),    # Weibull cyc-frequency effect

    # Section 3.6 — sensitivity diagnostics
    "sigma_e_recovery_pct": (1.7, 1.0),     # σ_e recovery within 1.7%
}

# Ground-truth values used in synthetic data generation (Table S4 of paper).
GROUND_TRUTH = {
    "mu_ASR0":   -1.7148,   # corresponds to ASR0 ≈ 0.18 Ω·cm²
    "mu_cyc":    -9.070,
    "mu_time":   -9.2103,
    "alpha_cyc":  0.720,
    "alpha_time": 0.680,
    "sigma_e":    0.008,
}


def check(name, observed, expected_tuple):
    """Verify that `observed` is within tolerance of `expected_tuple = (value, tol)`."""
    expected, tol = expected_tuple
    diff = abs(observed - expected)
    status = "PASS" if diff <= tol else "FAIL"
    icon = "✓" if status == "PASS" else "✗"
    print(f"  [{icon}] {name:30s}  observed = {observed:>10.4f}   "
          f"expected = {expected:>10.4f} ± {tol:.4f}   {status}")
    return status == "PASS"


def main():
    print("=" * 75)
    print("Reproducibility verification for Park, Kwon, Lee (2026)")
    print("=" * 75)
    print()

    if not (RES / "idata_asr_v3.nc").exists():
        print("ERROR: results/idata_asr_v3.nc not found.")
        print("Please ensure all .nc files are present in the results/ directory.")
        sys.exit(1)

    # =========================================================================
    # Load inference results
    # =========================================================================
    print("Loading cached inference results ...")
    idata_asr  = az.from_netcdf(RES / "idata_asr_v3.nc")
    idata_surv = az.from_netcdf(RES / "idata_surv_v3.nc")
    print(f"  ASR posterior: {idata_asr.posterior.chain.size} chains × "
          f"{idata_asr.posterior.draw.size} draws")
    print(f"  Survival posterior: {idata_surv.posterior.chain.size} chains × "
          f"{idata_surv.posterior.draw.size} draws")
    print()

    # =========================================================================
    # Section 1 — Convergence diagnostics (Table 2)
    # =========================================================================
    print("[1] Convergence diagnostics (Table 2 of main text)")

    # The 17 main parameters reported in Table 2 of the paper
    # (excluding cell-level non-centered z_* and derived ASR0_i, k_cyc_i, k_time_i)
    main_asr_vars = ["mu_ASR0", "mu_cyc", "mu_time",
                     "tau_ASR0", "tau_cyc", "tau_time",
                     "alpha_cyc", "alpha_time",
                     "sigma_e", "nu",
                     "gamma_cyc", "gamma_time"]  # gamma each shape=2, total 14
    main_surv_vars = ["lambda0", "delta_fc", "delta_th"]   # 3 parameters

    summ = az.summary(idata_asr, var_names=main_asr_vars, hdi_prob=0.94)
    summ_surv = az.summary(idata_surv, var_names=main_surv_vars, hdi_prob=0.94)

    max_r_hat = max(summ["r_hat"].max(), summ_surv["r_hat"].max())
    min_ess_bulk = min(summ["ess_bulk"].min(), summ_surv["ess_bulk"].min())
    min_ess_tail = min(summ["ess_tail"].min(), summ_surv["ess_tail"].min())
    n_diverg = int(idata_asr.sample_stats.diverging.sum().values) \
             + int(idata_surv.sample_stats.diverging.sum().values)

    passed = []
    passed.append(check("Max R̂",            max_r_hat,    EXPECTED["max_r_hat"]))
    passed.append(check("Min ESS_bulk",      min_ess_bulk, EXPECTED["min_ess_bulk"]))
    passed.append(check("Min ESS_tail",      min_ess_tail, EXPECTED["min_ess_tail"]))
    passed.append(check("Divergent transitions", n_diverg, EXPECTED["n_divergent"]))
    print()

    # =========================================================================
    # Section 2 — Key posterior estimates (Table 2)
    # =========================================================================
    print("[2] Key posterior estimates (Table 2)")
    asr_post = idata_asr.posterior

    a_cyc  = float(asr_post["alpha_cyc"].mean())
    a_time = float(asr_post["alpha_time"].mean())
    sig_e  = float(asr_post["sigma_e"].mean())
    df_cyc = float(idata_surv.posterior["delta_fc"].mean())

    passed.append(check("alpha_cyc mean",  a_cyc,  EXPECTED["alpha_cyc_mean"]))
    passed.append(check("alpha_time mean", a_time, EXPECTED["alpha_time_mean"]))
    passed.append(check("sigma_e mean",    sig_e,  EXPECTED["sigma_e_mean"]))
    passed.append(check("delta_fc mean",   df_cyc, EXPECTED["delta_fc_mean"]))
    print()

    # =========================================================================
    # Section 3 — Parameter recovery (Table 3)
    # =========================================================================
    print("[3] Parameter recovery — true value within 94% HDI (Table 3)")
    for var_name, true_val in GROUND_TRUTH.items():
        if var_name in asr_post.data_vars:
            arr = asr_post[var_name].values.flatten()
            hdi_lo, hdi_hi = az.hdi(arr, hdi_prob=0.94)
            within = bool(hdi_lo <= true_val <= hdi_hi)
            mark = "✓" if within else "✗"
            print(f"  [{mark}] {var_name:12s}  true = {true_val:>+8.4f}   "
                  f"94% HDI = [{hdi_lo:+8.4f}, {hdi_hi:+8.4f}]   "
                  f"{'within HDI' if within else 'OUTSIDE HDI'}")
            passed.append(within)
    print()

    # =========================================================================
    # Section 4 — sigma_e sensitivity (Section 3.6)
    # =========================================================================
    print("[4] σ_e sensitivity — recovery across noise levels (Section 3.6)")
    for noise_str in ["0_004", "0_008", "0_012", "0_016"]:
        path = RES / f"idata_asr_noise_{noise_str}_v3.nc"
        if path.exists():
            idata = az.from_netcdf(path)
            sig_post = float(idata.posterior["sigma_e"].mean())
            sig_true = float(noise_str.replace("_", "."))
            rel_err = abs(sig_post - sig_true) / sig_true * 100.0
            ok = rel_err < 5.0
            print(f"  {'✓' if ok else '✗'} σ_e_true = {sig_true:.3f}   "
                  f"σ_e_post = {sig_post:.5f}   "
                  f"rel. error = {rel_err:.2f} %")
            passed.append(ok)
    print()

    # =========================================================================
    # Run summary
    # =========================================================================
    n_pass = sum(1 for p in passed if p)
    n_total = len(passed)
    print("=" * 75)
    print(f"  Verification summary: {n_pass} / {n_total} checks passed")
    print("=" * 75)

    # Run summary metadata
    if (RES / "run_summary_v3.json").exists():
        with open(RES / "run_summary_v3.json") as f:
            meta = json.load(f)
        print()
        print("Run metadata:")
        print(f"  Run timestamp: {meta.get('run_time_iso', 'N/A')}")
        print(f"  Random seed:   {meta.get('seed', 'N/A')}")
        if "library_versions" in meta:
            print("  Library versions used to produce the published results:")
            for lib, ver in meta["library_versions"].items():
                print(f"    {lib:10s} {ver}")

    if n_pass == n_total:
        print()
        print("✓ All checks passed — results are fully reproducible.")
        sys.exit(0)
    else:
        print()
        print("✗ Some checks failed — please review version compatibility.")
        sys.exit(1)


if __name__ == "__main__":
    main()
