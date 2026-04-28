"""
common.py - Sections 0-5 of SOFC_NSR_v3_REVISION.ipynb
Import-safe module containing setup, config, equations, DGP, model factories,
and posterior helpers. No script execution at import time.
"""

# =============================================================================
# Section 0 -- Environment setup
# =============================================================================
import importlib, subprocess, sys

_REQUIRED = [("pymc", "5.16"), ("arviz", "0.18"),
             ("SALib", "1.5"), ("netCDF4", None)]
for pkg, min_ver in _REQUIRED:
    try:
        importlib.import_module(pkg)
    except ImportError:
        spec = f"{pkg}>={min_ver}" if min_ver else pkg
        print(f"Installing {spec} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "-q", spec])

import os, sys, time, json, shutil, platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
import scipy.stats as sps
from scipy.stats import weibull_min
import matplotlib as mpl
import matplotlib.pyplot as plt

import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze

mpl.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "figure.dpi": 110,
})


# =============================================================================
# Section 1 -- Configuration
# =============================================================================
OUTPUT_DIR = Path("./sofc_v3_output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEED = 2024
REUSE_CACHED_MCMC = True   # set False to force fresh sampling

# Data-generating process.  Step 5 tries to recover these.
P = dict(
    ASR0_mean=0.18,   ASR0_std=0.02,
    K_CYC_BASE=5e-5,  K_TIME_BASE=1e-4,
    ALPHA_CYC=0.72,   ALPHA_TIME=0.68,
    ASR_THRESHOLD=0.40,
    SIGMA_OBS=0.008,  NU_T=6.0,
    GAMMA_FCYC=0.45,  GAMMA_THEAT=0.003,
    TAU_CYC=0.15,     TAU_TIME=0.12,
    BETA_W=2.8,
    LAMBDA_CRACK_BASE=5000,
    DELTA_FCYC=-0.60, DELTA_THEAT=-0.008,
    N_CELLS=15, N_MAX=600, OBS_INT=10,
)

# Six operating scenarios.
SCENARIOS = {
    "S1a": {"name": "Coastal-fast",   "f_cyc": 27, "tau_op":   9,
            "T_heat":  30, "color": "#E74C3C", "ls": "-"},
    "S1b": {"name": "Coastal-slow",   "f_cyc": 20, "tau_op":  15,
            "T_heat":  45, "color": "#C0392B", "ls": "--"},
    "S2a": {"name": "Regional-fast",  "f_cyc": 10, "tau_op":  36,
            "T_heat":  60, "color": "#F39C12", "ls": "-"},
    "S2b": {"name": "Regional-slow",  "f_cyc":  6, "tau_op":  72,
            "T_heat":  75, "color": "#D35400", "ls": "--"},
    "S3a": {"name": "Long-haul-fast", "f_cyc":  3, "tau_op": 120,
            "T_heat": 120, "color": "#2980B9", "ls": "-"},
    "S3b": {"name": "Long-haul-slow", "f_cyc":  1, "tau_op": 240,
            "T_heat": 180, "color": "#1F618D", "ls": "--"},
}
S_KEYS = list(SCENARIOS.keys())
N_SCEN = len(SCENARIOS)

# MCMC budgets.
ASR_DRAWS,  ASR_TUNE  = 1500, 4000
SURV_DRAWS, SURV_TUNE = 2000, 6000

# Reduced budget for the four noise-sensitivity refits in Step 9.
SENS_DRAWS, SENS_TUNE, SENS_CHAINS = 500, 2000, 2

# Replicates per grid cell in Step 8 (data-generation level; no MCMC).
BOUND_REPLICATES = 200

# cores = 1 on Windows (multiprocessing quirk), else up to 4.
if platform.system() == "Windows":
    N_CORES = 1
else:
    N_CORES = min(4, os.cpu_count() or 1)

MCMC_COMMON = dict(
    chains=4, cores=N_CORES,
    target_accept=0.97, max_treedepth=15,
    progressbar=True, return_inferencedata=True,
)


# =============================================================================
# Section 1b -- Logging helpers
# =============================================================================
_LOG_PATH = OUTPUT_DIR / "execution_log_v3.txt"
_LOG_FH = open(_LOG_PATH, "a", encoding="utf-8")


def log(msg=""):
    line = str(msg)
    print(line)
    _LOG_FH.write(line + "\n")
    _LOG_FH.flush()


def section(title):
    log("")
    log("=" * 76)
    log("  " + title)
    log("=" * 76)


# =============================================================================
# Section 2 -- Physical model equations
# =============================================================================
def k_cyc_mean(f_cyc, T_heat, params=None):
    """Scenario-level mean of the cycle-driven degradation rate.

        log k_cyc = log K_CYC_BASE + gamma_f * log(f_cyc)
                    + gamma_T * T_heat
    """
    p = params if params is not None else P
    return np.exp(np.log(p["K_CYC_BASE"])
                  + p["GAMMA_FCYC"]  * np.log(f_cyc)
                  + p["GAMMA_THEAT"] * T_heat)


def lam_crack(f_cyc, T_heat, params=None):
    """Weibull scale parameter (cycles) for time-to-crack."""
    p = params if params is not None else P
    return np.exp(np.log(p["LAMBDA_CRACK_BASE"])
                  + p["DELTA_FCYC"]  * np.log(f_cyc)
                  + p["DELTA_THEAT"] * T_heat)


def asr_trajectory(N, tau_op, ASR0, k_cyc, k_time,
                   alpha_cyc=None, alpha_time=None, params=None):
    """Dual-degradation ASR curve."""
    p = params if params is not None else P
    a_c = p["ALPHA_CYC"]  if alpha_cyc  is None else alpha_cyc
    a_t = p["ALPHA_TIME"] if alpha_time is None else alpha_time
    return ASR0 + k_cyc * N**a_c + k_time * (N * tau_op)**a_t


def first_asr_crossing(ASR0, k_cyc, k_time, tau_op,
                       threshold=None, N_max=5000, resolution=30000,
                       params=None):
    """First cycle count N at which ASR(N) >= threshold."""
    p = params if params is not None else P
    thr = p["ASR_THRESHOLD"] if threshold is None else threshold
    Nv = np.linspace(1.0, N_max, resolution)
    curve = asr_trajectory(Nv, tau_op, ASR0, k_cyc, k_time, params=p)
    hits = np.where(curve >= thr)[0]
    return float(Nv[hits[0]]) if hits.size else float(N_max)


# =============================================================================
# Section 3 -- Synthetic data generator
# =============================================================================
def generate_synthetic_data(param_override=None, seed=BASE_SEED):
    """Return (obs_df, fail_df, gt_df) DataFrames."""
    p = {**P, **(param_override or {})}
    rng = np.random.default_rng(seed)

    obs_rows, fail_rows, gt_rows = [], [], []

    for si, (sk, sc) in enumerate(SCENARIOS.items()):
        kcm = k_cyc_mean(sc["f_cyc"], sc["T_heat"], params=p)
        lc  = lam_crack(sc["f_cyc"], sc["T_heat"], params=p)

        for ci in range(p["N_CELLS"]):
            cid     = f"{sk}_cell{ci:02d}"
            cid_int = ci + si * p["N_CELLS"]

            A0   = max(float(rng.normal(p["ASR0_mean"], p["ASR0_std"])),
                       0.10)
            kc_i = float(np.exp(np.log(kcm)
                                + rng.normal(0, p["TAU_CYC"])))
            kt_i = float(np.exp(np.log(p["K_TIME_BASE"])
                                + rng.normal(0, p["TAU_TIME"])))

            T_crack = float(weibull_min.rvs(
                c=p["BETA_W"], scale=lc,
                random_state=int(rng.integers(1 << 31))))
            T_asr = first_asr_crossing(A0, kc_i, kt_i, sc["tau_op"],
                                       params=p)

            T_fail   = min(T_asr, T_crack)
            cause    = "ASR" if T_asr <= T_crack else "Crack"
            censored = T_fail > p["N_MAX"]
            N_end    = min(T_fail, p["N_MAX"])

            Nobs = np.arange(p["OBS_INT"], N_end + p["OBS_INT"],
                             p["OBS_INT"])
            Nobs = Nobs[Nobs <= N_end]
            if Nobs.size == 0:
                Nobs = np.array([p["OBS_INT"]])

            for N in Nobs:
                asr_t = asr_trajectory(N, sc["tau_op"],
                                       A0, kc_i, kt_i, params=p)
                noise = p["SIGMA_OBS"] * rng.standard_t(df=p["NU_T"])
                obs_rows.append({
                    "scenario": sk, "name": sc["name"],
                    "cell_id": cid, "cell_id_int": cid_int,
                    "f_cyc": sc["f_cyc"], "tau_op": sc["tau_op"],
                    "T_heat": sc["T_heat"],
                    "N_cycle": int(N),
                    "t_hours": round(float(N) * sc["tau_op"], 2),
                    "ASR_true": round(float(asr_t), 6),
                    "ASR_obs":  round(max(float(asr_t + noise), 0.05), 6),
                })

            fail_rows.append({
                "scenario": sk, "name": sc["name"],
                "cell_id": cid, "cell_id_int": cid_int,
                "f_cyc": sc["f_cyc"], "tau_op": sc["tau_op"],
                "T_heat": sc["T_heat"],
                "T_fail_cycles":  round(T_fail,  2),
                "T_fail_hours":   round(T_fail * sc["tau_op"], 1),
                "T_asr_cycles":   round(T_asr,   2),
                "T_crack_cycles": round(T_crack, 2),
                "cause": cause, "censored": bool(censored),
                "delta_asr":   int((not censored) and cause == "ASR"),
                "delta_crack": int((not censored) and cause == "Crack"),
            })
            gt_rows.append({
                "scenario": sk, "cell_id": cid,
                "ASR0":   round(A0,   6),
                "k_cyc":  round(kc_i, 10),
                "k_time": round(kt_i, 10),
                "T_asr":   round(T_asr,   2),
                "T_crack": round(T_crack, 2),
                "T_fail":  round(T_fail,  2),
                "cause":   cause,
            })

    return (pd.DataFrame(obs_rows),
            pd.DataFrame(fail_rows),
            pd.DataFrame(gt_rows))


# =============================================================================
# Section 4 -- Bayesian model factories
# =============================================================================
def build_covariate_matrix(obs_df):
    """Standardise (log f_cyc, T_heat) across the 6 scenarios."""
    Z_raw = np.array([[np.log(sc["f_cyc"]), sc["T_heat"]]
                      for sc in SCENARIOS.values()])
    Z_mean = Z_raw.mean(axis=0)
    Z_std  = Z_raw.std(axis=0)
    Z_std[Z_std == 0] = 1.0
    Z_norm = (Z_raw - Z_mean) / Z_std

    cell_meta = (obs_df.drop_duplicates("cell_id_int")
                       [["cell_id", "cell_id_int", "scenario",
                         "f_cyc", "T_heat"]]
                 .sort_values("cell_id_int")
                 .reset_index(drop=True))
    Z_cell_norm = np.array([Z_norm[S_KEYS.index(s)]
                            for s in cell_meta["scenario"]])
    return Z_raw, Z_mean, Z_std, Z_norm, Z_cell_norm, cell_meta


def fit_asr_model(obs_df, Z_cell_norm, n_cells,
                  draws, tune, chains, cores,
                  target_accept=0.97, max_treedepth=15,
                  seed=BASE_SEED):
    """Hierarchical dual-degradation ASR model. Returns InferenceData."""
    cell_idx = obs_df["cell_id_int"].values.astype(int)
    N_arr    = obs_df["N_cycle"].values.astype(float)
    t_arr    = N_arr * obs_df["tau_op"].values.astype(float)
    y_arr    = obs_df["ASR_obs"].values.astype(float)

    with pm.Model():
        mu_ASR0 = pm.Normal("mu_ASR0", mu=-1.7, sigma=0.3)
        mu_cyc  = pm.Normal("mu_cyc",  mu=-9.0, sigma=1.0)
        mu_time = pm.Normal("mu_time", mu=-9.0, sigma=1.0)

        gamma_cyc  = pm.Normal("gamma_cyc",  0, 0.5, shape=2)
        gamma_time = pm.Normal("gamma_time", 0, 0.5, shape=2)

        tau_ASR0 = pm.HalfNormal("tau_ASR0", sigma=0.3)
        tau_cyc  = pm.HalfNormal("tau_cyc",  sigma=0.3)
        tau_time = pm.HalfNormal("tau_time", sigma=0.3)

        alpha_cyc  = pm.TruncatedNormal("alpha_cyc",
                                        mu=0.72, sigma=0.15,
                                        lower=0, upper=1.5)
        alpha_time = pm.TruncatedNormal("alpha_time",
                                        mu=0.68, sigma=0.15,
                                        lower=0, upper=1.5)
        nu      = pm.Gamma("nu", alpha=2, beta=0.1)
        sigma_e = pm.HalfNormal("sigma_e", sigma=0.02)

        z_ASR0 = pm.Normal("z_ASR0", 0, 1, shape=n_cells)
        z_cyc  = pm.Normal("z_cyc",  0, 1, shape=n_cells)
        z_time = pm.Normal("z_time", 0, 1, shape=n_cells)

        Zt = pt.as_tensor_variable(Z_cell_norm)
        logA = mu_ASR0 + tau_ASR0 * z_ASR0
        logc = mu_cyc  + pt.dot(Zt, gamma_cyc)  + tau_cyc  * z_cyc
        logt = mu_time + pt.dot(Zt, gamma_time) + tau_time * z_time

        ASR0_i   = pm.Deterministic("ASR0_i",   pt.exp(logA))
        k_cyc_i  = pm.Deterministic("k_cyc_i",  pt.exp(logc))
        k_time_i = pm.Deterministic("k_time_i", pt.exp(logt))

        mu_y = (ASR0_i[cell_idx]
                + k_cyc_i[cell_idx]  * N_arr**alpha_cyc
                + k_time_i[cell_idx] * t_arr**alpha_time)

        pm.StudentT("Y_obs", nu=nu, mu=mu_y, sigma=sigma_e,
                    observed=y_arr)

        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores,
            target_accept=target_accept, max_treedepth=max_treedepth,
            random_seed=[seed + i for i in range(chains)],
            progressbar=True, return_inferencedata=True,
        )
    return idata


def fit_survival_model(fail_df, Z_cell_norm,
                       draws, tune, chains, cores,
                       target_accept=0.97, max_treedepth=15,
                       seed=BASE_SEED + 100):
    """Weibull time-to-crack with log-linear covariates."""
    T_fail  = fail_df["T_fail_cycles"].values.astype(float)
    delta_c = fail_df["delta_crack"].values.astype(int)
    BETA    = P["BETA_W"]

    with pm.Model():
        lambda0  = pm.Normal("lambda0",  mu=7.0, sigma=1.5)
        delta_fc = pm.Normal("delta_fc", mu=0,   sigma=0.5)
        delta_th = pm.Normal("delta_th", mu=0,   sigma=0.1)

        log_lam = (lambda0
                   + delta_fc * Z_cell_norm[:, 0]
                   + delta_th * Z_cell_norm[:, 1])
        lam_s = pm.Deterministic("lam_s", pt.exp(log_lam))

        log_S = -(T_fail / lam_s) ** BETA
        log_f = (pt.log(BETA / lam_s)
                 + (BETA - 1) * pt.log(T_fail / lam_s + 1e-10)
                 + log_S)
        loglik = pt.switch(pt.eq(delta_c, 1), log_f, log_S)
        pm.Potential("surv_like", loglik.sum())

        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores,
            target_accept=target_accept, max_treedepth=max_treedepth,
            random_seed=seed,
            progressbar=True, return_inferencedata=True,
        )
    return idata


# =============================================================================
# Section 5 -- Posterior-based helpers
# =============================================================================
def classify_cells_from_posterior(idata_asr, idata_surv,
                                   cell_meta, Z_cell_norm,
                                   n_samples=2000, seed=77,
                                   N_max_inf=5000.0,
                                   grid_resolution=500):
    """Per-cell posterior P(failure mode = ASR)."""
    post   = az.extract(idata_asr)
    post_s = az.extract(idata_surv)
    rng = np.random.default_rng(seed)

    mu_A = post["mu_ASR0"].values.flatten()
    mu_c = post["mu_cyc"].values.flatten()
    mu_t = post["mu_time"].values.flatten()
    gc   = post["gamma_cyc"].values.reshape(-1, 2)
    gt   = post["gamma_time"].values.reshape(-1, 2)
    tA   = post["tau_ASR0"].values.flatten()
    tc   = post["tau_cyc"].values.flatten()
    tt   = post["tau_time"].values.flatten()
    ac   = post["alpha_cyc"].values.flatten()
    at   = post["alpha_time"].values.flatten()

    lam0 = post_s["lambda0"].values.flatten()
    dfc  = post_s["delta_fc"].values.flatten()
    dth  = post_s["delta_th"].values.flatten()

    p_asr = np.zeros(len(cell_meta))
    t_med = np.zeros(len(cell_meta))
    Ngrid = np.linspace(1.0, N_max_inf, grid_resolution)

    for i, row in cell_meta.iterrows():
        z  = Z_cell_norm[i]
        sc = SCENARIOS[row["scenario"]]
        tau_op = sc["tau_op"]

        n_A  = min(n_samples, len(mu_A))
        n_S  = min(n_samples, len(lam0))
        idxA = rng.integers(0, len(mu_A), n_A)
        idxS = rng.integers(0, len(lam0), n_S)

        zA = rng.standard_normal(n_A)
        zc = rng.standard_normal(n_A)
        zT = rng.standard_normal(n_A)

        A0   = np.exp(mu_A[idxA] + tA[idxA] * zA)
        kcyc = np.exp(mu_c[idxA] + gc[idxA] @ z + tc[idxA] * zc)
        ktim = np.exp(mu_t[idxA] + gt[idxA] @ z + tt[idxA] * zT)

        curves = (A0[:, None]
                  + kcyc[:, None] * Ngrid[None, :]**ac[idxA, None]
                  + ktim[:, None] * (Ngrid[None, :] * tau_op)**at[idxA, None])
        hit = curves >= P["ASR_THRESHOLD"]
        first_hit = np.where(hit.any(axis=1),
                             hit.argmax(axis=1),
                             Ngrid.size - 1)
        T_asr_s = Ngrid[first_hit]

        lam_sc = np.exp(lam0[idxS] + dfc[idxS] * z[0]
                        + dth[idxS] * z[1])
        u = rng.uniform(size=n_S)
        T_crack_s = lam_sc * (-np.log(1 - u))**(1 / P["BETA_W"])

        n = min(n_A, n_S)
        p_asr[i] = float(np.mean(T_asr_s[:n] < T_crack_s[:n]))
        t_med[i] = float(np.median(np.minimum(T_asr_s[:n],
                                              T_crack_s[:n])))

    out = cell_meta.copy()
    out["p_asr_posterior"]    = p_asr
    out["pred_mode"]          = np.where(p_asr > 0.5, "ASR", "Crack")
    out["pred_T_fail_median"] = t_med
    return out


def compute_maintenance_schedule(idata_asr, idata_surv, fail_df,
                                 Z_norm, n_mc=5000, seed=55):
    """Per-scenario cycle at which system P_fail first exceeds 10%."""
    post   = az.extract(idata_asr)
    post_s = az.extract(idata_surv)
    rng = np.random.default_rng(seed)

    results = []
    N_check = list(range(10, P["N_MAX"] + 1, 10))

    for si, (sk, sc) in enumerate(SCENARIOS.items()):
        z = Z_norm[si]
        tau_op = sc["tau_op"]

        idxA = rng.integers(0, post["mu_ASR0"].size, n_mc)
        idxS = rng.integers(0, post_s["lambda0"].size, n_mc)

        A0 = np.exp(post["mu_ASR0"].values.flatten()[idxA])
        kc = np.exp(post["mu_cyc"].values.flatten()[idxA]
                    + post["gamma_cyc"].values.reshape(-1, 2)[idxA] @ z)
        kt = np.exp(post["mu_time"].values.flatten()[idxA]
                    + post["gamma_time"].values.reshape(-1, 2)[idxA] @ z)
        a_c = post["alpha_cyc"].values.flatten()[idxA]
        a_t = post["alpha_time"].values.flatten()[idxA]

        lam = np.exp(post_s["lambda0"].values.flatten()[idxS]
                     + post_s["delta_fc"].values.flatten()[idxS] * z[0]
                     + post_s["delta_th"].values.flatten()[idxS] * z[1])

        N_maint, pfail_at = None, None
        pfail_curve = []
        for N_t in N_check:
            asr_pred = A0 + kc * N_t**a_c + kt * (N_t * tau_op)**a_t
            p_asr   = float((asr_pred >= P["ASR_THRESHOLD"]).mean())
            p_crack = float((1 - np.exp(-(N_t / lam)**P["BETA_W"])).mean())
            p_fail  = 1 - (1 - p_asr) * (1 - p_crack)
            pfail_curve.append(p_fail)
            if N_maint is None and p_fail > 0.10:
                N_maint, pfail_at = N_t, p_fail

        if N_maint is None:
            N_maint, pfail_at = P["N_MAX"], pfail_curve[-1]

        fd = fail_df[fail_df["scenario"] == sk]
        n_asr   = int(fd["delta_asr"].sum())
        n_crack = int(fd["delta_crack"].sum())
        n_cens  = int(fd["censored"].sum())
        dom = ("ASR"   if n_asr  > n_crack else
               "Crack" if n_crack > n_asr  else "Balanced")

        results.append({
            "scenario": sk, "name": sc["name"],
            "f_cyc": sc["f_cyc"], "tau_op": sc["tau_op"],
            "T_heat": sc["T_heat"],
            "N_maint": int(N_maint),
            "months":  round(N_maint / sc["f_cyc"], 1),
            "pfail_at_maint": round(pfail_at, 4),
            "dominant": dom,
            "lambda_cr_mean": round(float(lam.mean()), 1),
            "lambda_cr_lo":   round(float(np.percentile(lam, 2.5)), 1),
            "lambda_cr_hi":   round(float(np.percentile(lam, 97.5)), 1),
            "n_asr": n_asr, "n_crack": n_crack, "n_censored": n_cens,
            "pfail_curve": pfail_curve,
            "N_check": N_check,
        })
    return results
