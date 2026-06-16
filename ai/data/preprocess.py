"""
Preprocessing — Resample to RESAMPLE_HZ (100Hz), z-score normalize using pre-ISC segment.
"""
import numpy as np
from typing import List
from ai.data.loader import SimRun
from ai.config import RESAMPLE_HZ, T_ONSET


def resample_run(run: SimRun) -> SimRun:
    """Resample a run to uniform RESAMPLE_HZ."""
    t_end = run.t[-1]
    dt = 1.0 / RESAMPLE_HZ
    t_new = np.arange(0, t_end, dt)

    V_new = np.interp(t_new, run.t, run.V)
    T_new = np.interp(t_new, run.t, run.T)

    return SimRun(
        run_id=run.run_id,
        scenario=run.scenario,
        scenario_name=run.scenario_name,
        k_R=run.k_R,
        k_R_mOhm=run.k_R_mOhm,
        tau=run.tau,
        t=t_new, V=V_new, T=T_new
    )


def normalize_run(run: SimRun) -> SimRun:
    """Z-score normalize using pre-ISC segment (t < T_ONSET) as reference."""
    pre_mask = run.t < T_ONSET

    if pre_mask.sum() < 10:
        raise ValueError(f"Run {run.run_id}: too few pre-ISC samples ({pre_mask.sum()})")

    V_mean = run.V[pre_mask].mean()
    V_std = run.V[pre_mask].std()
    T_mean = run.T[pre_mask].mean()
    T_std = run.T[pre_mask].std()

    # Avoid division by zero
    V_std = max(V_std, 1e-8)
    T_std = max(T_std, 1e-8)

    V_norm = (run.V - V_mean) / V_std
    T_norm = (run.T - T_mean) / T_std

    return SimRun(
        run_id=run.run_id,
        scenario=run.scenario,
        scenario_name=run.scenario_name,
        k_R=run.k_R,
        k_R_mOhm=run.k_R_mOhm,
        tau=run.tau,
        t=run.t, V=V_norm, T=T_norm
    )


def preprocess_all(runs: List[SimRun]) -> List[SimRun]:
    """Resample and normalize all runs."""
    processed = []
    for run in runs:
        r = resample_run(run)
        r = normalize_run(r)
        processed.append(r)
    print(f"Preprocessed {len(processed)} runs (resampled to {RESAMPLE_HZ}Hz, z-score normalized)")
    return processed
