"""
EKF Detection Delay Evaluation — Run EKF on continuous signal,
detect ISC via SOC inconsistency (ΔSOC = SOC_CC - SOC_EKF).
Returns results in same dict format as AI evaluate_detection_delay().
"""
import numpy as np
from typing import List
from ai.config import (
    T_ONSET, DEADLINE_FACTOR, SCENARIO_NAMES,
    RESAMPLE_HZ, EKF_WINDOW_SEC, EKF_SIGMA_MULT,
    R_INIT, R_FINAL, R_LABEL_THRESHOLD,
)
from ai.models.ekf import BatteryEKF
from ai.data.loader import SimRun


def calibrate_ekf(calib_runs: List[SimRun], dt: float) -> dict:
    """
    Calibrate EKF from pre-ISC segments of given runs.
    Uses the SAME vibration environment as the target evaluation.

    Time structure (per run):
        [0, WARMUP_SEC):           EKF initial-transient warmup (data discarded)
        [WARMUP_SEC, CALIB_END):   calibration window (sigma estimation)
        [CALIB_END, T_ONSET):      steady-state buffer (data not used)
        [T_ONSET, T_end]:          ISC onset and detection (handled by evaluate_*)

    The buffer between calibration end and ISC onset addresses the
    timing concern that calibration data should not be adjacent to the
    fault, allowing residual EKF transients to fully decay.

    Args:
        calib_runs: List of SimRun objects for calibration (resampled).
                    Should match the target vibration scenario for fair comparison.
        dt: Sampling interval (1/RESAMPLE_HZ).

    Returns:
        dict with sigma_residual, mean/sigma_delta_soc, R_calibrated.
    """
    WARMUP_SEC = 30.0     # Skip first 30s for EKF convergence
    CALIB_END_SEC = 100.0 # Calibration window ends at 100s; 100-T_ONSET acts as steady-state buffer
    all_residuals = []
    all_dsoc = []
    win_samples = int(EKF_WINDOW_SEC * RESAMPLE_HZ)

    for run in calib_runs:
        ekf = BatteryEKF(dt=dt, R_scalar=1e-6)  # Initial small R
        pre_mask = run.t < CALIB_END_SEC
        V_pre = run.V[pre_mask]
        t_pre = run.t[pre_mask]

        result = ekf.run(V_pre)
        # Skip warmup for convergence
        warmup_samples = int(WARMUP_SEC * RESAMPLE_HZ)
        all_residuals.append(result["residuals"][warmup_samples:])

        # Windowed ΔSOC means (same windowing as detection)
        dsoc = result["delta_soc"]
        n_win = len(dsoc) // win_samples
        for i in range(n_win):
            s, e = i * win_samples, (i + 1) * win_samples
            t_end = t_pre[min(e - 1, len(t_pre) - 1)]
            if t_end > WARMUP_SEC:  # skip warmup
                all_dsoc.append(np.mean(dsoc[s:e]))

    sigma_res = np.std(np.concatenate(all_residuals))
    mean_dsoc = np.mean(all_dsoc)
    sigma_dsoc = np.std(all_dsoc)

    scenarios = set(r.scenario_name for r in calib_runs)
    print(f"  EKF Calibration: {len(calib_runs)} runs ({', '.join(scenarios)})")
    print(f"  Warmup: {WARMUP_SEC}s skipped")
    print(f"  Residual sigma = {sigma_res*1000:.4f} mV  (for R)")
    print(f"  DSOC baseline mean = {mean_dsoc*100:.6f} %SOC")
    print(f"  DSOC window sigma = {sigma_dsoc*100:.6f} %SOC")
    print(f"  3-sigma DSOC threshold = {sigma_dsoc*3*100:.6f} %SOC")

    return {
        "sigma_residual": sigma_res,
        "mean_delta_soc": mean_dsoc,
        "sigma_delta_soc": sigma_dsoc,
        "R_calibrated": sigma_res ** 2,
    }


def evaluate_ekf_detection_delay(
    runs: List[SimRun],
    calib: dict,
    dt: float = None,
) -> List[dict]:
    """
    Run-level EKF detection delay via SOC inconsistency.

    For each run:
      1. Run EKF → SOC_EKF (voltage-based) and SOC_CC (current-based)
      2. Compute ΔSOC = SOC_CC - SOC_EKF per 10s window
      3. First window where |ΔSOC_mean| > 3σ (after T_ONSET) = detection
      4. Compute detection delay vs deadline

    Args:
        runs: List of SimRun objects (resampled, physical units, NOT normalized).
        calib: Calibration dict from calibrate_ekf_from_novib().
        dt: Sampling interval. Default: 1/RESAMPLE_HZ.

    Returns:
        List of per-run result dicts (same format as AI evaluation).
    """
    if dt is None:
        dt = 1.0 / RESAMPLE_HZ

    WARMUP_SEC = 30.0
    sigma_dsoc = calib["sigma_delta_soc"]
    mean_dsoc_baseline = calib["mean_delta_soc"]
    R_cal = calib["R_calibrated"]
    threshold = EKF_SIGMA_MULT * sigma_dsoc
    win_samples = int(EKF_WINDOW_SEC * RESAMPLE_HZ)

    results = []
    for run in runs:
        ekf = BatteryEKF(dt=dt, R_scalar=R_cal)
        ekf_result = ekf.run(run.V)
        delta_soc = ekf_result["delta_soc"]

        # Windowed mean of ΔSOC (non-overlapping 10s windows)
        N = len(delta_soc)
        n_windows = N // win_samples
        t_window_ends = []
        window_dsoc = []

        for i in range(n_windows):
            start = i * win_samples
            end = start + win_samples
            window_dsoc.append(np.mean(delta_soc[start:end]))
            t_window_ends.append(run.t[min(end - 1, N - 1)])

        t_window_ends = np.array(t_window_ends)
        window_dsoc = np.array(window_dsoc)

        # Label onset (R_ISC <= 50 Ohm)
        t_label_onset = _compute_label_onset(run.tau)

        # Subtract baseline bias and skip warmup windows
        window_dsoc_corrected = window_dsoc - mean_dsoc_baseline
        valid = t_window_ends > WARMUP_SEC  # skip warmup

        # Find first detection after T_ONSET (and after warmup)
        post_onset = (t_window_ends > T_ONSET) & valid
        post_dsoc = window_dsoc_corrected[post_onset]
        post_times = t_window_ends[post_onset]

        detect_idx = np.where(np.abs(post_dsoc) > threshold)[0]
        if len(detect_idx) > 0:
            t_detect = float(post_times[detect_idx[0]])
            delay_from_onset = t_detect - T_ONSET
        else:
            t_detect = None
            delay_from_onset = None

        # Deadline
        deadline = DEADLINE_FACTOR * run.tau
        t_deadline = T_ONSET + deadline

        if delay_from_onset is not None:
            passed = delay_from_onset <= deadline
            margin = deadline - delay_from_onset
            delay_from_label = (t_detect - t_label_onset) if t_label_onset else None
        else:
            passed = False
            margin = None
            delay_from_label = None

        # False alarms before ISC onset (after warmup)
        pre_onset = (t_window_ends <= T_ONSET) & valid
        pre_dsoc = window_dsoc_corrected[pre_onset]
        n_pre_fa = int(np.sum(np.abs(pre_dsoc) > threshold))

        results.append({
            "run_id": run.run_id,
            "scenario": run.scenario,
            "scenario_name": SCENARIO_NAMES.get(run.scenario, str(run.scenario)),
            "k_R_mOhm": run.k_R_mOhm,
            "tau": run.tau,
            "t_label_onset": t_label_onset,
            "t_detect": t_detect,
            "delay_from_onset": delay_from_onset,
            "delay_from_label": delay_from_label,
            "deadline": deadline,
            "t_deadline": t_deadline,
            "passed": passed,
            "margin": margin,
            "n_pre_fa": n_pre_fa,
        })

    _print_ekf_delay_report(results, sigma_dsoc, threshold)
    return results


def _compute_label_onset(tau):
    """Compute time when R_ISC <= R_LABEL_THRESHOLD (signal emergence)."""
    # R_ISC(t) = R_FINAL + (R_INIT - R_FINAL) * exp(-(t - T_ONSET) / tau)
    # Solve for R_ISC = R_LABEL_THRESHOLD:
    ratio = (R_LABEL_THRESHOLD - R_FINAL) / (R_INIT - R_FINAL)
    if ratio <= 0:
        return T_ONSET
    dt_label = -tau * np.log(ratio)
    return T_ONSET + dt_label


def _print_ekf_delay_report(results, sigma_dsoc, threshold):
    """Print EKF detection delay report (mirrors AI report format)."""
    if not results:
        return

    tau = results[0]["tau"]
    deadline = results[0]["deadline"]
    t_deadline = results[0]["t_deadline"]

    print(f"\n  [EKF (DSOC)] Detection Delay Analysis")
    print(f"  {'=' * 80}")
    print(f"  tau={tau}s, deadline={deadline:.0f}s (t={t_deadline:.0f}s)")
    print(f"  DSOC sigma={sigma_dsoc*100:.6f}%SOC, "
          f"threshold(3-sigma)={threshold*100:.6f}%SOC")

    # Per-run table
    print(f"\n  {'Run':<24} {'k_R':>5} {'Scenario':<9} "
          f"{'t_detect':>8} {'Delay':>8} {'Reaction':>8} {'Margin':>8} {'Result':>6}")
    print(f"  {'-' * 80}")

    sorted_results = sorted(results, key=lambda r: (r["k_R_mOhm"], r["scenario"]))
    for r in sorted_results:
        td = f"{r['t_detect']:.0f}" if r["t_detect"] else "  N/A"
        dl = f"{r['delay_from_onset']:.0f}" if r["delay_from_onset"] is not None else "  N/A"
        rx = f"{r['delay_from_label']:.0f}" if r["delay_from_label"] is not None else "  N/A"
        mg = f"{r['margin']:.0f}" if r["margin"] is not None else "  N/A"
        rs = "PASS" if r["passed"] else "FAIL"
        fa = f" FA:{r['n_pre_fa']}" if r["n_pre_fa"] > 0 else ""
        print(f"  {r['run_id']:<24} {r['k_R_mOhm']:>5.2f} {r['scenario_name']:<9} "
              f"{td:>8}s {dl:>7}s {rx:>7}s {mg:>7}s {rs:>6}{fa}")

    # Summary
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    detected = [r for r in results if r["delay_from_onset"] is not None]

    print(f"\n  Summary: PASS {n_pass}/{n_total} ({n_pass/n_total*100:.0f}%)")
    if detected:
        delays = [r["delay_from_onset"] for r in detected]
        print(f"  Avg delay: {np.mean(delays):.0f}s, "
              f"Min: {np.min(delays):.0f}s, Max: {np.max(delays):.0f}s")
    fa_total = sum(r["n_pre_fa"] for r in results)
    if fa_total > 0:
        print(f"  Pre-onset false alarms: {fa_total} total")
