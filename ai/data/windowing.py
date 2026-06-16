"""
Windowing — Adaptive stride to equalize windows per run, physics-based labeling.
"""
import numpy as np
from typing import List
from dataclasses import dataclass
from ai.data.loader import SimRun
from ai.config import (
    WINDOW_SIZE, MIN_STRIDE, TARGET_WINDOWS_PER_RUN,
    R_LABEL_THRESHOLD, T_ONSET, R_INIT, R_FINAL, INPUT_CHANNELS,
)


@dataclass
class WindowSet:
    """Windowed data from a single run."""
    run_id: str
    scenario: int
    k_R: float
    k_R_mOhm: float
    tau: float
    windows: np.ndarray     # (N, 2, W) — channels: [V, T]
    labels: np.ndarray      # (N,) — 0=Normal, 1=ISC
    t_ends: np.ndarray      # (N,) — end time of each window
    stride_used: int        # actual stride used


def _compute_adaptive_stride(n_samples: int) -> int:
    """Compute stride to yield ~TARGET_WINDOWS_PER_RUN, with MIN_STRIDE floor."""
    W = WINDOW_SIZE
    if n_samples <= W:
        return MIN_STRIDE
    ideal_stride = (n_samples - W) / (TARGET_WINDOWS_PER_RUN - 1)
    return max(int(round(ideal_stride)), MIN_STRIDE)


def _compute_label(t_end: float, tau: float) -> int:
    """Compute binary label based on R_ISC at window end time."""
    if t_end <= T_ONSET:
        return 0
    r_isc = R_FINAL + (R_INIT - R_FINAL) * np.exp(-(t_end - T_ONSET) / tau)
    return 1 if r_isc <= R_LABEL_THRESHOLD else 0


def extract_windows(run: SimRun) -> WindowSet:
    """Extract sliding windows with adaptive stride."""
    W = WINDOW_SIZE
    n_samples = len(run.V)
    stride = _compute_adaptive_stride(n_samples)

    windows = []
    labels = []
    t_ends = []

    for start in range(0, n_samples - W + 1, stride):
        end = start + W
        t_end = run.t[end - 1]

        if INPUT_CHANNELS == 1:
            window = run.V[start:end][np.newaxis, :]  # (1, W) — V-only ablation
        else:
            window = np.stack([run.V[start:end], run.T[start:end]], axis=0)  # (2, W) — V+T
        label = _compute_label(t_end, run.tau)

        windows.append(window)
        labels.append(label)
        t_ends.append(t_end)

    return WindowSet(
        run_id=run.run_id,
        scenario=run.scenario,
        k_R=run.k_R,
        k_R_mOhm=run.k_R_mOhm,
        tau=run.tau,
        windows=np.array(windows, dtype=np.float32),
        labels=np.array(labels, dtype=np.int64),
        t_ends=np.array(t_ends),
        stride_used=stride
    )


def extract_all_windows(runs: List[SimRun]) -> List[WindowSet]:
    """Extract windows from all runs with adaptive stride."""
    window_sets = []
    total_windows = 0
    total_normal = 0
    total_isc = 0

    for run in runs:
        ws = extract_windows(run)
        window_sets.append(ws)
        total_windows += len(ws.labels)
        total_normal += (ws.labels == 0).sum()
        total_isc += (ws.labels == 1).sum()

    print(f"Extracted {total_windows} windows from {len(runs)} runs")
    print(f"  Normal: {total_normal} ({total_normal/total_windows*100:.1f}%)")
    print(f"  ISC:    {total_isc} ({total_isc/total_windows*100:.1f}%)")

    # Per-tau summary
    for tau in sorted(set(ws.tau for ws in window_sets)):
        subset = [ws for ws in window_sets if ws.tau == tau]
        counts = [len(ws.labels) for ws in subset]
        strides = [ws.stride_used for ws in subset]
        print(f"  tau={tau:>5}s: stride={strides[0]:>4}, "
              f"windows/run={counts[0]:>4}, "
              f"total={sum(counts):>5}")

    return window_sets
