"""
Train/Val/Test split — tau-based test, stratified val from train pool.
All k_R values appear in every split for per-k_R evaluation.
"""
from typing import List, Tuple
import numpy as np
from ai.data.windowing import WindowSet
from ai.config import TRAIN_TAUS, TEST_TAU, VAL_RATIO, RANDOM_SEED


def split_window_sets(
    window_sets: List[WindowSet]
) -> Tuple[List[WindowSet], List[WindowSet], List[WindowSet]]:
    """
    Split:
      Test:  tau = 1800              → 15 runs (all k_R, all scenarios)
      Train+Val pool: tau in {50, 300, 3600} → 45 runs
        Val:  ~20% stratified by (scenario, tau) → 9 runs
        Train: remaining → 36 runs
    """
    test = []
    pool = []

    for ws in window_sets:
        if ws.tau == TEST_TAU:
            test.append(ws)
        elif ws.tau in TRAIN_TAUS:
            pool.append(ws)

    # Stratified val: 1 run per (scenario, tau) group
    rng = np.random.RandomState(RANDOM_SEED)
    val_ids = set()
    for tau in TRAIN_TAUS:
        for sc in sorted(set(ws.scenario for ws in pool)):
            candidates = [ws for ws in pool if ws.tau == tau and ws.scenario == sc]
            n_val = max(1, round(len(candidates) * VAL_RATIO))
            chosen = rng.choice(len(candidates), size=n_val, replace=False)
            for i in chosen:
                val_ids.add(candidates[i].run_id)

    train = [ws for ws in pool if ws.run_id not in val_ids]
    val = [ws for ws in pool if ws.run_id in val_ids]

    _print_split_info("Train", train)
    _print_split_info("Val", val)
    _print_split_info("Test", test)

    return train, val, test


def _print_split_info(name: str, sets: List[WindowSet]):
    if not sets:
        print(f"  {name}: 0 runs")
        return

    n_runs = len(sets)
    n_windows = sum(len(ws.labels) for ws in sets)
    n_normal = sum((ws.labels == 0).sum() for ws in sets)
    n_isc = sum((ws.labels == 1).sum() for ws in sets)
    taus = sorted(set(ws.tau for ws in sets))
    krs = sorted(set(ws.k_R_mOhm for ws in sets))

    print(f"  {name}: {n_runs} runs, {n_windows} windows "
          f"(Normal={n_normal}, ISC={n_isc})")
    print(f"    k_R: {krs}, tau: {taus}")
