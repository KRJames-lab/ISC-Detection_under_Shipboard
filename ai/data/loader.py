"""
Data Loader — Load tau_sweep_results.mat (HDF5) and extract Module runs.
"""
import h5py
import numpy as np
from dataclasses import dataclass
from typing import List
from ai.config import (
    TAU_SWEEP_FILE, EVAL_TAU_FILE, MODEL_ID, SCENARIOS, SCENARIO_NAMES,
    TAU_VALUES, KR_VALUES, KR_LABELS, EVAL_TAUS, T_ONSET, R_INIT, R_FINAL
)


@dataclass
class SimRun:
    """Single simulation run data."""
    run_id: str
    scenario: int
    scenario_name: str
    k_R: float          # Ohm/g
    k_R_mOhm: float     # mOhm/g (display)
    tau: float           # seconds
    t: np.ndarray        # time vector (s)
    V: np.ndarray        # terminal voltage (V)
    T: np.ndarray        # temperature (K)

    @property
    def T_celsius(self) -> np.ndarray:
        return self.T - 273.15

    def R_ISC(self, t_arr: np.ndarray = None) -> np.ndarray:
        """Compute R_ISC at given times."""
        if t_arr is None:
            t_arr = self.t
        r = np.where(
            t_arr < T_ONSET,
            R_INIT,
            R_FINAL + (R_INIT - R_FINAL) * np.exp(-(t_arr - T_ONSET) / self.tau)
        )
        return r


def load_all_runs() -> List[SimRun]:
    """Load all Module runs from tau_sweep_results.mat."""
    runs = []

    with h5py.File(str(TAU_SWEEP_FILE), 'r') as f:
        results = f['results']

        for si, sc in enumerate(SCENARIOS):
            for ki, (kr, kr_label) in enumerate(zip(KR_VALUES, KR_LABELS)):
                for tau in TAU_VALUES:
                    field = f"m{MODEL_ID}_s{sc:02d}_k{ki+1}_tau{tau}"

                    if field not in results:
                        print(f"Warning: {field} not found, skipping")
                        continue

                    entry = results[field]
                    V = np.array(entry['V']).flatten()
                    T = np.array(entry['T']).flatten()
                    t = np.array(entry['t']).flatten()

                    run = SimRun(
                        run_id=field,
                        scenario=sc,
                        scenario_name=SCENARIO_NAMES[sc],
                        k_R=kr,
                        k_R_mOhm=kr_label,
                        tau=tau,
                        t=t, V=V, T=T
                    )
                    runs.append(run)

    print(f"Loaded {len(runs)} runs from {TAU_SWEEP_FILE.name}")
    _print_summary(runs)
    return runs


def load_eval_runs() -> List[SimRun]:
    """Load evaluation-only runs from eval_tau_results.mat (unseen τ values)."""
    runs = []

    with h5py.File(str(EVAL_TAU_FILE), 'r') as f:
        results = f['results']

        for si, sc in enumerate(SCENARIOS):
            for ki, (kr, kr_label) in enumerate(zip(KR_VALUES, KR_LABELS)):
                for tau in EVAL_TAUS:
                    field = f"m{MODEL_ID}_s{sc:02d}_k{ki+1}_tau{tau}"

                    if field not in results:
                        print(f"Warning: {field} not found, skipping")
                        continue

                    entry = results[field]
                    V = np.array(entry['V']).flatten()
                    T = np.array(entry['T']).flatten()
                    t = np.array(entry['t']).flatten()

                    run = SimRun(
                        run_id=field,
                        scenario=sc,
                        scenario_name=SCENARIO_NAMES[sc],
                        k_R=kr,
                        k_R_mOhm=kr_label,
                        tau=tau,
                        t=t, V=V, T=T
                    )
                    runs.append(run)

    print(f"Loaded {len(runs)} eval runs from {EVAL_TAU_FILE.name}")
    _print_summary(runs)
    return runs


def _print_summary(runs: List[SimRun]):
    """Print summary of loaded data."""
    print(f"  Scenarios: {sorted(set(r.scenario for r in runs))}")
    print(f"  k_R (mOhm/g): {sorted(set(r.k_R_mOhm for r in runs))}")
    print(f"  tau (s): {sorted(set(r.tau for r in runs))}")

    for tau in sorted(set(r.tau for r in runs)):
        subset = [r for r in runs if r.tau == tau]
        durations = [r.t[-1] for r in subset]
        samples = [len(r.t) for r in subset]
        print(f"  tau={tau:>5}s: {len(subset)} runs, "
              f"duration={np.mean(durations):.0f}s, "
              f"samples={np.mean(samples):.0f}")
