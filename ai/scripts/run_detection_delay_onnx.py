"""
Evaluate detection delay on int8 quantized ONNX models.
Compares float32 ONNX vs int8 ONNX for both ModernTCN and LITE.

Usage:
    cd D:/01_Projects/05_Ship_Battery
    python -m ai.scripts.run_detection_delay_onnx
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import onnxruntime as ort
import torch

from ai.config import (
    DEADLINE_FACTOR, EVAL_TAUS, RANDOM_SEED, RESULTS_DIR,
    SCENARIO_NAMES, T_ONSET, TEST_TAU,
)
from ai.data.loader import load_all_runs, load_eval_runs
from ai.data.preprocess import preprocess_all
from ai.data.splits import split_window_sets
from ai.data.windowing import extract_all_windows


class OnnxModel:
    """Wraps an ONNX model to match the PyTorch model interface for evaluate_detection_delay."""

    def __init__(self, onnx_path: str):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.name = Path(onnx_path).stem

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Batch inference: loop since ONNX was exported with batch=1
        logits = []
        x_np = x.numpy()
        for i in range(len(x_np)):
            out = self.sess.run(None, {"input": x_np[i:i+1]})[0]
            logits.append(out.item())
        return torch.tensor(logits)

    def to(self, device):
        return self  # ONNX always runs on CPU

    def eval(self):
        return self


def evaluate_detection_delay_onnx(model, test_sets, model_name):
    """Run-level detection delay evaluation (same logic as evaluate.py)."""
    results = []
    for ws in test_sets:
        x = torch.FloatTensor(ws.windows)
        logits = model(x).numpy().flatten()
        preds = (logits > 0).astype(int)  # logit > 0 = ISC

        t_ends = ws.t_ends

        # First detection after T_ONSET
        post_onset = t_ends > T_ONSET
        post_preds = preds[post_onset]
        post_times = t_ends[post_onset]

        detect_idx = np.where(post_preds == 1)[0]
        if len(detect_idx) > 0:
            t_detect = float(post_times[detect_idx[0]])
            delay_from_onset = t_detect - T_ONSET
        else:
            t_detect = None
            delay_from_onset = None

        # Deadline
        deadline = DEADLINE_FACTOR * ws.tau

        if delay_from_onset is not None:
            passed = delay_from_onset <= deadline
            margin = deadline - delay_from_onset
        else:
            passed = False
            margin = None

        # False alarms before ISC onset
        pre_onset = t_ends <= T_ONSET
        n_pre_fa = int(preds[pre_onset].sum()) if pre_onset.any() else 0

        results.append({
            "run_id": ws.run_id,
            "scenario": ws.scenario,
            "scenario_name": SCENARIO_NAMES.get(ws.scenario, str(ws.scenario)),
            "k_R_mOhm": ws.k_R_mOhm,
            "tau": ws.tau,
            "delay_from_onset": delay_from_onset,
            "deadline": deadline,
            "passed": passed,
            "margin": margin,
            "n_pre_fa": n_pre_fa,
        })

    return results


def print_summary(name, results_by_tau):
    """Print pass/fail and average delay summary."""
    taus = sorted(results_by_tau.keys())
    total_pass = 0
    total_runs = 0
    total_fa = 0

    print(f"\n  [{name}]")
    print(f"  {'tau':>6} {'Deadline':>10} {'PASS':>8} {'Avg Delay':>10} {'FA':>5}")
    print(f"  {'-'*45}")

    for tau in taus:
        res = results_by_tau[tau]
        n_pass = sum(1 for r in res if r["passed"])
        n_total = len(res)
        detected = [r for r in res if r["delay_from_onset"] is not None]
        avg_delay = np.mean([r["delay_from_onset"] for r in detected]) if detected else float("nan")
        fa = sum(r["n_pre_fa"] for r in res)

        total_pass += n_pass
        total_runs += n_total
        total_fa += fa

        deadline = DEADLINE_FACTOR * tau
        status = "ALL PASS" if n_pass == n_total else f"{n_pass}/{n_total}"
        print(f"  {tau:>6} {deadline:>9.0f}s {status:>8} {avg_delay:>9.0f}s {fa:>5}")

    print(f"  {'-'*45}")
    print(f"  Total: {total_pass}/{total_runs} PASS, FA={total_fa}")
    return total_pass, total_runs, total_fa


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("  int8 Quantization Validation: Detection Delay")
    print("  float32 ONNX vs int8 ONNX")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    runs = load_all_runs()
    runs = preprocess_all(runs)
    window_sets = extract_all_windows(runs)
    _, _, test_sets_1800 = split_window_sets(window_sets)

    eval_runs = load_eval_runs()
    eval_runs = preprocess_all(eval_runs)
    eval_window_sets = extract_all_windows(eval_runs)

    eval_sets_by_tau = {}
    for tau in EVAL_TAUS:
        eval_sets_by_tau[tau] = [ws for ws in eval_window_sets if ws.tau == tau]
    eval_sets_by_tau[TEST_TAU] = test_sets_1800

    # Models to evaluate
    onnx_dir = RESULTS_DIR / "onnx"
    model_configs = [
        ("ModernTCN f32", onnx_dir / "modern_tcn.onnx"),
        ("ModernTCN int8", onnx_dir / "modern_tcn_int8.onnx"),
        ("LITE f32", onnx_dir / "lite.onnx"),
        ("LITE int8", onnx_dir / "lite_int8.onnx"),
    ]

    # Evaluate each model
    all_summaries = {}
    for name, path in model_configs:
        if not path.exists():
            print(f"\n  [SKIP] {name}: {path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"  Evaluating: {name}")
        print(f"  Model: {path.name} ({path.stat().st_size/1024:.1f} KB)")
        print(f"{'='*70}")

        model = OnnxModel(str(path))
        results_by_tau = {}

        for tau in sorted(eval_sets_by_tau.keys()):
            sets = eval_sets_by_tau[tau]
            print(f"  tau={tau}s ({len(sets)} runs)...", end=" ", flush=True)
            results = evaluate_detection_delay_onnx(model, sets, name)
            n_pass = sum(1 for r in results if r["passed"])
            fa = sum(r["n_pre_fa"] for r in results)
            print(f"{n_pass}/{len(results)} PASS, FA={fa}")
            results_by_tau[tau] = results

        total_pass, total_runs, total_fa = print_summary(name, results_by_tau)
        all_summaries[name] = {
            "pass": total_pass, "total": total_runs, "fa": total_fa,
            "results": results_by_tau,
        }

    # Final comparison table
    print(f"\n{'='*70}")
    print("  COMPARISON: float32 vs int8")
    print(f"{'='*70}")
    print(f"  {'Model':<20} {'PASS':>10} {'FA':>6} {'Status':>10}")
    print(f"  {'-'*50}")
    for name, s in all_summaries.items():
        status = "OK" if s["pass"] == s["total"] and s["fa"] == 0 else "CHECK"
        print(f"  {name:<20} {s['pass']:>4}/{s['total']:<4} {s['fa']:>5} {status:>10}")

    # Per-tau delay comparison (f32 vs int8)
    print(f"\n  Average Detection Delay (s)")
    print(f"  {'tau':>6}", end="")
    for name in all_summaries:
        print(f"  {name:>18}", end="")
    print()
    print(f"  {'-'*80}")

    taus = sorted(eval_sets_by_tau.keys())
    for tau in taus:
        print(f"  {tau:>6}", end="")
        for name, s in all_summaries.items():
            res = s["results"][tau]
            detected = [r for r in res if r["delay_from_onset"] is not None]
            if detected:
                avg = np.mean([r["delay_from_onset"] for r in detected])
                print(f"  {avg:>17.0f}s", end="")
            else:
                print(f"  {'N/A':>18}", end="")
        print()


if __name__ == "__main__":
    main()
