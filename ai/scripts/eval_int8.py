"""
Evaluate TFLite int8 models on Detection Delay — verify quantization preserves 45/45 PASS.

Compares float32 (PyTorch) vs int8 (TFLite) predictions on the same test set.

Usage:
    cd D:/01_Projects/05_Ship_Battery
    python -m ai.scripts.eval_int8
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.config import (
    RESULTS_DIR, T_ONSET, DEADLINE_FACTOR, SCENARIO_NAMES,
    INPUT_CHANNELS, WINDOW_SIZE,
)
from ai.models.npu_conv2d import RESHAPE_H, RESHAPE_W


class TFLitePredictor:
    """Wraps a TFLite int8 model for batch prediction."""

    def __init__(self, tflite_path, is_2d=False):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.is_2d = is_2d

        # Quantization params
        q_in = self.input_details["quantization_parameters"]
        self.in_scale = q_in["scales"][0]
        self.in_zp = q_in["zero_points"][0]

        q_out = self.output_details["quantization_parameters"]
        self.out_scale = q_out["scales"][0]
        self.out_zp = q_out["zero_points"][0]

    def _preprocess(self, window):
        """Convert single window (2, 1000) NCHW → TFLite input shape NHWC, quantized int8."""
        if self.is_2d:
            # NPU-Conv2D: (2, 1000) → (1, 20, 50, 2)
            x = window.reshape(INPUT_CHANNELS, RESHAPE_H, RESHAPE_W)
            x = np.transpose(x, (1, 2, 0))  # (20, 50, 2)
        else:
            # 1D models: (2, 1000) → (1, 1000, 2)
            x = np.transpose(window, (1, 0))  # (1000, 2)
            x = x[np.newaxis, :, :]  # (1, 1000, 2)

        x = x[np.newaxis].astype(np.float32)  # (1, H, W, C)

        # Quantize to int8
        x_q = np.clip(np.round(x / self.in_scale + self.in_zp), -128, 127).astype(np.int8)
        return x_q

    def predict_windows(self, windows):
        """Predict all windows. Returns logits (float) array."""
        logits = []
        for i in range(len(windows)):
            x_q = self._preprocess(windows[i])
            self.interpreter.set_tensor(self.input_details["index"], x_q)
            self.interpreter.invoke()
            out_q = self.interpreter.get_tensor(self.output_details["index"])
            # Dequantize
            out_f = (out_q.astype(np.float32) - self.out_zp) * self.out_scale
            logits.append(out_f.flatten()[0])
        return np.array(logits)


def evaluate_detection_delay_tflite(predictor, test_sets, model_name):
    """Run-level detection delay evaluation using TFLite predictor."""
    results = []

    for ws in test_sets:
        logits = predictor.predict_windows(ws.windows)
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        preds = (probs > 0.5).astype(int)

        t_ends = ws.t_ends
        labels = ws.labels

        isc_mask = labels == 1
        t_label_onset = float(t_ends[isc_mask][0]) if isc_mask.any() else None

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

        deadline = DEADLINE_FACTOR * ws.tau
        t_deadline = T_ONSET + deadline

        if delay_from_onset is not None:
            passed = delay_from_onset <= deadline
            margin = deadline - delay_from_onset
            delay_from_label = (t_detect - t_label_onset) if t_label_onset else None
        else:
            passed = False
            margin = None
            delay_from_label = None

        pre_onset = t_ends <= T_ONSET
        n_pre_fa = int(preds[pre_onset].sum()) if pre_onset.any() else 0

        results.append({
            "run_id": ws.run_id,
            "scenario": ws.scenario,
            "scenario_name": SCENARIO_NAMES.get(ws.scenario, str(ws.scenario)),
            "k_R_mOhm": ws.k_R_mOhm,
            "tau": ws.tau,
            "t_detect": t_detect,
            "delay_from_onset": delay_from_onset,
            "delay_from_label": delay_from_label,
            "deadline": deadline,
            "passed": passed,
            "margin": margin,
            "n_pre_fa": n_pre_fa,
        })

    _print_report(model_name, results)
    return results


def evaluate_window_accuracy(predictor, test_sets, model_name):
    """Window-level F1/accuracy comparison."""
    all_preds, all_labels = [], []
    for ws in test_sets:
        logits = predictor.predict_windows(ws.windows)
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(ws.labels)

    labels = np.array(all_labels)
    preds = np.array(all_preds)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"  [{model_name}] Window-level: F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} "
          f"(TP={tp} FP={fp} FN={fn} TN={tn})")
    return f1


def _print_report(model_name, results):
    """Print detection delay summary."""
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    detected = [r for r in results if r["delay_from_onset"] is not None]
    n_fa = sum(r["n_pre_fa"] for r in results)

    print(f"\n  [{model_name}] Detection Delay: {n_pass}/{n_total} PASS, FA={n_fa}")

    if detected:
        delays = [r["delay_from_onset"] for r in detected]
        margins = [r["margin"] for r in detected]
        print(f"    Avg delay: {np.mean(delays):.0f}s, Avg margin: {np.mean(margins):.0f}s")
        print(f"    Min/Max delay: {min(delays):.0f}s / {max(delays):.0f}s")

    # Per k_R
    print(f"    {'k_R':>6}  {'PASS':>8}  {'Avg Delay':>10}")
    for kr in sorted(set(r["k_R_mOhm"] for r in results)):
        sub = [r for r in results if r["k_R_mOhm"] == kr]
        n_p = sum(1 for r in sub if r["passed"])
        det = [r for r in sub if r["delay_from_onset"] is not None]
        avg_d = np.mean([r["delay_from_onset"] for r in det]) if det else float("nan")
        print(f"    {kr:>6.2f}  {n_p}/{len(sub):>5}  {avg_d:>9.0f}s")


def main():
    import torch
    from ai.data.loader import load_all_runs
    from ai.data.preprocess import preprocess_all
    from ai.data.windowing import extract_all_windows
    from ai.data.splits import split_window_sets

    print("=" * 65)
    print("  Int8 Quantization Accuracy Verification")
    print("=" * 65)

    # Load test data
    print("\nLoading data...")
    runs = load_all_runs()
    runs = preprocess_all(runs)
    window_sets = extract_all_windows(runs)
    _, _, test_sets = split_window_sets(window_sets)
    n_windows = sum(len(ws.windows) for ws in test_sets)
    print(f"  Test: {len(test_sets)} runs, {n_windows} windows")

    tflite_dir = RESULTS_DIR / "tflite"

    models = [
        # PTQ int8
        ("ModernTCN-PTQ-int8", tflite_dir / "modern_tcn_gelu_int8.tflite", False),
        ("LITE-PTQ-int8", tflite_dir / "lite_gelu_int8.tflite", False),
        ("NPU-Conv2D-PTQ-int8", tflite_dir / "npu_conv2d_relu_int8.tflite", True),
        # QAT int8
        ("ModernTCN-QAT-int8", tflite_dir / "modern_tcn_gelu_qat_int8.tflite", False),
        ("LITE-QAT-int8", tflite_dir / "lite_gelu_qat_int8.tflite", False),
        ("NPU-Conv2D-QAT-int8", tflite_dir / "npu_conv2d_relu_qat_int8.tflite", True),
    ]

    # Also run float32 PyTorch models for comparison
    print("\n--- Float32 Reference (PyTorch) ---")
    from ai.models.modern_tcn import ModernTCN
    from ai.models.lite import LITE
    from ai.models.npu_conv2d import NPUConv2D
    from ai.training.evaluate import evaluate_detection_delay

    pt_models = [
        ("ModernTCN-f32", ModernTCN, RESULTS_DIR / "moderntcn_best.pt"),
        ("LITE-f32", LITE, RESULTS_DIR / "lite_best.pt"),
        ("NPU-Conv2D-f32", NPUConv2D, RESULTS_DIR / "npu_conv2d_best.pt"),
    ]

    pt_results = {}
    for name, cls, pt_path in pt_models:
        if not pt_path.exists():
            print(f"  [SKIP] {name}: {pt_path} not found")
            continue
        model = cls()
        sd = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        res = evaluate_detection_delay(model, test_sets, name, "cpu")
        pt_results[name] = res

    # TFLite int8 models
    print("\n--- Int8 Quantized (TFLite) ---")
    tfl_results = {}
    for name, path, is_2d in models:
        if not path.exists():
            print(f"  [SKIP] {name}: {path} not found")
            continue
        pred = TFLitePredictor(path, is_2d=is_2d)
        evaluate_window_accuracy(pred, test_sets, name)
        res = evaluate_detection_delay_tflite(pred, test_sets, name)
        tfl_results[name] = res

    # Comparison summary
    print(f"\n{'=' * 65}")
    print(f"  Float32 vs Int8 Comparison")
    print(f"{'=' * 65}")
    print(f"  {'Model':<25} {'PASS':>6} {'FA':>4} {'Avg Delay':>10} {'Avg Margin':>11}")
    print(f"  {'-' * 60}")

    all_results = {}
    all_results.update({k: v for k, v in pt_results.items()})
    all_results.update({k: v for k, v in tfl_results.items()})

    for name, results in all_results.items():
        n_pass = sum(1 for r in results if r["passed"])
        n_total = len(results)
        n_fa = sum(r["n_pre_fa"] for r in results)
        detected = [r for r in results if r["delay_from_onset"] is not None]
        avg_d = np.mean([r["delay_from_onset"] for r in detected]) if detected else float("nan")
        avg_m = np.mean([r["margin"] for r in detected]) if detected else float("nan")
        tag = "***" if n_pass < n_total else ""
        print(f"  {name:<25} {n_pass:>2}/{n_total:<2} {n_fa:>4} {avg_d:>9.0f}s {avg_m:>10.0f}s {tag}")

    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
