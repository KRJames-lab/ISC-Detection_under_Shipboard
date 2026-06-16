"""
Post-Training Quantization (PTQ) for STM32N6 NPU deployment.
W8A8 static quantization using onnxruntime — int8 weights + int8 activations.

Usage:
    cd D:/01_Projects/05_Ship_Battery
    python -m ai.export.quantize_onnx

Input:  ai/results/onnx/modern_tcn.onnx, lite.onnx (float32)
Output: ai/results/onnx/modern_tcn_int8.onnx, lite_int8.onnx (int8 QDQ)
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from ai.config import (
    INPUT_CHANNELS,
    RANDOM_SEED,
    RESULTS_DIR,
    WINDOW_SIZE,
)

N_CALIBRATION = 200


class WindowCalibrationReader(CalibrationDataReader):
    """Feeds calibration windows one-by-one to the quantizer."""

    def __init__(self, windows: np.ndarray):
        self.data = windows.astype(np.float32)
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.data):
            return None
        sample = self.data[self.idx : self.idx + 1]  # (1, 2, 1000)
        self.idx += 1
        return {"input": sample}


def load_calibration_windows() -> np.ndarray:
    """Load N_CALIBRATION random windows from train set."""
    from ai.data.loader import load_all_runs
    from ai.data.preprocess import preprocess_all
    from ai.data.splits import split_window_sets
    from ai.data.windowing import extract_all_windows

    print(f"  Loading train data for calibration ({N_CALIBRATION} samples)...")
    runs = load_all_runs()
    runs = preprocess_all(runs)
    window_sets = extract_all_windows(runs)
    train_sets, _, _ = split_window_sets(window_sets)

    # Gather all train windows
    all_windows = np.concatenate([ws.windows for ws in train_sets], axis=0)
    print(f"  Total train windows: {len(all_windows)}")

    # Random sample
    rng = np.random.RandomState(RANDOM_SEED)
    indices = rng.choice(len(all_windows), size=N_CALIBRATION, replace=False)
    calib = all_windows[indices]
    print(f"  Calibration samples: {calib.shape}")
    return calib


def quantize_model(f32_path: Path, int8_path: Path, calib_windows: np.ndarray, name: str):
    """Apply W8A8 static quantization to a single model."""
    print(f"\n--- {name} ---")
    print(f"  Input:  {f32_path}")

    # Pre-process: shape inference + optimization (recommended by onnxruntime)
    preproc_path = f32_path.parent / f"{f32_path.stem}_preproc.onnx"
    quant_pre_process(str(f32_path), str(preproc_path))
    print(f"  Pre-processed: {preproc_path.name}")

    reader = WindowCalibrationReader(calib_windows)

    # Quantize ALL operators (Conv, Add, AveragePool, etc.) to int8.
    # Full int8 avoids SW fallback epochs on NPU — required for Nucleo NPU execution.
    quantize_static(
        model_input=str(preproc_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )

    # Clean up preprocessed file
    preproc_path.unlink(missing_ok=True)

    f32_kb = f32_path.stat().st_size / 1024
    int8_kb = int8_path.stat().st_size / 1024
    print(f"  float32: {f32_kb:.1f} KB")
    print(f"  int8:    {int8_kb:.1f} KB  ({int8_kb / f32_kb * 100:.0f}%)")
    print(f"  Saved:   {int8_path}")

    # Validate: compare float32 vs int8 outputs
    validate_quantized(f32_path, int8_path, calib_windows, name)


def validate_quantized(f32_path: Path, int8_path: Path, windows: np.ndarray, name: str):
    """Compare float32 vs int8 output on calibration data."""
    sess_f32 = ort.InferenceSession(str(f32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    diffs = []
    agree = 0
    for i in range(len(windows)):
        x = windows[i : i + 1].astype(np.float32)
        out_f32 = sess_f32.run(None, {"input": x})[0].item()
        out_int8 = sess_int8.run(None, {"input": x})[0].item()
        diffs.append(abs(out_f32 - out_int8))
        if (out_f32 > 0) == (out_int8 > 0):
            agree += 1

    max_diff = max(diffs)
    mean_diff = np.mean(diffs)
    agreement = agree / len(windows) * 100

    print(f"  Validation ({len(windows)} samples):")
    print(f"    Max logit diff:   {max_diff:.4f}")
    print(f"    Mean logit diff:  {mean_diff:.4f}")
    print(f"    Sign agreement:   {agree}/{len(windows)} ({agreement:.1f}%)")

    if agreement < 95:
        print(f"  [!] WARNING: Low sign agreement -- quantization may hurt accuracy")


def main():
    onnx_dir = RESULTS_DIR / "onnx"

    print("=" * 60)
    print("  PTQ Quantization: float32 -> int8 (W8A8, QDQ)")
    print("  For STM32N6 Neural-ART NPU deployment")
    print("=" * 60)
    print(f"  Calibration: {N_CALIBRATION} random train windows (seed={RANDOM_SEED})")
    print(f"  Format: QDQ (QuantizeLinear/DequantizeLinear nodes)")
    print(f"  Input shape: (1, {INPUT_CHANNELS}, {WINDOW_SIZE})")

    # Load calibration data once (reused for both models)
    calib_windows = load_calibration_windows()

    targets = [
        ("modern_tcn_relu.onnx", "modern_tcn_relu_int8.onnx", "ModernTCN"),
        ("lite_relu.onnx", "lite_relu_int8.onnx", "LITE"),
    ]

    for f32_name, int8_name, display_name in targets:
        f32_path = onnx_dir / f32_name
        int8_path = onnx_dir / int8_name
        if not f32_path.exists():
            print(f"\n  [SKIP] {display_name}: {f32_path} not found")
            continue
        quantize_model(f32_path, int8_path, calib_windows, display_name)

    print(f"\n{'=' * 60}")
    print("  Next: python -m ai.scripts.run_detection_delay_onnx")
    print("=" * 60)


if __name__ == "__main__":
    main()
