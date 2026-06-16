"""
Convert NPU-Conv2D from PyTorch to TFLite (full int8) for STM32N6 NPU.

Approach: Build equivalent Keras model → transfer PyTorch weights →
          TFLiteConverter with full int8 quantization.

This bypasses onnx2tf which produces partially-quantized models
(only filter weights int8, activations float32 → NPU SW fallback).

Usage:
    cd D:/01_Projects/05_Ship_Battery
    python -m ai.export.convert_tflite

Input:  ai/results/npu_conv2d_best.pt (PyTorch weights)
Output: ai/results/tflite/npu_conv2d_relu_int8.tflite (full int8)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.config import INPUT_CHANNELS, RANDOM_SEED, RESULTS_DIR, WINDOW_SIZE
from ai.models.npu_conv2d import RESHAPE_H, RESHAPE_W

N_CALIBRATION = 200


def load_calibration_data() -> np.ndarray:
    """Load calibration windows from train set. Returns (N, 2, 1000)."""
    from ai.data.loader import load_all_runs
    from ai.data.preprocess import preprocess_all
    from ai.data.splits import split_window_sets
    from ai.data.windowing import extract_all_windows

    print(f"  Loading calibration data ({N_CALIBRATION} samples)...")
    runs = load_all_runs()
    runs = preprocess_all(runs)
    window_sets = extract_all_windows(runs)
    train_sets, _, _ = split_window_sets(window_sets)

    all_windows = np.concatenate([ws.windows for ws in train_sets], axis=0)
    rng = np.random.RandomState(RANDOM_SEED)
    indices = rng.choice(len(all_windows), size=N_CALIBRATION, replace=False)
    calib = all_windows[indices].astype(np.float32)
    print(f"  Calibration shape: {calib.shape}")
    return calib


def build_keras_model():
    """Build Keras model equivalent to NPUConv2D (PyTorch).

    Input:  (batch, RESHAPE_H, RESHAPE_W, 2)  — NHWC
    Output: (batch, 1)
    """
    import tensorflow as tf
    from tensorflow import keras

    inp = keras.Input(shape=(RESHAPE_H, RESHAPE_W, INPUT_CHANNELS), name="input")
    x = inp

    # Stem: Conv2D(2→16, 3x3) + BN + ReLU
    x = keras.layers.Conv2D(16, 3, padding="same", use_bias=False, name="stem_conv")(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, name="stem_bn")(x)
    x = keras.layers.ReLU(name="stem_relu")(x)

    # Block configs: (in_ch, out_ch, stride, expand_ratio)
    block_configs = [
        (16, 32, 2, 2),   # block0
        (32, 32, 1, 2),   # block1 (residual)
        (32, 64, 2, 2),   # block2
        (64, 64, 1, 2),   # block3 (residual)
        (64, 64, 2, 2),   # block4
    ]

    for i, (in_ch, out_ch, stride, expand_ratio) in enumerate(block_configs):
        mid_ch = in_ch * expand_ratio
        use_residual = (stride == 1 and in_ch == out_ch)
        prefix = f"block{i}"

        residual = x

        # Expand: 1x1 Conv
        x = keras.layers.Conv2D(
            mid_ch, 1, use_bias=False, name=f"{prefix}_expand_conv"
        )(x)
        x = keras.layers.BatchNormalization(
            epsilon=1e-5, name=f"{prefix}_expand_bn"
        )(x)
        x = keras.layers.ReLU(name=f"{prefix}_expand_relu")(x)

        # Depthwise: 3x3
        # For stride>1, use explicit symmetric padding to match PyTorch
        # (Keras 'same' uses asymmetric padding for even input sizes)
        if stride > 1:
            x = keras.layers.ZeroPadding2D(padding=1, name=f"{prefix}_dw_pad")(x)
            x = keras.layers.DepthwiseConv2D(
                3, strides=stride, padding="valid", use_bias=False,
                name=f"{prefix}_dw_conv",
            )(x)
        else:
            x = keras.layers.DepthwiseConv2D(
                3, strides=stride, padding="same", use_bias=False,
                name=f"{prefix}_dw_conv",
            )(x)
        x = keras.layers.BatchNormalization(
            epsilon=1e-5, name=f"{prefix}_dw_bn"
        )(x)
        x = keras.layers.ReLU(name=f"{prefix}_dw_relu")(x)

        # Project: 1x1 Conv (no activation)
        x = keras.layers.Conv2D(
            out_ch, 1, use_bias=False, name=f"{prefix}_proj_conv"
        )(x)
        x = keras.layers.BatchNormalization(
            epsilon=1e-5, name=f"{prefix}_proj_bn"
        )(x)

        if use_residual:
            x = keras.layers.Add(name=f"{prefix}_add")([residual, x])

    # Head: GAP → Dense(1)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(1, name="head_dense")(x)

    model = keras.Model(inputs=inp, outputs=x, name="npu_conv2d")
    return model


def transfer_weights(keras_model, pt_weight_path: Path):
    """Transfer PyTorch weights to Keras model.

    Key conversions:
      - Conv2D weight: PyTorch (out, in, H, W) → Keras (H, W, in, out)
      - DepthwiseConv2D: PyTorch (ch, 1, H, W) → Keras (H, W, ch, 1)
      - BN: gamma, beta, moving_mean, moving_var (same order, same shape)
      - Dense weight: PyTorch (out, in) → Keras (in, out)
    """
    import torch

    pt_state = torch.load(str(pt_weight_path), map_location="cpu", weights_only=True)
    print(f"  PyTorch state dict keys: {len(pt_state)}")

    def pt(key):
        return pt_state[key].numpy()

    # --- Stem ---
    stem_conv = keras_model.get_layer("stem_conv")
    w = pt("stem.0.weight")  # (16, 2, 3, 3)
    stem_conv.set_weights([np.transpose(w, (2, 3, 1, 0))])

    stem_bn = keras_model.get_layer("stem_bn")
    stem_bn.set_weights([
        pt("stem.1.weight"),        # gamma
        pt("stem.1.bias"),          # beta
        pt("stem.1.running_mean"),  # moving_mean
        pt("stem.1.running_var"),   # moving_var
    ])

    # --- Blocks ---
    block_configs = [
        (16, 32, 2, 2),
        (32, 32, 1, 2),
        (32, 64, 2, 2),
        (64, 64, 1, 2),
        (64, 64, 2, 2),
    ]

    for i, (in_ch, out_ch, stride, expand_ratio) in enumerate(block_configs):
        mid_ch = in_ch * expand_ratio
        prefix = f"block{i}"
        pt_prefix = f"blocks.{i}.conv"

        # Expand Conv: 1x1
        layer = keras_model.get_layer(f"{prefix}_expand_conv")
        w = pt(f"{pt_prefix}.0.weight")  # (mid_ch, in_ch, 1, 1)
        layer.set_weights([np.transpose(w, (2, 3, 1, 0))])

        layer = keras_model.get_layer(f"{prefix}_expand_bn")
        layer.set_weights([
            pt(f"{pt_prefix}.1.weight"),
            pt(f"{pt_prefix}.1.bias"),
            pt(f"{pt_prefix}.1.running_mean"),
            pt(f"{pt_prefix}.1.running_var"),
        ])

        # Depthwise Conv: 3x3
        layer = keras_model.get_layer(f"{prefix}_dw_conv")
        w = pt(f"{pt_prefix}.3.weight")  # (mid_ch, 1, 3, 3)
        # PyTorch DW: (out_ch=groups, 1, H, W) → Keras DW: (H, W, ch, 1)
        w = np.transpose(w, (2, 3, 0, 1))  # (3, 3, mid_ch, 1)
        layer.set_weights([w])

        layer = keras_model.get_layer(f"{prefix}_dw_bn")
        layer.set_weights([
            pt(f"{pt_prefix}.4.weight"),
            pt(f"{pt_prefix}.4.bias"),
            pt(f"{pt_prefix}.4.running_mean"),
            pt(f"{pt_prefix}.4.running_var"),
        ])

        # Project Conv: 1x1
        layer = keras_model.get_layer(f"{prefix}_proj_conv")
        w = pt(f"{pt_prefix}.6.weight")  # (out_ch, mid_ch, 1, 1)
        layer.set_weights([np.transpose(w, (2, 3, 1, 0))])

        layer = keras_model.get_layer(f"{prefix}_proj_bn")
        layer.set_weights([
            pt(f"{pt_prefix}.7.weight"),
            pt(f"{pt_prefix}.7.bias"),
            pt(f"{pt_prefix}.7.running_mean"),
            pt(f"{pt_prefix}.7.running_var"),
        ])

    # --- Head ---
    head_dense = keras_model.get_layer("head_dense")
    w = pt("head.3.weight")  # (1, 64)
    b = pt("head.3.bias")    # (1,)
    head_dense.set_weights([w.T, b])  # Keras: (64, 1), (1,)

    print(f"  Weights transferred successfully")


def verify_numerical(keras_model, pt_weight_path: Path, calib_data: np.ndarray):
    """Verify Keras output matches PyTorch output."""
    import torch
    from ai.models.npu_conv2d import NPUConv2D

    # PyTorch inference
    pt_model = NPUConv2D()
    pt_state = torch.load(str(pt_weight_path), map_location="cpu", weights_only=True)
    pt_model.load_state_dict(pt_state)
    pt_model.eval()

    sample = calib_data[:5]  # (5, 2, 1000)
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(sample)).numpy()

    # Keras inference: reshape + NHWC
    sample_2d = sample.reshape(-1, INPUT_CHANNELS, RESHAPE_H, RESHAPE_W)
    sample_nhwc = np.transpose(sample_2d, (0, 2, 3, 1)).astype(np.float32)
    # Use direct call with training=False (predict() has Keras 3 issues)
    keras_out = keras_model(sample_nhwc, training=False).numpy().flatten()

    diff = np.max(np.abs(pt_out - keras_out))
    print(f"\n  Numerical verification:")
    print(f"    PyTorch output:  {pt_out}")
    print(f"    Keras output:    {keras_out}")
    print(f"    Max diff:        {diff:.6e}")
    ok = diff < 1e-4
    print(f"    Status:          {'PASS' if ok else 'FAIL'}")
    return ok


def convert_to_tflite_int8(keras_model, tflite_path: Path, calib_data: np.ndarray):
    """Convert Keras model to full int8 TFLite with representative dataset.

    Uses from_concrete_functions() to bypass Keras 3 / TF 2.19 compatibility
    bug in from_keras_model() (_freeze_keras_model TypeError).
    """
    import tensorflow as tf

    print(f"\n--- TFLite Full Int8 Conversion ---")

    # Trace model to concrete function (bypasses Keras 3 internals)
    input_spec = tf.TensorSpec([1, RESHAPE_H, RESHAPE_W, INPUT_CHANNELS], tf.float32)
    model_fn = tf.function(lambda x: keras_model(x, training=False))
    concrete_fn = model_fn.get_concrete_function(input_spec)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_fn], keras_model
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Calibration: NCHW (N,2,1000) → reshape to (N,2,20,50) → NHWC (N,20,50,2)
    calib_nhwc = calib_data.reshape(-1, INPUT_CHANNELS, RESHAPE_H, RESHAPE_W)
    calib_nhwc = np.transpose(calib_nhwc, (0, 2, 3, 1)).astype(np.float32)

    def representative_dataset():
        for i in range(len(calib_nhwc)):
            yield [calib_nhwc[i:i+1]]

    converter.representative_dataset = representative_dataset

    # Force ALL operators to int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)

    kb = len(tflite_model) / 1024
    print(f"  TFLite size: {kb:.1f} KB")
    print(f"  Saved: {tflite_path}")
    return tflite_path


def verify_full_int8(tflite_path: Path):
    """Verify ALL tensors are quantized (no float32 activations)."""
    import tensorflow as tf
    from collections import Counter

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    details = interpreter.get_tensor_details()
    dtype_counts = Counter(d["dtype"].__name__ for d in details)

    print(f"\n--- Quantization Verification ---")
    print(f"  Total tensors: {len(details)}")
    for dtype, count in dtype_counts.most_common():
        print(f"    {dtype}: {count}")

    n_float = dtype_counts.get("float32", 0)
    if n_float == 0:
        print(f"  PASS: Full int8 - no float32 tensors")
        return True
    else:
        print(f"  FAIL: {n_float} float32 tensors remain")
        for d in details:
            if d["dtype"].__name__ == "float32":
                print(f"    float32: {d['name']}")
        return False


def main():
    pt_weight_path = RESULTS_DIR / "npu_conv2d_best.pt"
    tflite_path = RESULTS_DIR / "tflite" / "npu_conv2d_relu_int8.tflite"

    print("=" * 60)
    print("  PyTorch → Keras → TFLite (full int8)")
    print("  For STM32N6 Neural-ART NPU")
    print("=" * 60)
    print(f"  Model input:  (1, {RESHAPE_H}, {RESHAPE_W}, {INPUT_CHANNELS}) NHWC")
    print(f"  Calibration:  {N_CALIBRATION} samples")

    if not pt_weight_path.exists():
        print(f"\n  [ERROR] {pt_weight_path} not found")
        print(f"  Run: python -m ai.scripts.run_training")
        return

    # Load calibration data
    calib_data = load_calibration_data()

    # Build Keras model
    print(f"\n--- Building Keras Model ---")
    keras_model = build_keras_model()
    keras_model.summary(print_fn=lambda x: print(f"  {x}"))

    # Transfer weights
    print(f"\n--- Transferring Weights ---")
    transfer_weights(keras_model, pt_weight_path)

    # Verify numerical match
    ok = verify_numerical(keras_model, pt_weight_path, calib_data)
    if not ok:
        print("  WARNING: Numerical mismatch > 1e-4. Proceeding anyway.")

    # Convert to TFLite full int8
    convert_to_tflite_int8(keras_model, tflite_path, calib_data)

    # Verify quantization
    is_full_int8 = verify_full_int8(tflite_path)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Full int8:   {'CONFIRMED' if is_full_int8 else 'FAILED'}")
    print(f"  Output:      {tflite_path}")
    print(f"  Note:        Model input is (1, {RESHAPE_H}, {RESHAPE_W}, {INPUT_CHANNELS})")
    print(f"               Reshape from (1,2,1000) is done in preprocessing")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
