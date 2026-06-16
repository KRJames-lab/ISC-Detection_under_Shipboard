"""
Convert all models (ModernTCN, LITE, NPU-Conv2D) to TFLite full int8.

PyTorch → Keras (manual build + weight transfer) → TFLite int8
Same approach that achieved 18/18 HW epochs on NPU-Conv2D.

Conv1D models (ModernTCN, LITE) are expanded to Conv2D with H=1:
  Input (B, T, C) → (B, 1, T, C) → Conv2D((1,K)) → ...

Usage:
    cd D:/01_Projects/05_Ship_Battery
    python -m ai.export.convert_tflite_all
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.config import (
    INPUT_CHANNELS, RANDOM_SEED, RESULTS_DIR, WINDOW_SIZE,
    MODERN_TCN_D_MODEL, MODERN_TCN_STEM_KERNEL,
    MODERN_TCN_KERNEL, MODERN_TCN_BLOCKS, MODERN_TCN_FFN_RATIO,
    LITE_INCEPTION_KERNELS, LITE_INCEPTION_FILTERS,
    LITE_DWS_CHANNELS, LITE_DWS1_KERNEL, LITE_DWS2_KERNEL,
)
from ai.models.npu_conv2d import RESHAPE_H, RESHAPE_W

N_CALIBRATION = 500


# ============================================================
# Calibration data
# ============================================================

_calib_cache = None

def load_calibration_data() -> np.ndarray:
    global _calib_cache
    if _calib_cache is not None:
        return _calib_cache
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
    all_labels = np.concatenate([ws.labels for ws in train_sets], axis=0)

    # Balanced sampling: 50% Normal, 50% ISC for better range estimation
    rng = np.random.RandomState(RANDOM_SEED)
    n_half = N_CALIBRATION // 2
    idx_normal = np.where(all_labels == 0)[0]
    idx_isc = np.where(all_labels == 1)[0]
    sel_n = rng.choice(idx_normal, size=min(n_half, len(idx_normal)), replace=False)
    sel_i = rng.choice(idx_isc, size=min(n_half, len(idx_isc)), replace=False)
    indices = np.concatenate([sel_n, sel_i])
    rng.shuffle(indices)

    _calib_cache = all_windows[indices].astype(np.float32)
    n_norm = int((all_labels[indices] == 0).sum())
    n_isc = int((all_labels[indices] == 1).sum())
    print(f"  Calibration: {_calib_cache.shape} (Normal={n_norm}, ISC={n_isc})")
    return _calib_cache


# ============================================================
# Keras model builders
# ============================================================

def build_keras_modern_tcn():
    """ModernTCN as Conv2D with H=1. Input: (B, 1, 1000, 2) NHWC."""
    from tensorflow import keras
    d = MODERN_TCN_D_MODEL       # 32
    sk = MODERN_TCN_STEM_KERNEL  # 51
    bk = MODERN_TCN_KERNEL       # 121
    ffn = MODERN_TCN_FFN_RATIO   # 2
    n_blocks = MODERN_TCN_BLOCKS # 3

    inp = keras.Input(shape=(1, WINDOW_SIZE, INPUT_CHANNELS), name="input")
    x = inp

    # Stem: Conv2D(2->32, (1,51), same) + BN + GELU
    x = keras.layers.Conv2D(d, (1, sk), padding="same", use_bias=True, name="stem_conv")(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, name="stem_bn")(x)
    x = keras.layers.Activation("gelu", name="stem_gelu")(x)

    # Blocks
    for i in range(n_blocks):
        p = f"blk{i}"
        residual = x
        # DWConv: DepthwiseConv2D((1, 121), same)
        x = keras.layers.DepthwiseConv2D(
            (1, bk), padding="same", use_bias=True, name=f"{p}_dw",
        )(x)
        x = keras.layers.BatchNormalization(epsilon=1e-5, name=f"{p}_bn")(x)
        # ConvFFN: pw1 + GELU + pw2
        x = keras.layers.Conv2D(d * ffn, (1, 1), use_bias=True, name=f"{p}_pw1")(x)
        x = keras.layers.Activation("gelu", name=f"{p}_gelu")(x)
        x = keras.layers.Conv2D(d, (1, 1), use_bias=True, name=f"{p}_pw2")(x)
        x = keras.layers.Add(name=f"{p}_add")([residual, x])

    # Head: GAP -> Dense(1)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(1, name="head")(x)

    return keras.Model(inp, x, name="modern_tcn")


def build_keras_lite():
    """LITE as Conv2D with H=1. Input: (B, 1, 1000, 2) NHWC."""
    from tensorflow import keras
    kernels = LITE_INCEPTION_KERNELS   # [21,41,81,161,321]
    n_filt = LITE_INCEPTION_FILTERS    # 8
    dws_ch = LITE_DWS_CHANNELS         # 32
    k1 = LITE_DWS1_KERNEL              # 51
    k2 = LITE_DWS2_KERNEL              # 41

    inp = keras.Input(shape=(1, WINDOW_SIZE, INPUT_CHANNELS), name="input")
    x = inp

    # Inception: multi-scale parallel Conv
    branches = []
    for j, k in enumerate(kernels):
        b = keras.layers.Conv2D(
            n_filt, (1, k), padding="same", use_bias=True, name=f"inc_br{j}",
        )(x)
        branches.append(b)
    x = keras.layers.Concatenate(axis=-1, name="inc_cat")(branches)
    x = keras.layers.BatchNormalization(epsilon=1e-5, name="inc_bn")(x)
    x = keras.layers.Activation("gelu", name="inc_gelu")(x)

    # DWS1
    inc_out_ch = n_filt * len(kernels)  # 40
    x = keras.layers.DepthwiseConv2D(
        (1, k1), padding="same", use_bias=True, name="dws1_dw",
    )(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, name="dws1_dw_bn")(x)
    x = keras.layers.Activation("gelu", name="dws1_dw_gelu")(x)
    x = keras.layers.Conv2D(dws_ch, (1, 1), use_bias=True, name="dws1_pw")(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, name="dws1_pw_bn")(x)
    x = keras.layers.Activation("gelu", name="dws1_pw_gelu")(x)

    # DWS2
    x = keras.layers.DepthwiseConv2D(
        (1, k2), padding="same", use_bias=True, name="dws2_dw",
    )(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, name="dws2_dw_bn")(x)
    x = keras.layers.Activation("gelu", name="dws2_dw_gelu")(x)
    x = keras.layers.Conv2D(dws_ch, (1, 1), use_bias=True, name="dws2_pw")(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, name="dws2_pw_bn")(x)
    x = keras.layers.Activation("gelu", name="dws2_pw_gelu")(x)

    # Head
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(1, name="head")(x)

    return keras.Model(inp, x, name="lite")


def build_keras_npu_conv2d():
    """NPU-Conv2D (already 2D). Input: (B, 20, 50, 2) NHWC."""
    from tensorflow import keras
    inp = keras.Input(shape=(RESHAPE_H, RESHAPE_W, INPUT_CHANNELS), name="input")
    x = inp

    x = keras.layers.Conv2D(16, 3, padding="same", use_bias=False, name="stem_conv")(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, name="stem_bn")(x)
    x = keras.layers.ReLU(name="stem_relu")(x)

    block_configs = [
        (16, 32, 2, 2), (32, 32, 1, 2), (32, 64, 2, 2),
        (64, 64, 1, 2), (64, 64, 2, 2),
    ]
    for i, (in_ch, out_ch, stride, er) in enumerate(block_configs):
        mid = in_ch * er
        use_res = (stride == 1 and in_ch == out_ch)
        p = f"b{i}"
        residual = x
        x = keras.layers.Conv2D(mid, 1, use_bias=False, name=f"{p}_exp")(x)
        x = keras.layers.BatchNormalization(epsilon=1e-5, name=f"{p}_exp_bn")(x)
        x = keras.layers.ReLU(name=f"{p}_exp_relu")(x)
        if stride > 1:
            x = keras.layers.ZeroPadding2D(padding=1, name=f"{p}_dw_pad")(x)
            x = keras.layers.DepthwiseConv2D(3, strides=stride, padding="valid", use_bias=False, name=f"{p}_dw")(x)
        else:
            x = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False, name=f"{p}_dw")(x)
        x = keras.layers.BatchNormalization(epsilon=1e-5, name=f"{p}_dw_bn")(x)
        x = keras.layers.ReLU(name=f"{p}_dw_relu")(x)
        x = keras.layers.Conv2D(out_ch, 1, use_bias=False, name=f"{p}_proj")(x)
        x = keras.layers.BatchNormalization(epsilon=1e-5, name=f"{p}_proj_bn")(x)
        if use_res:
            x = keras.layers.Add(name=f"{p}_add")([residual, x])

    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(1, name="head")(x)
    return keras.Model(inp, x, name="npu_conv2d")


# ============================================================
# Weight transfer
# ============================================================

def transfer_conv1d(keras_layer, pt_weight, pt_bias=None):
    """PyTorch Conv1d (out, in, K) → Keras Conv2D (1, K, in, out)."""
    w = np.transpose(pt_weight, (2, 1, 0))    # (K, in, out)
    w = np.expand_dims(w, axis=0)               # (1, K, in, out)
    if pt_bias is not None:
        keras_layer.set_weights([w, pt_bias])
    else:
        keras_layer.set_weights([w])


def transfer_dw_conv1d(keras_layer, pt_weight, pt_bias=None):
    """PyTorch DW Conv1d (ch, 1, K) → Keras DepthwiseConv2D (1, K, ch, 1)."""
    w = np.transpose(pt_weight, (2, 0, 1))     # (K, ch, 1)
    w = np.expand_dims(w, axis=0)               # (1, K, ch, 1)
    if pt_bias is not None:
        keras_layer.set_weights([w, pt_bias])
    else:
        keras_layer.set_weights([w])


def transfer_bn(keras_layer, gamma, beta, mean, var):
    keras_layer.set_weights([gamma, beta, mean, var])


def pt_np(state, key):
    return state[key].numpy()


def transfer_modern_tcn(model, state):
    """Transfer ModernTCN weights."""
    # Stem (has bias)
    transfer_conv1d(
        model.get_layer("stem_conv"),
        pt_np(state, "stem.0.weight"), pt_np(state, "stem.0.bias"),
    )
    transfer_bn(model.get_layer("stem_bn"),
        pt_np(state, "stem.1.weight"), pt_np(state, "stem.1.bias"),
        pt_np(state, "stem.1.running_mean"), pt_np(state, "stem.1.running_var"),
    )

    for i in range(MODERN_TCN_BLOCKS):
        p = f"blk{i}"
        bp = f"blocks.{i}"
        # DW conv (has bias)
        transfer_dw_conv1d(
            model.get_layer(f"{p}_dw"),
            pt_np(state, f"{bp}.dw_conv.weight"), pt_np(state, f"{bp}.dw_conv.bias"),
        )
        transfer_bn(model.get_layer(f"{p}_bn"),
            pt_np(state, f"{bp}.norm.weight"), pt_np(state, f"{bp}.norm.bias"),
            pt_np(state, f"{bp}.norm.running_mean"), pt_np(state, f"{bp}.norm.running_var"),
        )
        # pw1 (has bias)
        transfer_conv1d(
            model.get_layer(f"{p}_pw1"),
            pt_np(state, f"{bp}.pw1.weight"), pt_np(state, f"{bp}.pw1.bias"),
        )
        # pw2 (has bias)
        transfer_conv1d(
            model.get_layer(f"{p}_pw2"),
            pt_np(state, f"{bp}.pw2.weight"), pt_np(state, f"{bp}.pw2.bias"),
        )

    # Head
    w = pt_np(state, "head.3.weight")  # (1, 32)
    b = pt_np(state, "head.3.bias")
    model.get_layer("head").set_weights([w.T, b])


def transfer_lite(model, state):
    """Transfer LITE weights."""
    # Inception branches (have bias)
    for j in range(len(LITE_INCEPTION_KERNELS)):
        transfer_conv1d(
            model.get_layer(f"inc_br{j}"),
            pt_np(state, f"inception.branches.{j}.weight"),
            pt_np(state, f"inception.branches.{j}.bias"),
        )
    transfer_bn(model.get_layer("inc_bn"),
        pt_np(state, "inception.norm.weight"), pt_np(state, "inception.norm.bias"),
        pt_np(state, "inception.norm.running_mean"), pt_np(state, "inception.norm.running_var"),
    )

    # DWS1
    transfer_dw_conv1d(
        model.get_layer("dws1_dw"),
        pt_np(state, "dws1.dw.weight"), pt_np(state, "dws1.dw.bias"),
    )
    transfer_bn(model.get_layer("dws1_dw_bn"),
        pt_np(state, "dws1.norm1.weight"), pt_np(state, "dws1.norm1.bias"),
        pt_np(state, "dws1.norm1.running_mean"), pt_np(state, "dws1.norm1.running_var"),
    )
    transfer_conv1d(
        model.get_layer("dws1_pw"),
        pt_np(state, "dws1.pw.weight"), pt_np(state, "dws1.pw.bias"),
    )
    transfer_bn(model.get_layer("dws1_pw_bn"),
        pt_np(state, "dws1.norm2.weight"), pt_np(state, "dws1.norm2.bias"),
        pt_np(state, "dws1.norm2.running_mean"), pt_np(state, "dws1.norm2.running_var"),
    )

    # DWS2
    transfer_dw_conv1d(
        model.get_layer("dws2_dw"),
        pt_np(state, "dws2.dw.weight"), pt_np(state, "dws2.dw.bias"),
    )
    transfer_bn(model.get_layer("dws2_dw_bn"),
        pt_np(state, "dws2.norm1.weight"), pt_np(state, "dws2.norm1.bias"),
        pt_np(state, "dws2.norm1.running_mean"), pt_np(state, "dws2.norm1.running_var"),
    )
    transfer_conv1d(
        model.get_layer("dws2_pw"),
        pt_np(state, "dws2.pw.weight"), pt_np(state, "dws2.pw.bias"),
    )
    transfer_bn(model.get_layer("dws2_pw_bn"),
        pt_np(state, "dws2.norm2.weight"), pt_np(state, "dws2.norm2.bias"),
        pt_np(state, "dws2.norm2.running_mean"), pt_np(state, "dws2.norm2.running_var"),
    )

    # Head
    w = pt_np(state, "head.3.weight")  # (1, 32)
    b = pt_np(state, "head.3.bias")
    model.get_layer("head").set_weights([w.T, b])


def transfer_npu_conv2d(model, state):
    """Transfer NPU-Conv2D weights."""
    # Stem
    w = pt_np(state, "stem.0.weight")
    model.get_layer("stem_conv").set_weights([np.transpose(w, (2, 3, 1, 0))])
    transfer_bn(model.get_layer("stem_bn"),
        pt_np(state, "stem.1.weight"), pt_np(state, "stem.1.bias"),
        pt_np(state, "stem.1.running_mean"), pt_np(state, "stem.1.running_var"),
    )

    block_configs = [(16,32,2,2),(32,32,1,2),(32,64,2,2),(64,64,1,2),(64,64,2,2)]
    for i, (in_ch, out_ch, stride, er) in enumerate(block_configs):
        p = f"b{i}"
        bp = f"blocks.{i}.conv"
        # Expand
        w = pt_np(state, f"{bp}.0.weight")
        model.get_layer(f"{p}_exp").set_weights([np.transpose(w, (2, 3, 1, 0))])
        transfer_bn(model.get_layer(f"{p}_exp_bn"),
            pt_np(state, f"{bp}.1.weight"), pt_np(state, f"{bp}.1.bias"),
            pt_np(state, f"{bp}.1.running_mean"), pt_np(state, f"{bp}.1.running_var"),
        )
        # DW
        w = pt_np(state, f"{bp}.3.weight")
        model.get_layer(f"{p}_dw").set_weights([np.transpose(w, (2, 3, 0, 1))])
        transfer_bn(model.get_layer(f"{p}_dw_bn"),
            pt_np(state, f"{bp}.4.weight"), pt_np(state, f"{bp}.4.bias"),
            pt_np(state, f"{bp}.4.running_mean"), pt_np(state, f"{bp}.4.running_var"),
        )
        # Proj
        w = pt_np(state, f"{bp}.6.weight")
        model.get_layer(f"{p}_proj").set_weights([np.transpose(w, (2, 3, 1, 0))])
        transfer_bn(model.get_layer(f"{p}_proj_bn"),
            pt_np(state, f"{bp}.7.weight"), pt_np(state, f"{bp}.7.bias"),
            pt_np(state, f"{bp}.7.running_mean"), pt_np(state, f"{bp}.7.running_var"),
        )

    w = pt_np(state, "head.3.weight")
    b = pt_np(state, "head.3.bias")
    model.get_layer("head").set_weights([w.T, b])


# ============================================================
# Numerical verification
# ============================================================

def verify(name, keras_model, pt_model_class, pt_path, calib, is_2d=False):
    import torch

    pt_model = pt_model_class()
    sd = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    pt_model.load_state_dict(sd)
    pt_model.eval()

    sample = calib[:5]  # (5, 2, 1000)
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(sample)).numpy()

    if is_2d:
        # NPU-Conv2D: (5,2,1000) → (5,2,20,50) NCHW → (5,20,50,2) NHWC
        s = sample.reshape(-1, INPUT_CHANNELS, RESHAPE_H, RESHAPE_W)
        s = np.transpose(s, (0, 2, 3, 1)).astype(np.float32)
    else:
        # 1D models: (5,2,1000) NCHW → (5,1,1000,2) NHWC with H=1
        s = np.transpose(sample, (0, 2, 1))[:, np.newaxis, :, :]  # (5,1,1000,2)

    keras_out = keras_model(s, training=False).numpy().flatten()
    diff = float(np.max(np.abs(pt_out - keras_out)))
    ok = diff < 1e-3
    print(f"  [{name}] Verify: max_diff={diff:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================
# TFLite conversion
# ============================================================

def convert_to_tflite(name, keras_model, tflite_path, calib, is_2d=False):
    import tensorflow as tf
    from collections import Counter

    print(f"\n  [{name}] Converting to TFLite int8...")

    input_shape = list(keras_model.input_shape[1:])  # e.g. (1, 1000, 2) or (20, 50, 2)
    input_spec = tf.TensorSpec([1] + input_shape, tf.float32)
    model_fn = tf.function(lambda x: keras_model(x, training=False))
    concrete_fn = model_fn.get_concrete_function(input_spec)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_fn], keras_model
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Prepare calibration data
    if is_2d:
        calib_tf = calib.reshape(-1, INPUT_CHANNELS, RESHAPE_H, RESHAPE_W)
        calib_tf = np.transpose(calib_tf, (0, 2, 3, 1)).astype(np.float32)
    else:
        calib_tf = np.transpose(calib, (0, 2, 1))[:, np.newaxis, :, :]  # (N,1,T,2)

    def representative_dataset():
        for i in range(len(calib_tf)):
            yield [calib_tf[i:i+1]]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)

    # Verify quantization
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()
    dtype_counts = Counter(d["dtype"].__name__ for d in details)
    n_float = dtype_counts.get("float32", 0)
    kb = len(tflite_model) / 1024

    print(f"  [{name}] TFLite: {kb:.1f} KB")
    print(f"  [{name}] Tensors: {dict(dtype_counts)}")
    print(f"  [{name}] Full int8: {'PASS' if n_float == 0 else f'FAIL ({n_float} float32)'}")
    print(f"  [{name}] Saved: {tflite_path}")
    return n_float == 0


# ============================================================
# Main
# ============================================================

def main():
    import torch

    print("=" * 65)
    print("  PyTorch -> Keras -> TFLite (full int8) : All Models")
    print("=" * 65)

    calib = load_calibration_data()
    tflite_dir = RESULTS_DIR / "tflite"

    targets = [
        # --- Original PTQ ---
        {
            "name": "ModernTCN",
            "builder": build_keras_modern_tcn,
            "transferer": transfer_modern_tcn,
            "pt_path": RESULTS_DIR / "moderntcn_best.pt",
            "tflite_name": "modern_tcn_gelu_int8.tflite",
            "pt_class_path": "ai.models.modern_tcn.ModernTCN",
            "is_2d": False,
        },
        {
            "name": "LITE",
            "builder": build_keras_lite,
            "transferer": transfer_lite,
            "pt_path": RESULTS_DIR / "lite_best.pt",
            "tflite_name": "lite_gelu_int8.tflite",
            "pt_class_path": "ai.models.lite.LITE",
            "is_2d": False,
        },
        {
            "name": "NPU-Conv2D",
            "builder": build_keras_npu_conv2d,
            "transferer": transfer_npu_conv2d,
            "pt_path": RESULTS_DIR / "npu_conv2d_best.pt",
            "tflite_name": "npu_conv2d_relu_int8.tflite",
            "pt_class_path": "ai.models.npu_conv2d.NPUConv2D",
            "is_2d": True,
        },
        # --- QAT ---
        {
            "name": "ModernTCN-QAT",
            "builder": build_keras_modern_tcn,
            "transferer": transfer_modern_tcn,
            "pt_path": RESULTS_DIR / "moderntcn_qat_best.pt",
            "tflite_name": "modern_tcn_gelu_qat_int8.tflite",
            "pt_class_path": "ai.models.modern_tcn.ModernTCN",
            "is_2d": False,
        },
        {
            "name": "LITE-QAT",
            "builder": build_keras_lite,
            "transferer": transfer_lite,
            "pt_path": RESULTS_DIR / "lite_qat_best.pt",
            "tflite_name": "lite_gelu_qat_int8.tflite",
            "pt_class_path": "ai.models.lite.LITE",
            "is_2d": False,
        },
        {
            "name": "NPU-Conv2D-QAT",
            "builder": build_keras_npu_conv2d,
            "transferer": transfer_npu_conv2d,
            "pt_path": RESULTS_DIR / "npu_conv2d_qat_best.pt",
            "tflite_name": "npu_conv2d_relu_qat_int8.tflite",
            "pt_class_path": "ai.models.npu_conv2d.NPUConv2D",
            "is_2d": True,
        },
    ]

    results = []

    for t in targets:
        name = t["name"]
        pt_path = t["pt_path"]

        print(f"\n{'=' * 65}")
        print(f"  {name}")
        print(f"{'=' * 65}")

        if not pt_path.exists():
            print(f"  [SKIP] {pt_path} not found")
            results.append((name, "SKIP"))
            continue

        # Build Keras model
        keras_model = t["builder"]()
        n_params = keras_model.count_params()
        print(f"  Keras params: {n_params:,}")

        # Transfer weights
        sd = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        t["transferer"](keras_model, sd)

        # Verify
        module_path, class_name = t["pt_class_path"].rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        pt_cls = getattr(mod, class_name)
        verify(name, keras_model, pt_cls, pt_path, calib, t["is_2d"])

        # Convert
        tflite_path = tflite_dir / t["tflite_name"]
        ok = convert_to_tflite(name, keras_model, tflite_path, calib, t["is_2d"])
        results.append((name, "PASS" if ok else "FAIL"))

    # Summary
    print(f"\n{'=' * 65}")
    print(f"  Summary")
    print(f"{'=' * 65}")
    for name, status in results:
        print(f"  {name:20s} {status}")
    print(f"\n  Output: {tflite_dir}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
