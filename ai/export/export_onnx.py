"""
Export ModernTCN and LITE to ONNX for STM32Cube.AI deployment.

Usage:
    cd D:/01_Projects/05_Ship_Battery
    python -m ai.export.export_onnx

Output:
    ai/results/onnx/modern_tcn.onnx
    ai/results/onnx/lite.onnx

Next step (STM32Cube.AI CLI):
    stedgeai generate -m ai/results/onnx/modern_tcn.onnx --target stm32n6 --compression none
    stedgeai generate -m ai/results/onnx/lite.onnx --target stm32n6 --compression none
    (Add --GELU-recognition for NPU hardware GELU acceleration)
"""
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.config import RESULTS_DIR, INPUT_CHANNELS, WINDOW_SIZE
from ai.models.modern_tcn import ModernTCN
from ai.models.lite import LITE


class ExportWrapper(torch.nn.Module):
    """Wraps any model to output (B, 1) instead of (B,).
    Replaces squeeze(-1) with a clean 2D output for stedgeai/NPU compatibility."""
    def __init__(self, model):
        super().__init__()
        self._model = model
    def forward(self, x):
        # Replicate forward path up to head, skipping squeeze(-1)
        from ai.models.npu_conv2d import NPUConv2D, RESHAPE_H, RESHAPE_W
        if isinstance(self._model, NPUConv2D):
            # NPUConv2D: reshape 1D→2D before stem
            x = x.view(x.shape[0], 2, RESHAPE_H, RESHAPE_W)
            x = self._model.stem(x)
            x = self._model.blocks(x)
        elif hasattr(self._model, 'stem'):
            # ModernTCN (1D model)
            x = self._model.stem(x)
            x = self._model.blocks(x)
        elif hasattr(self._model, 'inception'):
            # LITE
            x = self._model.inception(x)
            x = self._model.dws1(x)
            x = self._model.dws2(x)
        return self._model.head(x)  # (B, 1) — clean 2D, no squeeze


def export_model(model_class, weight_path, onnx_path, model_name):
    """Export a single model to ONNX and validate output consistency."""
    print(f"\n--- {model_name} ---")

    # Load trained weights
    model = model_class()
    state_dict = torch.load(str(weight_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    # Wrap to output (B, 1) without Squeeze node for NPU compatibility
    if hasattr(model, 'stem') and hasattr(model, 'blocks') and hasattr(model, 'head'):
        model = ExportWrapper(model)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:  {params:,}")

    # Fixed input shape for embedded (no dynamic axes)
    dummy = torch.randn(1, INPUT_CHANNELS, WINDOW_SIZE)

    # PyTorch reference output
    with torch.no_grad():
        pt_out = model(dummy).numpy()

    # --- ONNX Export ---
    # opset 17: well-supported by STM32Cube.AI
    # GELU is decomposed to Erf-based formula; stedgeai --GELU-recognition fuses it
    # dynamo=False: use legacy TorchScript exporter (no onnxscript dependency)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes=None,
        dynamo=False,
    )

    file_kb = onnx_path.stat().st_size / 1024
    print(f"  ONNX size:   {file_kb:.1f} KB")
    print(f"  Input shape: (1, {INPUT_CHANNELS}, {WINDOW_SIZE})")
    print(f"  Output:      logit (>0 = ISC detected)")
    print(f"  Saved:       {onnx_path}")

    # --- Validate with ONNX Runtime ---
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(None, {"input": dummy.numpy()})[0]
        diff = float(np.max(np.abs(pt_out - ort_out)))
        ok = diff < 1e-4
        print(f"  ORT check:   max_diff={diff:.2e} -> {'PASS' if ok else 'FAIL'}")
        return ok
    except ImportError:
        print("  ORT check:   skipped (pip install onnxruntime)")
        return True


def main():
    out_dir = RESULTS_DIR / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  ONNX Export for STM32Cube.AI (STM32N6)")
    print("=" * 55)
    print(f"  Input:  ({INPUT_CHANNELS} channels, {WINDOW_SIZE} samples)")
    print(f"  Rate:   100 Hz -> 10s window")
    print(f"  Output: single logit (sigmoid threshold = 0)")

    targets = [
        (ModernTCN, "modern_tcn_relu_best.pt", "modern_tcn_relu.onnx", "ModernTCN"),
        (LITE, "lite_relu_best.pt", "lite_relu.onnx", "LITE"),
    ]

    all_ok = True
    for cls, wt_name, ox_name, name in targets:
        wt = RESULTS_DIR / wt_name
        if not wt.exists():
            print(f"\n  [SKIP] {name}: {wt} not found")
            continue
        ok = export_model(cls, wt, out_dir / ox_name, name)
        all_ok = all_ok and ok

    # Summary
    print(f"\n{'=' * 55}")
    if all_ok:
        print("  All exports PASS")
    else:
        print("  WARNING: Some validations FAILED")
    print()
    print("  Next steps:")
    print(f"  1. stedgeai generate \\")
    print(f"       -m {out_dir / 'modern_tcn.onnx'} \\")
    print(f"       --target stm32n6 --compression none \\")
    print(f"       --GELU-recognition")
    print(f"  2. stedgeai generate \\")
    print(f"       -m {out_dir / 'lite.onnx'} \\")
    print(f"       --target stm32n6 --compression none \\")
    print(f"       --GELU-recognition")
    print(f"  3. Copy generated network files into STM32CubeIDE project")
    print(f"  4. Add embedded/ekf.c + ekf.h for EKF comparison")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
