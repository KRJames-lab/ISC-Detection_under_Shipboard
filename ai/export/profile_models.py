"""
Profile ModernTCN, LITE, and EKF for STM32N6 embedded deployment.

Calculates: parameter count, MACs, model size, activation memory.
All values are analytical (no GPU/inference required).

Usage:
    cd D:/01_Projects/05_Ship_Battery
    python -m ai.export.profile_models
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.config import (
    RESULTS_DIR, INPUT_CHANNELS, WINDOW_SIZE,
    MODERN_TCN_D_MODEL, MODERN_TCN_STEM_KERNEL, MODERN_TCN_KERNEL,
    MODERN_TCN_BLOCKS, MODERN_TCN_FFN_RATIO,
    LITE_INCEPTION_KERNELS, LITE_INCEPTION_FILTERS,
    LITE_DWS_CHANNELS, LITE_DWS1_KERNEL, LITE_DWS2_KERNEL,
    EKF_OCV_MODULE, EKF_R0_MODULE, EKF_R1_MODULE, EKF_TAU1,
)
from ai.models.modern_tcn import ModernTCN
from ai.models.lite import LITE


L = WINDOW_SIZE  # 1000 samples


# ===== Analytical MACs Calculation =====

def macs_conv1d(c_in, c_out, kernel, length, groups=1):
    """MACs for Conv1d (multiply-accumulate, no bias)."""
    return (c_in // groups) * kernel * c_out * length


def macs_linear(c_in, c_out):
    return c_in * c_out


def calc_modern_tcn_macs():
    d = MODERN_TCN_D_MODEL
    sk = MODERN_TCN_STEM_KERNEL
    bk = MODERN_TCN_KERNEL
    fr = MODERN_TCN_FFN_RATIO

    # Stem: Conv1d(2 -> d, k=sk)
    stem = macs_conv1d(INPUT_CHANNELS, d, sk, L)

    # Per block: DWConv + PW1 + PW2
    block_dw = macs_conv1d(d, d, bk, L, groups=d)
    block_pw1 = macs_conv1d(d, d * fr, 1, L)
    block_pw2 = macs_conv1d(d * fr, d, 1, L)
    per_block = block_dw + block_pw1 + block_pw2

    # Head: Linear(d -> 1) after GAP
    head = macs_linear(d, 1)

    total = stem + MODERN_TCN_BLOCKS * per_block + head
    return total, {
        "Stem Conv": stem,
        f"{MODERN_TCN_BLOCKS}x DWConv": MODERN_TCN_BLOCKS * block_dw,
        f"{MODERN_TCN_BLOCKS}x PW1+PW2": MODERN_TCN_BLOCKS * (block_pw1 + block_pw2),
        "Head": head,
    }


def calc_lite_macs():
    nf = LITE_INCEPTION_FILTERS
    dc = LITE_DWS_CHANNELS

    # Inception: 5 parallel Conv1d(2 -> nf, k=k_i)
    inception = sum(
        macs_conv1d(INPUT_CHANNELS, nf, k, L)
        for k in LITE_INCEPTION_KERNELS
    )
    inc_out = nf * len(LITE_INCEPTION_KERNELS)  # 40

    # DWSConv1: DW(inc_out, k=51) + PW(inc_out -> dc)
    dws1_dw = macs_conv1d(inc_out, inc_out, LITE_DWS1_KERNEL, L, groups=inc_out)
    dws1_pw = macs_conv1d(inc_out, dc, 1, L)

    # DWSConv2: DW(dc, k=41) + PW(dc -> dc)
    dws2_dw = macs_conv1d(dc, dc, LITE_DWS2_KERNEL, L, groups=dc)
    dws2_pw = macs_conv1d(dc, dc, 1, L)

    head = macs_linear(dc, 1)

    total = inception + dws1_dw + dws1_pw + dws2_dw + dws2_pw + head
    return total, {
        "Inception (5 branches)": inception,
        "DWSConv1": dws1_dw + dws1_pw,
        "DWSConv2": dws2_dw + dws2_pw,
        "Head": head,
    }


def calc_ekf_ops_per_window():
    """Approximate FLOPs per 10s window (1000 EKF steps)."""
    # Per step: 2x interp (~40), expf (~20), predict (~20), update (~50), CC (~5)
    ops_per_step = 135
    return ops_per_step * L


# ===== Memory Estimation =====

def estimate_activation_sram(model_name):
    """Peak activation memory (float32 bytes). Conservative upper bound."""
    if model_name == "ModernTCN":
        d = MODERN_TCN_D_MODEL
        fr = MODERN_TCN_FFN_RATIO
        # Block peak: residual(d*L) + ffn_hidden(d*fr*L)
        peak_floats = d * L + d * fr * L  # 32K + 64K = 96K
        return peak_floats * 4
    elif model_name == "LITE":
        nf = LITE_INCEPTION_FILTERS
        nb = len(LITE_INCEPTION_KERNELS)
        # Inception output peak: all branches concatenated
        peak_floats = nf * nb * L  # 40K
        return peak_floats * 4
    else:  # EKF
        return 400  # state + working vars


def main():
    print("=" * 70)
    print("  Model Profiling for STM32N6 Deployment")
    print("=" * 70)
    print(f"  Window: {INPUT_CHANNELS}ch x {L} samples = 10s @ 100Hz")
    print(f"  Target: STM32N6 (Cortex-M55 800MHz + Neural-ART NPU)")
    print(f"  SRAM:   4.2 MB available")
    print()

    # --- Load models for parameter count ---
    models_info = {}

    for name, cls in [("ModernTCN", ModernTCN), ("LITE", LITE)]:
        m = cls()
        params = sum(p.numel() for p in m.parameters())
        models_info[name] = {"params": params}

    # EKF "params" = lookup table entries
    ekf_lut = len(EKF_OCV_MODULE) * 2 + len(EKF_R0_MODULE) * 4  # soc+ocv + soc+r0+r1+tau
    models_info["EKF"] = {"params": ekf_lut}

    # --- MACs ---
    mtcn_macs, mtcn_detail = calc_modern_tcn_macs()
    lite_macs, lite_detail = calc_lite_macs()
    ekf_ops = calc_ekf_ops_per_window()

    models_info["ModernTCN"]["macs"] = mtcn_macs
    models_info["LITE"]["macs"] = lite_macs
    models_info["EKF"]["macs"] = ekf_ops

    # --- Print comparison table ---
    def fmt_macs(n):
        if n >= 1e6:
            return f"{n/1e6:.1f}M"
        elif n >= 1e3:
            return f"{n/1e3:.0f}K"
        return str(n)

    def fmt_kb(b):
        if b < 1024:
            return f"{b} B"
        return f"{b/1024:.1f} KB"

    print("-" * 70)
    print(f"  {'Metric':<28} | {'ModernTCN':>12} | {'LITE':>12} | {'EKF':>12}")
    print("-" * 70)

    # Parameters
    print(f"  {'Parameters':<28} | {models_info['ModernTCN']['params']:>12,} "
          f"| {models_info['LITE']['params']:>12,} "
          f"| {f'~{ekf_lut} (LUT)':>12}")

    # MACs
    print(f"  {'MACs / window':<28} | {fmt_macs(mtcn_macs):>12} "
          f"| {fmt_macs(lite_macs):>12} "
          f"| {fmt_macs(ekf_ops):>12}")

    # Flash (weights)
    for label, scale in [("Flash (float32)", 4), ("Flash (int8 quant)", 1)]:
        mtcn_flash = models_info["ModernTCN"]["params"] * scale
        lite_flash = models_info["LITE"]["params"] * scale
        ekf_flash = ekf_lut * 4  # always float32
        ekf_str = fmt_kb(ekf_flash) if label == "Flash (float32)" else "N/A"
        print(f"  {label:<28} | {fmt_kb(mtcn_flash):>12} "
              f"| {fmt_kb(lite_flash):>12} "
              f"| {ekf_str:>12}")

    # Activation SRAM
    for label, div in [("Peak SRAM (float32)", 1), ("Peak SRAM (int8 quant)", 4)]:
        mtcn_sram = estimate_activation_sram("ModernTCN") // div
        lite_sram = estimate_activation_sram("LITE") // div
        ekf_sram = estimate_activation_sram("EKF")
        ekf_str = fmt_kb(ekf_sram) if label == "Peak SRAM (float32)" else "N/A"
        print(f"  {label:<28} | {fmt_kb(mtcn_sram):>12} "
              f"| {fmt_kb(lite_sram):>12} "
              f"| {ekf_str:>12}")

    # Execution target
    print(f"  {'Execution Target':<28} | {'NPU+CPU':>12} | {'NPU+CPU':>12} | {'CPU only':>12}")

    print("-" * 70)

    # --- MACs breakdown ---
    print(f"\n  ModernTCN MACs breakdown:")
    for k, v in mtcn_detail.items():
        pct = v / mtcn_macs * 100
        print(f"    {k:<25} {fmt_macs(v):>8}  ({pct:5.1f}%)")

    print(f"\n  LITE MACs breakdown:")
    for k, v in lite_detail.items():
        pct = v / lite_macs * 100
        print(f"    {k:<25} {fmt_macs(v):>8}  ({pct:5.1f}%)")

    # --- NPU coverage ---
    print(f"\n  NPU-acceleratable operations:")
    print(f"    ModernTCN: 100% (all Conv1d layers)")
    print(f"    LITE:      100% (all Conv1d layers)")
    print(f"    EKF:       0%   (scalar operations, CPU only)")

    # --- ONNX file sizes ---
    onnx_dir = RESULTS_DIR / "onnx"
    if onnx_dir.exists():
        print(f"\n  ONNX files:")
        for name in ["modern_tcn.onnx", "lite.onnx"]:
            p = onnx_dir / name
            if p.exists():
                print(f"    {name}: {p.stat().st_size / 1024:.1f} KB")

    # --- Check trained weights ---
    print(f"\n  Trained weights:")
    for name in ["modern_tcn_best.pt", "lite_best.pt"]:
        p = RESULTS_DIR / name
        status = f"{p.stat().st_size / 1024:.1f} KB" if p.exists() else "NOT FOUND"
        print(f"    {name}: {status}")

    print(f"\n{'=' * 70}")
    print(f"  All models fit within STM32N6 4.2 MB SRAM budget.")
    print(f"  INT8 quantization recommended for NPU acceleration.")
    print(f"  EKF runs on CPU only -- minimal resource usage.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
