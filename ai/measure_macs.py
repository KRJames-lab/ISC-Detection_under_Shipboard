"""
Measure MACs for all detector models (ModernTCN, LITE, NPU-Conv2D)
using PyTorch forward hooks. Output: per-layer breakdown + total.
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ai.models.modern_tcn import ModernTCN
from ai.models.lite import LITE
from ai.models.npu_conv2d import NPUConv2D


def count_macs(model, input_shape, name=""):
    layer_macs = []
    hooks = []

    def make_conv1d_hook(mod_name):
        def hook(module, inp, out):
            B, C_out, L_out = out.shape
            C_in = inp[0].shape[1]
            k = module.kernel_size[0]
            groups = module.groups
            m = C_out * L_out * (C_in // groups) * k
            layer_macs.append((mod_name, 'Conv1d',
                              f'in={C_in},out={C_out},k={k},L={L_out},g={groups}', m))
        return hook

    def make_conv2d_hook(mod_name):
        def hook(module, inp, out):
            B, C_out, H_out, W_out = out.shape
            C_in = inp[0].shape[1]
            kH, kW = module.kernel_size
            groups = module.groups
            m = C_out * H_out * W_out * (C_in // groups) * kH * kW
            layer_macs.append((mod_name, 'Conv2d',
                              f'in={C_in},out={C_out},k={kH}x{kW},HxW={H_out}x{W_out},g={groups}', m))
        return hook

    def make_linear_hook(mod_name):
        def hook(module, inp, out):
            m = module.in_features * module.out_features
            layer_macs.append((mod_name, 'Linear',
                              f'in={module.in_features},out={module.out_features}', m))
        return hook

    for nm, mod in model.named_modules():
        if isinstance(mod, nn.Conv1d):
            hooks.append(mod.register_forward_hook(make_conv1d_hook(nm)))
        elif isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(make_conv2d_hook(nm)))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(make_linear_hook(nm)))

    model.eval()
    with torch.no_grad():
        x = torch.randn(*input_shape)
        _ = model(x)

    for h in hooks:
        h.remove()

    total = sum(m for *_, m in layer_macs)
    return total, layer_macs


def report(name, model, shape):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total, breakdown = count_macs(model, shape, name)
    print(f"\n{'='*78}")
    print(f"  {name}")
    print(f"{'='*78}")
    print(f"  Input shape: {shape}")
    print(f"  Parameters:  {n_params:,}")
    print(f"  Total MACs:  {total:,}  (~{total/1e6:.3f} M)")
    print(f"  ---- layer breakdown ----")
    for nm, kind, info, m in breakdown:
        print(f"    {nm:30s} {kind:7s} {info:50s} {m:>15,}")


if __name__ == '__main__':
    report('ModernTCN', ModernTCN(), (1, 2, 1000))
    report('LITE', LITE(), (1, 2, 1000))
    report('NPU-Conv2D', NPUConv2D(), (1, 2, 1000))
