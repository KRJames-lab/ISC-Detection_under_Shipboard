"""
Per-k_R window-level F1 under MIL-STD vibration (test set, tau=1800 s).
Recomputes ModernTCN/LITE for sanity check and adds NPU-Conv2D.

Usage: python -m ai.scripts.run_per_kr_f1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json

from ai.config import RANDOM_SEED, RESULTS_DIR, ABLATION_V_ONLY, ABLATION_NOVIB_TRAIN
from ai.data.loader import load_all_runs, SimRun
from ai.data.preprocess import preprocess_all
from ai.data.windowing import extract_all_windows
from ai.data.splits import split_window_sets
from ai.models.modern_tcn import ModernTCN
from ai.models.lite import LITE
from ai.models.npu_conv2d import NPUConv2D


MIL_STD_SCENARIO = 2    # from config: SCENARIOS = [1, 2, 10]


def _f1(preds, labels):
    p, y = np.array(preds), np.array(labels)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def eval_model_per_kr(model, test_sets, device, scenario=MIL_STD_SCENARIO):
    """Group MIL-STD τ=1800 test windows by k_R_mOhm and compute window-level F1."""
    model.eval()
    model.to(device)
    mil_sets = [ws for ws in test_sets if ws.scenario == scenario]
    kr_groups = {}
    for ws in mil_sets:
        kr_groups.setdefault(ws.k_R_mOhm, []).append(ws)

    out = {}
    for kr in sorted(kr_groups.keys()):
        preds_all, y_all = [], []
        for ws in kr_groups[kr]:
            with torch.no_grad():
                x = torch.FloatTensor(ws.windows).to(device)
                preds = (torch.sigmoid(model(x)) > 0.5).long().cpu().numpy()
            preds_all.extend(preds)
            y_all.extend(ws.labels)
        out[float(kr)] = _f1(preds_all, y_all)
    return out


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading data...")
    all_raw = load_all_runs()
    all_ai = preprocess_all([SimRun(**r.__dict__) for r in all_raw])
    ws = extract_all_windows(all_ai)
    _, _, test_sets = split_window_sets(ws)
    print(f"  test_sets total = {len(test_sets)} (tau=1800 only)")

    suffix = ""
    if ABLATION_V_ONLY:
        suffix += "_v_only"
    if ABLATION_NOVIB_TRAIN:
        suffix += "_novib_train"
    models_info = [
        ("ModernTCN",  ModernTCN(),  f"moderntcn{suffix}_best.pt"),
        ("LITE",       LITE(),       f"lite{suffix}_best.pt"),
    ]
    if not ABLATION_V_ONLY:
        models_info.append(("NPU-Conv2D", NPUConv2D(), "npu_conv2d_best.pt"))
    results = {}
    print("\n{:<12}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}".format(
        "Model", "k=0.05", "k=0.10", "k=0.20", "k=0.50", "k=1.00"))
    print("-" * 65)
    for name, model, pt in models_info:
        model.load_state_dict(torch.load(
            RESULTS_DIR / pt, map_location=device, weights_only=True))
        f1_by_kr = eval_model_per_kr(model, test_sets, device)
        results[name] = f1_by_kr
        kr_vals = sorted(f1_by_kr.keys())
        row = f"{name:<12}  " + "  ".join(f"{f1_by_kr[kr]:>7.4f}" for kr in kr_vals)
        print(row)

    out_json = Path(__file__).parent.parent.parent / "data" / f"per_kr_f1{suffix}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
