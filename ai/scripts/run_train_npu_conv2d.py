"""
Train NPU-Conv2D model only.
Run from project root: python -m ai.scripts.run_train_npu_conv2d
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

from ai.config import RANDOM_SEED, RESULTS_DIR, SCENARIO_NAMES
from ai.data.loader import load_all_runs
from ai.data.preprocess import preprocess_all
from ai.data.windowing import extract_all_windows
from ai.data.splits import split_window_sets
from ai.data.dataset import ISCDataset
from ai.models.npu_conv2d import NPUConv2D
from ai.training.train_supervised import train_model
from ai.training.evaluate import evaluate_supervised


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("=" * 60)
    print("Loading and preprocessing data...")
    runs = load_all_runs()
    runs = preprocess_all(runs)
    window_sets = extract_all_windows(runs)
    train_sets, val_sets, test_sets = split_window_sets(window_sets)

    train_ds = ISCDataset(train_sets, augment=True)
    val_ds = ISCDataset(val_sets, augment=False)
    test_ds = ISCDataset(test_sets, augment=False)

    model = NPUConv2D()
    print(f"\nNPU-Conv2D: {model.count_params():,} params ({model.count_params()*4/1024:.1f} KB)")

    print("\n" + "=" * 60)
    model, hist = train_model(model, train_ds, val_ds, "npu_conv2d", device, lr=1e-3)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (tau=1800, all k_R, all scenarios)")
    print(f"  Total: {len(test_ds)} windows "
          f"(Normal={int((test_ds.labels == 0).sum())}, "
          f"ISC={int((test_ds.labels == 1).sum())})")
    evaluate_supervised(model, test_ds, "NPU-Conv2D", device)

    print(f"\nModel saved: {RESULTS_DIR / 'npu_conv2d_best.pt'}")


if __name__ == "__main__":
    main()
