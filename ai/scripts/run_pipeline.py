"""
End-to-end data pipeline test.
Run from project root: python -m ai.scripts.run_pipeline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.data.loader import load_all_runs
from ai.data.preprocess import preprocess_all
from ai.data.windowing import extract_all_windows
from ai.data.splits import split_window_sets
from ai.data.dataset import ISCDataset, AEDataset
from ai.config import RESULTS_DIR

import numpy as np


def main():
    print("=" * 60)
    print("Phase 2: AI Training Data Pipeline")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/5] Loading simulation data...")
    runs = load_all_runs()

    # Step 2: Preprocess
    print("\n[2/5] Preprocessing (resample + normalize)...")
    runs = preprocess_all(runs)

    # Step 3: Windowing
    print("\n[3/5] Extracting windows...")
    window_sets = extract_all_windows(runs)

    # Step 4: Split
    print("\n[4/5] Splitting train/val/test...")
    train_sets, val_sets, test_sets = split_window_sets(window_sets)

    # Step 5: Create PyTorch datasets
    print("\n[5/5] Creating PyTorch datasets...")
    train_ds = ISCDataset(train_sets, augment=True)
    val_ds = ISCDataset(val_sets, augment=False)
    test_ds = ISCDataset(test_sets, augment=False)
    ae_train_ds = AEDataset(train_sets, augment=True)

    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"  Train:    {len(train_ds)} windows (pos_weight={train_ds.get_class_weights():.2f})")
    print(f"  Val:      {len(val_ds)} windows")
    print(f"  Test:     {len(test_ds)} windows")
    print(f"  AE Train: {len(ae_train_ds)} windows (Normal only)")

    # Verify shapes
    x, y = train_ds[0]
    print(f"\n  Sample shape: x={tuple(x.shape)}, y={y.item():.0f}")
    x_ae, target_ae = ae_train_ds[0]
    print(f"  AE sample:   x={tuple(x_ae.shape)}, target={tuple(target_ae.shape)}")

    # Label distribution per split
    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        n = len(ds)
        n1 = ds.labels.sum().item()
        print(f"  {name:6s}: Normal={n-n1:.0f} ({(n-n1)/n*100:.1f}%), ISC={n1:.0f} ({n1/n*100:.1f}%)")

    print(f"\n{'='*60}")
    print("Data pipeline complete. Ready for model training.")


if __name__ == "__main__":
    main()
