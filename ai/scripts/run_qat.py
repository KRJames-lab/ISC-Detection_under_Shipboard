"""
Quantization-Aware Training (QAT) for ISC detection models.

Fine-tunes pretrained float32 models with activation fake quantization
(straight-through estimator). Models learn to be robust to int8 rounding,
so subsequent TFLite int8 PTQ produces better accuracy.

Pipeline: pretrained .pt -> QAT fine-tune -> *_qat_best.pt -> convert_tflite_all.py -> eval_int8.py

Usage: python -m ai.scripts.run_qat
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from ai.config import (RANDOM_SEED, RESULTS_DIR, BATCH_SIZE, WEIGHT_DECAY)
from ai.data.loader import load_all_runs
from ai.data.preprocess import preprocess_all
from ai.data.windowing import extract_all_windows
from ai.data.splits import split_window_sets
from ai.data.dataset import ISCDataset
from ai.models.modern_tcn import ModernTCN
from ai.models.lite import LITE
from ai.models.npu_conv2d import NPUConv2D
from ai.training.evaluate import evaluate_supervised

# === QAT Hyperparameters ===
QAT_LR = 1e-5
QAT_EPOCHS = 20
QAT_PATIENCE = 5


# === Fake Quantization via Forward Hooks ===

class ActivationFakeQuantize:
    """Per-tensor asymmetric int8 fake quantization with EMA statistics.

    Attached as a forward hook to Conv/Linear layers.
    Active only during training (model.train()); passes through in eval().
    Uses straight-through estimator (torch.fake_quantize_per_tensor_affine).
    """

    def __init__(self, momentum=0.1):
        self.momentum = momentum
        self.min_val = None
        self.max_val = None

    def __call__(self, module, input, output):
        if not module.training:
            return output

        # Update EMA statistics
        with torch.no_grad():
            batch_min = output.min().item()
            batch_max = output.max().item()
            if self.min_val is None:
                self.min_val = batch_min
                self.max_val = batch_max
            else:
                self.min_val = (1 - self.momentum) * self.min_val + self.momentum * batch_min
                self.max_val = (1 - self.momentum) * self.max_val + self.momentum * batch_max

        # Compute int8 quantization parameters
        scale = max((self.max_val - self.min_val) / 255.0, 1e-8)
        zero_point = int(round(-128 - self.min_val / scale))
        zero_point = max(-128, min(127, zero_point))

        # Straight-through fake quantize
        return torch.fake_quantize_per_tensor_affine(
            output, scale, zero_point, -128, 127
        )


def attach_qat_hooks(model):
    """Attach fake quantization hooks to all Conv and Linear layers."""
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            fq = ActivationFakeQuantize()
            hook = module.register_forward_hook(fq)
            hooks.append(hook)
    return hooks


# === QAT Training Loop ===

def train_qat(model, train_ds, val_ds, model_name, device):
    """QAT fine-tuning with early stopping on validation loss."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULTS_DIR / f"{model_name}_qat_best.pt"

    hooks = attach_qat_hooks(model)
    model.to(device)

    pos_weight = train_ds.get_class_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=QAT_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    patience_counter = 0
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nQAT Fine-tuning {model_name} ({n_params:,} params)")
    print(f"  lr={QAT_LR}, max_epochs={QAT_EPOCHS}, patience={QAT_PATIENCE}")

    for epoch in range(1, QAT_EPOCHS + 1):
        t0 = time.time()

        # Train with fake quantization hooks active
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate without fake quantization (eval mode)
        model.eval()
        val_losses, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                preds = (torch.sigmoid(logits) > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_f1 = _f1_score(all_labels, all_preds)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        if epoch % 2 == 1 or val_loss < best_val_loss:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"F1={val_f1:.4f}, lr={lr_now:.1e} ({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= QAT_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

    # Remove hooks and load best
    for h in hooks:
        h.remove()
    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"  Best val loss: {best_val_loss:.4f}")
    return model


def _f1_score(labels, preds):
    labels, preds = np.array(labels), np.array(preds)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# === Main ===

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Data pipeline (same as run_training.py)
    print("=" * 60)
    print("Loading and preprocessing data...")
    runs = load_all_runs()
    runs = preprocess_all(runs)
    window_sets = extract_all_windows(runs)
    train_sets, val_sets, test_sets = split_window_sets(window_sets)

    train_ds = ISCDataset(train_sets, augment=True)
    val_ds = ISCDataset(val_sets, augment=False)
    test_ds = ISCDataset(test_sets, augment=False)

    # Models: (save_name, class, pretrained_file)
    models = [
        ("moderntcn", ModernTCN, "moderntcn_best.pt"),
        ("lite", LITE, "lite_best.pt"),
        ("npu_conv2d", NPUConv2D, "npu_conv2d_best.pt"),
    ]

    trained = {}
    for save_name, cls, pt_file in models:
        print(f"\n{'=' * 60}")
        pt_path = RESULTS_DIR / pt_file
        if not pt_path.exists():
            print(f"  SKIP: {pt_path} not found")
            continue

        model = cls()
        print(f"Loading pretrained: {pt_file}")
        model.load_state_dict(torch.load(pt_path, weights_only=True))
        model = train_qat(model, train_ds, val_ds, save_name, device)
        trained[save_name] = model

    # Quick test evaluation (float32, no fake quant)
    if trained:
        print(f"\n{'=' * 60}")
        print("QAT MODEL TEST EVALUATION (float32)")
        print("=" * 60)
        for name, model in trained.items():
            evaluate_supervised(model, test_ds, f"{name}_qat", device)

    print(f"\n{'=' * 60}")
    print("QAT complete. Next steps:")
    print("  1. python -m ai.export.convert_tflite_all   (add QAT weights)")
    print("  2. python -m ai.scripts.eval_int8            (evaluate QAT int8)")


if __name__ == "__main__":
    main()
