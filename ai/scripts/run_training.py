"""
Train 2 AI models: ModernTCN, LITE.
Run from project root: python -m ai.scripts.run_training
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai.config import RANDOM_SEED, RESULTS_DIR, SCENARIO_NAMES, ABLATION_V_ONLY, ABLATION_NOVIB_TRAIN
from ai.data.loader import load_all_runs
from ai.data.preprocess import preprocess_all
from ai.data.windowing import extract_all_windows
from ai.data.splits import split_window_sets
from ai.data.dataset import ISCDataset
from ai.models.modern_tcn import ModernTCN
from ai.models.lite import LITE
from ai.training.train_supervised import train_model
from ai.training.evaluate import evaluate_supervised


def plot_training_curves(histories, save_path):
    """Plot training curves for all models."""
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (name, hist) in enumerate(histories.items()):
        epochs = range(1, len(hist["train_loss"]) + 1)

        ax = axes[0, col]
        ax.plot(epochs, hist["train_loss"], label="Train", linewidth=1.5)
        ax.plot(epochs, hist["val_loss"], label="Val", linewidth=1.5)
        ax.set_title(f"{name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.plot(epochs, hist["val_f1"], color="green", linewidth=1.5)
        best_idx = int(np.argmax(hist["val_f1"]))
        best_f1 = hist["val_f1"][best_idx]
        ax.axhline(y=best_f1, color="red", linestyle="--", alpha=0.5)
        ax.annotate(f"Best: {best_f1:.4f} (ep {best_idx+1})",
                    xy=(best_idx + 1, best_f1), fontsize=9,
                    xytext=(5, -15), textcoords="offset points", color="red")
        ax.set_ylabel("Val F1")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.8, 1.01)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {save_path}")


def evaluate_breakdown(model, model_name, test_sets, device):
    """Per-scenario and per-k_R evaluation on test set."""
    print(f"\n  [{model_name}] Breakdown")

    def calc(labels, preds):
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return tp, fp, fn, tn, prec, rec, f1

    def predict(windows):
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(windows).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            return (probs > 0.5).long().cpu().numpy()

    # By k_R
    print(f"  {'k_R(mOhm/g)':<14} {'Norm':>5} {'ISC':>5} | {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} | {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-'*74}")
    for kr in sorted(set(ws.k_R_mOhm for ws in test_sets)):
        sub = [ws for ws in test_sets if ws.k_R_mOhm == kr]
        w = np.concatenate([ws.windows for ws in sub])
        l = np.concatenate([ws.labels for ws in sub])
        p = predict(w)
        tp, fp, fn, tn, prec, rec, f1 = calc(l, p)
        n0, n1 = int((l == 0).sum()), int((l == 1).sum())
        print(f"  {kr:<14} {n0:>5} {n1:>5} | {tp:>5} {fp:>5} {fn:>5} {tn:>5} | {prec:>6.4f} {rec:>6.4f} {f1:>6.4f}")

    # By scenario
    print(f"\n  {'Scenario':<14} {'Norm':>5} {'ISC':>5} | {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} | {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-'*74}")
    for sc in sorted(set(ws.scenario for ws in test_sets)):
        sub = [ws for ws in test_sets if ws.scenario == sc]
        w = np.concatenate([ws.windows for ws in sub])
        l = np.concatenate([ws.labels for ws in sub])
        p = predict(w)
        tp, fp, fn, tn, prec, rec, f1 = calc(l, p)
        n0, n1 = int((l == 0).sum()), int((l == 1).sum())
        nm = SCENARIO_NAMES.get(sc, str(sc))
        print(f"  {nm:<14} {n0:>5} {n1:>5} | {tp:>5} {fp:>5} {fn:>5} {tn:>5} | {prec:>6.4f} {rec:>6.4f} {f1:>6.4f}")

    # Scenario x k_R (F1 matrix)
    krs = sorted(set(ws.k_R_mOhm for ws in test_sets))
    scs = sorted(set(ws.scenario for ws in test_sets))
    print(f"\n  F1 matrix (Scenario x k_R):")
    hdr = f"  {'':14}" + "".join(f"{'k=' + str(k):>10}" for k in krs)
    print(hdr)
    print(f"  {'-'*64}")
    for sc in scs:
        row = f"  {SCENARIO_NAMES.get(sc, str(sc)):14}"
        for kr in krs:
            ms = [ws for ws in test_sets if ws.scenario == sc and ws.k_R_mOhm == kr]
            if ms:
                w = np.concatenate([x.windows for x in ms])
                l = np.concatenate([x.labels for x in ms])
                p = predict(w)
                _, _, _, _, _, _, f1 = calc(l, p)
                row += f"{f1:>10.4f}"
            else:
                row += f"{'N/A':>10}"
        print(row)


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Data pipeline
    print("=" * 60)
    print("Loading and preprocessing data...")
    runs = load_all_runs()
    runs = preprocess_all(runs)
    window_sets = extract_all_windows(runs)
    train_sets, val_sets, test_sets = split_window_sets(window_sets)

    if ABLATION_NOVIB_TRAIN:
        # NoVib-train ablation: restrict train/val to scenario==1 (NoVib);
        # test_sets unchanged to evaluate generalisation to vibration scenarios.
        before_train = len(train_sets)
        before_val = len(val_sets)
        train_sets = [ws for ws in train_sets if ws.scenario == 1]
        val_sets = [ws for ws in val_sets if ws.scenario == 1]
        print(f"\n[ABLATION_NOVIB_TRAIN] train_sets {before_train} -> {len(train_sets)} (NoVib only)")
        print(f"[ABLATION_NOVIB_TRAIN] val_sets   {before_val} -> {len(val_sets)} (NoVib only)")
        print(f"[ABLATION_NOVIB_TRAIN] test_sets unchanged ({len(test_sets)} runs, all scenarios)")

    # Datasets
    train_ds = ISCDataset(train_sets, augment=True)
    val_ds = ISCDataset(val_sets, augment=False)
    test_ds = ISCDataset(test_sets, augment=False)

    # Model summaries
    models_to_train = [
        ("ModernTCN", ModernTCN, 3e-4),
        ("LITE", LITE, 1e-3),
    ]
    print("\n" + "=" * 60)
    print("Model Architectures:")
    for name, cls, _ in models_to_train:
        m = cls()
        print(f"  {name}: {m.count_params()} params ({m.count_params()*4/1024:.1f} KB)")

    histories = {}
    trained_models = {}

    suffix = ""
    if ABLATION_V_ONLY:
        suffix += "_v_only"
    if ABLATION_NOVIB_TRAIN:
        suffix += "_novib_train"
    for name, cls, lr in models_to_train:
        print("\n" + "=" * 60)
        model = cls()
        save_name = name.lower().replace("-", "_") + suffix
        model, hist = train_model(model, train_ds, val_ds, save_name, device, lr=lr)
        histories[name] = hist
        trained_models[name] = model

    # Plot training curves
    print("\n" + "=" * 60)
    plot_training_curves(histories, RESULTS_DIR / f"training_curves{suffix}.png")

    # === Test Set Evaluation ===
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (tau=1800, all k_R, all scenarios)")
    print("=" * 60)
    print(f"  Total: {len(test_ds)} windows "
          f"(Normal={int((test_ds.labels == 0).sum())}, "
          f"ISC={int((test_ds.labels == 1).sum())})")

    # Overall
    for name, model in trained_models.items():
        evaluate_supervised(model, test_ds, name, device)

    # Breakdown
    print("\n" + "=" * 60)
    print("DETAILED BREAKDOWN")
    print("=" * 60)
    for name, model in trained_models.items():
        model.to(device)
        evaluate_breakdown(model, name, test_sets, device)

    print("\n" + "=" * 60)
    print("All results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
