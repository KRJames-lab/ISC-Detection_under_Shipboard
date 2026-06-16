"""
Supervised training loop for binary classification (ModernTCN, LITE).
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from ai.config import BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, RESULTS_DIR


def train_model(model, train_ds, val_ds, model_name: str, device: str = "cpu", lr: float = None):
    """Train a supervised model with early stopping on val loss."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lr = lr or LEARNING_RATE

    pos_weight = train_ds.get_class_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model.to(device)
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    print(f"\nTraining {model_name} ({model.count_params()} params)")
    print(f"  pos_weight={pos_weight.item():.2f}, lr={lr}, device={device}")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        # Train
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

        # Validate
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

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if epoch % 5 == 1 or val_loss < best_val_loss:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_F1={val_f1:.4f} ({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), RESULTS_DIR / f"{model_name}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val_loss={best_val_loss:.4f})")
                break

    # Load best model
    model.load_state_dict(torch.load(RESULTS_DIR / f"{model_name}_best.pt", weights_only=True))
    print(f"  Best val loss: {best_val_loss:.4f}")

    return model, history


def _f1_score(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
