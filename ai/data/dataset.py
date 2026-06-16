"""
PyTorch Datasets for supervised (LSTM/CNN) and unsupervised (Autoencoder) training.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
from ai.data.windowing import WindowSet


class ISCDataset(Dataset):
    """Supervised dataset for LSTM/CNN binary classification."""

    def __init__(self, window_sets: List[WindowSet], augment: bool = False):
        all_windows = []
        all_labels = []
        for ws in window_sets:
            all_windows.append(ws.windows)
            all_labels.append(ws.labels)

        self.windows = torch.FloatTensor(np.concatenate(all_windows, axis=0))
        self.labels = torch.FloatTensor(np.concatenate(all_labels, axis=0))
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.windows[idx]  # (2, W)
        y = self.labels[idx]

        if self.augment:
            x = self._augment(x)

        return x, y

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        # Gaussian noise (std = 5% of signal std)
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(x) * 0.05
            x = x + noise
        # Amplitude scaling (±2%)
        if torch.rand(1) < 0.5:
            scale = 1.0 + (torch.rand(1) - 0.5) * 0.04
            x = x * scale
        return x

    def get_class_weights(self) -> torch.Tensor:
        """Compute pos_weight for BCEWithLogitsLoss."""
        n_pos = self.labels.sum()
        n_neg = len(self.labels) - n_pos
        if n_pos == 0:
            return torch.tensor(1.0)
        return n_neg / n_pos


class AEDataset(Dataset):
    """Autoencoder dataset — Normal windows only, input=target."""

    def __init__(self, window_sets: List[WindowSet], augment: bool = False):
        normal_windows = []
        for ws in window_sets:
            mask = ws.labels == 0
            if mask.any():
                normal_windows.append(ws.windows[mask])

        if not normal_windows:
            raise ValueError("No Normal windows found for AE training")

        self.windows = torch.FloatTensor(np.concatenate(normal_windows, axis=0))
        self.augment = augment

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        if self.augment:
            x = ISCDataset._augment(x)
        return x, x  # input = target
