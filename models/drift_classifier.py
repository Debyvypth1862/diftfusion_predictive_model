"""
Drift-Type Classifier (Algorithm 1, Step 6): a single-layer LSTM encodes
the fingerprint sequence, its final hidden state is passed to an MLP
head, softmax over the five drift categories. Meta-trained on synthetic
fingerprint sequences.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fingerprint import FINGERPRINT_DIM
from ..synthetic.fingerprint_sequences import CATEGORIES, build_synthetic_dataset

N_CATEGORIES = len(CATEGORIES)


class DriftTypeClassifier(nn.Module):
    def __init__(self, input_dim: int = FINGERPRINT_DIM, hidden_dim: int = 32,
                 n_layers: int = 1, n_classes: int = N_CATEGORIES):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (batch, seq_len, input_dim) -> logits (batch, n_classes)."""
        _, (h_n, _) = self.lstm(seq)        # h_n: (n_layers, batch, hidden_dim)
        last_hidden = h_n[-1]               # final layer's hidden state at the last step
        return self.mlp(last_hidden)

    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> tuple[str, np.ndarray]:
        self.eval()
        x = torch.from_numpy(sequence).float().unsqueeze(0)
        probs = F.softmax(self.forward(x), dim=-1).squeeze(0).numpy()
        return CATEGORIES[int(np.argmax(probs))], probs


def train_drift_classifier(n_per_class: int = 2000, epochs: int = 30,
                            batch_size: int = 64, lr: float = 1e-3,
                            seed: int = 42, verbose: bool = True) -> DriftTypeClassifier:
    torch.manual_seed(seed)
    X, y = build_synthetic_dataset(n_per_class=n_per_class, seed=seed)
    n_val = int(0.1 * len(X))
    X_train, y_train = X[n_val:], y[n_val:]
    X_val, y_val = X[:n_val], y[:n_val]

    model = DriftTypeClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    n = len(X_train_t)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_train_t[idx])
            loss = F.cross_entropy(logits, y_train_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(idx)

        if verbose and (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_acc = (model(X_val_t).argmax(-1) == y_val_t).float().mean().item()
            print(f"  [drift-classifier] epoch {epoch+1}/{epochs} "
                  f"train_loss={total_loss/n:.4f} val_acc={val_acc:.4f}")

    return model


if __name__ == "__main__":
    train_drift_classifier(n_per_class=500, epochs=20)
