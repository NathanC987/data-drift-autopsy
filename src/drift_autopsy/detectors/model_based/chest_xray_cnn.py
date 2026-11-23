"""CNN classifier for Chest X-ray binary classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ChestXrayClassifier(nn.Module):
    """Lightweight CNN for binary Chest X-ray classification (No Finding vs Disease)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def train_classifier(
    model: ChestXrayClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 8,
    device: str | torch.device = "cpu",
) -> float:
    """Train classifier with BCEWithLogitsLoss and Adam (lr=1e-3), returning val accuracy."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            preds = (torch.sigmoid(model(x)).squeeze(1) >= 0.5).long()
            correct += int((preds == y.long()).sum().item())
            total += len(y)

    return float(correct / total) if total > 0 else 0.0
