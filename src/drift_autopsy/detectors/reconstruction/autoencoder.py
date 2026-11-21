"""Convolutional autoencoder and drift simulation for Chest X-ray images."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import functional as TF


class ChestXrayAutoencoder(nn.Module):
    """Lightweight convolutional autoencoder with a 32-D latent bottleneck."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return a 32-D latent vector per image."""
        feat = self.encoder(x)
        pooled = self.pool(feat)
        return pooled.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.decoder(feat)


def simulate_drift(
    images: torch.Tensor,
    brightness: float = 1.0,
    contrast: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> torch.Tensor:
    """Apply brightness, contrast, and additive Gaussian noise clamped to [0, 1]."""
    out = images.clone()
    if brightness != 1.0:
        out = TF.adjust_brightness(out, brightness)
    if contrast != 1.0:
        out = TF.adjust_contrast(out, contrast)
    if noise_std > 0.0:
        gen = torch.Generator(device=out.device).manual_seed(seed)
        noise = torch.randn(out.shape, generator=gen, device=out.device, dtype=out.dtype) * noise_std
        out = out + noise
    return torch.clamp(out, 0.0, 1.0)


def train_autoencoder(
    model: ChestXrayAutoencoder,
    train_loader: DataLoader,
    epochs: int = 10,
    device: str | torch.device = "cpu",
) -> None:
    """Train the autoencoder with Adam (lr=1e-3) and MSELoss."""
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def extract_embeddings(
    model: ChestXrayAutoencoder,
    images: torch.Tensor,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Extract (N, 32) latent embeddings from the autoencoder."""
    model.to(device)
    model.eval()
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    chunks = []
    for (x,) in loader:
        z = model.encode(x.to(device))
        chunks.append(z.cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def reconstruction_errors(
    model: ChestXrayAutoencoder,
    images: torch.Tensor,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Compute per-image MSE reconstruction errors."""
    model.to(device)
    model.eval()
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    errors = []
    for (x,) in loader:
        x_dev = x.to(device)
        recon = model(x_dev)
        mse = ((recon - x_dev) ** 2).mean(dim=(1, 2, 3))
        errors.append(mse.cpu().numpy())
    return np.concatenate(errors, axis=0)
