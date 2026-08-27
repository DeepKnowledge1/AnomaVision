"""Native AnomaVision implementation of EfficientAD."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class _FeatureTeacher(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = efficientnet_b0(weights=weights)
        self.features = nn.Sequential(*list(net.features[:6]))
        self.out_channels = 112
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.features(x)


class _Student(nn.Module):
    def __init__(self, out_channels: int = 112) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 112, 3, stride=2, padding=1),
            nn.BatchNorm2d(112), nn.ReLU(inplace=True),
            nn.Conv2d(112, out_channels, 3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class EfficientAD(nn.Module):
    """EfficientAD-compatible anomaly detector for the AnomaVision pipeline."""

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        model_size: str = "s",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        pretrained_teacher: bool = True,
        teacher_weights: Optional[str] = None,
        feature_weight: float = 1.0,
        reconstruction_weight: float = 0.1,
    ) -> None:
        super().__init__()
        model_size = str(model_size).lower()
        if model_size not in {"s", "m", "small", "medium"}:
            raise ValueError("EfficientAD model_size must be one of: s, m")
        if lr <= 0 or weight_decay < 0:
            raise ValueError("lr must be > 0 and weight_decay must be >= 0")

        self.device = torch.device(device)
        self.model_size = "m" if model_size in {"m", "medium"} else "s"
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.feature_weight = float(feature_weight)
        self.reconstruction_weight = float(reconstruction_weight)

        self.teacher = _FeatureTeacher(pretrained=pretrained_teacher)
        if teacher_weights:
            state = torch.load(teacher_weights, map_location="cpu", weights_only=False)
            self.teacher.load_state_dict(state, strict=False)
        self.student = _Student(self.teacher.out_channels)
        self.autoencoder = _AutoEncoder()
        self.register_buffer("score_mean", torch.tensor(0.0))
        self.register_buffer("score_std", torch.tensor(1.0))
        self.register_buffer("trained", torch.tensor(False, dtype=torch.bool))
        self.to(self.device)

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _signals(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self._normalise(x)
        teacher = self.teacher(x_norm)
        student = self.student(x_norm)
        feature_map = (student - teacher).pow(2).mean(dim=1)
        reconstruction = (self.autoencoder(x) - x).abs().mean(dim=1)
        feature_map = F.interpolate(
            feature_map.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        return feature_map, reconstruction

    def forward(self, x: torch.Tensor, return_map: bool = True, export: bool = False):
        feature_map, reconstruction = self._signals(x)
        score_map = feature_map + self.reconstruction_weight * reconstruction
        scores = score_map.flatten(1).amax(1)
        scores = (scores - self.score_mean) / self.score_std.clamp_min(1e-6)
        return scores, score_map if return_map else None

    def fit(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1) -> None:
        self.train()
        self.teacher.eval()
        optimizer = torch.optim.Adam(
            list(self.student.parameters()) + list(self.autoencoder.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        for _ in range(int(epochs)):
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(self.device, non_blocking=True).float()
                with torch.no_grad():
                    teacher = self.teacher(self._normalise(batch))
                student = self.student(self._normalise(batch))
                reconstructed = self.autoencoder(batch)
                feature_loss = F.mse_loss(student, teacher)
                reconstruction_loss = F.l1_loss(reconstructed, batch)
                loss = self.feature_weight * feature_loss + self.reconstruction_weight * reconstruction_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self.eval()
        values = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(self.device).float()
                fmap, recon = self._signals(batch)
                values.append((fmap + self.reconstruction_weight * recon).flatten(1).amax(1))
        if values:
            scores = torch.cat(values)
            self.score_mean.copy_(scores.mean())
            self.score_std.copy_(scores.std(unbiased=False).clamp_min(1e-6))
        self.trained.fill_(True)

    def predict(self, batch: torch.Tensor, export: bool = False):
        # Do not inspect the tensor-backed ``trained`` flag while exporting.
        # torch.export treats ``trained.item()`` as data-dependent control flow
        # and cannot specialize that condition. Training validation belongs to
        # the Python lifecycle, while the exported graph must contain only the
        # tensor computation.
        if not export and not bool(self.trained.item()):
            raise RuntimeError("EfficientAD model is not trained. Call fit() first.")
        self.eval()
        with torch.no_grad():
            return self.forward(batch.to(self.device).float(), export=export)

    def to_device(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.to(self.device)

    def save_statistics(self, path: str, half: Optional[bool] = None) -> None:
        if not bool(self.trained.item()):
            raise RuntimeError("Model is not trained. Call fit() first.")
        torch.save({
            "algorithm": "efficientad", "model_state": self.state_dict(),
            "model_size": self.model_size, "lr": self.lr,
            "weight_decay": self.weight_decay,
        }, path)

    @staticmethod
    def load_statistics(path: str, device: str = "cpu") -> "EfficientAD":
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("algorithm") != "efficientad":
            raise ValueError("Not an EfficientAD statistics artifact")
        model = EfficientAD(
            device=torch.device(device), model_size=data.get("model_size", "s"),
            pretrained_teacher=False,
        )
        model.load_state_dict(data["model_state"])
        return model
