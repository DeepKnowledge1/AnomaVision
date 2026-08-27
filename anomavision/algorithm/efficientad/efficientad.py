"""EfficientAD implementation integrated with the AnomaVision algorithm API.

The implementation follows the PaDiM/PatchCore fit/predict contract.
"""

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
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class _Student(nn.Module):
    def __init__(self, out_channels: int = 112) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 112, 3, stride=2, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True),
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
    def __init__(self, device: torch.device = torch.device("cpu"), model_size: str = "s",
                 lr: float = 1e-4, weight_decay: float = 1e-5,
                 pretrained_teacher: bool = True, teacher_weights: Optional[str] = None,
                 feature_weight: float = 1.0, reconstruction_weight: float = 0.1,
                 threshold_quantile: float = 0.995) -> None:
        super().__init__()
        model_size = str(model_size).lower()
        if model_size not in {"s", "m", "small", "medium"}:
            raise ValueError("EfficientAD model_size must be one of: s, m")
        if lr <= 0 or weight_decay < 0:
            raise ValueError("lr must be > 0 and weight_decay must be >= 0")
        if not 0.0 < float(threshold_quantile) < 1.0:
            raise ValueError("threshold_quantile must be between 0 and 1")
        self.device = torch.device(device)
        self.model_size = "m" if model_size in {"m", "medium"} else "s"
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.feature_weight = float(feature_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.threshold_quantile = float(threshold_quantile)
        self.teacher = _FeatureTeacher(pretrained=pretrained_teacher)
        if teacher_weights:
            self.teacher.load_state_dict(torch.load(teacher_weights, map_location="cpu", weights_only=False), strict=False)
        self.student = _Student(self.teacher.out_channels)
        self.autoencoder = _AutoEncoder()
        self.register_buffer("score_mean", torch.tensor(0.0))
        self.register_buffer("score_std", torch.tensor(1.0))
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("trained", torch.tensor(False, dtype=torch.bool))
        self.to(self.device)

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def _raw_signals(self, x: torch.Tensor, teacher: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self._normalise(x)
        teacher_features = self.teacher(x_norm) if teacher is None else teacher
        student_features = self.student(x_norm)
        feature_map = (student_features - teacher_features).pow(2).mean(dim=1)
        reconstruction = (self.autoencoder(x) - x).abs().mean(dim=1)
        feature_map = F.interpolate(feature_map.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        return feature_map, reconstruction

    def _score_map(self, feature_map: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        raw_map = feature_map + self.reconstruction_weight * reconstruction
        return (raw_map - self.score_mean) / self.score_std.clamp_min(1e-6)

    def forward(self, x: torch.Tensor, return_map: bool = True, export: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del export
        feature_map, reconstruction = self._raw_signals(x)
        score_map = self._score_map(feature_map, reconstruction)
        image_scores = score_map.flatten(1).amax(1)
        return image_scores, score_map if return_map else None

    def _iter_batches(self, dataloader):
        for item in dataloader:
            batch = item[0] if isinstance(item, (tuple, list)) else item
            yield batch.to(self.device, non_blocking=True).float()

    def fit(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1) -> None:
        epochs = int(epochs)
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        cached_inputs, cached_teacher = [], []
        self.eval()
        # Use no_grad, not inference_mode: cached teacher tensors are later used
        # as targets in an autograd-tracked student loss.
        with torch.no_grad():
            for batch in self._iter_batches(dataloader):
                cached_inputs.append(batch.detach().cpu())
                cached_teacher.append(self.teacher(self._normalise(batch)).detach().cpu())
        if not cached_inputs:
            raise RuntimeError("EfficientAD training requires at least one normal training image.")
        optimizer = torch.optim.Adam(list(self.student.parameters()) + list(self.autoencoder.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        self.student.train(); self.autoencoder.train()
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        for _ in range(epochs):
            for images_cpu, teacher_cpu in zip(cached_inputs, cached_teacher):
                images = images_cpu.to(self.device, non_blocking=True)
                teacher = teacher_cpu.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    student = self.student(self._normalise(images))
                    reconstructed = self.autoencoder(images)
                    feature_loss = F.mse_loss(student.float(), teacher.float())
                    reconstruction_loss = F.l1_loss(reconstructed.float(), images.float())
                    loss = self.feature_weight * feature_loss + self.reconstruction_weight * reconstruction_loss
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        self.eval()
        normal_scores = []
        with torch.no_grad():
            for images_cpu, teacher_cpu in zip(cached_inputs, cached_teacher):
                images = images_cpu.to(self.device, non_blocking=True)
                teacher = teacher_cpu.to(self.device, non_blocking=True)
                fmap, recon = self._raw_signals(images, teacher=teacher)
                normal_scores.append((fmap + self.reconstruction_weight * recon).flatten(1).amax(1))
        scores = torch.cat(normal_scores)
        mean = scores.mean(); std = scores.std(unbiased=False).clamp_min(1e-6)
        raw_threshold = torch.quantile(scores, self.threshold_quantile)
        self.score_mean.copy_(mean); self.score_std.copy_(std)
        self.threshold.copy_((raw_threshold - mean) / std); self.trained.fill_(True)

    def predict(self, batch: torch.Tensor, export: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not export and not bool(self.trained.item()):
            raise RuntimeError("EfficientAD model is not trained. Call fit() first.")
        self.eval()
        with torch.inference_mode():
            return self.forward(batch.to(self.device, non_blocking=True).float(), export=export)

    def to_device(self, device: torch.device) -> None:
        self.device = torch.device(device); self.to(self.device)

    def save_statistics(self, path: str, half: Optional[bool] = None) -> None:
        if not bool(self.trained.item()):
            raise RuntimeError("Model is not trained. Call fit() first.")
        torch.save({"algorithm": "efficientad", "model_state": self.state_dict(), "model_size": self.model_size,
                    "lr": self.lr, "weight_decay": self.weight_decay, "threshold_quantile": self.threshold_quantile}, path)

    @staticmethod
    def load_statistics(path: str, device: str = "cpu") -> "EfficientAD":
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("algorithm") != "efficientad":
            raise ValueError("Not an EfficientAD statistics artifact")
        model = EfficientAD(device=torch.device(device), model_size=data.get("model_size", "s"), pretrained_teacher=False,
                            threshold_quantile=data.get("threshold_quantile", 0.995))
        model.load_state_dict(data["model_state"])
        return model
