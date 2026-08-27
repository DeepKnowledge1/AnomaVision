"""EfficientAD implementation integrated with the AnomaVision algorithm API.

The implementation deliberately follows the PaDiM/PatchCore contract:
``fit(dataloader)`` trains/calibrates from normal data and ``predict(batch)``
returns image scores plus an image-sized anomaly map.  Teacher features are
cached once during training so additional epochs do not repeat the frozen
backbone extraction.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class _FeatureTeacher(nn.Module):
    """Frozen EfficientNet feature extractor used by EfficientAD."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = efficientnet_b0(weights=weights)
        # EfficientNet-B0 stage 6 produces a compact 112-channel 14x14 map for
        # the standard 224x224 AnomaVision input.
        self.features = nn.Sequential(*list(net.features[:6]))
        self.out_channels = 112
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class _Student(nn.Module):
    """Small trainable student matching the teacher feature resolution."""

    def __init__(self, out_channels: int = 112) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 112, 3, stride=2, padding=1),
            nn.BatchNorm2d(112),
            nn.ReLU(inplace=True),
            nn.Conv2d(112, out_channels, 3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _AutoEncoder(nn.Module):
    """Compact reconstruction branch used as the second anomaly signal."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class EfficientAD(nn.Module):
    """EfficientAD detector with the same fit/predict contract as PaDiM.

    Scores and maps use the same calibrated scale.  ``score_mean``,
    ``score_std`` and ``threshold`` are learned exclusively from normal
    training images, and all three are registered buffers so they survive PT
    and ONNX export.
    """

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
        threshold_quantile: float = 0.995,
    ) -> None:
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
            state = torch.load(teacher_weights, map_location="cpu", weights_only=False)
            self.teacher.load_state_dict(state, strict=False)
        self.student = _Student(self.teacher.out_channels)
        self.autoencoder = _AutoEncoder()

        self.register_buffer("score_mean", torch.tensor(0.0))
        self.register_buffer("score_std", torch.tensor(1.0))
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("trained", torch.tensor(False, dtype=torch.bool))
        self.to(self.device)

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        # AnodetDataset performs the ImageNet normalization shared by all
        # AnomaVision algorithms. Do not normalize a second time here.
        return x

    @torch.no_grad()
    def _raw_signals(
        self, x: torch.Tensor, teacher: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self._normalise(x)
        teacher_features = self.teacher(x_norm) if teacher is None else teacher
        student_features = self.student(x_norm)
        feature_map = (student_features - teacher_features).pow(2).mean(dim=1)
        reconstruction = (self.autoencoder(x) - x).abs().mean(dim=1)
        feature_map = F.interpolate(
            feature_map.unsqueeze(1),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return feature_map, reconstruction

    def _score_map(self, feature_map: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        """Combine signals and put the map on the calibrated image-score scale."""
        raw_map = feature_map + self.reconstruction_weight * reconstruction
        # Calibration is performed on raw image scores. Normalize the complete
        # map as well so localization uses exactly the same threshold units as
        # image classification.
        return (raw_map - self.score_mean) / self.score_std.clamp_min(1e-6)

    def forward(
        self, x: torch.Tensor, return_map: bool = True, export: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del export  # kept for the common AnomaVision export contract
        feature_map, reconstruction = self._raw_signals(x)
        score_map = self._score_map(feature_map, reconstruction)
        image_scores = score_map.flatten(1).amax(1)
        return image_scores, score_map if return_map else None

    def _iter_batches(self, dataloader):
        for item in dataloader:
            batch = item[0] if isinstance(item, (tuple, list)) else item
            yield batch.to(self.device, non_blocking=True).float()

    def fit(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1) -> None:
        """Train on normal images and calibrate only from those normal images.

        The frozen teacher is evaluated exactly once per training image. Inputs
        and teacher features are then reused for every requested epoch. This
        removes the expensive repeated EfficientNet pass that made the earlier
        implementation scale poorly with ``epochs`` and avoids a second
        dataloader pass for calibration.
        """
        epochs = int(epochs)
        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        # Cache the already-preprocessed training tensors and teacher features.
        # PatchCore/PaDiM also operate on the DataLoader's preprocessed tensors;
        # keeping this cache makes EfficientAD deterministic and avoids repeated
        # CPU image decoding/preprocessing for multi-epoch training.
        cached_inputs = []
        cached_teacher = []
        self.eval()
        with torch.inference_mode():
            for batch in self._iter_batches(dataloader):
                cached_inputs.append(batch.detach().cpu())
                cached_teacher.append(self.teacher(self._normalise(batch)).detach().cpu())

        if not cached_inputs:
            raise RuntimeError("EfficientAD training requires at least one normal training image.")

        optimizer = torch.optim.Adam(
            list(self.student.parameters()) + list(self.autoencoder.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.student.train()
        self.autoencoder.train()
        use_amp = self.device.type == "cuda"
        amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
                    loss = (
                        self.feature_weight * feature_loss
                        + self.reconstruction_weight * reconstruction_loss
                    )
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()

        # One final normal-only pass over the cached tensors calibrates the full
        # raw score distribution after training. No test/anomalous image is used.
        self.eval()
        normal_scores = []
        with torch.inference_mode():
            for images_cpu, teacher_cpu in zip(cached_inputs, cached_teacher):
                images = images_cpu.to(self.device, non_blocking=True)
                teacher = teacher_cpu.to(self.device, non_blocking=True)
                fmap, recon = self._raw_signals(images, teacher=teacher)
                raw_map = fmap + self.reconstruction_weight * recon
                normal_scores.append(raw_map.flatten(1).amax(1))

        scores = torch.cat(normal_scores)
        mean = scores.mean()
        std = scores.std(unbiased=False).clamp_min(1e-6)
        raw_threshold = torch.quantile(scores, self.threshold_quantile)

        self.score_mean.copy_(mean)
        self.score_std.copy_(std)
        self.threshold.copy_((raw_threshold - mean) / std)
        self.trained.fill_(True)

    def predict(
        self, batch: torch.Tensor, export: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not export and not bool(self.trained.item()):
            raise RuntimeError("EfficientAD model is not trained. Call fit() first.")
        self.eval()
        with torch.inference_mode():
            return self.forward(batch.to(self.device, non_blocking=True).float(), export=export)

    def to_device(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.to(self.device)

    def save_statistics(self, path: str, half: Optional[bool] = None) -> None:
        if not bool(self.trained.item()):
            raise RuntimeError("Model is not trained. Call fit() first.")
        state = self.state_dict()
        torch.save(
            {
                "algorithm": "efficientad",
                "model_state": state,
                "model_size": self.model_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "threshold_quantile": self.threshold_quantile,
            },
            path,
        )

    @staticmethod
    def load_statistics(path: str, device: str = "cpu") -> "EfficientAD":
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("algorithm") != "efficientad":
            raise ValueError("Not an EfficientAD statistics artifact")
        model = EfficientAD(
            device=torch.device(device),
            model_size=data.get("model_size", "s"),
            pretrained_teacher=False,
            threshold_quantile=data.get("threshold_quantile", 0.995),
        )
        model.load_state_dict(data["model_state"])
        return model
