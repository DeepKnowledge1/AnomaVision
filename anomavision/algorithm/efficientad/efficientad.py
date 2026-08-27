"""EfficientAD teacher/student anomaly detector using the AnomaVision API.

The deployment path contains only the teacher/student discrepancy. Normal training
images calibrate a per-pixel discrepancy distribution and the image threshold.
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
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class _Student(nn.Module):
    def __init__(self, out_channels: int = 112) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, 2, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, out_channels, 3, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EfficientAD(nn.Module):
    """EfficientAD-compatible teacher/student detector.

    It follows the same ``fit -> predict -> (scores, maps)`` contract as PaDiM
    and PatchCore. The old autoencoder branch was removed from deployment because
    it added a second full image network without improving the AnomaVision score
    calibration and made inference unnecessarily slow.
    """

    def __init__(
        self, device: torch.device = torch.device("cpu"), model_size: str = "s",
        lr: float = 1e-4, weight_decay: float = 1e-5,
        pretrained_teacher: bool = True, teacher_weights: Optional[str] = None,
        feature_weight: float = 1.0, reconstruction_weight: float = 0.1,
        threshold_quantile: float = 0.995,
    ) -> None:
        super().__init__()
        size = str(model_size).lower()
        if size not in {"s", "m", "small", "medium"}:
            raise ValueError("EfficientAD model_size must be one of: s, m")
        if lr <= 0 or weight_decay < 0:
            raise ValueError("lr must be > 0 and weight_decay must be >= 0")
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be between 0 and 1")

        self.device = torch.device(device)
        self.model_size = "m" if size in {"m", "medium"} else "s"
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.feature_weight = float(feature_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.threshold_quantile = float(threshold_quantile)

        self.teacher = _FeatureTeacher(pretrained_teacher)
        if teacher_weights:
            state = torch.load(teacher_weights, map_location="cpu", weights_only=False)
            self.teacher.load_state_dict(state, strict=False)
        self.student = _Student(self.teacher.out_channels)

        self.register_buffer("map_mean", torch.zeros(1, 1, 1))
        self.register_buffer("map_std", torch.ones(1, 1, 1))
        self.register_buffer("score_mean", torch.tensor(0.0))
        self.register_buffer("score_std", torch.tensor(1.0))
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("trained", torch.tensor(False, dtype=torch.bool))
        self.to(self.device)

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def _raw_map(self, x: torch.Tensor, teacher: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True).float()
        teacher_features = self.teacher(self._normalise(x)) if teacher is None else teacher
        student_features = self.student(self._normalise(x))
        discrepancy = (student_features - teacher_features).pow(2).mean(dim=1, keepdim=True)
        return F.interpolate(
            discrepancy, size=x.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

    def _score_map(self, raw_map: torch.Tensor) -> torch.Tensor:
        return (raw_map - self.map_mean.to(raw_map.device)) / self.map_std.to(raw_map.device).clamp_min(1e-6)

    def forward(self, x: torch.Tensor, return_map: bool = True, export: bool = False):
        del export
        score_map = self._score_map(self._raw_map(x))
        scores = score_map.flatten(1).amax(1)
        return scores, score_map if return_map else None

    @staticmethod
    def _batch(item):
        return item[0] if isinstance(item, (tuple, list)) else item

    def fit(self, dataloader: torch.utils.data.DataLoader, epochs: int = 3) -> None:
        epochs = int(epochs)
        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        cached = []
        self.teacher.eval()
        # Teacher is frozen: compute it exactly once per training batch.
        with torch.no_grad():
            for item in dataloader:
                images = self._batch(item).to(self.device, non_blocking=True).float()
                teacher = self.teacher(self._normalise(images)).detach()
                cached.append((images.detach().cpu(), teacher.cpu()))
        if not cached:
            raise RuntimeError("EfficientAD training requires normal training images")

        optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.student.train()
        for _ in range(epochs):
            for images_cpu, teacher_cpu in cached:
                images = images_cpu.to(self.device, non_blocking=True)
                teacher = teacher_cpu.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    student = self.student(self._normalise(images))
                    loss = F.mse_loss(student.float(), teacher.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Calibrate the exact map used in production from NORMAL images only.
        self.student.eval()
        maps = []
        with torch.no_grad():
            for images_cpu, teacher_cpu in cached:
                images = images_cpu.to(self.device, non_blocking=True)
                teacher = teacher_cpu.to(self.device, non_blocking=True)
                maps.append(self._raw_map(images, teacher).float())
        normal_maps = torch.cat(maps, 0)
        mean = normal_maps.mean(dim=0, keepdim=True)
        std = normal_maps.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        self.map_mean.copy_(mean.cpu())
        self.map_std.copy_(std.cpu())

        normalized = (normal_maps - mean) / std
        scores = normalized.flatten(1).amax(1)
        self.score_mean.copy_(scores.mean().cpu())
        self.score_std.copy_(scores.std(unbiased=False).clamp_min(1e-6).cpu())
        self.threshold.copy_(torch.quantile(scores, self.threshold_quantile).cpu())
        self.trained.fill_(True)
        self.to(self.device)

    def predict(self, batch: torch.Tensor, export: bool = False):
        if not export and not bool(self.trained.item()):
            raise RuntimeError("EfficientAD model is not trained. Call fit() first.")
        self.eval()
        with torch.inference_mode():
            return self.forward(batch.to(self.device, non_blocking=True).float(), export=export)

    def to_device(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.to(self.device)

    def save_statistics(self, path: str, half: Optional[bool] = None) -> None:
        """Save a directly loadable deployment artifact.

        The previous implementation saved a raw dictionary which the generic
        exporter could not reconstruct. Saving the runtime module here makes the
        existing AnomaVision exporter work unchanged for EfficientAD .pth files.
        """
        if not bool(self.trained.item()):
            raise RuntimeError("Model is not trained. Call fit() first.")
        torch.save(self.cpu(), path)
        self.to(self.device)

    @staticmethod
    def load_statistics(path: str, device: str = "cpu") -> "EfficientAD":
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, EfficientAD):
            obj.to_device(torch.device(device))
            return obj
        if isinstance(obj, dict) and obj.get("algorithm") == "efficientad":
            model = EfficientAD(
                device=torch.device(device),
                model_size=obj.get("model_size", "s"),
                pretrained_teacher=False,
                threshold_quantile=obj.get("threshold_quantile", 0.995),
            )
            model.load_state_dict(obj["model_state"])
            return model
        raise ValueError("Not an EfficientAD artifact")


def build_efficientad_from_stats(stats, device: str = "cpu") -> EfficientAD:
    """Build EfficientAD from either a runtime artifact or legacy stats dict."""
    if isinstance(stats, EfficientAD):
        stats.to_device(torch.device(device))
        return stats
    if isinstance(stats, dict):
        return EfficientAD.load_statistics_from_dict(stats, device)
    raise ValueError("Unsupported EfficientAD statistics artifact")


def _load_statistics_from_dict(stats, device: str = "cpu") -> EfficientAD:
    model = EfficientAD(
        device=torch.device(device),
        model_size=stats.get("model_size", "s"),
        pretrained_teacher=False,
        threshold_quantile=stats.get("threshold_quantile", 0.995),
    )
    model.load_state_dict(stats["model_state"])
    return model


# Keep the builder self-contained without changing the public API above.
EfficientAD.load_statistics_from_dict = staticmethod(_load_statistics_from_dict)
