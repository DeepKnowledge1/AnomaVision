"""Lightweight EfficientAD-style teacher-student anomaly detection."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.feature_extraction import ResnetEmbeddingsExtractor


class _Student(nn.Module):
    """Small CNN that learns normal ResNet layer-1 features."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EfficientAD(torch.nn.Module):
    """Minimal teacher-student EfficientAD implementation."""

    def __init__(
        self,
        backbone: str = "resnet18",
        device: torch.device = torch.device("cpu"),
        layer_indices: Optional[List[int]] = None,
        epochs: int = 5,
        learning_rate: float = 1e-3,
        threshold_percentile: float = 99.5,
    ) -> None:
        super().__init__()
        if backbone != "resnet18":
            raise ValueError("EfficientAD lightweight supports backbone='resnet18' only.")
        self.device = torch.device(device)
        self.backbone = backbone
        self.layer_indices = [0]
        self.epochs = max(1, int(epochs))
        self.learning_rate = float(learning_rate)
        self.threshold_percentile = float(threshold_percentile)
        if not 0.0 < self.threshold_percentile <= 100.0:
            raise ValueError("threshold_percentile must be in (0, 100].")
        self.threshold = 0.0

        self.teacher = ResnetEmbeddingsExtractor(backbone, self.device)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

        self.student = _Student(64).to(self.device)
        self._fitted = False

    @torch.no_grad()
    def _teacher_features(self, batch: torch.Tensor) -> torch.Tensor:
        features, width, height = self.teacher(batch, layer_indices=[0])
        return features.reshape(batch.shape[0], width, height, 64).permute(0, 3, 1, 2)

    def _loss(self, teacher: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        teacher = F.normalize(teacher, dim=1)
        student = F.normalize(self.student(batch), dim=1)
        return F.mse_loss(student, teacher)

    @torch.no_grad()
    def _raw_scores(
        self, batch: torch.Tensor, teacher: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = batch.to(self.device, non_blocking=True)
        if teacher is None:
            teacher = self._teacher_features(batch)
        teacher = F.normalize(teacher, dim=1)
        student = F.normalize(self.student(batch), dim=1)
        score = (teacher - student).pow(2).mean(dim=1)

        # Mean of the highest-scoring 1% of locations is more robust than a
        # single maximum while remaining sensitive to small defects.
        flat = score.flatten(1)
        k = max(1, int(flat.shape[1] * 0.01))
        image_scores = flat.topk(k, dim=1).values.mean(dim=1)

        score_map = F.interpolate(
            score.unsqueeze(1),
            size=batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return image_scores, score_map

    def fit(self, dataloader: torch.utils.data.DataLoader, extractions: int = 1) -> None:
        """Train the student and calibrate an adaptive threshold on normal images."""
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.learning_rate)

        # Compute frozen teacher features once. This avoids running ResNet for
        # every epoch and makes CPU training considerably faster.
        cached = []
        self.teacher.eval()
        with torch.no_grad():
            for item in dataloader:
                batch = item[0] if isinstance(item, (tuple, list)) else item
                batch = batch.to(self.device, non_blocking=True)
                cached.append((batch.detach(), self._teacher_features(batch).detach()))

        self.student.train()
        for _ in range(self.epochs):
            for batch, teacher in cached:
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(teacher, batch)
                loss.backward()
                optimizer.step()

        self.student.eval()

        # Calibrate from normal training scores using the exact score that is
        # used for inference.
        normal_scores = []
        for batch, teacher in cached:
            scores, _ = self._raw_scores(batch, teacher)
            normal_scores.append(scores.cpu())
        self.threshold = float(
            torch.quantile(torch.cat(normal_scores), self.threshold_percentile / 100.0)
        )
        self.threshold = max(self.threshold, 1e-8)
        self._fitted = True
        self.eval()

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, return_map: bool = True, export: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self._fitted:
            raise RuntimeError("EfficientAD is not fitted. Call fit() first.")
        image_scores, score_map = self._raw_scores(x)

        # Normalize scores by the training-derived threshold. This embeds the
        # adaptive threshold into the exported model: >= 1.0 means anomalous.
        scale = image_scores.new_tensor(self.threshold)
        image_scores = image_scores / scale
        score_map = score_map / scale
        return image_scores, score_map if return_map else None

    def predict(
        self, batch: torch.Tensor, export: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(batch, return_map=True, export=export)

    def to_device(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.teacher.to_device(self.device)
        self.student.to(self.device)

    def save_statistics(self, path: str, half: Optional[bool] = False) -> None:
        if not self._fitted:
            raise RuntimeError("EfficientAD is not fitted. Call fit() first.")
        student_state = {
            key: value.detach().cpu().half() if half and value.is_floating_point() else value.detach().cpu()
            for key, value in self.student.state_dict().items()
        }
        torch.save(
            {
                "student": student_state,
                "backbone": self.backbone,
                "layer_indices": self.layer_indices,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "threshold_percentile": self.threshold_percentile,
                "threshold": self.threshold,
                "model_type": "efficientad",
                "dtype": "fp16" if half else "fp32",
            },
            path,
        )


def build_efficientad_from_stats(
    stats: Dict, device: str = "cpu", force_precision: Optional[str] = None
) -> EfficientAD:
    """Build EfficientAD from a compact statistics artifact."""
    model = EfficientAD(
        backbone=str(stats.get("backbone", "resnet18")),
        device=torch.device(device),
        layer_indices=[0],
        epochs=int(stats.get("epochs", 5)),
        learning_rate=float(stats.get("learning_rate", 1e-3)),
        threshold_percentile=float(stats.get("threshold_percentile", 99.5)),
    )
    state = stats["student"]
    model.student.load_state_dict({key: value.float() for key, value in state.items()})
    model.student.eval()
    model.threshold = max(float(stats.get("threshold", 0.0)), 1e-8)
    model._fitted = True
    model.eval()
    return model
