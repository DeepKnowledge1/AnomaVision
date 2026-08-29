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
    """Minimal teacher-student EfficientAD implementation.

    A frozen ImageNet ResNet teacher provides layer-1 features and a very small
    CNN student learns those features from normal images. At inference, the
    teacher-student feature error is used as the anomaly map.

    The public API mirrors PaDiM/PatchCore: ``fit`` trains on normal images and
    ``predict`` returns image scores plus a spatial anomaly map.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        device: torch.device = torch.device("cpu"),
        layer_indices: Optional[List[int]] = None,
        epochs: int = 5,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        if backbone != "resnet18":
            raise ValueError("EfficientAD lightweight supports backbone='resnet18' only.")
        self.device = torch.device(device)
        self.backbone = backbone
        self.layer_indices = [0]
        self.epochs = max(1, int(epochs))
        self.learning_rate = float(learning_rate)

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

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            teacher = self._teacher_features(batch)
            teacher = F.normalize(teacher, dim=1)
        student = F.normalize(self.student(batch), dim=1)
        return F.mse_loss(student, teacher)

    @torch.no_grad()
    def _scores(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = batch.to(self.device, non_blocking=True)
        teacher = F.normalize(self._teacher_features(batch), dim=1)
        student = F.normalize(self.student(batch), dim=1)
        score = (teacher - student).pow(2).mean(dim=1)
        image_scores = score.flatten(1).amax(1)
        score_map = F.interpolate(
            score.unsqueeze(1),
            size=batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return image_scores, score_map

    def fit(self, dataloader: torch.utils.data.DataLoader, extractions: int = 1) -> None:
        """Train the small student using only normal training images."""
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.learning_rate)
        self.student.train()
        for _ in range(self.epochs):
            for item in dataloader:
                batch = item[0] if isinstance(item, (tuple, list)) else item
                batch = batch.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(batch)
                loss.backward()
                optimizer.step()
        self.student.eval()
        self._fitted = True

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, return_map: bool = True, export: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return image-level anomaly scores and an optional spatial map."""
        if not self._fitted:
            raise RuntimeError("EfficientAD is not fitted. Call fit() first.")
        image_scores, score_map = self._scores(x)
        return image_scores, score_map if return_map else None

    def predict(
        self, batch: torch.Tensor, export: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference using the standard AnomaVision prediction contract."""
        return self.forward(batch, return_map=True, export=export)

    def to_device(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.teacher.to_device(self.device)
        self.student.to(self.device)

    def save_statistics(self, path: str, half: Optional[bool] = False) -> None:
        """Save a compact student/teacher deployment artifact."""
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
    )
    state = stats["student"]
    model.student.load_state_dict({key: value.float() for key, value in state.items()})
    model.student.eval()
    model._fitted = True
    return model
