from pathlib import Path

import torch
import torch.nn.functional as F

from anomavision.quantize.model.backends.hef.exporter import export_onnx

from . import graphs as hailo_graphs


class FakeExtractor(torch.nn.Module):
    def __init__(self, backbone, device):
        super().__init__()

    def forward(self, image, layer_indices=None):
        batch = image.shape[0]
        features = F.adaptive_avg_pool2d(image, (4, 4))
        features = features.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
        return features.permute(0, 2, 3, 1).reshape(batch, 16, 4), 4, 4


def main():
    hailo_graphs.ResnetEmbeddingsExtractor = FakeExtractor
    output = Path(".artifacts/hailo_operator_audit")
    output.mkdir(parents=True, exist_ok=True)
    patchcore_artifact = output / "patchcore.pt"
    padim_artifact = output / "padim.pt"
    torch.save(
        {
            "backbone": "resnet18",
            "layer_indices": [0, 1],
            "memory_bank": torch.zeros(8, 4),
            "patch_grid": 4,
        },
        patchcore_artifact,
    )
    torch.save(
        {
            "backbone": "resnet18",
            "layer_indices": [0, 1],
            "channel_indices": torch.arange(4),
            "mean": torch.zeros(16, 4),
            "cov_inv": torch.eye(4).repeat(16, 1, 1),
        },
        padim_artifact,
    )
    for algorithm, artifact in (
        ("padim", padim_artifact),
        ("patchcore", patchcore_artifact),
    ):
        path = export_onnx(algorithm, artifact, output / algorithm, (32, 32))
        import onnx

        model = onnx.load(path)
        ops = sorted({node.op_type for node in model.graph.node})
        print(f"{algorithm}: {', '.join(ops)}")


if __name__ == "__main__":
    main()
