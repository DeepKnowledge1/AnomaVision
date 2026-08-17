import torch

from scripts.convert_to_tensorrt import _image_files, detect_artifact


def test_detect_padim_statistics(tmp_path):
    path = tmp_path / "padim.pth"
    torch.save(
        {
            "mean": torch.zeros(3),
            "cov_inv": torch.eye(3),
            "channel_indices": torch.tensor([0, 1, 2]),
            "layer_indices": [0],
            "backbone": "resnet18",
        },
        path,
    )
    model_type, _ = detect_artifact(path)
    assert model_type == "padim"


def test_detect_patchcore_statistics(tmp_path):
    path = tmp_path / "patchcore.pth"
    torch.save(
        {
            "memory_bank": torch.randn(4, 3),
            "layer_indices": [0],
            "backbone": "resnet18",
        },
        path,
    )
    model_type, _ = detect_artifact(path)
    assert model_type == "patchcore"


def test_calibration_images_are_recursive_and_deterministic(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "b.png").write_bytes(b"placeholder")
    (nested / "a.jpg").write_bytes(b"placeholder")
    (nested / "ignored.txt").write_bytes(b"placeholder")
    paths = list(_image_files(tmp_path))
    assert [path.name for path in paths] == ["b.png", "a.jpg"]
