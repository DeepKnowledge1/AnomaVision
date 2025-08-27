import pytest
import torch


from anodet.feature_extraction import concatenate_layers

def test_concatenate_layers_basic_shapes():
    # Create layers with different channels and spatial sizes
    # First layer determines target (H, W) after resize
    B = 2
    l1 = torch.randn(B, 8, 16, 16)   # (B, C1, H, W)  -> target size (16,16)
    l2 = torch.randn(B, 4, 8, 8)     # will be upsampled to (16,16)
    l3 = torch.randn(B, 6, 4, 4)     # will be upsampled to (16,16)

    out = concatenate_layers([l1, l2, l3])

    # Channels sum
    assert out.shape == (B, 8 + 4 + 6, 16, 16)


def test_concatenate_layers_uses_nearest_resize_correctly():
    # Small, easy-to-verify case
    B = 1
    # Target is 2x2
    base = torch.tensor([[[[1.0, 2.0],
                           [3.0, 4.0]]]])  # (1,1,2,2)
    # A 1x1 tensor that should expand to 2x2 with nearest -> constant fill
    small = torch.tensor([[[[7.0]]]])  # (1,1,1,1)

    out = concatenate_layers([base, small])
    # Expect channel 0 == base, channel 1 == all 7s after nearest upsample
    assert out.shape == (1, 2, 2, 2)
    assert torch.allclose(out[:, 0], base[:, 0])
    assert torch.allclose(out[:, 1], torch.full((1, 2, 2), 7.0))


def test_concatenate_layers_empty_raises():
    with pytest.raises(ValueError):
        _ = concatenate_layers([])


def test_concatenate_layers_non_tensor_raises():
    B = 1
    good = torch.randn(B, 2, 4, 4)
    bad = "not a tensor"
    with pytest.raises(TypeError):
        _ = concatenate_layers([good, bad])  # type: ignore[arg-type]


def test_concatenate_layers_insufficient_dims_raises():
    # Tensor with <2 spatial dims (e.g., (B, C) only) should raise
    bad = torch.randn(2, 3)
    with pytest.raises(ValueError):
        _ = concatenate_layers([bad])
