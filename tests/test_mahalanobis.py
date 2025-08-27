import pytest
import torch

from anodet.mahalanobis import MahalanobisDistance

def make_identity_cov_inv(n: int, d: int) -> torch.Tensor:
    """
    Build a (N, D, D) stacked identity inverse covariance.
    """
    eye = torch.eye(d)
    return eye.unsqueeze(0).repeat(n, 1, 1)


def test_mahalanobis_basic_shape_and_nonneg():
    B, N, D = 2, 9, 4
    width = height = 3  # width * height must equal N

    mean = torch.zeros(N, D)
    cov_inv = make_identity_cov_inv(N, D)

    m = MahalanobisDistance(mean, cov_inv)

    # random features
    features = torch.randn(B, N, D)

    out = m(features, width, height)
    assert out.shape == (B, width, height)
    # Mahalanobis distance is non-negative
    assert torch.all(out >= 0)


def test_mahalanobis_zeros_features_zero_mean_is_norm():
    B, N, D = 1, 4, 3
    width = height = 2

    mean = torch.zeros(N, D)
    cov_inv = make_identity_cov_inv(N, D)

    m = MahalanobisDistance(mean, cov_inv)

    # Features equal to mean -> distance should be all zeros
    features = torch.zeros(B, N, D)
    out = m(features, width, height)
    assert torch.allclose(out, torch.zeros_like(out))


def test_mahalanobis_mismatched_patch_count_raises():
    B, N, D = 1, 5, 3
    width, height = 2, 3  # width*height=6 != N(5)

    mean = torch.zeros(N, D)
    cov_inv = make_identity_cov_inv(N, D)
    m = MahalanobisDistance(mean, cov_inv)

    features = torch.randn(B, N, D)

    with pytest.raises(ValueError):
        _ = m(features, width, height)


def test_mahalanobis_invalid_features_shape_raises():
    N, D = 4, 3
    mean = torch.zeros(N, D)
    cov_inv = make_identity_cov_inv(N, D)
    m = MahalanobisDistance(mean, cov_inv)

    # Bad shape: (B, D) instead of (B, N, D)
    bad = torch.randn(2, D)
    with pytest.raises(ValueError):
        _ = m(bad, 2, 2)


def test_mahalanobis_type_check_raises():
    N, D = 4, 3
    mean = torch.zeros(N, D)
    cov_inv = make_identity_cov_inv(N, D)
    m = MahalanobisDistance(mean, cov_inv)

    with pytest.raises(TypeError):
        _ = m("not a tensor", 2, 2)  # type: ignore[arg-type]
