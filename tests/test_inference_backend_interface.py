import numpy as np
import pytest
import torch

from anomavision.inference.model.backends.base import InferenceBackend
from anomavision.inference.model.backends.hailo_backend import HailoBackend
from anomavision.inference.model.backends.k260_backend import KV260Backend
from anomavision.inference.model.backends.tensorrt_backend import TensorRTBackend


class DummyBackend(InferenceBackend):
    """Small test backend demonstrating the required backend contract."""

    def __init__(self, score: float = 0.5) -> None:
        self.score = float(score)
        self.closed = False

    def predict(self, batch: torch.Tensor | np.ndarray):
        """Return one score and one map per input image."""
        if self.closed:
            raise RuntimeError("backend is closed")
        array = (
            batch.detach().cpu().numpy() if isinstance(batch, torch.Tensor) else batch
        )
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4:
            raise ValueError("batch must be a 4D tensor or array")
        batch_size, _, height, width = array.shape
        scores = np.full((batch_size,), self.score, dtype=np.float32)
        maps = np.full((batch_size, height, width), self.score, dtype=np.float32)
        return scores, maps

    def close(self) -> None:
        """Mark the dummy backend as closed."""
        self.closed = True


def test_dummy_backend_follows_common_contract_with_numpy_batch():
    backend = DummyBackend(score=0.75)
    scores, maps = backend.predict(np.zeros((2, 3, 16, 12), dtype=np.float32))

    assert isinstance(scores, np.ndarray)
    assert isinstance(maps, np.ndarray)
    assert scores.shape == (2,)
    assert maps.shape == (2, 16, 12)
    assert scores.dtype == np.float32
    assert maps.dtype == np.float32
    np.testing.assert_allclose(scores, 0.75)
    np.testing.assert_allclose(maps, 0.75)


def test_dummy_backend_accepts_single_torch_image_and_lifecycle():
    backend = DummyBackend()
    scores, maps = backend.predict(torch.zeros(3, 8, 10))

    assert scores.shape == (1,)
    assert maps.shape == (1, 8, 10)
    assert not backend.closed

    backend.close()
    assert backend.closed
    with pytest.raises(RuntimeError, match="closed"):
        backend.predict(torch.zeros(1, 3, 8, 10))


def test_dummy_backend_rejects_invalid_batch_shape():
    with pytest.raises(ValueError, match="4D"):
        DummyBackend().predict(np.zeros((3, 8), dtype=np.float32))


def test_accelerator_backends_follow_the_same_concrete_pattern():
    for backend_class in (TensorRTBackend, HailoBackend, KV260Backend):
        assert InferenceBackend in backend_class.__mro__
        assert callable(backend_class.predict)
        assert callable(backend_class.close)
        assert backend_class.predict.__doc__
        assert backend_class.close.__doc__
