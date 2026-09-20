import numpy as np

from anomavision.inference.model.wrapper import ModelWrapper


class FakeBackend:
    def __init__(self):
        self.last_batch = None

    def predict(self, batch):
        self.last_batch = batch
        return np.zeros(batch.shape[0]), np.zeros((batch.shape[0], 1, 1))

    def extract_drift_embeddings(self, batch):
        assert batch is self.last_batch
        return np.asarray(batch).reshape(batch.shape[0], -1)

    def close(self):
        pass


def test_model_wrapper_exposes_backend_drift_representation(monkeypatch):
    backend = FakeBackend()

    monkeypatch.setattr(
        "anomavision.inference.model.wrapper.make_backend",
        lambda model_path, device: backend,
    )

    model = ModelWrapper("unused.pt", "cpu")
    batch = np.arange(12, dtype=np.float32).reshape(3, 4)

    model.predict(batch)
    embeddings = model.extract_drift_embeddings(batch)

    np.testing.assert_array_equal(embeddings, batch)
    model.close()
