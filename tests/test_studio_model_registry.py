from pathlib import Path

from apps.studio.services.model_registry import list_models


def test_list_models(tmp_path: Path):
    model = tmp_path / "models" / "patchcore" / "bottle" / "run-1"
    model.mkdir(parents=True)
    (model / "model.pt").write_bytes(b"model")
    (model / "config.yml").write_text(
        "algorithm: patchcore\nclass_name: bottle\nrun_name: run-1\n",
        encoding="utf-8",
    )

    models = list_models(tmp_path)

    assert len(models) == 1
    assert models[0]["algorithm"] == "patchcore"
    assert models[0]["class_name"] == "bottle"
    assert models[0]["id"] == "patchcore/bottle/run-1/model.pt"
