import json

import numpy as np

from anomavision.actions.EvidenceAction import EvidenceAction


def test_evidence_action_saves_fail_event(tmp_path):
    action = EvidenceAction(directory=str(tmp_path))
    assert action.connect() is True

    result = {
        "event_id": "evt-1",
        "decision": "FAIL",
        "anomaly_score": 0.93,
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
    }

    assert action.execute(result) is True

    event_dir = tmp_path / "evt-1"
    assert (event_dir / "event.json").exists()
    assert (event_dir / "image.png").exists()

    with (event_dir / "event.json").open(encoding="utf-8") as file:
        data = json.load(file)
    assert data["decision"] == "FAIL"
    assert "image" not in data


def test_evidence_action_skips_pass_by_default(tmp_path):
    action = EvidenceAction(directory=str(tmp_path))
    action.connect()

    assert action.execute({"event_id": "evt-2", "decision": "PASS"}) is True
    assert not (tmp_path / "evt-2").exists()
