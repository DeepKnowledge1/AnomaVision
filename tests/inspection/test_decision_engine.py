import pytest

from anomavision.inspection.DecisionEngine import DecisionEngine


def test_score_below_threshold_is_pass():
    engine = DecisionEngine(0.8)
    assert engine.evaluate(0.79) == "PASS"
    assert not engine.is_anomaly(0.79)


def test_score_at_threshold_is_fail():
    engine = DecisionEngine(0.8)
    assert engine.evaluate(0.8) == "FAIL"
    assert engine.is_anomaly(0.8)


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        DecisionEngine(-0.1)
