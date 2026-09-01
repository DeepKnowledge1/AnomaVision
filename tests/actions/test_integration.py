from anomavision.actions.integration import dispatch_inference_results


class DummyDispatcher:
    def __init__(self):
        self.results = []

    def execute_all(self, result):
        self.results.append(result)
        return [True]


def test_dispatch_inference_results_uses_existing_classification():
    dispatcher = DummyDispatcher()

    dispatch_inference_results(
        dispatcher,
        image_scores=[0.1, 0.9],
        classifications=[False, True],
        source_id="camera_01",
        model_name="padim",
        frame_start=10,
    )

    assert len(dispatcher.results) == 2
    assert dispatcher.results[0]["decision"] == "PASS"
    assert dispatcher.results[1]["decision"] == "FAIL"
    assert dispatcher.results[0]["frame_id"] == 10
    assert dispatcher.results[1]["frame_id"] == 11


def test_dispatch_is_optional():
    dispatch_inference_results(
        None,
        image_scores=[0.9],
        classifications=[True],
        source_id="camera_01",
        model_name="padim",
    )
