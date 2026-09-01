from unittest.mock import MagicMock, patch

from anomavision.actions.MQTTAction import MQTTAction


def test_mqtt_action_connect_publish_disconnect():
    fake_client = MagicMock()
    fake_client.connect.return_value = None
    fake_client.publish.return_value.rc = 0

    with patch("anomavision.actions.MQTTAction.mqtt.Client", return_value=fake_client), patch(
        "anomavision.actions.MQTTAction.mqtt.MQTT_ERR_SUCCESS", 0
    ):
        action = MQTTAction(
            broker="localhost",
            port=1883,
            topic="factory/test/inspection",
        )

        # Simulate the asynchronous on_connect callback used by paho-mqtt.
        def start_loop():
            action._on_connect(fake_client, None, {}, 0)

        fake_client.loop_start.side_effect = start_loop

        assert action.connect() is True
        assert action.is_connected() is True

        result = {
            "event_id": "evt-1",
            "decision": "FAIL",
            "anomaly_score": 0.91,
        }
        assert action.execute(result) is True

        fake_client.publish.assert_called_once()
        _, kwargs = fake_client.publish.call_args
        assert kwargs["topic"] == "factory/test/inspection"
        assert '"decision":"FAIL"' in kwargs["payload"]
        assert kwargs["qos"] == 1

        action.disconnect()
        assert action.is_connected() is False
        fake_client.loop_stop.assert_called_once()
        fake_client.disconnect.assert_called_once()
