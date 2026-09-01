# Industrial Actions

AnomaVision can route normalized inspection results to external systems without coupling models to industrial protocols.

## Flow

```text
Source -> Model -> InspectionResult -> DecisionEngine -> ActionDispatcher
                                                        |-> MQTTAction
                                                        |-> EvidenceAction
                                                        `-> OPCUAAction
```

## Configuration

See `examples/actions/industrial_inspection.yaml` for a complete example.

Actions are configured as a list. Multiple actions may receive the same inspection event. A failure in one action is isolated by `ActionDispatcher` so it does not terminate inference processing.

## MQTT

`MQTTAction` publishes a JSON representation of the inspection event. Configure broker, port, topic, QoS, retain, and optional credentials in the action configuration.

## Evidence

`EvidenceAction` stores event metadata and optional image/heatmap evidence under a directory organized by event ID.

## OPC UA

`OPCUAAction` is optional and lazily imports the OPC UA client dependency. Configure the endpoint and node IDs for the decision and optional score values.

## Design rule

Sources acquire data; models produce predictions; the decision layer determines PASS/FAIL/UNKNOWN; actions communicate the result. Industrial protocol details should stay inside action implementations.
