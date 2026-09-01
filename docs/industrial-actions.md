# Industrial Actions

AnomaVision can send existing inference results to external systems without coupling models to industrial protocols.

## Flow

```text
Source -> Model -> Existing AnomaVision classification -> ActionDispatcher
                                                     |-> MQTTAction
                                                     |-> EvidenceAction
                                                     `-> OPCUAAction
```

## Configuration

See `examples/actions/industrial_inspection.yaml` for a complete example.

Actions are configured as a list. Multiple actions may receive the same inspection result. A failure in one action is isolated by `ActionDispatcher` so it does not terminate inference processing.

## Design rule

Sources acquire data; models produce predictions; existing AnomaVision logic determines the classification; actions communicate or store the result. Industrial protocol details stay inside action implementations.
