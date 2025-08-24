"""Example usage of the model handling system"""

from anodet.inference import ModelService, ModelFormat, ModelConfig, DeviceType


def main():
    # Initialize service with custom config
    model_service = ModelService()

    # Create configurations for different models
    pytorch_config = ModelConfig(
        device=DeviceType.GPU, batch_size=8, precision="float16"
    )

    onnx_config = ModelConfig(
        device=DeviceType.CPU, batch_size=1, optimization_level="O2"
    )

    try:
        # Load different model types
        model_service.load_model(
            "models/pytorch_model.pt",
            ModelFormat.PYTORCH,
            "pytorch_model",
            pytorch_config,
        )

        model_service.load_model(
            "models/onnx_model.onnx", ModelFormat.ONNX, "onnx_model", onnx_config
        )

        model_service.load_model(
            "models/xmodel.xmodel", ModelFormat.XMODEL, "xmodel_model"
        )

        # Use models
        dummy_input = [[1.0, 2.0, 3.0]]  # Example input

        pytorch_result = model_service.predict("pytorch_model", dummy_input)
        onnx_result = model_service.predict("onnx_model", dummy_input)

        # Get model information
        pytorch_info = model_service.get_model_info("pytorch_model")
        print(f"PyTorch model info: {pytorch_info}")

        # List all loaded models
        loaded_models = model_service.list_loaded_models()
        print(f"Loaded models: {loaded_models}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Clean up
        for model_id in model_service.list_loaded_models():
            model_service.unload_model(model_id)


if __name__ == "__main__":
    main()
