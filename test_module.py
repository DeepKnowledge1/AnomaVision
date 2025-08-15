from anodet.inference.model.wrapper import ModelWrapper
import numpy as np

from anodet.utils import setup_logging

from anodet.utils import get_logger

# logger = get_logger(__name__)

logger = setup_logging("INFO")


logger = get_logger(__name__)

logger.info("Starting test module...")


# Path to your model (e.g., ONNX or PyTorch)
model_path = "distributions/padim_model.onnx"

# Create the wrapper. The backend is selected automatically based on the extension.
wrapper = ModelWrapper(model_path, device="cpu")

# Example input batch (B, C, H, W)
# Replace this with your real preprocessed data
dummy_batch = np.random.rand(3, 3, 224, 224).astype(np.float32)

# Run inference
scores, maps = wrapper.predict(dummy_batch)

print("Scores:", scores)
print("Score maps shape:", maps.shape)

# When you're done
wrapper.close()
