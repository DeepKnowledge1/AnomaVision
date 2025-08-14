import torch
import torch.nn.functional as F
from torchvision import transforms as T
from typing import Tuple, Optional

def export_onnx(model, filepath: str, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224), 
                min_batch_size: int = 1, max_batch_size: Optional[int] = None) -> None:
    """Export the model to ONNX format with dynamic batch size support.

    Args:
        model: The PyTorch model to export.
        filepath: Path where to save the ONNX model.
        input_shape: Shape of the input tensor (B, C, H, W). Batch size will be made dynamic.
        min_batch_size: Minimum batch size for dynamic axes (default: 1).
        max_batch_size: Maximum batch size for dynamic axes (default: None for unlimited).
    """
    model.eval()  # Set to evaluation mode
    
    # Create dummy input with the specified shape
    dummy_input = torch.randn(*input_shape, device=model.device)
    
    # Define dynamic axes configuration
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'image_scores': {0: 'batch_size'},
        'score_map': {0: 'batch_size'}
    }
    
    # If max_batch_size is specified, add range constraints
    if max_batch_size is not None:
        # Note: ONNX dynamic_axes doesn't directly support min/max constraints
        # This would need to be handled at inference time or with additional validation
        print(f"Note: Batch size will be dynamic from {min_batch_size} to {max_batch_size}")
        print("ONNX format doesn't enforce these limits - validate at inference time.")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['image_scores', 'score_map'],
        dynamic_axes=dynamic_axes,
        verbose=True
    )
    
    
    print(f"Model exported to {filepath} with dynamic batch size support")
    print(f"Input shape: (batch_size, {input_shape[1]}, {input_shape[2]}, {input_shape[3]})")


# Usage examples:
# 
# # Basic usage - accepts any number of images
# export_onnx(model, "model_dynamic.onnx")
# 
# # With custom input dimensions
# export_onnx(model, "model_512.onnx", input_shape=(1, 3, 512, 512))