
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from typing import Tuple

def export_onnx(model, filepath: str, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> None:
    """Export the model to ONNX format.

    Args:
        filepath: Path where to save the ONNX model.
        input_shape: Shape of the input tensor (B, C, H, W).
    """
    # assert model.mean is not None and model.cov_inv is not None, \
    #     "The model must be trained before exporting to ONNX"
    
    model.eval()  # Set to evaluation mode
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=model.device)
    
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
        dynamic_axes={
            'input': {0: 'batch_size'},
            'image_scores': {0: 'batch_size'},
            'score_map': {0: 'batch_size'}
        },
        verbose=True
    )
