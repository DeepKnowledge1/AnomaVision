import argparse
import torch
from typing import Tuple, Optional, List
import os

def export_onnx(model,
                filepath: str,
                input_shape: Tuple[int, int, int, int] = (2, 3, 224, 224),
                opset: int = 17,
                output_names: Optional[List[str]] = None) -> None:
    model.eval()

    # Safe device detection
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    dummy_input = torch.randn(*input_shape, device=device)

    # Probe outputs to name them if needed
    with torch.no_grad():
        sample_out = model(dummy_input)
    if output_names is None:
        if isinstance(sample_out, dict):
            output_names = list(sample_out.keys())
        elif isinstance(sample_out, (list, tuple)):
            output_names = [f"output_{i}" for i in range(len(sample_out))]
        else:
            output_names = ["output"]

    dynamic_axes = {"input": {0: "batch_size"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    print(f"✅ Exported with dynamic batch to {filepath}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PaDiM model for anomaly detection.")
    
    parser.add_argument('--model_data_path', type=str, default='./distributions/',
                        help='Directory to save model distributions and ONNX file.')

    parser.add_argument("--output_path", type=str,default="padim_model.onnx" ,help="Output ONNX file path")
    
    parser.add_argument('--model', type=str, default='padim_model.pt',
                        help='Model file (.pt for PyTorch, .onnx for ONNX)')

    parser.add_argument("--input-shape", type=int, nargs=4, default=(2, 3, 224, 224),
                        metavar=("BATCH", "CHANNELS", "HEIGHT", "WIDTH"),
                        help="Input shape for the dummy tensor")

    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")

    return parser.parse_args()



def main(args):

    # Load model
    
    
    MODEL_DATA_PATH = os.path.realpath(args.model_data_path)    
    # Load model
    model_path = os.path.join(MODEL_DATA_PATH, args.model)

    output_path = os.path.join(MODEL_DATA_PATH, args.output_path)
    model = torch.load(model_path, map_location="cpu")
    if hasattr(model, "module"):  # Handle DataParallel
        model = model.module

    export_onnx(model=model,
                filepath=output_path,
                input_shape=tuple(args.input_shape),
                opset=args.opset)


if __name__ == "__main__":
    args = parse_args()

    main(args)
