#!/usr/bin/env python3
"""
Model Export Utility
Export PyTorch models to ONNX and OpenVINO formats with dynamic batch support.
"""

import argparse
import time
import warnings
from pathlib import Path
from typing import Tuple, List, Optional

import torch


class ModelExporter:
    """Professional model exporter with clean interface."""
    
    def __init__(self, model_path: str, output_dir: str = "./"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_model(self) -> torch.nn.Module:
        """Load and prepare model for export."""
        print(f"Loading model: {self.model_path}")
        model = torch.load(self.model_path, map_location="cpu")
        
        # Handle DataParallel wrapper
        if hasattr(model, "module"):
            model = model.module
            
        model.eval()
        return model
    
    def _get_output_names(self, model: torch.nn.Module, dummy_input: torch.Tensor) -> List[str]:
        """Automatically determine output names from model."""
        with torch.no_grad():
            outputs = model(dummy_input)
            
        if isinstance(outputs, dict):
            return list(outputs.keys())
        elif isinstance(outputs, (list, tuple)):
            return [f"output_{i}" for i in range(len(outputs))]
        else:
            return ["output"]
    
    def export_onnx(self, 
                   input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                   output_name: str = "model.onnx",
                   opset_version: int = 17,
                   dynamic_batch: bool = True) -> Optional[Path]:
        """
        Export model to ONNX format.
        
        Args:
            input_shape: Model input shape (batch, channels, height, width)
            output_name: Output filename
            opset_version: ONNX opset version
            dynamic_batch: Enable dynamic batch size
            
        Returns:
            Path to exported file or None if failed
        """
        start_time = time.time()
        
        try:
            # Load model
            model = self._load_model()
            
            # Create dummy input
            dummy_input = torch.randn(*input_shape)
            
            # Warm up model
            with torch.no_grad():
                for _ in range(2):
                    _ = model(dummy_input)
                    
            # Get output names
            output_names = self._get_output_names(model, dummy_input)
            
            # Setup dynamic axes if requested
            dynamic_axes = None
            if dynamic_batch:
                dynamic_axes = {"input": {0: "batch_size"}}
                for name in output_names:
                    dynamic_axes[name] = {0: "batch_size"}
            
            # Export path
            output_path = self.output_dir / output_name
            
            # Suppress warnings
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            elapsed = time.time() - start_time
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            
            print(f"✅ Export successful ({elapsed:.1f}s)")
            print(f"   File: {output_path} ({file_size:.1f} MB)")
            print(f"   Dynamic batch: {dynamic_batch}")
            
            return output_path
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Export failed ({elapsed:.1f}s): {e}")
            return None
    
    def export_openvino(self,
                       input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                       output_name: str = "model_openvino",
                       fp16: bool = True,
                       dynamic_batch: bool = False) -> Optional[Path]:
        """
        Export model to OpenVINO format via ONNX.
        
        Args:
            input_shape: Model input shape (batch, channels, height, width)
            output_name: Output directory name
            fp16: Enable FP16 precision
            dynamic_batch: Enable dynamic batch size
            
        Returns:
            Path to exported directory or None if failed
        """
        start_time = time.time()
        
        try:
            # First export to ONNX (required for OpenVINO)
            temp_onnx = self.output_dir / "temp_model.onnx"
            onnx_path = self.export_onnx(
                input_shape=input_shape,
                output_name=temp_onnx.name,
                dynamic_batch=dynamic_batch  # Now respects the parameter
            )
            
            if onnx_path is None:
                raise RuntimeError("ONNX export failed")
            
            # Import OpenVINO tools
            try:
                from openvino.tools import mo
                import openvino.runtime as ov
            except ImportError:
                raise ImportError("OpenVINO not installed. Install with: pip install openvino")
            
            # Setup output directory
            output_dir = self.output_dir / output_name
            output_dir.mkdir(exist_ok=True)
            
            # Convert ONNX to OpenVINO
            print(f"Converting to OpenVINO (fp16={fp16}, dynamic_batch={dynamic_batch})...")
            ov_model = mo.convert_model(
                onnx_path,
                framework="onnx",
                compress_to_fp16=fp16
            )
            
            # Save OpenVINO model
            xml_path = output_dir / f"{output_name}.xml"
            ov.serialize(ov_model, str(xml_path))
            
            # Clean up temporary ONNX file
            if temp_onnx.exists():
                temp_onnx.unlink()
            
            elapsed = time.time() - start_time
            
            print(f"✅ OpenVINO export successful ({elapsed:.1f}s)")
            print(f"   Directory: {output_dir}")
            print(f"   Precision: {'FP16' if fp16 else 'FP32'}")
            print(f"   Dynamic batch: {dynamic_batch}")
            
            return output_dir
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ OpenVINO export failed ({elapsed:.1f}s): {e}")
            
            # Clean up on failure
            temp_onnx = self.output_dir / "temp_model.onnx"
            if temp_onnx.exists():
                temp_onnx.unlink()
                
            return None


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Export PaDiM models to ONNX and OpenVINO formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_data_path', type=str, default='./distributions/',
                        help='Directory containing model and output location')
    parser.add_argument('--model', type=str, default='padim_model.pt',
                        help='Model file (.pt for PyTorch)')
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output filename (if None, uses model name)")
    parser.add_argument("--format", choices=["onnx", "openvino", "both"], 
                       default="both", help="Export format")
    parser.add_argument("--input-shape", type=int, nargs=4, 
                       default=[2, 3, 224, 224],
                       metavar=("BATCH", "CHANNELS", "HEIGHT", "WIDTH"),
                       help="Input shape for the dummy tensor")
    parser.add_argument("--opset", type=int, default=17,
                       help="ONNX opset version")
    parser.add_argument("--static-batch", action="store_true",
                       help="Disable dynamic batch size")
    parser.add_argument("--fp32", action="store_true",
                       help="Use FP32 precision for OpenVINO (default: FP16)")
    
    args = parser.parse_args()
    
    # Setup paths using your naming convention
    model_path = Path(args.model_data_path) / args.model
    output_dir = Path(args.model_data_path)
    
    # Create exporter
    exporter = ModelExporter(model_path, output_dir)
    
    success = True
    
    # Generate output names based on original model name
    model_stem = Path(args.model).stem  # Gets 'padim_model' from 'padim_model.pt'
    
    # Use provided output_path or generate from model name
    if args.output_path:
        onnx_name = args.output_path
        openvino_name = Path(args.output_path).stem + "_openvino"
    else:
        onnx_name = f"{model_stem}.onnx"
        openvino_name = f"{model_stem}_openvino"
    
    # Determine if dynamic batch should be used
    use_dynamic_batch = not args.static_batch
    
    # Export based on format
    if args.format in ["onnx", "both"]:
        result = exporter.export_onnx(
            input_shape=tuple(args.input_shape),
            output_name=onnx_name,
            opset_version=args.opset,
            dynamic_batch=use_dynamic_batch
        )
        if result is None:
            success = False
    
    if args.format in ["openvino", "both"]:
        result = exporter.export_openvino(
            input_shape=tuple(args.input_shape),
            output_name=openvino_name,
            fp16=not args.fp32,
            dynamic_batch=use_dynamic_batch  # Now properly passed
        )
        if result is None:
            success = False
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()