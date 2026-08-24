import argparse
from pathlib import Path

# Vitis AI 3.5.0 deploy-optimizer workaround.
import nndct_shared.compile.deploy_optimizer as _deploy_optimizer
import torch
from PIL import Image
from pytorch_nndct.apis import torch_quantizer

from anomavision.quantize.model.backends.xmodel.patchcore import PatchCoreKV260

for _name, _obj in vars(_deploy_optimizer).items():
    if isinstance(_obj, type) and hasattr(_obj, "fuse_transpose_matmul"):
        _obj.fuse_transpose_matmul = lambda self: None

        print(f"[KV260] Disabled {_name}.fuse_transpose_matmul")


def load_calibration_images(directory, size=224, limit=50):
    paths = []
    directory = Path(directory)
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        paths.extend(directory.glob(ext))
    paths = sorted(paths)[:limit]
    if not paths:
        raise RuntimeError(f"No calibration images found in {directory}")

    tensors = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((size, size))
            tensor = torch.from_numpy(__import__("numpy").array(image))
            tensors.append(tensor.permute(2, 0, 1).float() / 255.0)
    return tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--output-dir", default="./quantize_result")
    parser.add_argument("--quant_mode", choices=["calib", "test"], default="calib")
    args = parser.parse_args()

    print("Loading:", args.model)
    model = torch.load(args.model, map_location="cpu", weights_only=False)
    model.eval()
    print("Original model:", type(model))
    print("Memory bank:", tuple(model.memory_bank.shape))

    wrapper = PatchCoreKV260(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, 224, 224)
    print("Testing wrapper...")
    with torch.no_grad():
        outputs = wrapper(dummy)
    print("Output 0:", tuple(outputs[0].shape))
    print("Output 1:", tuple(outputs[1].shape))

    calibration_images = load_calibration_images(args.calibration_dir)
    print("Calibration images:", len(calibration_images))

    quantizer = torch_quantizer(args.quant_mode, wrapper, (dummy,))
    quant_model = quantizer.quant_model
    quant_model.eval()

    print("Running calibration...")
    with torch.no_grad():
        for i, image in enumerate(calibration_images):
            quant_model(image.unsqueeze(0))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(calibration_images)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.quant_mode == "calib":
        quantizer.export_quant_config()
        print("\nCalibration finished.")
        print("Quant config exported.")
    else:
        quantizer.export_xmodel(output_dir=str(output_dir), deploy_check=False)
        print("\nXMODEL exported to:")
        print(output_dir)


if __name__ == "__main__":
    main()
