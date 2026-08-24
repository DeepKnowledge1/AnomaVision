import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from pytorch_nndct.apis import torch_quantizer

# Workaround for Vitis AI 3.5.0 PatchCore XModel export.
# The deploy optimizer crashes in fuse_transpose_matmul().
import nndct_shared.compile.deploy_optimizer as _deploy_optimizer


class PatchCoreKV260(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.extractor = model.embeddings_extractor
        self.register_buffer(
            "memory_bank",
            F.normalize(model.memory_bank.float(), dim=-1),
        )

    def forward(self, x):
        features, _, _ = self.extractor(x, layer_indices=[0])

        # (B, 3136, 64)
        features = F.normalize(features, dim=-1)

        # (B, 3136, 819)
        similarity = torch.matmul(
            features,
            self.memory_bank.transpose(0, 1),
        )

        # nearest memory-bank distance
        max_similarity = similarity.amax(dim=-1)

        distances = torch.sqrt(
            torch.clamp(2.0 - 2.0 * max_similarity, min=0.0)
        )

        # 56 x 56
        batch = distances.shape[0]
        score_map = distances.reshape(batch, 1, 56, 56)

        # output anomaly map
        score_map = F.interpolate(
            score_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # image-level anomaly score
        image_score = distances.amax(dim=1)

        return image_score, score_map


def load_calibration_images(directory, size=224, limit=50):
    paths = []

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        paths.extend(Path(directory).glob(ext))

    paths = sorted(paths)[:limit]

    if not paths:
        raise RuntimeError(
            f"No calibration images found in {directory}"
        )

    tensors = []

    for path in paths:
        image = Image.open(path).convert("RGB")
        image = image.resize((size, size))

        tensor = torch.from_numpy(
            __import__("numpy").array(image)
        ).permute(2, 0, 1).float() / 255.0

        tensors.append(tensor)

    return tensors



for _name, _obj in vars(_deploy_optimizer).items():
    if isinstance(_obj, type) and hasattr(_obj, "fuse_transpose_matmul"):
        setattr(
            _obj,
            "fuse_transpose_matmul",
            lambda self: None,
        )
        print(f"[KV260] Disabled {_name}.fuse_transpose_matmul")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--calibration-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="./quantize_result",
    )

    parser.add_argument(
        "--quant_mode",
        choices=["calib", "test"],
        default="calib",
    )

    args = parser.parse_args()

    print("Loading:", args.model)

    model = torch.load(
        args.model,
        map_location="cpu",
        weights_only=False,
    )

    model.eval()

    print("Original model:", type(model))
    print("Memory bank:", tuple(model.memory_bank.shape))

    wrapper = PatchCoreKV260(model)
    wrapper.eval()

    dummy = torch.randn(
        1,
        3,
        224,
        224,
    )

    print("Testing wrapper...")

    with torch.no_grad():
        outputs = wrapper(dummy)

    print("Output 0:", tuple(outputs[0].shape))
    print("Output 1:", tuple(outputs[1].shape))

    calibration_images = load_calibration_images(
        args.calibration_dir
    )

    print(
        "Calibration images:",
        len(calibration_images),
    )

    quantizer = torch_quantizer(
        args.quant_mode,
        wrapper,
        (dummy,),
    )

    quant_model = quantizer.quant_model
    quant_model.eval()

    print("Running calibration...")

    with torch.no_grad():
        for i, image in enumerate(calibration_images):
            x = image.unsqueeze(0)

            quant_model(x)

            if (i + 1) % 10 == 0:
                print(
                    f"  {i + 1}/{len(calibration_images)}"
                )

    if args.quant_mode == "calib":

        Path(args.output_dir).mkdir(
            parents=True,
            exist_ok=True,
        )

        quantizer.export_quant_config()

        print()
        print("Calibration finished.")
        print("Quant config exported.")

    elif args.quant_mode == "test":

        Path(args.output_dir).mkdir(
            parents=True,
            exist_ok=True,
        )

        quantizer.export_xmodel(
            output_dir=args.output_dir,
            deploy_check=False,
        )

        print()
        print("XMODEL exported to:")
        print(args.output_dir)


if __name__ == "__main__":
    main()
