# Synthetic Defect Studio

Synthetic Defect Studio creates controlled defect examples from a normal reference image. It is designed for demonstrations, localization checks, regression tests, and controlled experiments when real defect images are limited.

The generator is deliberately lightweight and CPU-friendly. It uses deterministic Pillow drawing and image-processing operations rather than a large generative model. The same input, defect type, severity, and seed produce the same result.

## Supported defects

| Defect | Description |
|---|---|
| `scratch` | A dark linear surface mark. |
| `crack` | A branched linear fracture-like mark. |
| `stain` | A soft, colored surface region. |
| `dent` | A shaded circular indentation with a highlight rim. |
| `hole` | A dark circular missing-material region. |
| `cutpaste` | A texture-preserving patch copied from the same image and blended at another location. |

## Gradio workflow

Start the UI from the repository root:

```bash
uv run python apps/ui/gradio_app.py
```

Open the local address printed by Gradio, then select **Synthetic Studio**. Upload a normal image, choose a defect type and severity, set a seed, and click **Generate Synthetic Defect**.

The studio displays the synthetic defect and its exact ground-truth mask. It also provides downloadable `synthetic_defect.png`, `ground_truth_mask.png`, and `metadata.json` files.

## Generate a dataset from the CLI

For repeatable experiments, generate a bounded dataset directly from a folder of normal images:

```powershell
anomavision synthesize `
  --input_dir .\normal_images `
  --output_dir .\synthetic_dataset `
  --defect_types scratch crack stain dent hole cutpaste `
  --severity medium `
  --copies_per_type 2 `
  --val_ratio 0.2 `
  --seed 42
```

The exporter writes:

```text
synthetic_dataset/
├── images/train/{normal,anomaly}/
├── images/val/{normal,anomaly}/
├── masks/train/{normal,anomaly}/
├── masks/val/{normal,anomaly}/
├── manifest.jsonl
└── dataset_manifest.json
```

`manifest.jsonl` links every image to its mask, label, source image, defect type, severity, and seed. The exporter uses a deterministic seed, validates the requested defect types, and stops when `--max_samples` is reached so an accidental large input folder cannot create an unbounded dataset.

## Python API

```python
from PIL import Image

from anomavision.synthetic_defects import generate_synthetic_defect

normal = Image.open("normal_product.png")
defective, mask, metadata = generate_synthetic_defect(
    normal,
    defect_type="scratch",
    severity="medium",
    seed=42,
)

defective.save("synthetic_defect.png")
mask.save("ground_truth_mask.png")
print(metadata)
```

The mask is a grayscale image in which white pixels identify the synthetic defect region. The metadata includes the defect type, severity, seed, image dimensions, and mask coverage percentage.

## Important limitation

Synthetic defects are useful for controlled testing, but they are not a substitute for real factory examples. Use them together with real normal and defective images because generated scratches, stains, dents, and holes may not reproduce the full texture, lighting, camera, and manufacturing variation of real defects.
