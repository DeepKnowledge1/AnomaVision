# Synthetic Defect Studio

Synthetic Defect Studio creates controlled defect examples from a normal reference image. It is designed for demonstrations, localization checks, regression tests, and controlled experiments when real defect images are limited.

The current generator uses a deterministic, CPU-friendly **surface-aware v2** pipeline. It combines irregular geometry, multi-scale mask variation, soft internal alpha, local luminance and texture modulation, and subtle surrounding cues instead of painting a fixed geometric shape with a flat color. The same input, defect type, severity, and seed produce the same result.

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

The mask is a binary grayscale image in which white pixels identify the synthetic defect region. The metadata includes the defect type, severity, seed, image dimensions, mask coverage percentage, and `synthesis_profile` (`surface_aware_v2`). The internal alpha used for blending is soft, but the exported annotation remains binary for training compatibility.

## Important limitation

Synthetic defects are useful for controlled testing, but they are not a substitute for real factory examples. Use them together with real normal and defective images because generated scratches, stains, dents, and holes may not reproduce the full texture, lighting, camera, and manufacturing variation of real defects.

## Reuse real defects at new locations

The **Reuse Real Defects** tab is a reference-driven workflow. Upload one normal target image and one or more defective reference images. The studio extracts each defect from its paired mask when supplied, or uses a conservative local-contrast heuristic when a mask is not available.

The studio then applies controlled random transformations and places the defect at new locations. Before compositing, the reference patch is partially matched to the target region's luminance and contrast statistics, and the alpha boundary is feathered to reduce pasted-edge artifacts. You can control the number of copies, scale range, rotation range, automatic-mask sensitivity, and seed. The output contains one combined binary image-level mask and a metadata record containing every placement coordinate, scale, rotation, source index, and seed.

For the most accurate results, upload masks in the same order as the defective reference images. A real defect image without a paired mask may include normal edges or background texture in the extracted region, so heuristic extraction should be reviewed before using the output for training.

This workflow is especially useful when a factory has a small number of real defect examples but needs more variation in location, orientation, and scale. It preserves the appearance of the uploaded defect while adapting its local intensity to the target surface. It is still an augmentation method, not a guarantee that synthetic samples are indistinguishable from factory captures; validate generated data against held-out real defects before deploying a model.
