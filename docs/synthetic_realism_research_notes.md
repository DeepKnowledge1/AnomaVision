# Synthetic defect realism research notes

## Evidence from industrial inspection literature and datasets

- Industrial surface-defect inspection is strongly affected by reflective materials and lighting; metal surfaces are especially common and difficult to image consistently.
- Real benchmark anomalies vary in size, color, structure, and appearance rather than forming one fixed geometric pattern.
- MVTec AD contains over 5,000 high-resolution images across 15 object and texture categories with pixel-precise anomaly masks.
- MVTec AD 2 adds challenging scenarios with more than 8,000 high-resolution images, including lighting conditions that may not occur in training data.
- Production-like synthesis should therefore model local surface texture, illumination, scale, boundary uncertainty, context, and small/subtle anomalies instead of relying on clean geometric primitives.
- Real-defect reuse should preserve surface-dependent appearance and avoid pasted rectangular crops, hard alpha boundaries, inconsistent illumination, and texture discontinuities.

## Sources

1. https://www.mdpi.com/2313-433X/9/10/193 — A Systematic Review on Deep Learning with CNNs Applied to Surface Defect Detection.
2. https://www.mvtec.com/research-teaching/datasets/mvtec-ad — MVTec AD benchmark dataset.
3. https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2 — MVTec AD 2 benchmark dataset.
