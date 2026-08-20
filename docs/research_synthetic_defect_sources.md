# Synthetic defect implementation references

These external references informed the Synthetic Defect Studio upgrade.

1. Anomalib CutPaste proposal: https://github.com/open-edge-platform/anomalib/issues/3462

Key finding: Anomalib’s discussed synthetic pipeline relies heavily on Perlin-noise synthesis; the proposal identifies missing structural realism and localized defect patterns. It recommends CutPaste-Normal, CutPaste-Scar, and CutPaste-Union variants, plus blending and intensity perturbation, while retaining CPU efficiency and backward compatibility.

2. Awesome Industrial Anomaly Detection: https://github.com/m-3lab/awesome-industrial-anomaly-detection

Key finding: the field includes anomaly synthesis benchmarks and surveys, with emphasis on realistic industrial anomaly generation, segmentation, and reproducible datasets.

3. Synthetic MVTec AD defect-detection dataset: https://huggingface.co/datasets/anywaylabs/synthetic-mvtec-ad-defect-detection

Key finding: a practical synthetic dataset should expose explicit annotations, stable directory structure, defect-level controls, placement/size/severity variation, and a distinction between synthetic training data and real validation data. The dataset uses image/label splits and documents limitations of synthetic-to-real transfer.

4. CutPaste paper: https://arxiv.org/abs/2104.04015

Key finding: CutPaste creates synthetic anomalies by cutting an image patch and pasting it at another location, supporting self-supervised anomaly detection and localization.

Implementation implications for AnomaVision: add CutPaste, deterministic seeds, exact masks, metadata manifests, bounded dataset export, train/validation structure, real-defect reuse with optional paired masks, controlled scale/rotation/location transforms, and clear warnings that synthetic data must be reviewed against real factory data.
