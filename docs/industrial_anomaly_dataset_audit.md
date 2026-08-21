# Industrial Anomaly Dataset Audit

## Purpose

Synthetic Defect Studio should not use one generic defect generator for every industrial task. The listed datasets cover several different problem families: surface damage, texture anomalies, logical or structural violations, PCB manufacturing defects, and RGB–3D inspection. This audit maps those families to the kind of synthesis that is technically justified.

> **Central conclusion:** MVTec AD, VisA, BTAD, MPDD, KolektorSDD, KolektorSDD2, and real PCB datasets are evidence about industrial appearance. DTD and DAGM are useful texture or algorithmic references but are not equivalent to real production-defect data. Eyecandies is explicitly synthetic. MVTec 3D-AD requires geometry-aware synthesis rather than RGB-only compositing.

## Dataset inventory

| Dataset | What the primary source reports | Defect / modality characteristics | Synthesis relevance | License or access caution |
|---|---|---|---|---|
| **MVTec AD** | More than 5,000 high-resolution images across 15 object and texture categories, with defect-free training images, normal and anomalous test images, and pixel-precise anomaly annotations [1]. | Object-surface defects and texture anomalies; category-specific appearance and masks. | Core benchmark for surface-conditioned image synthesis and mask statistics. Profiles must be category-specific rather than universal. | Official page states CC BY-NC-SA 4.0 and explicitly prohibits commercial use without clarification [1]. |
| **MVTec LOCO AD** | 3,644 images across five industrial categories, with both structural and logical anomalies and pixel-precise ground truth [2]. | Structural anomalies include scratches, dents, and contamination. Logical anomalies violate constraints, such as an object in an invalid position or a required object missing. | Requires two generators: appearance/texture defects and scene-constraint defects. A scratch generator cannot represent logical anomalies. | Official page states CC BY-NC-SA 4.0 and non-commercial restriction [2]. |
| **BTAD** | Primary dataset references describe 2,830 real-world images from three industrial products [3]. | Real product imagery with surface and body anomalies; product-specific capture and defect distributions. | Useful for cross-product generalization and subtle industrial context. Requires per-product calibration. | Distribution and license must be checked from the original release before redistribution or commercial use. |
| **VisA** | AWS Registry describes 10,821 images across 12 object classes, including 9,621 normal and 1,200 anomalous images [4]. | Multi-object visual anomaly data with classification and segmentation use; object-specific anomaly appearance. | Useful for measuring whether generated anomalies preserve object identity and category context. | AWS Registry links the dataset to CC BY 4.0 [4]; retain attribution and confirm upstream terms before commercial packaging. |
| **MPDD** | Maintainer repository describes more than 1,000 metal-part images with pixel-precise defect masks; anomaly-free training and mixed normal/anomalous validation [5]. | Industrial metal parts including brackets, tubes, connectors, and plates; examples include scratches, bends/missing parts, painting defects, defective parts, and major rust. | Strong source for metal-surface appearance, paint variation, rust morphology, and part-specific geometry. | Repository has a license file, but the exact commercial permissions should be read and recorded before redistribution [5]. |
| **KolektorSDD** | Official ViCoS page reports 399 controlled industrial images: 52 with visible defects and 347 defect-free; original images are approximately 500 by 1240–1270 pixels, with fine and box annotations available [6]. | Grayscale electrical-commutator imagery; extremely imbalanced production data; defects may be visible in only one view of an item. | Important for tiny, low-contrast, elongated surface defects and high-resolution grayscale context. | CC BY-NC-SA 4.0; commercial use requires contacting the authors [6]. |
| **KolektorSDD2** | Official ViCoS page reports 356 visible-defect images and 2,979 defect-free images at approximately 230 by 630 pixels [7]. | Controlled industrial production images with scratches, minor spots, surface imperfections, and several defect types. | Useful for subtle, thin, irregular defects and strong class imbalance; a good test of whether synthesis creates obvious shortcuts. | CC BY-NC-SA 4.0; commercial use requires contacting the authors [7]. |
| **DAGM 2007** | The DAGM competition source describes defect-free and defective subsets on statistically textured backgrounds [8]. | Texture inspection benchmark; commonly treated as synthetic/statistical texture-defect data rather than real factory capture. | Useful for texture-anomaly algorithm tests and hard-negative texture generation, but not a primary photorealism target. | Verify the competition terms before redistribution; do not assume that a mirror or re-upload grants commercial rights [8]. |
| **PCB defect datasets** | The PKU-Market-PCB source is described as 1,386 defect images in six categories: missing hole, mouse bite, open, short, spur, and spurious copper [9]. Other PCB releases differ in scale and license. | Highly structured planar geometry, repeated traces, solder/copper patterns, and discrete manufacturing failures. | Needs registration-aware, geometry-constrained synthesis. Naive free-form blending can create impossible traces and invalid electrical geometry. | PCB datasets are not interchangeable; verify the exact release, source, and license before use. |
| **DTD** | Oxford reports 5,640 images in 47 human-centric texture categories, 120 images per category, with images sourced from Google and Flickr [10]. | Natural texture images in the wild, not industrial defect captures. | Useful as a texture prior, hard-negative bank, or source for controlled texture perturbations, but not evidence of production defect morphology. | Oxford makes it available for research purposes [10]; check terms before commercial use. |
| **Eyecandies** | The project describes a synthetic dataset for unsupervised multimodal anomaly detection, with RGB, depth, and normal maps in an industrial conveyor scenario [11]. | Procedurally generated multimodal scenes; synthetic RGB–3D anomalies. | Useful for testing multimodal pipelines and geometry-aware rendering ideas, not for validating realism against real factory defects. | Follow the project’s own distribution terms; synthetic does not automatically mean unrestricted commercial use. |
| **MVTec 3D-AD** | Official page reports more than 4,000 high-resolution scans acquired by an industrial 3D sensor across 10 object categories, with precise anomalous-test annotations [12]. | RGB plus 3D/depth geometry; defects may be geometric rather than purely photometric. | Requires displacement/depth/normal synthesis, sensor noise modeling, and synchronized RGB–3D masks. RGB-only output must not claim 3D realism. | Official page states CC BY-NC-SA 4.0 and non-commercial restriction [12]. |

## What the audit changes

### 1. Separate the problem into generation families

The studio should expose distinct generation profiles rather than a single list of `scratch`, `crack`, `stain`, `dent`, and `hole` primitives.

| Profile | Datasets that motivate it | Required behavior |
|---|---|---|
| **Surface-conditioned appearance** | MVTec AD, BTAD, VisA, MPDD, KolektorSDD, KolektorSDD2 | Preserve target object identity and local texture while generating defects with category-specific size, contrast, morphology, and boundary statistics. |
| **Logical and structural constraints** | MVTec LOCO AD, PCB datasets | Modify object count, position, orientation, connectivity, or required-part presence while preserving scene geometry and camera context. |
| **Texture anomaly** | DTD, DAGM, MVTec texture categories | Model texture frequency, orientation, periodicity, and local phase; distinguish texture perturbation from object damage. |
| **3D geometry** | MVTec 3D-AD, Eyecandies | Generate synchronized RGB, depth, and normal changes; model dents, missing material, bulges, and sensor response. |
| **Reference-conditioned defect transfer** | KolektorSDD/2, MPDD, VisA, MVTec AD | Learn a defect appearance from a masked reference and re-render it into a target context instead of pasting raw pixels. |

### 2. Replace generic procedural realism with a learned or calibrated pipeline

The current Pillow-based implementation can be retained as a fast baseline, but it cannot be the production-realism backend. A serious pipeline needs the following stages:

1. **Dataset adapter and profiling.** Read images, masks, product/category labels, resolution, color space, and available depth. Estimate defect area, aspect ratio, connected components, edge softness, local contrast, texture frequency, and placement relative to the object.
2. **Normal-surface conditioning.** Encode the target object and its local surface context. The generator must know where a defect can plausibly occur and which background texture, illumination, and reflectance should be preserved.
3. **Context-aware mask generation.** Generate irregular masks from measured distributions, not arbitrary polygons. For logical anomalies, generate scene-constraint violations separately from pixel-surface anomalies.
4. **Reference-conditioned inpainting or rendering.** For photorealistic 2D output, use a trained or fine-tuned inpainting diffusion/GAN model conditioned on the target image, defect mask, and reference defect features. For 3D tasks, use geometry/displacement and synchronized RGB rendering.
5. **Quality filtering.** Reject images with pasted-edge artifacts, abnormal color shifts, implausible object geometry, excessive defect contrast, or a real-vs-synthetic classifier confidence that is too high.
6. **Held-out evaluation.** Train the detector only on the allowed training split, evaluate on real held-out anomalies, and compare real-only, synthetic-only, and mixed training. Visual inspection alone is not enough.

## Acceptance criteria for “production-ready”

The feature should not be labeled production-ready until it satisfies all of the following on at least one fully specified product/category profile:

| Criterion | Required evidence |
|---|---|
| Morphology fidelity | Generated mask distributions for area, aspect ratio, connected components, skeleton length, and boundary complexity are compared with held-out real masks. |
| Appearance fidelity | Local contrast, color/luminance residuals, texture spectra, and edge statistics overlap with real defects without a trivial synthetic shortcut. |
| Context fidelity | A reviewer or automated quality gate confirms that target geometry, reflections, repeated texture, and object identity remain valid. |
| Detector hardness | A real-vs-synthetic classifier performs near chance after calibration, or its failure modes are documented and filtered. |
| Downstream utility | Mixed real/synthetic training improves or preserves AUROC, pixel AUROC, PRO, and localization quality on held-out real defects. |
| Reproducibility | Every sample records dataset profile, source references, seed, model version, mask provenance, and generation parameters. |
| Licensing | The exact dataset release and model/data usage rights are recorded; non-commercial datasets are not silently bundled into commercial training or distribution. |

## Immediate implementation consequence

The next engineering branch should not merely tune the existing primitive functions. It should add a **dataset profile and calibration layer**, keep the current procedural engine explicitly labeled as `baseline_procedural`, and introduce a separate **reference-conditioned learned backend** behind an opt-in interface. Without real image-mask pairs for the target product, the system can be benchmarked on public datasets but cannot honestly guarantee production photorealism for an unknown factory domain.

## References

[1]: https://www.mvtec.com/research-teaching/datasets/mvtec-ad "MVTec AD official dataset page"
[2]: https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad "MVTec LOCO AD official dataset page"
[3]: https://github.com/dataset-ninja/btad "BTAD dataset repository"
[4]: https://registry.opendata.aws/visa/ "VisA AWS Open Data Registry entry"
[5]: https://github.com/stepanje/MPDD "MPDD maintainer repository"
[6]: https://www.vicos.si/resources/kolektorsdd/ "KolektorSDD official ViCoS page"
[7]: https://www.vicos.si/resources/kolektorsdd2/ "KolektorSDD2 official ViCoS page"
[8]: https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html "DAGM 2007 competition source"
[9]: https://www.nature.com/articles/s41597-024-03656-8 "PKU-Market-PCB dataset paper"
[10]: https://www.robots.ox.ac.uk/~vgg/data/dtd/ "Oxford Describable Textures Dataset"
[11]: https://eyecan-ai.github.io/eyecandies/ "Eyecandies project page"
[12]: https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad "MVTec 3D-AD official dataset page"
