# Real Core Regression Assets

This directory contains the immutable assets used by the GitHub Actions core-safety gate.

Required files:

- `model/model.pt` — validated known-good model artifact.
- `images/good_01.png` — known-good normal image.
- `images/defect_01.png` — known-good defective image.
- `baseline.json` — expected predictions and score tolerances.

The model is stored with Git LFS. Do not replace it casually: changing the model changes the regression baseline.

To update the baseline intentionally:

1. Validate the new model locally with the normal `anomavision detect` command.
2. Record the new scores.
3. Update `baseline.json`.
4. Run the real regression gate locally.
5. Commit the model and baseline together.

Private/proprietary images must not be committed. If these regression images are proprietary, use a private GitHub repository or replace this asset mechanism with a private artifact store.
