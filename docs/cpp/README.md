
# 🖥️ AnomaVision C++ Inference

This folder contains the **C++ implementation of AnomaVision inference** using **ONNX Runtime** and **OpenCV**.
It is designed for **real-time anomaly detection** on edge devices and environments where Python is not available.

---

## 📦 Requirements

* C++17 or later
* [CMake ≥ 3.15](https://cmake.org/)
* [ONNX Runtime](https://onnxruntime.ai/) (prebuilt or built from source)
* [OpenCV](https://opencv.org/)

---

## ⚙️ Build

```bash
# From project root
cmake -S . -B build
cmake --build build --config Release
```

Make sure to update `CMakeLists.txt` with the correct paths to **ONNX Runtime** and **OpenCV** on your system.

---

## ▶️ Run Inference

```bash
./build/Release/onnx_inference.exe model.onnx D:/01-DATA/test \
    --save_viz D:/output --alpha 0.5 --thresh 13.0
```

Options:

* `--save_viz out_dir` → save annotated results
* `--alpha` → heatmap blending factor (default: 0.5)
* `--thresh` → anomaly threshold (default: 13.0)

---

## 📊 Example Output

```text
Found 10 images

Processing 000.png (1/10)
Score: 29.88  Anomalous: YES  Time: 31.53 ms
Number of anomalous pixels: 4271  Ratio: 0.085

Processing 007.png (2/10)
Score: 59.71  Anomalous: YES  Time: 33.93 ms
Number of anomalous pixels: 6068  Ratio: 0.121
...
```

* Average inference time ≈ **38.9 ms** (\~25.7 FPS)
* Visualization windows display heatmaps + bounding boxes
* Results saved if `--save_viz` is provided

---

## 🧩 Code Structure

* **`detect.cpp`** → entry point (CLI parser, launches app)
* **`oop_anomaly_detector.h/.cpp`** → modular pipeline

  * `Config` → runtime parameters
  * `Preprocessor` → image normalization & resize
  * `ONNXModel` → inference with ONNX Runtime
  * `Postprocessor` → thresholding, scores, masks
  * `Visualizer` → overlay, annotations, bounding boxes
  * `App` → orchestrates the full workflow

---

✅ This C++ module makes AnomaVision truly **edge-ready**, combining Python flexibility with C++ performance.
