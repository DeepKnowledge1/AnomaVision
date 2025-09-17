# oop_anomaly_detector.py
# Requires: pip install onnxruntime opencv-python numpy

import argparse
import glob
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# -------------------- Config --------------------
@dataclass
class Config:
    model: str
    images_dir: Path
    out_dir: Path = Path("out")
    save_viz: bool = False
    alpha: float = 0.5
    thresh: float = 13.0
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    @staticmethod
    def parse():
        p = argparse.ArgumentParser(
            description="Python OOP Anomaly Detector (vis compatible with C++)."
        )
        p.add_argument("model", type=str, help="Path to ONNX model")
        p.add_argument("images_dir", type=Path, help="Directory of input images")
        p.add_argument("--save_viz", type=Path, help="Directory to save visualizations")
        p.add_argument(
            "--alpha",
            type=float,
            default=0.5,
            help="Overlay alpha (0 disables heat overlay)",
        )
        p.add_argument("--thresh", type=float, default=13.0, help="Anomaly threshold")
        args = p.parse_args()

        if not Path(args.model).exists():
            sys.exit(f"Model not found: {args.model}")
        if not args.images_dir.exists():
            sys.exit(f"Images dir not found: {args.images_dir}")

        save = args.save_viz is not None
        out_dir = args.save_viz if save else Path("out")
        if save:
            out_dir.mkdir(parents=True, exist_ok=True)

        return Config(
            model=args.model,
            images_dir=args.images_dir,
            out_dir=out_dir,
            save_viz=save,
            alpha=args.alpha,
            thresh=float(args.thresh),
        )


# -------------------- Result --------------------
@dataclass
class Result:
    score: float = 0.0
    heatmap: np.ndarray | None = None
    mask: np.ndarray | None = None
    anomalous: bool = False
    pixels: int = 0
    ratio: float = 0.0
    time_ms: float = 0.0


# -------------------- ImageList --------------------
class ImageList:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    @staticmethod
    def list(dirpath: Path):
        files = []
        for ext in ImageList.exts:
            files.extend(sorted(map(Path, glob.glob(str(dirpath / f"*{ext}")))))
        return files


# -------------------- Preprocessor --------------------
class Preprocessor:
    def __init__(self, H: int, W: int, mean: tuple, stdev: tuple):
        self.H, self.W = H, W
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(stdev, dtype=np.float32).reshape(3, 1, 1)

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        r = r.astype(np.float32) / 255.0
        r = r.transpose(2, 0, 1)  # HWC->CHW
        r = (r - self.mean) / self.std
        return r  # CHW float32


# -------------------- ONNXModel --------------------
class ONNXModel:
    def __init__(self, model_path: str):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
        self._input_shape = self.session.get_inputs()[
            0
        ].shape  # e.g. [1,3,H,W] or [N,3,H,W]
        self._output_names = [o.name for o in self.session.get_outputs()]

    @property
    def input_name(self):
        return self._input_name

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_names(self):
        return self._output_names

    def run(self, input_array: np.ndarray):
        # Ensure batch dimension = 1 if dynamic/None
        shape = list(self._input_shape)
        if shape[0] in (None, "None", -1):
            shape[0] = 1
        # Expect CHW -> add batch
        x = np.expand_dims(input_array, 0).astype(np.float32)  # [1,3,H,W]
        feeds = {self._input_name: x}
        return self.session.run(self._output_names, feeds)


# -------------------- Postprocessor --------------------
class Postprocessor:
    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def process(self, outputs: list) -> Result:
        r = Result()
        # C++ logic: either (score, heatmap) or single tensor that is scalar or HxW
        if len(outputs) >= 2:
            score = float(np.array(outputs[0]).reshape(-1)[0])
            heat = np.array(outputs[1]).squeeze()
            r.score = score
            if heat.ndim == 3:  # e.g. 1xHxW
                heat = heat[0]
            r.heatmap = heat.astype(np.float32)
        elif len(outputs) == 1:
            t = np.array(outputs[0]).squeeze()
            if t.size == 1:
                r.score = float(t)
            else:
                # assume heatmap; score = mean(heatmap)
                if t.ndim == 3:
                    t = t[0]
                r.heatmap = t.astype(np.float32)
                r.score = float(r.heatmap.mean())
        # Extra: blur + threshold + stats (match C++)
        if r.heatmap is not None:
            r.heatmap = cv2.GaussianBlur(r.heatmap, (33, 33), 4.0)
            _, mask = cv2.threshold(r.heatmap, self.threshold, 255.0, cv2.THRESH_BINARY)
            r.mask = mask.astype(np.uint8)
            r.pixels = int(cv2.countNonZero(r.mask))
            h, w = r.heatmap.shape[-2:]
            r.ratio = r.pixels / float(h * w)
        r.anomalous = r.score > self.threshold
        return r


# -------------------- Visualizer --------------------
class Visualizer:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def overlay(self, orig: np.ndarray, heat: np.ndarray | None) -> np.ndarray:

        return orig

    @staticmethod
    def _to_fixed(v: float, prec: int) -> str:
        return f"{v:.{prec}f}"

    def annotate(self, img: np.ndarray, r: Result, t_ms: float):
        txt_color = (0, 0, 255) if r.anomalous else (0, 255, 0)
        cv2.rectangle(img, (5, 5), (420, 110), (0, 0, 0), -1)
        cv2.putText(
            img,
            f"Score: {self._to_fixed(r.score,3)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            txt_color,
            2,
        )
        cv2.putText(
            img,
            "ANOMALOUS" if r.anomalous else "NORMAL",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            txt_color,
            2,
        )
        cv2.putText(
            img,
            f"Time: {self._to_fixed(t_ms,2)} ms",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    def draw_boundaries(self, img: np.ndarray, mask: np.ndarray | None):
        if mask is None:
            return
        rm = cv2.resize(
            mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        contours, _ = cv2.findContours(rm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 50:
                cv2.drawContours(img, [c], -1, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


# -------------------- App --------------------
class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = ONNXModel(cfg.model)

        # Expect input [N,3,H,W]
        ishape = self.model.input_shape
        if len(ishape) < 4:
            raise RuntimeError(f"Unexpected input shape: {ishape}")
        self.H = int(ishape[2] if ishape[2] not in (None, -1) else 224)
        self.W = int(ishape[3] if ishape[3] not in (None, -1) else 224)

        self.pre = Preprocessor(self.H, self.W, cfg.mean, cfg.std)
        self.post = Postprocessor(cfg.thresh)
        self.vis = Visualizer(cfg.alpha)

    def run(self):
        images = ImageList.list(self.cfg.images_dir)
        if not images:
            print("No images found")
            return 1
        print(f"Found {len(images)} images")

        for i, p in enumerate(images, 1):
            print(f"\nProcessing {p.name} ({i}/{len(images)})")
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to open {p}")
                continue

            x = self.pre.preprocess(img)

            t0 = time.perf_counter()
            outputs = self.model.run(x)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0

            res = self.post.process(outputs)
            res.time_ms = elapsed_ms

            disp = self.vis.overlay(img, res.heatmap)
            self.vis.annotate(disp, res, elapsed_ms)
            self.vis.draw_boundaries(disp, res.mask)

            win = f"Result: {p.name}"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 800, 600)
            cv2.imshow(win, disp)

            if self.cfg.save_viz:
                outp = self.cfg.out_dir / f"{p.stem}_res.png"
                cv2.imwrite(str(outp), disp)

            key = cv2.waitKey(1) & 0xFF
            cv2.destroyWindow(win)
            if key in (27, ord("q"), ord("Q")):
                print("User quit")
                break

            print(
                f"Score: {res.score:.4f}  Anomalous: {'YES' if res.anomalous else 'NO'}  Time: {res.time_ms:.2f} ms"
            )
            print(
                f"Number of anomalous pixels: {res.pixels}  Ratio to total pixels: {res.ratio:.6f}"
            )

        print("\nDone")
        return 0


# -------------------- Main --------------------
if __name__ == "__main__":
    cfg = Config.parse()
    sys.exit(App(cfg).run())
