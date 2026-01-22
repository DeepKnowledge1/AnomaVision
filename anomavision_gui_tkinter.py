"""
@file anomavision_gui_tkinter.py
AnomaVision — Professional, modern Tkinter GUI with gradient header, logo, drag & drop,
optimized inference (AMP/TF32), fast Gaussian blur, safe parsing, card layout, toasts.

This version fixes:
- Pillow 10+ removal of ImageDraw.textsize -> uses textbbox
- Windows Tk error for bg="" -> uses real colors everywhere
- Ensures self.logo_tk is created before use (with safe fallback)
"""

import ast
import logging
import os
import sys
import threading
import time
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageTk
from torch.utils.data import DataLoader

# Optional drag & drop support: pip install TkinterDnD2
_DND_AVAILABLE = False
try:
    from TkinterDnD2 import DND_FILES, TkinterDnD

    _DND_AVAILABLE = True
except Exception:
    _DND_AVAILABLE = False

# ----- Your package imports (existing project modules) -----
import anomavision
from anomavision.config import load_config
from anomavision.general import Profiler
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.inference.modelType import ModelType
from anomavision.padim import Padim
from anomavision.utils import get_logger
from PIL import Image, ImageTk
# -----------------------------------------------------------------------------
# Global fast paths and helpers
# -----------------------------------------------------------------------------

# torch cache
cache_dir = Path(".cache")
cache_dir.mkdir(exist_ok=True)
os.environ["TORCH_HOME"] = str(cache_dir.absolute())

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable TF32 for faster matmul/conv on Ampere+
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def _safe_literal(s, default=None):
    try:
        if isinstance(s, str) and s.strip():
            return ast.literal_eval(s)
        return default
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Micro-UI helpers: tooltips, toasts, gradient, dnd
# -----------------------------------------------------------------------------


class Tooltip:
    def __init__(self, widget, text, delay=450):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._id = None
        self.tip = None
        widget.bind("<Enter>", self._enter)
        widget.bind("<Leave>", self._leave)

    def _enter(self, _):
        self._schedule()

    def _leave(self, _):
        self._unschedule()
        self._hide()

    def _schedule(self):
        self._unschedule()
        self._id = self.widget.after(self.delay, self._show)

    def _unschedule(self):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None

    def _show(self):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#222",
            foreground="#fff",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=4,
            font=("Segoe UI", 9),
        )
        lbl.pack()

    def _hide(self):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def toast(root, text, ms=2200, bg="#111827", fg="#E5E7EB"):
    """Non-blocking toast notification."""
    win = tk.Toplevel(root)
    win.wm_overrideredirect(True)
    win.configure(bg=bg)
    pad = 14
    lbl = tk.Label(
        win, text=text, bg=bg, fg=fg, font=("Segoe UI", 10), padx=pad, pady=pad
    )
    lbl.pack()
    # position bottom-right
    try:
        root.update_idletasks()
        x = root.winfo_rootx() + max(0, root.winfo_width() - win.winfo_reqwidth() - 24)
        y = root.winfo_rooty() + max(
            0, root.winfo_height() - win.winfo_reqheight() - 24
        )
        win.geometry(f"+{x}+{y}")
    except Exception:
        pass
    # subtle fade-in/out (best-effort; ignored if not supported)
    try:
        win.attributes("-alpha", 0.0)

        def _fade_in(step=0):
            a = step / 10.0
            win.attributes("-alpha", min(0.98, a))
            if step < 10:
                win.after(20, _fade_in, step + 1)

        _fade_in()
    except Exception:
        pass

    def _close():
        try:
            for step in range(10, -1, -1):
                a = step / 10.0
                win.attributes("-alpha", a)
                win.update()
                time.sleep(0.02)
        except Exception:
            pass
        win.destroy()

    win.after(ms, _close)


def draw_horizontal_gradient(width, height, left="#0EA5E9", right="#6366F1"):
    """Return a PIL.Image horizontal gradient."""
    img = Image.new("RGB", (width, height), left)
    drw = ImageDraw.Draw(img)
    r1, g1, b1 = ImageColor.getrgb(left)
    r2, g2, b2 = ImageColor.getrgb(right)
    for x in range(width):
        t = x / max(1, width - 1)
        r = int(r1 * (1 - t) + r2 * t)
        g = int(g1 * (1 - t) + g2 * t)
        b = int(b1 * (1 - t) + b2 * t)
        drw.line([(x, 0), (x, height)], fill=(r, g, b))
    return img


def make_logo(size=44, bg="#38BDF8", fg="#001225"):
    """Generate a simple circular 'AV' logo as a PIL image (Pillow 10+ safe)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    drw = ImageDraw.Draw(img)
    drw.ellipse((0, 0, size - 1, size - 1), fill=bg)

    text = "AV"
    # Try common fonts; fall back to default
    font = None
    for name in ("arial.ttf", "SegoeUI.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, int(size * 0.44))
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    # Pillow 10+: use textbbox instead of textsize
    bbox = drw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    drw.text(((size - tw) // 2, (size - th) // 2 - 1), text, fill=fg, font=font)
    return img


def enable_dnd_for_entry(entry, set_var):
    """Enable drag&drop for Entry if TkinterDnD2 exists."""
    if not _DND_AVAILABLE:
        return

    def _drop(event):
        data = event.data
        # On Windows paths may come quoted; on mac they are file://
        paths = entry.tk.splitlist(data)
        path = paths[0] if paths else ""
        if path.startswith("file://"):
            import urllib.parse

            path = urllib.parse.unquote(path.replace("file://", ""))
        # Strip surrounding braces on Windows
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        set_var(path)
        toast(entry.winfo_toplevel(), "Path set via drag & drop")
        return "break"

    entry.drop_target_register(DND_FILES)
    entry.dnd_bind("<<Drop>>", _drop)


# -----------------------------------------------------------------------------
# Worker Threads (performance-optimized)
# -----------------------------------------------------------------------------


class TrainingWorker(threading.Thread):
    def __init__(
        self,
        config,
        progress_callback,
        finished_callback,
        error_callback,
        status_callback=None,
    ):
        super().__init__(daemon=True)
        self.config = config
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
        self.error_callback = error_callback
        self.status_callback = status_callback or (lambda *_: None)
        self._stop_event = threading.Event()
        self.logger = get_logger("anomavision.train")

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            self.progress_callback("Starting model training…")
            self.status_callback("Training: preparing dataset…")
            root = os.path.join(
                os.path.realpath(self.config["dataset_path"]),
                self.config["class_name"],
                "train",
                "good",
            )
            if not os.path.isdir(root):
                raise FileNotFoundError(f"Train dir does not exist: {root}")
            ds = anomavision.AnodetDataset(
                root,
                resize=self.config["resize"],
                crop_size=self.config["crop_size"],
                normalize=self.config["normalize"],
                mean=self.config["norm_mean"],
                std=self.config["norm_std"],
            )
            if len(ds) == 0:
                raise ValueError(f"No images found: {root}")
            dl = DataLoader(
                ds, batch_size=int(self.config["batch_size"]), shuffle=False
            )
            self.progress_callback(f"Dataset loaded: {len(ds)} images")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.progress_callback(f"Using device: {device.type}")
            self.status_callback(f"Training on {device.type.upper()}…")
            layer_indices = self.config["layer_indices"]
            if isinstance(layer_indices, str):
                layer_indices = ast.literal_eval(layer_indices)
            padim = Padim(
                backbone=self.config["backbone"],
                device=device,
                layer_indices=layer_indices,
                feat_dim=int(self.config["feat_dim"]),
            )
            self.progress_callback("Fitting model…")
            t_fit = time.perf_counter()
            padim.fit(dl)
            self.progress_callback(
                f"Training completed in {time.perf_counter()-t_fit:.2f}s"
            )

            # Save model
            model_data_path = Path(self.config["model_data_path"])
            model_data_path.mkdir(parents=True, exist_ok=True)
            class_name = self.config["class_name"]
            prefixed_model_name = f"{class_name}_{self.config['output_model']}"
            model_path = model_data_path / prefixed_model_name
            torch.save(padim, str(model_path))
            self.progress_callback(f"Model saved to: {model_path}")
            try:
                stats_path = model_path.with_suffix(".pth")
                padim.save_statistics(str(stats_path), half=True)
                self.progress_callback(f"Statistics saved to: {stats_path}")
            except Exception as e:
                self.progress_callback(f"Failed to save statistics: {e}")
            self.finished_callback(padim)
            self.status_callback("Training finished ✔")
        except Exception as e:
            msg = f"Error during training: {str(e)}\n{traceback.format_exc()}"
            self.error_callback(msg)
            self.status_callback("Training failed ✖")


class InferenceWorker(threading.Thread):
    def __init__(
        self,
        model_path,
        img_path,
        config,
        progress_callback,
        finished_callback,
        error_callback,
        status_callback=None,
        perf_callback=None,
    ):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.img_path = img_path
        self.config = config
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
        self.error_callback = error_callback
        self.status_callback = status_callback or (lambda *_: None)
        self.perf_callback = perf_callback or (lambda *_: None)
        self._stop_event = threading.Event()
        self.logger = get_logger("anomavision.detect")

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            self.progress_callback("Starting inference…")
            self.status_callback("Inference: loading model…")
            prof = {
                k: Profiler()
                for k in [
                    "model_loading",
                    "data_loading",
                    "warmup",
                    "inference",
                    "postprocessing",
                    "visualization",
                ]
            }

            with prof["model_loading"]:
                req = (self.config.get("device") or "auto").lower()
                if req == "cuda" and torch.cuda.is_available():
                    device_str = "cuda"
                elif req == "cpu":
                    device_str = "cpu"
                else:
                    device_str = "cuda" if torch.cuda.is_available() else "cpu"
                if device_str == "cuda":
                    torch.backends.cudnn.benchmark = True
                model = ModelWrapper(self.model_path, device_str)
                model_type = ModelType.from_extension(self.model_path)
                self.progress_callback(
                    f"Model loaded: {model_type.value.upper()} on {device_str.upper()}"
                )

            with prof["data_loading"]:
                self.status_callback("Inference: preparing data…")
                test_dataset = anomavision.AnodetDataset(
                    self.img_path,
                    resize=self.config["resize"],
                    crop_size=self.config["crop_size"],
                    normalize=self.config["normalize"],
                    mean=self.config["norm_mean"],
                    std=self.config["norm_std"],
                )
                pin_mem = bool(self.config.get("pin_memory", False))
                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=self.config["batch_size"],
                    num_workers=self.config.get("num_workers", 1),
                    pin_memory=pin_mem,
                    persistent_workers=self.config.get("num_workers", 1) > 0,
                )
                self.progress_callback(f"Dataset loaded: {len(test_dataset)} images")

            try:
                with prof["warmup"]:
                    self.status_callback("Inference: warming up…")
                    first = next(iter(test_dataloader))
                    first_batch = first[0].to(device_str, non_blocking=True)
                    use_amp = device_str == "cuda"
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        model.warmup(batch=first_batch, runs=2)
                    self.progress_callback("Warmup complete")
            except StopIteration:
                self.progress_callback("Dataset empty, skipping warmup")
            except Exception as e:
                self.progress_callback(f"Warmup issue: {e}")

            all_images, all_scores, all_maps, all_classes, all_viz = [], [], [], [], []
            batch_count = 0
            self.status_callback("Inference: running…")
            for bidx, (batch, images, _, _) in enumerate(test_dataloader):
                if self._stop_event.is_set():
                    self.progress_callback("Inference stopped by user.")
                    break
                batch_count += 1
                self.progress_callback(
                    f"Processing batch {bidx+1}/{len(test_dataloader)}…"
                )
                with prof["inference"]:
                    use_amp = device_str == "cuda"
                    batch = batch.to(device_str, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        image_scores, score_maps = model.predict(batch)
                with prof["postprocessing"]:
                    score_maps = self.adaptive_gaussian_blur(
                        score_maps, kernel_size=33, sigma=4
                    )
                    img_clf = anomavision.classification(
                        image_scores, self.config["thresh"]
                    )
                    all_images.extend(images)
                    all_scores.extend(image_scores.tolist())
                    all_maps.extend(score_maps)
                    all_classes.extend(img_clf.tolist())
                with prof["visualization"]:
                    viz = self.generate_visualizations(
                        images, score_maps, img_clf, self.config
                    )
                    all_viz.extend(viz)

            total_images = len(all_images)
            fps = prof["inference"].get_fps(total_images)
            avg_ms = prof["inference"].get_avg_time_ms(batch_count)
            self.perf_callback(fps, avg_ms)
            self.progress_callback("Inference completed")
            self.status_callback("Inference finished ✔")
            self.finished_callback(
                all_images, all_scores, all_maps, all_classes, all_viz
            )
            model.close()
        except Exception as e:
            msg = f"Error during inference: {str(e)}\n{traceback.format_exc()}"
            self.error_callback(msg)
            self.status_callback("Inference failed ✖")

    @staticmethod
    def adaptive_gaussian_blur(input_array, kernel_size=33, sigma=4):
        try:
            import torchvision.transforms as T

            if torch.is_tensor(input_array):
                if input_array.dim() == 2:
                    x = input_array.unsqueeze(0).unsqueeze(0)
                    y = T.GaussianBlur(kernel_size, sigma=sigma)(x)
                    return y.squeeze(0).squeeze(0)
                elif input_array.dim() == 3:
                    x = input_array.unsqueeze(1)
                    y = T.GaussianBlur(kernel_size, sigma=sigma)(x)
                    return y.squeeze(1)
                elif input_array.dim() == 4:
                    return T.GaussianBlur(kernel_size, sigma=sigma)(input_array)
        except ImportError:
            pass
        if hasattr(input_array, "detach"):
            input_array = input_array.detach().cpu().numpy()
        if isinstance(input_array, np.ndarray) and input_array.dtype == np.float16:
            input_array = input_array.astype(np.float32)
        try:
            k = int(kernel_size) | 1
            if input_array.ndim == 2:
                return cv2.GaussianBlur(
                    input_array, (k, k), sigma, sigma, cv2.BORDER_REFLECT101
                )
            elif input_array.ndim == 3:
                return np.stack(
                    [
                        cv2.GaussianBlur(x, (k, k), sigma, sigma, cv2.BORDER_REFLECT101)
                        for x in input_array
                    ],
                    0,
                )
            elif input_array.ndim == 4:
                N, C, H, W = input_array.shape
                resh = input_array.transpose(0, 2, 3, 1).reshape(N * H, W, C)
                out = np.empty_like(resh)
                for c in range(C):
                    out[:, :, c] = cv2.GaussianBlur(
                        resh[:, :, c], (k, k), sigma, sigma, cv2.BORDER_REFLECT101
                    )
                out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
                return out
        except Exception:
            try:
                from scipy.ndimage import gaussian_filter

                truncate = (kernel_size - 1) / (2 * sigma)
                return gaussian_filter(input_array, sigma=sigma, truncate=truncate)
            except Exception as e:
                raise e

    @staticmethod
    def generate_visualizations(images, score_maps, classifications, config):
        visualizations = []
        try:
            score_maps_np = (
                score_maps.detach().cpu().numpy()
                if isinstance(score_maps, torch.Tensor)
                else np.array(score_maps)
            )
            score_map_clf = anomavision.classification(score_maps_np, config["thresh"])
            image_clf = classifications
            heatmap_images = anomavision.visualization.heatmap_images(
                np.asarray(images), score_maps_np, alpha=config.get("viz_alpha", 0.5)
            )
            highlighted_images = anomavision.visualization.highlighted_images(
                [images[i] for i in range(len(images))],
                score_map_clf,
                color=tuple(map(int, config.get("viz_color", "128,0,128").split(","))),
                alpha=0.5,
            )
            boundary_images = anomavision.visualization.framed_boundary_images(
                np.asarray(images),
                score_map_clf,
                image_clf,
                padding=config.get("viz_padding", 30),
            )
            for i in range(len(images)):
                visualizations.append(
                    {
                        "original": images[i],
                        "heatmap": heatmap_images[i],
                        "highlighted": (
                            highlighted_images[i]
                            if i < len(highlighted_images)
                            else None
                        ),
                        "boundary": (
                            boundary_images[i] if i < len(boundary_images) else None
                        ),
                        "classification": image_clf[i] if i < len(image_clf) else 0,
                    }
                )
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            for i in range(len(images)):
                visualizations.append(
                    {
                        "original": images[i],
                        "heatmap": None,
                        "highlighted": None,
                        "boundary": None,
                        "classification": 0,
                    }
                )
        return visualizations


class ExportWorker(threading.Thread):
    def __init__(
        self,
        model_path,
        output_dir,
        export_format,
        config,
        progress_callback,
        finished_callback,
        error_callback,
        status_callback=None,
    ):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.output_dir = output_dir
        self.export_format = export_format
        self.config = config
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
        self.error_callback = error_callback
        self.status_callback = status_callback or (lambda *_: None)
        self._stop_event = threading.Event()
        self.logger = get_logger("anomavision.export")

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            self.progress_callback("Starting export…")
            self.status_callback("Export: preparing…")
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from export import ModelExporter

            exporter = ModelExporter(
                model_path=Path(self.model_path),
                output_dir=Path(self.output_dir),
                logger=logging.getLogger("anomavision.export"),
                device=self.config.get("device", "cpu"),
            )
            h, w = (
                self.config["crop_size"]
                if self.config["crop_size"] is not None
                else self.config["resize"]
            )
            input_shape = [1, 3, h, w]
            name = Path(self.model_path).stem
            if self.export_format == "onnx":
                self.status_callback("Exporting to ONNX…")
                out = exporter.export_onnx(
                    tuple(input_shape),
                    f"{name}.onnx",
                    opset_version=self.config.get("opset", 18),
                    dynamic_batch=self.config.get("dynamic_batch", True),
                )
            elif self.export_format == "torchscript":
                self.status_callback("Exporting to TorchScript…")
                out = exporter.export_torchscript(
                    tuple(input_shape),
                    f"{name}.torchscript",
                    optimize=self.config.get("optimize", False),
                )
            elif self.export_format == "openvino":
                self.status_callback("Exporting to OpenVINO…")
                out = exporter.export_openvino(
                    tuple(input_shape),
                    f"{name}_openvino",
                    fp16=self.config.get("fp16", True),
                    dynamic_batch=self.config.get("dynamic_batch", True),
                )
            else:
                raise ValueError("Unsupported export format")
            if not out:
                raise RuntimeError("Export failed")
            self.progress_callback(f"Exported to: {out}")
            self.status_callback("Export finished ✔")
            self.finished_callback(str(out))
        except Exception as e:
            msg = f"Error during export: {str(e)}\n{traceback.format_exc()}"
            self.error_callback(msg)
            self.status_callback("Export failed ✖")


# -----------------------------------------------------------------------------
# GUI — Professional & Catchy
# -----------------------------------------------------------------------------


class AnomaVisionGUI:
    LIGHT = {
        "bg": "#F7F8FA",
        "panel": "#FFFFFF",
        "fg": "#1F2937",
        "muted": "#6B7280",
        "accent": "#3B82F6",
        "accent_fg": "#FFFFFF",
        "border": "#E5E7EB",
        "badge": "#EEF2FF",
        "ok": "#10B981",
        "warn": "#F59E0B",
        "err": "#EF4444",
        "slot": "#0B1020",
    }
    DARK = {
        "bg": "#0B1220",
        "panel": "#0F172A",
        "fg": "#E5E7EB",
        "muted": "#9CA3AF",
        "accent": "#60A5FA",
        "accent_fg": "#0B1220",
        "border": "#1F2937",
        "badge": "#111827",
        "ok": "#34D399",
        "warn": "#FBBF24",
        "err": "#F87171",
        "slot": "#000000",
    }

    def __init__(self, root):
        self.root = root

        try:
            self.root.iconbitmap("av.ico")
        except Exception:
            print("Could not load icon – ensure av.ico is in the same folder.")

        self.root.title("AnomaVision — Anomaly Detection (Pro)")
        self.root.geometry("1366x860")
        self.root.minsize(1180, 760)
        self.theme = "DARK"
        self.colors = self.DARK.copy()
        self._build_style()

        # runtime vars
        self.current_model = None
        self.inference_results = None
        self.current_result_index = 0

        # config
        self.config = self.load_default_config()

        # layout
        self._build_header()
        self._build_body()
        self._build_statusbar()
        self.setup_logging()
        self._bind_shortcuts()

        self._apply_theme()
        toast(self.root, "Welcome to AnomaVision (Pro UI) ✨")

    # ---------- Style & Theme ----------
    def _build_style(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure("TButton", font=("Segoe UI", 10), padding=6)
        self.style.configure("TLabel", font=("Segoe UI", 10))
        self.style.configure("TEntry", padding=4)
        self.style.configure("TFrame", background=self.colors["bg"])
        self.style.configure(
            "TNotebook.Tab", padding=(12, 8, 12, 8), font=("Segoe UI", 10, "bold")
        )
        self.style.configure(
            "Green.Horizontal.TProgressbar",
            troughcolor=self.colors["panel"],
            background="#10B981",      # ✅ green bar
            thickness=14
        )

    def toggle_theme(self):
        self.theme = "LIGHT" if self.theme == "DARK" else "DARK"
        self.colors = self.LIGHT.copy() if self.theme == "LIGHT" else self.DARK.copy()
        self._apply_theme()

    def _apply_theme(self):
        c = self.colors
        self.root.configure(bg=c["bg"])
        self.header_canvas.configure(bg=c["panel"])
        self.header_overlay.configure(bg=c["panel"])
        self.sidebar.configure(bg=c["panel"], bd=0, highlightthickness=0)
        for b in self.nav_buttons:
            b.configure(
                bg=c["panel"],
                fg=c["fg"],
                activebackground=c["badge"],
                activeforeground=c["fg"],
            )
        for frame in [self.content]:
            frame.configure(bg=c["bg"])
        for card in getattr(self, "_cards", []):
            card.configure(
                bg=c["panel"],
                highlightbackground=c["border"],
                highlightcolor=c["border"],
            )
        for lbl in getattr(self, "_card_titles", []):
            lbl.configure(bg=c["panel"], fg=c["fg"])
        for lbl in getattr(self, "_image_labels", []):
            lbl.configure(bg=c["slot"], fg="#FFFFFF")
        # status
        self.statusbar.configure(bg=c["panel"])
        self.status_left.configure(bg=c["panel"], fg=c["muted"])
        self.status_right.configure(bg=c["panel"], fg=c["muted"])
        # redraw gradient header
        self._redraw_header_gradient()

    # ---------- Header with Gradient + Logo ----------
    def _build_header(self):
        c = self.colors
        self.header = tk.Frame(self.root, height=84, bg=c["panel"])
        self.header.pack(side=tk.TOP, fill=tk.X)

        # Gradient canvas
        self.header_canvas = tk.Canvas(
            self.header, height=84, highlightthickness=0, bd=0, bg=c["panel"]
        )
        self.header_canvas.pack(fill=tk.BOTH, expand=True)
        self.header_canvas.bind("<Configure>", lambda e: self._redraw_header_gradient())

        # Overlay frame (uses real bg color; no empty/transparent values)
        self.header_overlay = tk.Frame(
            self.header_canvas, bg=c["panel"], highlightthickness=0
        )
        self.header_canvas.create_window(
            0,
            0,
            anchor="nw",
            window=self.header_overlay,
            width=self.root.winfo_width(),
            height=84,
        )

        left = tk.Frame(self.header_overlay, bg=c["panel"], padx=22, pady=12)
        left.pack(side=tk.LEFT, fill=tk.Y)

        # --- Create logo image BEFORE using it (safe fallback if PIL font missing) ---



        try:
            icon_image = Image.open("av.png")  # use PNG for clean scaling; ICO also works
            icon_image = icon_image.resize((44, 44), Image.LANCZOS)
            self.logo_tk = ImageTk.PhotoImage(icon_image)
        except Exception:
            # fallback to text icon if file missing
            self.logo_tk = None

        if self.logo_tk is not None:
            self.logo_lbl = tk.Label(left, image=self.logo_tk, bg=c["panel"], bd=0)
        else:
            self.logo_lbl = tk.Label(
                left,
                text="AV",
                bg=c["panel"],
                fg="#001225",
                font=("Segoe UI", 16, "bold"),
                bd=0,
            )
        self.logo_lbl.pack(side=tk.LEFT, padx=(0, 10))

        title_wrap = tk.Frame(left, bg=c["panel"])
        title_wrap.pack(side=tk.LEFT)
        self.title_lbl = tk.Label(
            title_wrap,
            text="AnomaVision",
            font=("Segoe UI", 20, "bold"),
            bg=c["panel"],
            fg="#FFFFFF",
        )
        self.title_lbl.pack(anchor="w")
        self.subtitle_lbl = tk.Label(
            title_wrap,
            text="Anomaly Detection • PaDiM • Professional Suite",
            font=("Segoe UI", 10),
            bg=c["panel"],
            fg="#DBEAFE",
        )
        self.subtitle_lbl.pack(anchor="w")

        right = tk.Frame(self.header_overlay, bg=c["panel"], padx=16, pady=14)
        right.pack(side=tk.RIGHT)

        self.device_badge = tk.Label(
            right,
            text="Device: Auto",
            padx=10,
            pady=4,
            font=("Segoe UI", 10, "bold"),
            bg="#0EA5E9",
            fg="#001225",
        )
        self.device_badge.pack(side=tk.LEFT, padx=(0, 10))

        self.theme_btn = tk.Button(
            right,
            text="🌗  Theme",
            command=self.toggle_theme,
            padx=12,
            pady=6,
            relief="flat",
            bg=c["panel"],
            fg=c["fg"],
            bd=0,
            activebackground=c["badge"],
        )
        self.theme_btn.pack(side=tk.LEFT, padx=4)
        Tooltip(self.theme_btn, "Toggle Light/Dark Theme")

        self.quick_infer = tk.Button(
            right,
            text="▶  Inference",
            command=lambda: self.notebook.select(self.inference_tab),
            padx=12,
            pady=6,
            relief="flat",
            bg=c["panel"],
            fg=c["fg"],
            bd=0,
            activebackground=c["badge"],
        )
        self.quick_infer.pack(side=tk.LEFT, padx=4)
        Tooltip(self.quick_infer, "Jump to Inference")

        self.quick_export = tk.Button(
            right,
            text="⤴  Export",
            command=lambda: self.notebook.select(self.export_tab),
            padx=12,
            pady=6,
            relief="flat",
            bg=c["panel"],
            fg=c["fg"],
            bd=0,
            activebackground=c["badge"],
        )
        self.quick_export.pack(side=tk.LEFT, padx=4)
        Tooltip(self.quick_export, "Jump to Export")

    def _redraw_header_gradient(self):
        try:
            self.header_canvas.delete("all")
            w = max(300, self.header_canvas.winfo_width())
            h = max(84, self.header_canvas.winfo_height())
            grad = draw_horizontal_gradient(w, h, left="#0EA5E9", right="#6366F1")
            self.grad_tk = ImageTk.PhotoImage(grad)
            self.header_canvas.create_image(0, 0, anchor="nw", image=self.grad_tk)
            # reattach overlay
            self.header_canvas.create_window(
                0, 0, anchor="nw", window=self.header_overlay, width=w, height=h
            )
        except Exception as e:
            logger.warning(f"Header gradient redraw failed: {e}")

    # ---------- Body (Sidebar + Content) ----------
    def _build_body(self):
        c = self.colors
        body = tk.Frame(self.root, bg=c["bg"])
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Sidebar
        self.sidebar = tk.Frame(body, width=220, bg=c["panel"])
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self._build_sidebar()

        # Content
        self.content = tk.Frame(body, bg=c["bg"])
        self.content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Notebook
        self.notebook = ttk.Notebook(self.content)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        self.training_tab = ttk.Frame(self.notebook)
        self.inference_tab = ttk.Frame(self.notebook)
        self.export_tab = ttk.Frame(self.notebook)
        self.logs_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.inference_tab, text="Inference")
        self.notebook.add(self.export_tab, text="Export")
        self.notebook.add(self.logs_tab, text="Logs")

        # Build tab content
        self._cards, self._card_titles, self._image_labels = [], [], []
        self.create_training_tab()
        self.create_inference_tab()
        self.create_export_tab()
        self.create_log_tab()

    def _build_sidebar(self):
        c = self.colors

        def nav_btn(text, cmd):
            b = tk.Button(
                self.sidebar,
                text=text,
                relief="flat",
                font=("Segoe UI", 11, "bold"),
                padx=16,
                pady=10,
                anchor="w",
                command=cmd,
                bg=c["panel"],
                fg=c["fg"],
                activebackground=c["badge"],
            )
            b.pack(fill=tk.X, padx=14, pady=6)
            self.nav_buttons.append(b)
            return b

        self.nav_buttons = []
        tk.Label(
            self.sidebar,
            text="Navigation",
            font=("Segoe UI", 12, "bold"),
            bg=c["panel"],
            fg=c["fg"],
        ).pack(anchor="w", padx=16, pady=(16, 4))
        nav_btn("🏋  Training", lambda: self.notebook.select(self.training_tab))
        nav_btn("🧪  Inference", lambda: self.notebook.select(self.inference_tab))
        nav_btn("📦  Export", lambda: self.notebook.select(self.export_tab))
        nav_btn("📝  Logs", lambda: self.notebook.select(self.logs_tab))

        sep = tk.Frame(self.sidebar, height=1, bg=c["border"])
        sep.pack(fill=tk.X, padx=12, pady=12)

        hlp = tk.Button(
            self.sidebar,
            text="❓  About / Help",
            relief="flat",
            command=self._show_about,
            padx=16,
            pady=10,
            anchor="w",
            font=("Segoe UI", 10),
            bg=c["panel"],
            fg=c["fg"],
            activebackground=c["badge"],
        )
        hlp.pack(fill=tk.X, padx=12, pady=(4, 12))
        Tooltip(hlp, "About the app & shortcuts")

    def _card(self, master, title):
        c = self.colors
        outer = tk.Frame(master, bg=c["panel"], highlightthickness=1)
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        header = tk.Label(
            outer, text=title, font=("Segoe UI", 12, "bold"), bg=c["panel"], fg=c["fg"]
        )
        header.pack(anchor="w", padx=14, pady=(10, 6))
        self._cards.append(outer)
        self._card_titles.append(header)
        return outer

    # ---------- Status Bar ----------
    def _build_statusbar(self):
        c = self.colors
        self.statusbar = tk.Frame(self.root, height=28, bg=c["panel"])
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_left = tk.Label(
            self.statusbar,
            text="Ready",
            font=("Segoe UI", 9),
            bg=c["panel"],
            fg=c["muted"],
        )
        self.status_left.pack(side=tk.LEFT, padx=12)
        self.status_right = tk.Label(
            self.statusbar,
            text="FPS: —   •  avg/batch: — ms",
            font=("Segoe UI", 9),
            bg=c["panel"],
            fg=c["muted"],
        )
        self.status_right.pack(side=tk.RIGHT, padx=12)

    def set_status(self, text):
        self.status_left.config(text=text)

    def set_perf(self, fps, avg_ms):
        self.status_right.config(
            text=(
                f"FPS: {fps:.2f}   •  avg/batch: {avg_ms:.2f} ms"
                if fps
                else "FPS: —   •  avg/batch: — ms"
            )
        )

    # ---------- Tabs ----------
    def create_training_tab(self):
        container = tk.Frame(self.training_tab, bg=self.colors["bg"])
        container.pack(fill=tk.BOTH, expand=True)

        cfg = self._card(container, "Training Configuration")
        form = tk.Frame(cfg, bg=self.colors["panel"])
        form.pack(fill=tk.X, padx=14, pady=(0, 12))

        def row(lbl):
            r = tk.Frame(form, bg=self.colors["panel"])
            r.pack(fill=tk.X, pady=4)
            tk.Label(
                r,
                text=lbl,
                width=18,
                anchor="w",
                bg=self.colors["panel"],
                fg=self.colors["muted"],
            ).pack(side=tk.LEFT)
            return r

        # dataset path (DnD-enabled)
        r = row("Dataset Path")
        self.dataset_path_var = tk.StringVar(value=self.config.get("dataset_path", ""))
        e = ttk.Entry(r, textvariable=self.dataset_path_var)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(r, text="Browse", command=self.browse_dataset_path).pack(
            side=tk.LEFT
        )
        if _DND_AVAILABLE:
            enable_dnd_for_entry(e, self.dataset_path_var.set)

        # class name
        r = row("Class Name")
        self.class_name_var = tk.StringVar(value=self.config.get("class_name", "E85"))
        ttk.Entry(r, textvariable=self.class_name_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6
        )

        # backbone
        r = row("Backbone")
        self.backbone_var = tk.StringVar(value=self.config.get("backbone", "resnet18"))
        ttk.Combobox(
            r,
            textvariable=self.backbone_var,
            values=["resnet18", "wide_resnet50"],
            state="readonly",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        # batch size
        r = row("Batch Size")
        self.batch_size_var = tk.IntVar(value=self.config.get("batch_size", 2))
        ttk.Spinbox(r, from_=1, to=128, textvariable=self.batch_size_var).pack(
            side=tk.LEFT, padx=6
        )

        # feat dim
        r = row("Feature Dim")
        self.feat_dim_var = tk.IntVar(value=self.config.get("feat_dim", 50))
        ttk.Spinbox(r, from_=1, to=1000, textvariable=self.feat_dim_var).pack(
            side=tk.LEFT, padx=6
        )

        # layer indices
        r = row("Layer Indices")
        layer_indices = self.config.get("layer_indices", [0, 1])
        li_str = str(layer_indices if isinstance(layer_indices, list) else [0, 1])
        self.layer_indices_var = tk.StringVar(value=li_str)
        ttk.Entry(r, textvariable=self.layer_indices_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6
        )

        # resize
        r = row("Resize [h,w]")
        self.resize_var = tk.StringVar(value=str(self.config.get("resize", [224, 224])))
        ttk.Entry(r, textvariable=self.resize_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6
        )

        # crop size
        r = row("Crop Size [h,w]")
        self.crop_size_var = tk.StringVar(value=str(self.config.get("crop_size", "")))
        ttk.Entry(r, textvariable=self.crop_size_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6
        )

        # model output path (DnD)
        r = row("Model Output Dir")
        self.model_output_path_var = tk.StringVar(
            value=self.config.get("model_data_path", "./distributions")
        )
        e = ttk.Entry(r, textvariable=self.model_output_path_var)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(r, text="Browse", command=self.browse_model_output_path).pack(
            side=tk.LEFT
        )
        if _DND_AVAILABLE:
            enable_dnd_for_entry(e, self.model_output_path_var.set)

        # output model name
        r = row("Output Model Name")
        self.output_model_name_var = tk.StringVar(
            value=self.config.get("output_model", "padim_model.pt")
        )
        ttk.Entry(r, textvariable=self.output_model_name_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6
        )

        # actions
        actions = tk.Frame(cfg, bg=self.colors["panel"])
        actions.pack(fill=tk.X, padx=14, pady=(0, 14))
        self.train_button = ttk.Button(
            actions, text="▶ Start Training (Ctrl+T)", command=self.start_training
        )
        self.train_button.pack(side=tk.LEFT)
        self.train_stop_button = ttk.Button(
            actions, text="■ Stop", command=self.stop_training, state=tk.DISABLED
        )
        self.train_stop_button.pack(side=tk.LEFT, padx=8)

        self.train_progress_var = tk.IntVar(value=0)
        self.train_progress = ttk.Progressbar(
            cfg,
            variable=self.train_progress_var,
            maximum=100,
            mode="determinate",
            style="Green.Horizontal.TProgressbar"
        )
        self.train_progress.pack(fill=tk.X, padx=14, pady=(0, 12))
        self.train_progress.pack_forget()
        self.train_percent_label = tk.Label(cfg, text="0%", bg=self.colors["panel"], fg=self.colors["fg"])
        self.train_percent_label.pack(anchor="w", padx=14)

    def update_training_progress(self, value):
        value = max(0, min(100, int(value)))
        self.train_progress_var.set(value)
        self.train_percent_label.config(text=f"{value}%")
        self.train_progress.update_idletasks()

    def create_inference_tab(self):
        container = tk.Frame(self.inference_tab, bg=self.colors["bg"])
        container.pack(fill=tk.BOTH, expand=True)

        cfg = self._card(container, "Model & Data")
        form = tk.Frame(cfg, bg=self.colors["panel"])
        form.pack(fill=tk.X, padx=14, pady=(0, 12))

        def row(lbl):
            r = tk.Frame(form, bg=self.colors["panel"])
            r.pack(fill=tk.X, pady=4)
            tk.Label(
                r,
                text=lbl,
                width=18,
                anchor="w",
                bg=self.colors["panel"],
                fg=self.colors["muted"],
            ).pack(side=tk.LEFT)
            return r

        # model path (DnD)
        r = row("Model Path")
        self.model_path_var = tk.StringVar(
            value=self.config.get("model", "./distributions/padim_model.pt")
        )
        e = ttk.Entry(r, textvariable=self.model_path_var)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(r, text="Browse", command=self.browse_model_path).pack(side=tk.LEFT)
        if _DND_AVAILABLE:
            enable_dnd_for_entry(e, self.model_path_var.set)

        # image folder (DnD)
        r = row("Image Folder")
        self.image_path_var = tk.StringVar(value=self.config.get("img_path", ""))
        e = ttk.Entry(r, textvariable=self.image_path_var)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(r, text="Browse", command=self.browse_image_path).pack(side=tk.LEFT)
        if _DND_AVAILABLE:
            enable_dnd_for_entry(e, self.image_path_var.set)

        params = self._card(container, "Inference Parameters")
        p = tk.Frame(params, bg=self.colors["panel"])
        p.pack(fill=tk.X, padx=14, pady=(0, 10))

        # threshold
        r = tk.Frame(p, bg=self.colors["panel"])
        r.pack(fill=tk.X, pady=4)
        tk.Label(
            r,
            text="Threshold",
            width=18,
            anchor="w",
            bg=self.colors["panel"],
            fg=self.colors["muted"],
        ).pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=self.config.get("thresh", 13.0))
        ttk.Spinbox(
            r,
            from_=0.0,
            to=100.0,
            increment=0.1,
            textvariable=self.threshold_var,
            width=12,
        ).pack(side=tk.LEFT, padx=6)

        # device
        r = tk.Frame(p, bg=self.colors["panel"])
        r.pack(fill=tk.X, pady=4)
        tk.Label(
            r,
            text="Device",
            width=18,
            anchor="w",
            bg=self.colors["panel"],
            fg=self.colors["muted"],
        ).pack(side=tk.LEFT)
        self.device_var = tk.StringVar(value=self.config.get("device", "auto"))
        ttk.Combobox(
            r,
            textvariable=self.device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT, padx=6)

        # batch size
        r = tk.Frame(p, bg=self.colors["panel"])
        r.pack(fill=tk.X, pady=4)
        tk.Label(
            r,
            text="Batch Size",
            width=18,
            anchor="w",
            bg=self.colors["panel"],
            fg=self.colors["muted"],
        ).pack(side=tk.LEFT)
        self.infer_batch_size_var = tk.IntVar(value=self.config.get("batch_size", 1))
        ttk.Spinbox(
            r, from_=1, to=128, textvariable=self.infer_batch_size_var, width=12
        ).pack(side=tk.LEFT, padx=6)

        # padding
        r = tk.Frame(p, bg=self.colors["panel"])
        r.pack(fill=tk.X, pady=4)
        tk.Label(
            r,
            text="Boundary Padding",
            width=18,
            anchor="w",
            bg=self.colors["panel"],
            fg=self.colors["muted"],
        ).pack(side=tk.LEFT)
        self.viz_padding_var = tk.IntVar(value=self.config.get("viz_padding", 30))
        ttk.Spinbox(
            r, from_=0, to=100, textvariable=self.viz_padding_var, width=12
        ).pack(side=tk.LEFT, padx=6)

        actions = tk.Frame(params, bg=self.colors["panel"])
        actions.pack(fill=tk.X, padx=14, pady=(0, 12))
        self.infer_button = ttk.Button(
            actions, text="▶ Start Inference (Ctrl+I)", command=self.start_inference
        )
        self.infer_button.pack(side=tk.LEFT)
        self.infer_stop_button = ttk.Button(
            actions, text="■ Stop", command=self.stop_inference, state=tk.DISABLED
        )
        self.infer_stop_button.pack(side=tk.LEFT, padx=8)
        self.prev_result_button = ttk.Button(
            actions, text="⟨  Prev", command=self.show_prev_result, state=tk.DISABLED
        )
        self.prev_result_button.pack(side=tk.LEFT, padx=(20, 6))
        self.next_result_button = ttk.Button(
            actions, text="Next  ⟩", command=self.show_next_result, state=tk.DISABLED
        )
        self.next_result_button.pack(side=tk.LEFT)

        # results
        results = self._card(container, "Results")
        top = tk.Frame(results, bg=self.colors["panel"])
        top.pack(fill=tk.X, padx=14, pady=(6, 0))
        self.results_info_var = tk.StringVar(value="No inference results")
        ttk.Label(top, textvariable=self.results_info_var).pack(side=tk.LEFT)
        self.anomaly_score_var = tk.StringVar(value="Anomaly Score: 0.00")
        ttk.Label(
            top, textvariable=self.anomaly_score_var, font=("Segoe UI", 11, "bold")
        ).pack(side=tk.RIGHT)

        grid = tk.Frame(results, bg=self.colors["panel"])
        grid.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
        grid.columnconfigure((0, 1, 2, 3), weight=1)
        grid.rowconfigure(0, weight=1)

        def slot(text, col):
            lbl = tk.Label(
                grid,
                text=text,
                bg=self.colors["slot"],
                fg="#FFFFFF",
                relief="solid",
                bd=1,
            )
            lbl.grid(row=0, column=col, padx=6, pady=6, sticky="nsew")
            self._image_labels.append(lbl)
            return lbl

        self.original_image_label = slot("Original", 0)
        self.heatmap_label = slot("Heatmap", 1)
        self.highlighted_label = slot("Highlighted", 2)
        self.boundary_label = slot("Boundary", 3)

    def create_export_tab(self):
        container = tk.Frame(self.export_tab, bg=self.colors["bg"])
        container.pack(fill=tk.BOTH, expand=True)

        cfg = self._card(container, "Export Configuration")
        form = tk.Frame(cfg, bg=self.colors["panel"])
        form.pack(fill=tk.X, padx=14, pady=(0, 12))

        def row(lbl):
            r = tk.Frame(form, bg=self.colors["panel"])
            r.pack(fill=tk.X, pady=4)
            tk.Label(
                r,
                text=lbl,
                width=18,
                anchor="w",
                bg=self.colors["panel"],
                fg=self.colors["muted"],
            ).pack(side=tk.LEFT)
            return r

        r = row("Model Path")
        self.export_model_path_var = tk.StringVar(
            value=self.config.get("model", "./distributions/padim_model.pt")
        )
        e = ttk.Entry(r, textvariable=self.export_model_path_var)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(r, text="Browse", command=self.browse_export_model_path).pack(
            side=tk.LEFT
        )
        if _DND_AVAILABLE:
            enable_dnd_for_entry(e, self.export_model_path_var.set)

        r = row("Output Directory")
        self.export_output_dir_var = tk.StringVar(
            value=self.config.get("model_data_path", "./distributions/anomav_exp")
        )
        e = ttk.Entry(r, textvariable=self.export_output_dir_var)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(r, text="Browse", command=self.browse_export_output_dir).pack(
            side=tk.LEFT
        )
        if _DND_AVAILABLE:
            enable_dnd_for_entry(e, self.export_output_dir_var.set)

        r = row("Format")
        self.export_format_var = tk.StringVar(value="onnx")
        ttk.Combobox(
            r,
            textvariable=self.export_format_var,
            values=["onnx", "torchscript", "openvino"],
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT, padx=6)

        r = row("ONNX Opset")
        self.onnx_opset_var = tk.IntVar(value=self.config.get("opset", 18))
        ttk.Spinbox(r, from_=1, to=20, textvariable=self.onnx_opset_var, width=10).pack(
            side=tk.LEFT, padx=6
        )

        r = tk.Frame(form, bg=self.colors["panel"])
        r.pack(fill=tk.X, pady=4)
        self.dynamic_batch_var = tk.BooleanVar(
            value=self.config.get("dynamic_batch", True)
        )
        ttk.Checkbutton(r, text="Dynamic Batch", variable=self.dynamic_batch_var).pack(
            side=tk.LEFT, padx=(0, 14)
        )
        tk.Label(
            r,
            text="Export Device",
            width=14,
            anchor="w",
            bg=self.colors["panel"],
            fg=self.colors["muted"],
        ).pack(side=tk.LEFT)
        self.export_device_var = tk.StringVar(value=self.config.get("device", "cpu"))
        ttk.Combobox(
            r,
            textvariable=self.export_device_var,
            values=["cpu", "cuda"],
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT)

        actions = tk.Frame(cfg, bg=self.colors["panel"])
        actions.pack(fill=tk.X, padx=14, pady=(0, 10))
        self.export_button = ttk.Button(
            actions, text="⤴ Start Export (Ctrl+E)", command=self.start_export
        )
        self.export_button.pack(side=tk.LEFT)
        self.export_stop_button = ttk.Button(
            actions, text="■ Stop", command=self.stop_export, state=tk.DISABLED
        )
        self.export_stop_button.pack(side=tk.LEFT, padx=8)

        self.export_progress = ttk.Progressbar(cfg, mode="indeterminate")
        self.export_progress.pack(fill=tk.X, padx=14, pady=(0, 8))
        self.export_progress.pack_forget()

        self.export_info_var = tk.StringVar(
            value="Files will be saved to the selected output directory"
        )
        ttk.Label(cfg, textvariable=self.export_info_var).pack(
            anchor="w", padx=14, pady=(0, 10)
        )

    def create_log_tab(self):
        container = tk.Frame(self.logs_tab, bg=self.colors["bg"])
        container.pack(fill=tk.BOTH, expand=True)

        cfg = self._card(container, "Log Console")
        controls = tk.Frame(cfg, bg=self.colors["panel"])
        controls.pack(fill=tk.X, padx=14, pady=(6, 8))
        self.autosave_log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls, text="Auto-save logs", variable=self.autosave_log_var
        ).pack(side=tk.LEFT)
        ttk.Label(controls, text="Log level:", padding=(10, 0)).pack(side=tk.LEFT)
        self.log_level_var = tk.StringVar(value="INFO")
        cmb = ttk.Combobox(
            controls,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            width=10,
            state="readonly",
        )
        cmb.pack(side=tk.LEFT, padx=(6, 10))
        cmb.bind("<<ComboboxSelected>>", self.change_log_level)
        ttk.Button(controls, text="Clear", command=self.clear_log).pack(side=tk.RIGHT)
        ttk.Button(controls, text="Save", command=self.save_log).pack(
            side=tk.RIGHT, padx=(0, 8)
        )

        info = tk.Frame(cfg, bg=self.colors["panel"])
        info.pack(fill=tk.X, padx=14, pady=(0, 8))
        self.log_file_var = tk.StringVar(value="Not set")
        self.log_size_var = tk.StringVar(value="0 bytes")
        self.log_lines_var = tk.StringVar(value="0 lines")
        for label, var in [
            ("Log File:", self.log_file_var),
            ("File Size:", self.log_size_var),
            ("Lines:", self.log_lines_var),
        ]:
            row = tk.Frame(info, bg=self.colors["panel"])
            row.pack(anchor="w")
            tk.Label(
                row, text=label, bg=self.colors["panel"], fg=self.colors["muted"]
            ).pack(side=tk.LEFT)
            ttk.Label(row, textvariable=var).pack(side=tk.LEFT, padx=8)

        log_text_card = self._card(container, "Output")
        lf = tk.Frame(log_text_card, bg=self.colors["panel"])
        lf.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
        self.log_text = scrolledtext.ScrolledText(
            lf, wrap=tk.WORD, font=("Consolas", 10), height=18
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ---------- Logging ----------
    def setup_logging(self):
        self.logs_dir = Path("./logs")
        self.logs_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = self.logs_dir / f"anomavision_gui_{ts}.log"
        # Below variables are defined in create_log_tab() before setup_logging() is called
        self.log_file_var.set(str(self.log_file_path))

        class GuiLogHandler(logging.Handler):
            def __init__(
                self, text_widget, log_file_path, autosave_var, log_info_callback
            ):
                super().__init__()
                self.text_widget = text_widget
                self.log_file_path = log_file_path
                self.autosave_var = autosave_var
                self.log_info_callback = log_info_callback

            def emit(self, record):
                msg = self.format(record)
                try:
                    self.text_widget.configure(state="normal")
                    self.text_widget.insert(tk.END, msg + "\n")
                    self.text_widget.configure(state="disabled")
                    self.text_widget.see(tk.END)
                except Exception:
                    pass
                if self.autosave_var.get():
                    try:
                        with open(self.log_file_path, "a", encoding="utf-8") as f:
                            f.write(msg + "\n")
                        self.log_info_callback(increment_lines=1)
                    except Exception as e:
                        print(f"Failed to write log file: {e}")

        self.gui_handler = GuiLogHandler(
            self.log_text,
            self.log_file_path,
            self.autosave_log_var,
            self.update_log_info,
        )
        self.gui_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)s  %(name)s  —  %(message)s")
        )
        logging.getLogger("anomavision").addHandler(self.gui_handler)
        logging.getLogger().addHandler(self.gui_handler)
        logging.getLogger("anomavision").setLevel(logging.INFO)
        logger.info(f"AnomaVision GUI started — Log file: {self.log_file_path}")
        self.update_log_info()

    def update_log_info(self, increment_lines: int = 0):
        try:
            if self.log_file_path.exists():
                self.log_size_var.set(f"{self.log_file_path.stat().st_size:,} bytes")
                try:
                    current = int(
                        self.log_lines_var.get().split()[0].replace(",", "") or 0
                    )
                except Exception:
                    current = 0
                current += increment_lines
                self.log_lines_var.set(f"{current:,} lines")
        except Exception as e:
            print(f"Failed to update log info: {e}")

    def change_log_level(self, _=None):
        level_mapping = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        lvl = level_mapping.get(self.log_level_var.get(), logging.INFO)
        logging.getLogger("anomavision").setLevel(lvl)
        logging.getLogger().setLevel(lvl)
        logger.info(f"Log level changed to: {self.log_level_var.get()}")

    def save_log(self):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.logs_dir / f"anomavision_gui_manual_{ts}.log"
            with open(path, "w", encoding="utf-8") as f:
                self.log_text.configure(state="normal")
                f.write(self.log_text.get(1.0, tk.END))
                self.log_text.configure(state="disabled")
            messagebox.showinfo("Saved", f"Log saved to:\n{path}")
            toast(self.root, "Log saved")
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))

    def clear_log(self):
        try:
            self.log_text.configure(state="normal")
            self.log_text.delete(1.0, tk.END)
            self.log_text.configure(state="disabled")
        except Exception:
            pass

    # ---------- File Choosers ----------
    def browse_dataset_path(self):
        path = filedialog.askdirectory()
        if path:
            self.dataset_path_var.set(path)

    def browse_model_output_path(self):
        path = filedialog.askdirectory()
        if path:
            self.model_output_path_var.set(path)

    def browse_model_path(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Model Files", "*.pt *.pth *.onnx *.torchscript *.xml")]
        )
        if file_path:
            self.model_path_var.set(file_path)

    def browse_image_path(self):
        path = filedialog.askdirectory()
        if path:
            self.image_path_var.set(path)

    def browse_export_model_path(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Model Files", "*.pt *.pth *.onnx *.torchscript *.xml")]
        )
        if file_path:
            self.export_model_path_var.set(file_path)

    def browse_export_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.export_output_dir_var.set(path)

    # ---------- Config ----------
    def load_default_config(self):
        try:
            config = load_config("config.yml")
            if "viz_alpha" not in config:
                config["viz_alpha"] = 0.5
            if "viz_color" not in config:
                config["viz_color"] = "128,0,128"
            return config or {}
        except Exception as e:
            logger.warning(f"Failed to load configuration file: {e}")
            return {"viz_alpha": 0.5, "viz_color": "128,0,128"}

    def parse_layer_indices(self, s):
        try:
            if isinstance(s, str):
                cleaned = s.strip("[] ")
                return (
                    [int(x.strip()) for x in cleaned.split(",")] if cleaned else [0, 1]
                )
            if isinstance(s, list):
                return [int(x) for x in s]
            return [0, 1]
        except Exception:
            logger.warning("Bad layer_indices, defaulting to [0,1]")
            return [0, 1]

    # ---------- Training ----------
    def start_training(self):
        try:
            config = {
                "dataset_path": self.dataset_path_var.get(),
                "class_name": self.class_name_var.get(),
                "backbone": self.backbone_var.get(),
                "batch_size": self.batch_size_var.get(),
                "feat_dim": self.feat_dim_var.get(),
                "layer_indices": self.parse_layer_indices(self.layer_indices_var.get()),
                "resize": _safe_literal(self.resize_var.get(), [224, 224]),
                "crop_size": _safe_literal(self.crop_size_var.get(), None),
                "normalize": self.config.get("normalize", True),
                "norm_mean": self.config.get("norm_mean", [0.485, 0.456, 0.406]),
                "norm_std": self.config.get("norm_std", [0.229, 0.224, 0.225]),
                "model_data_path": self.model_output_path_var.get(),
                "output_model": self.output_model_name_var.get(),
                "run_name": self.config.get("run_name", "anomav_exp"),
                "log_level": self.config.get("log_level", "INFO"),
            }
            if not config["dataset_path"]:
                return messagebox.showwarning("Warning", "Please select dataset path")
            if not config["class_name"]:
                return messagebox.showwarning("Warning", "Please enter class name")
            self.train_button.config(state=tk.DISABLED)
            self.train_stop_button.config(state=tk.NORMAL)
            self.train_progress.pack(fill=tk.X, padx=14, pady=(0, 12))
            self.train_progress.start()
            self.training_worker = TrainingWorker(
                config,
                self.update_log,
                self.training_finished,
                self.training_error,
                status_callback=self.set_status,
            )
            self.training_worker.start()
            toast(self.root, "Training started")
        except Exception as e:
            messagebox.showerror("Error", f"Could not start training:\n{e}")
            self.train_button.config(state=tk.NORMAL)
            self.train_stop_button.config(state=tk.DISABLED)
            self.train_progress.pack_forget()

    def stop_training(self):
        try:
            if hasattr(self, "training_worker") and self.training_worker.is_alive():
                self.training_worker.stop()
        except Exception:
            pass
        self.train_button.config(state=tk.NORMAL)
        self.train_stop_button.config(state=tk.DISABLED)
        self.train_progress.stop()
        self.train_progress.pack_forget()
        self.update_log("Training stop requested")
        toast(self.root, "Training stop requested")

    def training_finished(self, model):
        self.current_model = model
        self.train_button.config(state=tk.NORMAL)
        self.train_stop_button.config(state=tk.DISABLED)
        self.train_progress.stop()
        self.train_progress.pack_forget()
        self.update_log("Training completed!")
        messagebox.showinfo("Completed", "Model training completed!")
        self.set_status("Ready")
        toast(self.root, "Training completed ✔", bg="#065F46")

    def training_error(self, error_msg):
        self.train_button.config(state=tk.NORMAL)
        self.train_stop_button.config(state=tk.DISABLED)
        self.train_progress.stop()
        self.train_progress.pack_forget()
        self.update_log(f"Training error: {error_msg}")
        messagebox.showerror("Training Error", error_msg)
        self.set_status("Ready")
        toast(self.root, "Training failed ✖", bg="#7F1D1D")

    # ---------- Inference ----------
    def start_inference(self):
        try:
            config = {
                "resize": _safe_literal(self.resize_var.get(), [224, 224]),
                "crop_size": _safe_literal(self.crop_size_var.get(), None),
                "normalize": self.config.get("normalize", True),
                "norm_mean": self.config.get("norm_mean", [0.485, 0.456, 0.406]),
                "norm_std": self.config.get("norm_std", [0.229, 0.224, 0.225]),
                "batch_size": self.infer_batch_size_var.get(),
                "thresh": self.threshold_var.get(),
                "device": self.device_var.get(),
                "num_workers": self.config.get("num_workers", 1),
                "pin_memory": (
                    self.device_var.get().lower() == "cuda"
                    and torch.cuda.is_available()
                ),
                "viz_alpha": self.config.get("viz_alpha", 0.5),
                "viz_color": self.config.get("viz_color", "128,0,128"),
                "viz_padding": self.viz_padding_var.get(),
                "layer_indices": self.parse_layer_indices(
                    self.config.get("layer_indices", [0, 1])
                ),
            }
            model_path, img_path = self.model_path_var.get(), self.image_path_var.get()
            if not model_path:
                return messagebox.showwarning("Warning", "Please select model file")
            if not img_path:
                return messagebox.showwarning(
                    "Warning", "Please select image directory"
                )
            if not os.path.exists(model_path):
                return messagebox.showwarning("Warning", "Model file does not exist")
            if not os.path.exists(img_path):
                return messagebox.showwarning(
                    "Warning", "Image directory does not exist"
                )

            self.device_badge.config(
                text=f"Device: {config['device'].upper() if config['device']!='auto' else 'Auto'}"
            )

            self.infer_button.config(state=tk.DISABLED)
            self.infer_stop_button.config(state=tk.NORMAL)
            self.prev_result_button.config(state=tk.DISABLED)
            self.next_result_button.config(state=tk.DISABLED)
            self.results_info_var.set("Processing…")
            self.set_status("Preparing inference…")

            self.inference_worker = InferenceWorker(
                model_path,
                img_path,
                config,
                self.update_log,
                self.inference_finished,
                self.inference_error,
                status_callback=self.set_status,
                perf_callback=self.set_perf,
            )
            self.inference_worker.start()
            toast(self.root, "Inference started")
        except Exception as e:
            messagebox.showerror("Error", f"Could not start inference:\n{e}")
            self.infer_button.config(state=tk.NORMAL)
            self.infer_stop_button.config(state=tk.DISABLED)
            self.set_status("Ready")

    def stop_inference(self):
        try:
            if hasattr(self, "inference_worker") and self.inference_worker.is_alive():
                self.inference_worker.stop()
        except Exception:
            pass
        self.infer_button.config(state=tk.NORMAL)
        self.infer_stop_button.config(state=tk.DISABLED)
        self.results_info_var.set("Inference stop requested")
        self.update_log("Inference stop requested")
        self.set_status("Ready")
        toast(self.root, "Inference stop requested")

    def inference_finished(self, images, scores, maps, classifications, visualizations):
        self.inference_results = {
            "images": images,
            "scores": scores,
            "maps": maps,
            "classifications": classifications,
            "visualizations": visualizations,
        }
        self.current_result_index = 0
        self.infer_button.config(state=tk.NORMAL)
        self.infer_stop_button.config(state=tk.DISABLED)
        if visualizations:
            if len(visualizations) > 1:
                self.prev_result_button.config(state=tk.NORMAL)
                self.next_result_button.config(state=tk.NORMAL)
            self.show_result(0)
            self.results_info_var.set(f"Showing result: 1/{len(visualizations)}")
        else:
            self.results_info_var.set("No visualization results")
        self.update_log(f"Inference completed! Processed {len(images)} images")
        messagebox.showinfo(
            "Completed", f"Inference completed! Processed {len(images)} images"
        )
        self.set_status("Ready")
        toast(self.root, "Inference completed ✔", bg="#065F46")

    def inference_error(self, error_msg):
        self.infer_button.config(state=tk.NORMAL)
        self.infer_stop_button.config(state=tk.DISABLED)
        self.results_info_var.set("Inference failed")
        self.update_log(f"Inference error: {error_msg}")
        messagebox.showerror("Inference Error", error_msg)
        self.set_status("Ready")
        toast(self.root, "Inference failed ✖", bg="#7F1D1D")

    def show_prev_result(self):
        if self.inference_results and self.inference_results["visualizations"]:
            self.current_result_index = (self.current_result_index - 1) % len(
                self.inference_results["visualizations"]
            )
            self.show_result(self.current_result_index)
            self.results_info_var.set(
                f"Showing result: {self.current_result_index+1}/{len(self.inference_results['visualizations'])}"
            )

    def show_next_result(self):
        if self.inference_results and self.inference_results["visualizations"]:
            self.current_result_index = (self.current_result_index + 1) % len(
                self.inference_results["visualizations"]
            )
            self.show_result(self.current_result_index)
            self.results_info_var.set(
                f"Showing result: {self.current_result_index+1}/{len(self.inference_results['visualizations'])}"
            )

    def show_result(self, index):
        if not self.inference_results or not self.inference_results["visualizations"]:
            return
        if index >= len(self.inference_results["visualizations"]):
            return
        vis = self.inference_results["visualizations"][index]
        score = 0.0
        if "scores" in self.inference_results and index < len(
            self.inference_results["scores"]
        ):
            score = self.inference_results["scores"][index]
        self.anomaly_score_var.set(f"Anomaly Score: {score:.2f}")
        is_anomaly = vis.get("classification", 0) == 0
        border = "red" if is_anomaly else "green"
        self.display_image(self.original_image_label, vis.get("original"), border)
        self.display_image(self.heatmap_label, vis.get("heatmap"), border)
        self.display_image(self.highlighted_label, vis.get("highlighted"), border)
        self.display_image(self.boundary_label, vis.get("boundary"), border)

    def display_image(self, label, image, border_color="black"):
        try:
            if hasattr(image, "detach"):
                image = image.detach().cpu().numpy()
            if isinstance(image, np.ndarray):
                if image.ndim == 4 and image.shape[0] == 1:
                    image = image[0]
                if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                    if image.shape[0] == 1:
                        image = image[0]
                    elif image.shape[0] in [3, 4]:
                        image = np.transpose(image, (1, 2, 0))
                if image.dtype != np.uint8:
                    if image.dtype in [np.float32, np.float64]:
                        mn, mx = float(image.min()), float(image.max())
                        if mx > mn:
                            image = (image - mn) / (mx - mn) * 255.0
                        image = image.astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.shape[2] == 4:
                    image = image[:, :, :3]

                h, w = image.shape[:2]
                disp_w, disp_h = 340, 270
                scale = min(disp_w / w, disp_h / h)
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                pil_resized = Image.fromarray(image).resize((nw, nh), Image.BILINEAR)
                canvas = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                y0, x0 = (disp_h - nh) // 2, (disp_w - nw) // 2
                canvas[y0 : y0 + nh, x0 : x0 + nw] = np.asarray(pil_resized)
                pil_image = Image.fromarray(canvas)
                tk_image = ImageTk.PhotoImage(pil_image)
                label.config(
                    image=tk_image,
                    text="",
                    bg=self.colors["slot"],
                    relief="solid",
                    bd=2,
                    highlightbackground=border_color,
                    highlightthickness=2,
                )
                label.image = tk_image
            else:
                label.config(
                    text="(no image)",
                    bg=self.colors["slot"],
                    fg="#FFFFFF",
                    relief="solid",
                    bd=1,
                )
        except Exception as e:
            logger.error(f"Image display error: {e}")
            label.config(
                text="Unable to display image",
                bg=self.colors["slot"],
                fg="#FFFFFF",
                relief="solid",
                bd=1,
            )

    # ---------- Export ----------
    def start_export(self):
        try:
            model_path = self.export_model_path_var.get()
            output_dir = self.export_output_dir_var.get()
            export_format = self.export_format_var.get()
            if not model_path:
                return messagebox.showwarning("Warning", "Please select model file")
            if not output_dir:
                return messagebox.showwarning(
                    "Warning", "Please select output directory"
                )
            if not os.path.exists(model_path):
                return messagebox.showwarning("Warning", "Model file does not exist")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            config = {
                "resize": _safe_literal(self.resize_var.get(), [224, 224]),
                "crop_size": _safe_literal(self.crop_size_var.get(), None),
                "normalize": self.config.get("normalize", True),
                "norm_mean": self.config.get("norm_mean", [0.485, 0.456, 0.406]),
                "norm_std": self.config.get("norm_std", [0.229, 0.224, 0.225]),
                "opset": self.onnx_opset_var.get(),
                "dynamic_batch": self.dynamic_batch_var.get(),
                "device": self.export_device_var.get(),
                "fp16": self.config.get("fp16", True),
                "optimize": self.config.get("optimize", False),
                "layer_indices": self.parse_layer_indices(
                    self.config.get("layer_indices", [0, 1])
                ),
            }
            self.export_button.config(state=tk.DISABLED)
            self.export_stop_button.config(state=tk.NORMAL)
            self.export_progress.pack(fill=tk.X, padx=14, pady=(0, 8))
            self.export_progress.start()
            self.export_worker = ExportWorker(
                model_path,
                output_dir,
                export_format,
                config,
                self.update_log,
                self.export_finished,
                self.export_error,
                status_callback=self.set_status,
            )
            self.export_worker.start()
            toast(self.root, "Export started")
        except Exception as e:
            messagebox.showerror("Error", f"Could not start export:\n{e}")
            self.export_button.config(state=tk.NORMAL)
            self.export_stop_button.config(state=tk.DISABLED)
            self.export_progress.pack_forget()
            self.set_status("Ready")

    def stop_export(self):
        try:
            if hasattr(self, "export_worker") and self.export_worker.is_alive():
                self.export_worker.stop()
        except Exception:
            pass
        self.export_button.config(state=tk.NORMAL)
        self.export_stop_button.config(state=tk.DISABLED)
        self.export_progress.stop()
        self.export_progress.pack_forget()
        self.update_log("Export stop requested")
        self.set_status("Ready")
        toast(self.root, "Export stop requested")

    def export_finished(self, output_path):
        self.export_button.config(state=tk.NORMAL)
        self.export_stop_button.config(state=tk.DISABLED)
        self.export_progress.stop()
        self.export_progress.pack_forget()
        self.export_info_var.set(f"Export completed: {output_path}")
        self.update_log(f"Export completed: {output_path}")
        messagebox.showinfo("Completed", f"Model export completed!\n{output_path}")
        self.set_status("Ready")
        toast(self.root, "Export completed ✔", bg="#065F46")

    def export_error(self, error_msg):
        self.export_button.config(state=tk.NORMAL)
        self.export_stop_button.config(state=tk.DISABLED)
        self.export_progress.stop()
        self.export_progress.pack_forget()
        self.export_info_var.set("Export failed")
        self.update_log(f"Export error: {error_msg}")
        messagebox.showerror("Export Error", error_msg)
        self.set_status("Ready")
        toast(self.root, "Export failed ✖", bg="#7F1D1D")

    # ---------- Misc ----------
    def update_log(self, message):
        try:
            self.log_text.after(0, self._append_log_message, message)
        except Exception:
            pass

    def _append_log_message(self, message):
        try:
            self.log_text.configure(state="normal")
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.configure(state="disabled")
            self.log_text.see(tk.END)
        except Exception:
            pass

    def _show_about(self):
        import webbrowser

        c = self.colors

        # ---- Modal ----
        win = tk.Toplevel(self.root)
        win.title("About · AnomaVision")
        win.transient(self.root)
        win.grab_set()
        win.configure(bg=c["panel"])
        win.resizable(False, False)

        # center
        self.root.update_idletasks()
        px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
        pw, ph = self.root.winfo_width(), self.root.winfo_height()
        ww, wh = 760, 600
        wx, wy = px + (pw - ww) // 2, py + (ph - wh) // 2
        win.geometry(f"{ww}x{wh}+{max(wx,0)}+{max(wy,0)}")

        def close():
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()

        win.bind("<Escape>", lambda e: close())
        win.bind("<Return>", lambda e: close())

        # ---- Header (gradient + logo + tagline) ----
        header_h = 130
        header_canvas = tk.Canvas(
            win, height=header_h, highlightthickness=0, bd=0, bg=c["panel"]
        )
        header_canvas.pack(fill=tk.X)
        grad_img = draw_horizontal_gradient(
            ww, header_h, left="#0EA5E9", right="#6366F1"
        )
        grad_tk = ImageTk.PhotoImage(grad_img)
        win._about_grad_ref = grad_tk
        header_canvas.create_image(0, 0, anchor="nw", image=grad_tk)

        try:
            lg = make_logo(size=58, bg="#38BDF8", fg="#001225")
            lg_tk = ImageTk.PhotoImage(lg)
            win._about_logo_ref = lg_tk
            header_canvas.create_image(24, (header_h // 2), image=lg_tk, anchor="w")
            title_x = 24 + 58 + 16
        except Exception:
            title_x = 24

        header_canvas.create_text(
            title_x,
            44,
            anchor="w",
            text="AnomaVision",
            fill="#FFFFFF",
            font=("Segoe UI", 24, "bold"),
        )
        header_canvas.create_text(
            title_x,
            78,
            anchor="w",
            text="Lightweight • Edge-Ready • Visual • Production-Focused",
            fill="#E0E7FF",
            font=("Segoe UI", 11),
        )
        # optional version
        try:
            import anomavision as _av

            ver = getattr(_av, "__version__", None)
            if ver:
                header_canvas.create_text(
                    ww - 24,
                    22,
                    anchor="e",
                    text=f"v{ver}",
                    fill="#DBEAFE",
                    font=("Segoe UI", 10, "bold"),
                )
        except Exception:
            pass

        # ---- Body ----
        body = tk.Frame(win, bg=c["panel"])
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=(10, 16))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)

        # ===== Highlights / Features =====
        feat_card = tk.Frame(
            body, bg=c["panel"], highlightthickness=1, highlightbackground=c["border"]
        )
        feat_card.grid(row=0, column=0, sticky="nsew", padx=(2, 10), pady=(8, 10))

        tk.Label(
            feat_card,
            text="✨ Features",
            bg=c["panel"],
            fg=c["fg"],
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 6))

        bullets = [
            (
                "🔥",
                " Lightweight, fast, and production-ready anomaly detection (PaDiM)",
            ),
            ("🌍", "Deploy anywhere: edge devices, servers, or the cloud"),
            ("⚡", "Optimized inference (AMP / TF32) with warmup & batching"),
            (
                "📦",
                "Multi-backend export: PyTorch, ONNX, TorchScript, OpenVINO, TensorRT",
            ),
            ("🔧", "INT8 / FP16 quantization options for edge efficiency"),
            (
                "🎨",
                "Visualizations: heatmaps, anomaly masks, boundary overlays, ROC curves",
            ),
            ("🖥️", "Unified Python GUI + CLI workflows"),
            ("📁", "Edge-first design with compact .pth statistics"),
            ("🚀", "Optional C++ runtime for ultra-low-latency deployment"),
        ]
        for icon, text in bullets:
            row = tk.Frame(feat_card, bg=c["panel"])
            row.pack(fill=tk.X, padx=14, pady=3)
            tk.Label(
                row, text=icon, bg=c["panel"], fg=c["fg"], font=("Segoe UI", 12)
            ).pack(side=tk.LEFT)
            tk.Label(
                row, text=text, bg=c["panel"], fg=c["muted"], font=("Segoe UI", 10)
            ).pack(side=tk.LEFT, padx=10)

        # Links
        link_row = tk.Frame(feat_card, bg=c["panel"])
        link_row.pack(fill=tk.X, padx=14, pady=(10, 14))

        def link(parent, text, url):
            import functools

            lbl = tk.Label(
                parent,
                text=text,
                bg=c["panel"],
                fg="#93C5FD",
                font=("Segoe UI", 10, "underline"),
                cursor="hand2",
            )
            lbl.bind("<Button-1>", lambda _e, u=url: webbrowser.open(u))
            lbl.pack(side=tk.LEFT, padx=(0, 14))
            return lbl

        link(link_row, "GitHub Repo", "https://github.com/DeepKnowledge1/AnomaVision")
        link(
            link_row,
            "Open Issues",
            "https://github.com/DeepKnowledge1/AnomaVision/issues",
        )
        link(
            link_row,
            "Releases",
            "https://github.com/DeepKnowledge1/AnomaVision/releases",
        )

        # ===== Shortcuts =====
        sc_card = tk.Frame(
            body, bg=c["panel"], highlightthickness=1, highlightbackground=c["border"]
        )
        sc_card.grid(row=0, column=1, sticky="nsew", padx=(10, 2), pady=(8, 10))

        tk.Label(
            sc_card,
            text="⌨ Shortcuts",
            bg=c["panel"],
            fg=c["fg"],
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 6))

        def keycap(parent, text):
            return tk.Label(
                parent,
                text=text,
                bg="#111827",
                fg="#E5E7EB",
                font=("Segoe UI", 10, "bold"),
                padx=8,
                pady=2,
                relief="raised",
                bd=1,
            )

        shortcuts = [
            ("Ctrl", "I", "Start Inference"),
            ("Ctrl", "T", "Start Training"),
            ("Ctrl", "E", "Start Export"),
            ("Ctrl", "O", "Open Model (Inference)"),
            ("F1", None, "About / Help"),
            ("Esc", None, "Close this dialog"),
        ]
        for k1, k2, desc in shortcuts:
            row = tk.Frame(sc_card, bg=c["panel"])
            row.pack(fill=tk.X, padx=14, pady=4)
            keycap(row, k1).pack(side=tk.LEFT)
            if k2:
                tk.Label(
                    row, text="+", bg=c["panel"], fg=c["muted"], font=("Segoe UI", 11)
                ).pack(side=tk.LEFT, padx=6)
                keycap(row, k2).pack(side=tk.LEFT)
            tk.Label(
                row,
                text=f"  {desc}",
                bg=c["panel"],
                fg=c["muted"],
                font=("Segoe UI", 10),
            ).pack(side=tk.LEFT, padx=8)

        # ===== Footer =====
        footer = tk.Frame(win, bg=c["panel"])
        footer.pack(fill=tk.X, padx=18, pady=(4, 14))
        tk.Label(
            footer,
            text="Built for practical anomaly detection at speed and scale.",
            bg=c["panel"],
            fg=c["muted"],
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)
        tk.Button(
            footer,
            text="Close  ✕",
            command=close,
            padx=14,
            pady=6,
            relief="flat",
            bg=c["badge"],
            fg=c["fg"],
            activebackground=c["badge"],
        ).pack(side=tk.RIGHT)

    def _bind_shortcuts(self):
        """Register global keyboard shortcuts."""
        # Training / Inference / Export
        self.root.bind("<Control-i>", lambda e: self.start_inference())
        self.root.bind(
            "<Control-I>", lambda e: self.start_inference()
        )  # uppercase fallback
        self.root.bind("<Control-t>", lambda e: self.start_training())
        self.root.bind("<Control-T>", lambda e: self.start_training())
        self.root.bind("<Control-e>", lambda e: self.start_export())
        self.root.bind("<Control-E>", lambda e: self.start_export())
        self.root.bind("<Control-o>", lambda e: self.browse_model_path())
        self.root.bind("<Control-O>", lambda e: self.browse_model_path())

        # Help/About
        self.root.bind("<F1>", lambda e: self._show_about())

        # Quality-of-life: close modals via Esc (handled in modal too)
        self.root.bind("<Escape>", lambda e: None)


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------


def main():
    # Use TkinterDnD base class if available, else regular Tk
    if _DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = AnomaVisionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
