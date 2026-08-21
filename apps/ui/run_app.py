"""
AnomaVision - Hugging Face Spaces Entry Point
Starts both FastAPI backend and Gradio frontend in a single process
"""

import importlib.util
import multiprocessing
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FASTAPI_MODULE_PATH = PROJECT_ROOT / "apps" / "api" / "fastapi_app.py"
GRADIO_MODULE_PATH = PROJECT_ROOT / "apps" / "ui" / "gradio_app.py"


def load_module_from_path(module_name: str, module_path: Path):
    """Load a project module by path, avoiding apps.py/package name collisions."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fastapi():
    """Run FastAPI server in background process"""
    try:
        import uvicorn

        fastapi_module = load_module_from_path(
            "anomavision_fastapi_app", FASTAPI_MODULE_PATH
        )
        app = fastapi_module.app

        print("🚀 Starting FastAPI backend on port 8000...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=False,  # Reduce log noise
        )
    except Exception as e:
        print(f"❌ FastAPI failed to start: {e}")
        sys.exit(1)


def run_gradio():
    """Run Gradio interface in main process"""
    import gradio as gr
    import requests

    gradio_module = load_module_from_path("anomavision_gradio_app", GRADIO_MODULE_PATH)

    # Wait for FastAPI to be ready
    print("⏳ Waiting for FastAPI backend to start...")
    time.sleep(8)  # Initial wait

    # Health check with retries
    max_retries = 15
    health_url = "http://127.0.0.1:8000/health"

    for attempt in range(max_retries):
        try:
            response = requests.get(health_url, timeout=3)
            if response.status_code == 200:
                print("✅ FastAPI backend is ready!")
                print(f"   Model status: {response.json().get('status', 'unknown')}")
                break
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                print(f"⏳ Backend starting... ({attempt + 1}/{max_retries})")
                time.sleep(3)
            else:
                print("❌ FastAPI backend failed to respond after multiple retries!")
                print("   Check logs above for errors.")
                sys.exit(1)

    # Start Gradio
    print("🎨 Starting Gradio interface on port 7860...")
    try:
        demo = gradio_module.demo
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False,
        )
    except Exception as e:
        print(f"❌ Gradio failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("🔍 AnomaVision - Visual Anomaly Detection")
    print("=" * 60)
    print("Starting dual-server application...")
    print("  • FastAPI backend: http://127.0.0.1:8000")
    print("  • Gradio frontend: http://127.0.0.1:7860")
    print("=" * 60)

    # Set multiprocessing start method (important for some platforms)
    multiprocessing.set_start_method("spawn", force=True)

    # Start FastAPI in a separate process
    fastapi_process = multiprocessing.Process(
        target=run_fastapi, daemon=True  # Dies when main process dies
    )
    fastapi_process.start()

    # Run Gradio in main process (blocks until interrupted)
    try:
        run_gradio()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    finally:
        if fastapi_process.is_alive():
            fastapi_process.terminate()
            fastapi_process.join(timeout=5)
        print("✅ Shutdown complete")
