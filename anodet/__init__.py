"""
Complete backward compatibility layer for anodet -> anomavision migration.

This module provides bulletproof backward compatibility for existing code that imports anodet.
All functionality has been moved to the 'anomavision' package.

Usage:
    # Old way (still works with deprecation warning)
    import anodet
    from anodet.utils import adaptive_gaussian_blur
    from anodet.inference.model.wrapper import ModelWrapper
    model = anodet.Padim()

    # New way (recommended)
    import anomavision
    from anomavision.utils import adaptive_gaussian_blur
    from anomavision.inference.model.wrapper import ModelWrapper
    model = anomavision.Padim()

This compatibility layer will be removed in AnomaVision 4.0.0
"""

import sys
import warnings
from pathlib import Path

# Issue deprecation warning
warnings.warn(
    "\n"
    "🔄 PACKAGE MIGRATION NOTICE 🔄\n"
    "═══════════════════════════════════════════════════════════\n"
    "The 'anodet' package has been renamed to 'anomavision'.\n"
    "\n"
    "Please update your imports:\n"
    "  ❌ OLD: import anodet\n"
    "  ✅ NEW: import anomavision\n"
    "\n"
    "  ❌ OLD: from anodet import Padim\n"
    "  ✅ NEW: from anomavision import Padim\n"
    "\n"
    "  ❌ OLD: from anodet.utils import get_logger\n"
    "  ✅ NEW: from anomavision.utils import get_logger\n"
    "\n"
    "Migration timeline:\n"
    "  • Now - v3.x: Both packages work (with warnings)\n"
    "  • v4.0.0: Legacy 'anodet' support will be removed\n"
    "\n"
    "Migration guide: https://github.com/DeepKnowledge1/AnomaVision#migration\n"
    "═══════════════════════════════════════════════════════════\n",
    DeprecationWarning,
    stacklevel=2,
)

# Import everything from the new package
try:
    # Import the main package first
    import anomavision

    # Import modules as modules (for module-level imports)
    # Test and evaluation functions
    # Feature extraction
    # Datasets
    # === EXPLICIT IMPORTS FOR BULLETPROOF COMPATIBILITY ===
    # Core models
    # Import ALL public APIs from anomavision to make them available via anodet
    from anomavision import *
    from anomavision import (
        AnodetDataset,
        MVTecDataset,
        Padim,
        PatchCore,
        ResnetEmbeddingsExtractor,
        optimal_threshold,
        test,
        utils,
        visualization,
        visualize_eval_data,
        visualize_eval_pair,
    )

    # All utility functions that scripts commonly import
    from anomavision.utils import (  # Image processing utilities; Logging and configuration; The specific function that was causing the import error; File utilities; Gaussian blur utilities
        adaptive_gaussian_blur,
        classification,
        create_image_transform,
        create_mask_transform,
        get_logger,
        image_score,
        mahalanobis,
        merge_config,
        pytorch_cov,
        rename_files,
        save_args_to_yaml,
        setup_logging,
        split_tensor_and_run_function,
        standard_image_transform,
        standard_mask_transform,
        to_batch,
    )

    # === COMPLETE MODULE MAPPING FOR sys.modules ===
    # This ensures that deep imports like "from anodet.inference.model.wrapper import X" work
    # Map this current module as anodet
    sys.modules["anodet"] = sys.modules[__name__]

    # 1. Map all main modules
    main_modules = [
        "config",
        "general",
        "utils",
        "padim",
        "padim_lite",
        "patch_core",
        "feature_extraction",
        "mahalanobis",
        "test",
        "visualization",
    ]

    for module_name in main_modules:
        try:
            # Import the actual module
            anomavision_module = __import__(
                f"anomavision.{module_name}", fromlist=[module_name]
            )
            # Map it in sys.modules
            sys.modules[f"anodet.{module_name}"] = anomavision_module
            print(f"✓ Mapped anodet.{module_name}")
        except ImportError as e:
            print(f"⚠ Could not map anodet.{module_name}: {e}")

    # 2. Map datasets module and submodules
    try:
        import anomavision.datasets

        sys.modules["anodet.datasets"] = anomavision.datasets
        print("✓ Mapped anodet.datasets")

        # Map datasets submodules
        datasets_submodules = ["dataset", "mvtec_dataset"]
        for sub in datasets_submodules:
            try:
                submodule = __import__(f"anomavision.datasets.{sub}", fromlist=[sub])
                sys.modules[f"anodet.datasets.{sub}"] = submodule
                print(f"✓ Mapped anodet.datasets.{sub}")
            except ImportError as e:
                print(f"⚠ Could not map anodet.datasets.{sub}: {e}")

    except ImportError as e:
        print(f"⚠ Could not map anodet.datasets: {e}")

    # 3. Map inference module and ALL nested modules (CRITICAL FOR DEEP IMPORTS)
    try:
        import anomavision.inference

        sys.modules["anodet.inference"] = anomavision.inference
        print("✓ Mapped anodet.inference")

        # Map inference.model
        try:
            import anomavision.inference.model

            sys.modules["anodet.inference.model"] = anomavision.inference.model
            print("✓ Mapped anodet.inference.model")

            # Map inference.model.wrapper (THIS IS WHAT train.py NEEDS)
            try:
                import anomavision.inference.model.wrapper

                sys.modules["anodet.inference.model.wrapper"] = (
                    anomavision.inference.model.wrapper
                )
                print("✓ Mapped anodet.inference.model.wrapper - CRITICAL FIX")
            except ImportError as e:
                print(f"❌ CRITICAL: Could not map anodet.inference.model.wrapper: {e}")

            # Map inference.model.backends
            try:
                import anomavision.inference.model.backends

                sys.modules["anodet.inference.model.backends"] = (
                    anomavision.inference.model.backends
                )
                print("✓ Mapped anodet.inference.model.backends")

                # Map all individual backend modules
                backend_modules = [
                    "base",
                    "torch_backend",
                    "onnx_backend",
                    "openvino_backend",
                    "tensorrt_backend",
                    "torchscript_backend",
                ]

                for backend in backend_modules:
                    try:
                        backend_module = __import__(
                            f"anomavision.inference.model.backends.{backend}",
                            fromlist=[backend],
                        )
                        sys.modules[f"anodet.inference.model.backends.{backend}"] = (
                            backend_module
                        )
                        print(f"✓ Mapped anodet.inference.model.backends.{backend}")
                    except ImportError as e:
                        print(
                            f"⚠ Could not map anodet.inference.model.backends.{backend}: {e}"
                        )

            except ImportError as e:
                print(f"⚠ Could not map anodet.inference.model.backends: {e}")

        except ImportError as e:
            print(f"❌ CRITICAL: Could not map anodet.inference.model: {e}")

        # Map inference.modelType
        try:
            import anomavision.inference.modelType

            sys.modules["anodet.inference.modelType"] = anomavision.inference.modelType
            print("✓ Mapped anodet.inference.modelType")
        except ImportError as e:
            print(f"⚠ Could not map anodet.inference.modelType: {e}")

    except ImportError as e:
        print(f"❌ CRITICAL: Could not map anodet.inference: {e}")

    # 4. Map visualization submodules
    try:
        import anomavision.visualization

        sys.modules["anodet.visualization"] = anomavision.visualization
        print("✓ Mapped anodet.visualization")

        viz_submodules = ["boundary", "frame", "heatmap", "highlight", "utils"]
        for viz_module in viz_submodules:
            try:
                viz_submodule = __import__(
                    f"anomavision.visualization.{viz_module}", fromlist=[viz_module]
                )
                sys.modules[f"anodet.visualization.{viz_module}"] = viz_submodule
                print(f"✓ Mapped anodet.visualization.{viz_module}")
            except ImportError as e:
                print(f"⚠ Could not map anodet.visualization.{viz_module}: {e}")

    except ImportError as e:
        print(f"⚠ Could not map anodet.visualization: {e}")

    # 5. Map sampling_methods and submodules
    try:
        import anomavision.sampling_methods

        sys.modules["anodet.sampling_methods"] = anomavision.sampling_methods
        print("✓ Mapped anodet.sampling_methods")

        sampling_submodules = ["kcenter_greedy", "sampling_def"]
        for sub in sampling_submodules:
            try:
                submodule = __import__(
                    f"anomavision.sampling_methods.{sub}", fromlist=[sub]
                )
                sys.modules[f"anodet.sampling_methods.{sub}"] = submodule
                print(f"✓ Mapped anodet.sampling_methods.{sub}")
            except ImportError as e:
                print(f"⚠ Could not map anodet.sampling_methods.{sub}: {e}")

    except ImportError as e:
        print(f"⚠ Could not map anodet.sampling_methods: {e}")

    # Set package metadata for compatibility
    __version__ = getattr(anomavision, "__version__", "3.0.0")
    __author__ = getattr(anomavision, "__author__", "Deep Knowledge")
    __email__ = getattr(anomavision, "__email__", "Deepp.Knowledge@gmail.com")

    print("🎉 Backward compatibility mapping completed successfully!")
    print("🔧 All anodet imports should now work with anomavision backend.")

except ImportError as e:
    print(f"❌ FATAL: Failed to import from anomavision: {e}")
    raise ImportError(
        f"Failed to import from 'anomavision' package. "
        f"Please ensure AnomaVision is properly installed: {e}"
    ) from e

# Define what gets imported with "from anodet import *"
__all__ = [
    # Core models
    "Padim",
    "PatchCore",
    # Datasets
    "MVTecDataset",
    "AnodetDataset",
    # Feature extraction
    "ResnetEmbeddingsExtractor",
    # Modules (for "import anodet.utils" style imports)
    "utils",
    "visualization",
    "test",
    # Utility functions (for "from anodet import function" style imports)
    "to_batch",
    "pytorch_cov",
    "mahalanobis",
    "standard_image_transform",
    "standard_mask_transform",
    "create_image_transform",
    "create_mask_transform",
    "image_score",
    "classification",
    "split_tensor_and_run_function",
    # Logging and config functions
    "get_logger",
    "setup_logging",
    "merge_config",
    "save_args_to_yaml",
    # Image processing functions
    "adaptive_gaussian_blur",
    # Testing functions
    "visualize_eval_data",
    "visualize_eval_pair",
    "optimal_threshold",
]

# Show additional helpful message for interactive users
if hasattr(sys, "ps1"):  # Interactive session
    print("\n🔔 Note: You're using the legacy 'anodet' package.")
    print("   Consider updating to 'anomavision' for the best experience!")
    print(
        "   Quick migration: https://github.com/DeepKnowledge1/AnomaVision#migration\n"
    )


# Final compatibility check - verify critical imports work
def _verify_compatibility():
    """Verify that critical imports work correctly."""
    try:
        # Test critical imports that commonly fail
        from anodet.config import load_config
        from anodet.general import determine_device
        from anodet.inference.model.wrapper import ModelWrapper
        from anodet.utils import adaptive_gaussian_blur, get_logger

        print("✅ Compatibility verification passed!")
        return True
    except ImportError as e:
        print(f"❌ Compatibility verification failed: {e}")
        return False


# Run verification (comment out if you don't want the output)
_verify_compatibility()
