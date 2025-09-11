"""
Backward compatibility layer for anodet -> anomavision migration.

This module provides backward compatibility for existing code that imports anodet.
All functionality has been moved to the 'anomavision' package.

Usage:
    # Old way (still works with deprecation warning)
    import anodet
    model = anodet.Padim()

    # New way (recommended)
    import anomavision
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
    # Import all public APIs from anomavision
    # Map old module paths to new ones for deep imports
    import anomavision

    # Import specific modules for compatibility
    from anomavision import *
    from anomavision import (  # Core algorithms; Datasets; Feature extraction; Utilities; Functions; Testing
        AnodetDataset,
        MVTecDataset,
        Padim,
        PatchCore,
        ResnetEmbeddingsExtractor,
        classification,
        image_score,
        mahalanobis,
        optimal_threshold,
        pytorch_cov,
        split_tensor_and_run_function,
        standard_image_transform,
        standard_mask_transform,
        test,
        to_batch,
        utils,
        visualization,
        visualize_eval_data,
        visualize_eval_pair,
    )

    # Create module aliases for backward compatibility
    sys.modules["anodet.padim"] = (
        anomavision.padim if hasattr(anomavision, "padim") else None
    )
    sys.modules["anodet.datasets"] = (
        anomavision.datasets if hasattr(anomavision, "datasets") else None
    )
    sys.modules["anodet.datasets.dataset"] = (
        getattr(anomavision.datasets, "dataset", None)
        if hasattr(anomavision, "datasets")
        else None
    )
    sys.modules["anodet.datasets.mvtec_dataset"] = (
        getattr(anomavision.datasets, "mvtec_dataset", None)
        if hasattr(anomavision, "datasets")
        else None
    )
    sys.modules["anodet.feature_extraction"] = (
        anomavision.feature_extraction
        if hasattr(anomavision, "feature_extraction")
        else None
    )
    sys.modules["anodet.visualization"] = (
        anomavision.visualization if hasattr(anomavision, "visualization") else None
    )
    sys.modules["anodet.utils"] = (
        anomavision.utils if hasattr(anomavision, "utils") else None
    )
    sys.modules["anodet.inference"] = (
        anomavision.inference if hasattr(anomavision, "inference") else None
    )
    sys.modules["anodet.mahalanobis"] = (
        anomavision.mahalanobis if hasattr(anomavision, "mahalanobis") else None
    )
    sys.modules["anodet.test"] = (
        anomavision.test if hasattr(anomavision, "test") else None
    )
    sys.modules["anodet.general"] = (
        anomavision.general if hasattr(anomavision, "general") else None
    )
    sys.modules["anodet.config"] = (
        anomavision.config if hasattr(anomavision, "config") else None
    )

    # Set package metadata for compatibility
    __version__ = getattr(anomavision, "__version__", "3.0.0")
    __author__ = getattr(anomavision, "__author__", "Deep Knowledge")
    __email__ = getattr(anomavision, "__email__", "Deepp.Knowledge@gmail.com")

except ImportError as e:
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
    # Modules
    "visualization",
    "utils",
    "test",
    # Utility functions
    "to_batch",
    "pytorch_cov",
    "mahalanobis",
    "standard_image_transform",
    "standard_mask_transform",
    "image_score",
    "classification",
    "split_tensor_and_run_function",
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
