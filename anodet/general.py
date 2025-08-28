from pathlib import Path
import os
import matplotlib.pyplot as plt
import torch



def increment_path(path, exist_ok=False, sep="", mkdir=False):
    p = Path(path)

    if exist_ok or not p.exists():
        if mkdir:
            (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
        return p

    # path exists -> append 2,3,4,...
    base, suffix = (p.with_suffix(""), p.suffix) if p.is_file() else (p, "")
    i = 2
    while (base.parent / f"{base.name}{sep}{i}{suffix}").exists():
        i += 1
    newp = base.parent / f"{base.name}{sep}{i}{suffix}"

    if mkdir:
        (newp if suffix == "" else newp.parent).mkdir(parents=True, exist_ok=True)
    return newp


def determine_device(device_arg):
    """Determine the best device to use for inference"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return device_arg

def save_visualization(images, filename, output_dir):
    """Save visualization images to disk"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    if len(images.shape) == 4:  # Batch of images
        for i, img in enumerate(images):
            individual_filepath = os.path.join(
                output_dir, f"{filename.split('.')[0]}_batch_{i}.png"
            )
            plt.imsave(individual_filepath, img)
    else:  # Single image
        plt.imsave(filepath, images)


import contextlib
import time
import torch


class Profiler(contextlib.ContextDecorator):
    """
    AnomaVision Performance Profiler for accurate timing measurements.

    Designed for anomaly detection inference pipelines with CUDA synchronization
    support for precise GPU timing measurements.

    Usage:
        @AnomaVisionProfiler() decorator or 'with AnomaVisionProfiler():' context manager

    Example:
        # Context manager usage
        profiler = AnomaVisionProfiler()
        with profiler:
            model.predict(batch)
        print(f"Inference time: {profiler.elapsed_time * 1000:.2f} ms")

        # Decorator usage
        @AnomaVisionProfiler()
        def inference_step():
            return model.predict(batch)
    """

    def __init__(self, accumulated_time=0.0):
        """
        Initialize AnomaVision profiler.

        Args:
            accumulated_time (float): Initial accumulated time in seconds
        """
        self.accumulated_time = accumulated_time  # Total time accumulated across multiple runs
        self.elapsed_time = 0.0                   # Time for the last measurement
        self.cuda_available = torch.cuda.is_available()  # Check if CUDA timing sync is needed
        self._start_time = 0.0                    # Internal start time marker

    def __enter__(self):
        """Enter context manager - start timing for AnomaVision operation."""
        self._start_time = self._get_precise_time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit context manager - complete timing measurement.

        Calculates elapsed time and adds to accumulated total for AnomaVision metrics.
        """
        self.elapsed_time = self._get_precise_time() - self._start_time  # delta-time for this operation
        self.accumulated_time += self.elapsed_time                        # accumulate for total AnomaVision runtime

    def _get_precise_time(self):
        """
        Get precise timestamp with CUDA synchronization if available.

        For AnomaVision GPU inference, this ensures accurate timing by
        synchronizing CUDA operations before measuring time.

        Returns:
            float: Precise timestamp in seconds
        """
        if self.cuda_available:
            torch.cuda.synchronize()  # Ensure all CUDA operations complete for accurate timing
        return time.time()

    def reset(self):
        """Reset accumulated time counter for new AnomaVision measurement session."""
        self.accumulated_time = 0.0
        self.elapsed_time = 0.0

    def get_fps(self, num_samples):
        """
        Calculate FPS (Frames Per Second) for AnomaVision inference.

        Args:
            num_samples (int): Number of images/samples processed

        Returns:
            float: FPS based on accumulated time
        """
        if self.accumulated_time > 0:
            return num_samples / self.accumulated_time
        return 0.0

    def get_avg_time_ms(self, num_operations):
        """
        Get average time per operation in milliseconds.

        Args:
            num_operations (int): Number of operations performed

        Returns:
            float: Average time per operation in milliseconds
        """
        if num_operations > 0:
            return (self.accumulated_time / num_operations) * 1000
        return 0.0