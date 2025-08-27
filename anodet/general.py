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
