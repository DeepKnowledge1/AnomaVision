import torch

files = [
    "benchmark_results/anomavision_patchcore_candle_state_dict.pt",
    "benchmark_results/anomalib_patchcore_candle_state_dict.pt",
]

for path in files:
    state = torch.load(path, map_location="cpu", weights_only=True)
    print(f"\n{'=' * 70}\n{path}")
    total = 0

    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            size_mb = value.numel() * value.element_size() / 1024**2
            total += size_mb
            print(f"{key:65s} {str(tuple(value.shape)):22s} {size_mb:8.3f} MB")

    print(f"Total tensor size: {total:.3f} MB")
