import argparse
import time

import torch
from tqdm import trange


def stress_vram_transfer(
        batch_size=10,
        warmup=5,
        repeats=100,
        frame_shape=(3, 3840, 2160),
        use_pinned_memory=True,
):
    tensor = torch.randn((batch_size, *frame_shape))
    if use_pinned_memory:
        tensor = tensor.pin_memory()

    for device_id in range(torch.cuda.device_count()):
        print(f"Starting test for device {device_id}: {torch.cuda.get_device_properties(device_id)}")
        for _ in trange(warmup, desc="warmup"):
            tensor = tensor.to(device=device_id, non_blocking=use_pinned_memory)
            tensor = tensor.to(device="cpu", non_blocking=use_pinned_memory)
            if use_pinned_memory:
                torch.cuda.current_stream(device=device_id).synchronize()
        start = time.perf_counter()
        for _ in trange(repeats, desc="test"):
            tensor = tensor.to(device=device_id, non_blocking=use_pinned_memory)
            tensor = tensor.to(device="cpu", non_blocking=use_pinned_memory)
            if use_pinned_memory:
                torch.cuda.current_stream(device=device_id).synchronize()
        end = time.perf_counter()
        print(f"Total time taken: {end-start:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--frame_shape", type=int, nargs=3, default=(3, 3840, 2160))
    parser.add_argument("--use_pinned_memory", type=bool, default=True)
    parser.add_argument('--no_pin', dest='use_pinned_memory', action='store_false')
    args = parser.parse_args()

    args = dict(vars(args))
    print(args)
    stress_vram_transfer(**args)