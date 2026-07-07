"""Utilities for comparing and benchmarking TY6 decompression backends."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Union

import numpy as np

from .rod_image_reader import HAS_NATIVE_CPP, read_rod_image


def list_ty6_frames(root: Union[Path, str], limit: Union[int, None] = None) -> List[Path]:
    """Return a deterministic subset of TY6 frames from a directory."""
    root_path = Path(root)
    frames = sorted(root_path.glob("*.rodhypix"))
    if limit is None or limit >= len(frames):
        return frames
    if limit <= 1:
        return frames[:1]

    positions = np.linspace(0, len(frames) - 1, num=limit)
    selected = []
    seen = set()
    for pos in positions:
        index = int(round(float(pos)))
        if index not in seen:
            selected.append(frames[index])
            seen.add(index)

    if len(selected) < limit:
        for frame in frames:
            if frame not in selected:
                selected.append(frame)
                if len(selected) == limit:
                    break

    return selected[:limit]


def read_backend_frame(path: Union[Path, str], backend: str) -> np.ndarray:
    """Read a single frame with an explicit TY6 backend."""
    return read_rod_image(path, use_native=True, backend=backend)


def decode_frames(paths: Sequence[Union[Path, str]], backend: str) -> Dict[str, np.ndarray]:
    """Decode a sequence of frames with one backend."""
    decoded: Dict[str, np.ndarray] = {}
    for path in paths:
        frame_path = Path(path)
        decoded[str(frame_path)] = read_backend_frame(frame_path, backend)
    return decoded


def assert_backend_equivalence(paths: Sequence[Union[Path, str]], backends: Sequence[str]) -> None:
    """Assert that every backend returns exactly the same image data."""
    if len(backends) < 2:
        raise ValueError("Need at least two backends to compare")

    decoded_by_backend = {
        backend: decode_frames(paths, backend) for backend in backends
    }

    for left_backend, right_backend in combinations(backends, 2):
        for frame_name, left_image in decoded_by_backend[left_backend].items():
            right_image = decoded_by_backend[right_backend][frame_name]
            if not np.array_equal(left_image, right_image):
                diff = np.argwhere(left_image != right_image)
                first_diff = tuple(diff[0]) if diff.size else None
                raise AssertionError(
                    f"Backend mismatch for {frame_name}: "
                    f"{left_backend} != {right_backend}; "
                    f"first differing index={first_diff}"
                )


def benchmark_backends(
    paths: Sequence[Union[Path, str]],
    backends: Sequence[str],
    warmup_runs: int = 1,
    timed_runs: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark explicit TY6 backends.

    Returns:
        Mapping of backend name to aggregate timings.
    """
    if timed_runs < 1:
        raise ValueError("timed_runs must be >= 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")

    timings: Dict[str, Dict[str, float]] = {}

    for backend in backends:
        if backend == "native" and not HAS_NATIVE_CPP:
            raise RuntimeError("Native C++ backend is not available")

        for _ in range(warmup_runs):
            for path in paths:
                read_backend_frame(path, backend)

        per_run_totals: List[float] = []
        per_frame_samples: List[float] = []
        for _ in range(timed_runs):
            run_start = perf_counter()
            for path in paths:
                frame_start = perf_counter()
                read_backend_frame(path, backend)
                per_frame_samples.append(perf_counter() - frame_start)
            per_run_totals.append(perf_counter() - run_start)

        total_images = len(paths) * timed_runs
        timings[backend] = {
            "warmup_runs": float(warmup_runs),
            "timed_runs": float(timed_runs),
            "frames": float(len(paths)),
            "total_time_s": float(sum(per_run_totals)),
            "avg_run_time_s": float(sum(per_run_totals) / len(per_run_totals)),
            "avg_frame_time_s": float(sum(per_frame_samples) / len(per_frame_samples)),
            "min_frame_time_s": float(min(per_frame_samples)),
            "max_frame_time_s": float(max(per_frame_samples)),
            "images_decoded": float(total_images),
        }

    return timings
