"""Benchmark TY6 decompression backends on sample .rodhypix frames."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rodhypix import HAS_NATIVE_CPP  # noqa: E402
from rodhypix.ty6_backend_tools import (  # noqa: E402
    assert_backend_equivalence,
    benchmark_backends,
    list_ty6_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "example_data" / "exp_11317" / "frames",
        help="Directory containing .rodhypix frames",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Number of frames to benchmark from the corpus",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup passes before timing each backend",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed passes over the selected sample frames",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["python", "native"],
        help="Backends to benchmark",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        zip_path = REPO_ROOT / "example_data" / "exp_11317.zip"
        if args.root == REPO_ROOT / "example_data" / "exp_11317" / "frames" and zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(REPO_ROOT / "example_data")

    frames = list_ty6_frames(args.root, limit=args.limit)
    if not frames:
        raise RuntimeError(f"No .rodhypix frames found in {args.root}")

    if "native" in args.backends and not HAS_NATIVE_CPP:
        raise RuntimeError("Native C++ backend is not available")

    print(f"Selected {len(frames)} frames from {args.root}")
    for frame in frames:
        print(f"  {frame.name}")

    assert_backend_equivalence(frames, args.backends)
    print("All selected backends produced identical arrays.")

    timings = benchmark_backends(
        frames,
        args.backends,
        warmup_runs=args.warmup,
        timed_runs=args.repeats,
    )

    print()
    print(
        f"{'backend':<8} {'frames':>6} {'warmups':>7} {'runs':>4} "
        f"{'total_s':>10} {'avg_run_s':>10} {'avg_frame_s':>12} "
        f"{'min_frame_s':>12} {'max_frame_s':>12}"
    )
    for backend in args.backends:
        stats = timings[backend]
        print(
            f"{backend:<8} "
            f"{int(stats['frames']):>6} "
            f"{int(stats['warmup_runs']):>7} "
            f"{int(stats['timed_runs']):>4} "
            f"{stats['total_time_s']:>10.6f} "
            f"{stats['avg_run_time_s']:>10.6f} "
            f"{stats['avg_frame_time_s']:>12.6f} "
            f"{stats['min_frame_time_s']:>12.6f} "
            f"{stats['max_frame_time_s']:>12.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
