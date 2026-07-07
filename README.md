# rodhypix

Standalone reader for Rigaku `.rodhypix` detector images.

Import as:

```python
from rodhypix import read_rod_image
```

## Features

- Read native `.rodhypix` images without CrysAlisPro
- Access detector and experiment metadata
- Optional native C++ TY6 backend
- Pure Python fallback

## Installation

```bash
pip install rodhypix
```

Optional plotting support:

```bash
pip install "rodhypix[viz]"
```

## Quick Start

```python
from rodhypix import RODImageReader, get_rod_info, read_rod_image

image = read_rod_image("snapshot.rodhypix")
info = get_rod_info("snapshot.rodhypix")

reader = RODImageReader("snapshot.rodhypix")
print(reader.get_decompression_method())
print(info["exposure_time_sec"])
```

## Public API

- `RODImageReader`
- `read_rod_image`
- `get_rod_info`
- `HAS_NATIVE_CPP`

## Included Material

- `rodhypix/rod_image_reader.py`: main reader implementation
- `rodhypix/ty6_backend_tools.py`: backend comparison and benchmark helpers
- `scripts/benchmark_ty6_backends.py`: CLI benchmark
- `examples/01_image_visualization.ipynb`: image visualization notebook

## Example Data

The extracted example dataset is stored in `example_data/exp_11317.zip`.
