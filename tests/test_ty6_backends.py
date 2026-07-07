from pathlib import Path
import zipfile

import pytest

from rodhypix import HAS_NATIVE_CPP
from rodhypix.ty6_backend_tools import assert_backend_equivalence, list_ty6_frames


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "example_data"
ZIP_PATH = EXAMPLE_ROOT / "exp_11317.zip"
DATA_ROOT = EXAMPLE_ROOT / "exp_11317"

if not DATA_ROOT.exists() and ZIP_PATH.exists():
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(EXAMPLE_ROOT)

FRAME_ROOT = DATA_ROOT / "frames"
SAMPLE_FRAMES = list_ty6_frames(FRAME_ROOT, limit=12)


@pytest.mark.skipif(not HAS_NATIVE_CPP, reason="Native C++ backend is not available")
def test_ty6_python_and_native_match_on_sample_frames():
    assert_backend_equivalence(SAMPLE_FRAMES, ["python", "native"])
