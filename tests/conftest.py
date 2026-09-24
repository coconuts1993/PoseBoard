import os
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# One person in a warrior yoga pose, 1000 x 667 (MediaPipe's test image)
PERSON_IMAGE_URL = "https://storage.googleapis.com/mediapipe-assets/pose.jpg"


def cache_dir() -> Path:
    """Download cache of the tests: $POSEBOARD_TEST_CACHE or ~/.cache/poseboard-tests."""
    return Path(os.environ.get("POSEBOARD_TEST_CACHE")
                or Path.home() / ".cache" / "poseboard-tests")


def download_cached(url: str, name: str, timeout: float = 30.0) -> Path:
    """``url`` downloaded once into the test cache; pytest.skip when that is not possible."""
    path = cache_dir() / name
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.part")  # parallel test runs
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.replace(path)
    except Exception as e:  # noqa: BLE001  (offline, proxy, ...)
        try:
            tmp.unlink()
        except OSError:
            pass
        pytest.skip(f"cannot download {url}: {e}")
    return path


@pytest.fixture(scope="session")
def person_image_path() -> Path:
    """Path of a JPEG with one standing person (downloaded once; skips when offline)."""
    return download_cached(PERSON_IMAGE_URL, "person_pose.jpg")


@pytest.fixture
def person_image(person_image_path):
    """BGR image (667 x 1000, a fresh copy per test) with one person in a warrior yoga pose
    (skips when offline)."""
    import cv2

    img = cv2.imread(str(person_image_path))
    if img is None:
        pytest.skip(f"cannot read {person_image_path}")
    return img
