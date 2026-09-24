"""Every 2D pose backend on the same real photo, compared with MediaPipe.

For each backend of the registry that is available in this environment and whose model files
can be obtained, the detector is created through the registry (``create_detector``, as the GUI
does) and run on the test image (one person in a warrior yoga pose, facing the camera). The
subject's shoulders, hips, knees and ankles must lie within 8 % of the image diagonal of
MediaPipe's, and left/right must be the subject's left/right (their left side appears on the
image right). Backends that are not installed, or whose weights cannot be downloaded (e.g. a
blocked host), are skipped with the reason; any other error fails.

``POSEBOARD_CONSISTENCY_BACKENDS=key1,key2`` limits the backends that are run. The rtmlib
backends use their ``lightweight`` models (``POSEBOARD_RTMLIB_SMOKE_MODE`` overrides it), the
OpenPose backend the BODY_25 weights of ``tests/test_det_openpose_dnn.py`` (downloaded into the
test cache; ``POSEBOARD_TEST_SKIP_LARGE_DOWNLOADS=1`` skips it).
"""

import os
import re

import numpy as np
import pytest

from poseboard.pose.detectors import BACKENDS, backend_available, create_detector
from poseboard.pose.subject import select_subject

JOINTS = ("shoulder", "hip", "knee", "ankle")
TOLERANCE = 0.08  # of the image diagonal
RTMLIB_MODE = os.environ.get("POSEBOARD_RTMLIB_SMOKE_MODE", "lightweight")

# Options per backend for this test: CPU, and the small models where there is a choice
TEST_OPTIONS = {
    "rtmpose": {"mode": RTMLIB_MODE, "device": "cpu"},
    "rtmpose_halpe26": {"mode": RTMLIB_MODE, "device": "cpu"},
    "rtmw_wholebody": {"mode": RTMLIB_MODE, "device": "cpu"},
    "rtmo": {"mode": RTMLIB_MODE, "device": "cpu"},
    "vitpose_onnx": {"mode": RTMLIB_MODE, "device": "cpu"},
    "rtmpose3d": {"mode": "balanced", "device": "cpu"},
    "yolo_pose": {"model": "n", "device": "cpu"},
    "keypoint_rcnn": {"device": "cpu"},
    "vitpose_hf": {"device": "cpu"},
    "openpose_dnn": {"model": "body25", "device": "cpu"},
    "mmpose": {"device": "cpu"},
    "movenet": {"model": "lightning", "device": "cpu"},
}
COMPARED = sorted(k for k in BACKENDS if k != "mediapipe")


def _selected(key: str) -> bool:
    only = os.environ.get("POSEBOARD_CONSISTENCY_BACKENDS", "").strip()
    return not only or key in {k.strip() for k in only.split(",")}


def weights_unavailable(e: BaseException) -> bool:
    """True for the backends' "the model file cannot be obtained" errors (download blocked,
    offline, ...), which skip the test; everything else is a failure."""
    for x in (e, e.__cause__):
        if x is None:
            continue
        if type(x).__name__ in ("ModelUnavailable", "WeightsUnavailable"):
            return True
        if isinstance(x, RuntimeError) and re.search(r"cannot download", str(x), re.I):
            return True
    return False


def _openpose_files(options: dict) -> dict:
    """The OpenPose model files (as in tests/test_det_openpose_dnn.py; skips if unavailable)."""
    from poseboard.pose.detectors import openpose_dnn_det as od
    from tests.conftest import cache_dir
    from tests.test_det_openpose_dnn import _fetch_weights

    key = options.get("model", "body25")
    weights = _fetch_weights(key)
    spec = od.MODELS[key]
    try:
        proto, _ = od.resolve_model_files(key, caffemodel=weights,
                                          folder=cache_dir() / "openpose" / spec.folder)
    except RuntimeError as e:
        pytest.skip(f"OpenPose prototxt not available: {e}")
    return {"prototxt": str(proto), "caffemodel": str(weights)}


@pytest.fixture(scope="module")
def mediapipe_reference(person_image_path):
    """MediaPipe's keypoints of the subject on the test image: {name: (x, y)}."""
    import cv2

    ok, why = backend_available("mediapipe")
    if not ok:
        pytest.skip(why)
    from poseboard.pose.mediapipe_backend import ensure_model

    try:
        ensure_model("full")
    except RuntimeError as e:
        pytest.skip(f"MediaPipe model not available: {e}")
    det = create_detector("mediapipe", model="full")
    try:
        people = det.detect(cv2.imread(str(person_image_path)), 0.0, "ref")
    finally:
        det.close()
    assert len(people) == 1
    fmt = det.format
    return {n: people[0].keypoints[fmt.index(n)] for n in fmt.names}


def test_mediapipe_reference_orientation(mediapipe_reference, person_image):
    """The reference itself: the subject faces the camera (their left is on the image right)."""
    mp = mediapipe_reference
    assert mp["left_shoulder"][0] > mp["right_shoulder"][0]
    assert mp["left_hip"][0] > mp["right_hip"][0]
    for side in ("left", "right"):
        ys = [mp[f"{side}_{j}"][1] for j in JOINTS]
        assert ys == sorted(ys), (side, ys)


@pytest.mark.parametrize("key", COMPARED)
def test_backend_agrees_with_mediapipe(key, person_image, mediapipe_reference):
    if not _selected(key):
        pytest.skip("not in POSEBOARD_CONSISTENCY_BACKENDS")
    ok, why = backend_available(key)
    if not ok:
        pytest.skip(why)
    options = dict(TEST_OPTIONS.get(key, {}))
    if key == "openpose_dnn":
        options.update(_openpose_files(options))
    try:
        det = create_detector(key, **options)
    except Exception as e:  # noqa: BLE001
        if weights_unavailable(e):
            pytest.skip(f"{key}: model files not available here: {e}")
        raise
    try:
        people = det.detect(person_image, 0.0, "cam0")
        fmt = det.format
    finally:
        det.close()
    assert people, f"{key}: nobody detected"
    subject = select_subject(people, fmt=fmt)
    kp = {n: subject.keypoints[fmt.index(n)] for n in fmt.names}
    mp = mediapipe_reference
    h, w = person_image.shape[:2]
    tol = TOLERANCE * np.hypot(w, h)
    errors = {}
    for side in ("left", "right"):
        for j in JOINTS:
            name = f"{side}_{j}"
            assert np.all(np.isfinite(kp[name])), f"{key}: {name} not found"
            errors[name] = float(np.linalg.norm(kp[name] - mp[name]))
    far = {n: round(d, 1) for n, d in errors.items() if d >= tol}
    assert not far, f"{key}: farther than {tol:.0f} px from MediaPipe: {far}"
    # left/right: the subject faces the camera, so their left side is on the image right
    assert kp["left_shoulder"][0] > kp["right_shoulder"][0], key
    assert kp["left_hip"][0] > kp["right_hip"][0], key
    if det.provides_3d:  # single-view 3D backends: a metric body-centred skeleton
        assert subject.keypoints_3d is not None
        k3 = subject.keypoints_3d
        body = [fmt.index(f"{s}_{j}") for s in ("left", "right") for j in JOINTS]
        assert np.all(np.isfinite(k3[body]))
        width = np.linalg.norm(k3[fmt.index("left_shoulder")] - k3[fmt.index("right_shoulder")])
        assert 0.2 < width < 0.6, f"{key}: shoulder width {width:.2f} m"


def test_download_errors_are_recognised():
    """Only "cannot download" errors skip; other errors must fail the comparison test."""

    class ModelUnavailable(RuntimeError):
        pass

    assert weights_unavailable(ModelUnavailable("x"))
    assert weights_unavailable(RuntimeError("Cannot download the YOLO pose weights (403)"))
    wrapped = RuntimeError("load failed")
    wrapped.__cause__ = ModelUnavailable("blocked")
    assert weights_unavailable(wrapped)
    assert not weights_unavailable(RuntimeError("shape mismatch"))
    assert not weights_unavailable(ValueError("cannot download"))
