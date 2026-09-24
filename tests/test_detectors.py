"""Backend registry (no heavy imports, clear errors) and the MediaPipe detector on a real image."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from poseboard.calibration import approximate_calibration
from poseboard.pose import detectors
from poseboard.pose.base import MODE_SINGLE_VIEW_3D
from poseboard.pose.detectors import (BACKENDS, BackendSpec, BackendUnavailable, Detector2D,
                                      backend_available, create_detector, list_backends)
from poseboard.pose.formats import FORMATS
from poseboard.pose.multiview import MultiViewEstimator
from tests.fakes import FakeDetector

ROOT = Path(__file__).resolve().parents[1]
ALL_KEYS = {"mediapipe", "rtmpose", "rtmpose_halpe26", "rtmw_wholebody", "rtmo", "vitpose_onnx",
            "rtmpose3d", "yolo_pose", "keypoint_rcnn", "vitpose_hf", "openpose_dnn", "mmpose",
            "movenet"}
HEAVY = ("torch", "torchvision", "ultralytics", "rtmlib", "onnxruntime", "transformers",
         "mmpose", "mmcv", "mediapipe")


def _has(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def test_registry_is_complete_and_consistent():
    assert set(BACKENDS) == ALL_KEYS and [s.key for s in list_backends()] == list(BACKENDS)
    for key, spec in BACKENDS.items():
        assert isinstance(spec, BackendSpec) and spec.key == key
        assert spec.label and spec.install and spec.license and spec.notes
        assert spec.module.startswith("poseboard.pose.detectors.") and spec.factory
        assert spec.keypoint_format in FORMATS
        assert set(spec.defaults) <= set(spec.options), key
        for opt, choices in spec.options.items():
            if choices and opt in spec.defaults and opt not in spec.needs_files:
                assert spec.defaults[opt] in choices, (key, opt)
        assert set(spec.needs_files) <= set(spec.options)
    assert BACKENDS["mediapipe"].provides_3d and BACKENDS["rtmpose3d"].provides_3d
    assert BACKENDS["rtmpose_halpe26"].keypoint_format == "halpe26"
    assert BACKENDS["openpose_dnn"].needs_files == ("prototxt", "caffemodel")
    for k in ("rtmpose", "rtmpose_halpe26", "rtmw_wholebody", "rtmo", "vitpose_onnx", "rtmpose3d"):
        assert BACKENDS[k].module == "poseboard.pose.detectors.rtmlib_det"


@pytest.mark.parametrize("key", sorted(ALL_KEYS))
def test_backend_available_reports_missing_packages_or_modules(key):
    spec = BACKENDS[key]
    ok, why = backend_available(key)
    missing = [m for m in spec.requires if not _has(m)]
    if missing:
        assert not ok and spec.install in why and all(m in why for m in missing)
    elif not _has(spec.module):
        assert not ok and "not implemented" in why
    else:
        assert ok and why == "available"
    if not ok:
        with pytest.raises(BackendUnavailable) as e:
            create_detector(key)
        assert str(e.value) == why


def test_unknown_backend():
    ok, why = backend_available("nope")
    assert not ok and "Unknown pose backend" in why
    with pytest.raises(BackendUnavailable, match="Unknown"):
        create_detector("nope")


def test_create_detector_merges_defaults(monkeypatch):
    spec = BackendSpec(key="fake", label="Fake", module="tests.fakes", factory="create_fake",
                       requires=("numpy",), install="-", options={"fmt": ("coco17", "halpe26")},
                       defaults={"fmt": "coco17", "extra": 1}, keypoint_format="coco17",
                       license="-", notes="-")
    monkeypatch.setitem(BACKENDS, "fake", spec)
    det = create_detector("fake", fmt="halpe26", extra=None, more=2)
    assert isinstance(det, FakeDetector) and det.format.key == "halpe26"
    assert det.options == {"fmt": "halpe26", "provides_3d": False, "extra": 1, "more": 2}
    assert det.info()["keypoint_format"] == "halpe26"
    monkeypatch.setitem(BACKENDS, "bad", BackendSpec(
        key="bad", label="Bad", module="tests.fakes", factory="nope", requires=(), install="-"))
    with pytest.raises(BackendUnavailable, match="nope"):
        create_detector("bad")
    monkeypatch.setitem(BACKENDS, "missing", BackendSpec(
        key="missing", label="Missing", module="tests.fakes", factory="create_fake",
        requires=("surely_not_installed_pkg",), install="pip install surely"))
    ok, why = backend_available("missing")
    assert not ok and "surely_not_installed_pkg" in why and "pip install surely" in why


def test_detector_base_class():
    d = Detector2D()
    with pytest.raises(NotImplementedError):
        d.detect(np.zeros((2, 2, 3), np.uint8), 0.0, "c")
    d.close()
    assert d.info()["keypoint_format"] == "coco17"


def test_importing_the_app_imports_no_model_library():
    code = ("import sys; import poseboard.pose.detectors, poseboard.pose.multiview, "
            "poseboard.pose.mediapipe_backend, poseboard.session, poseboard.gui.app; "
            "from poseboard.pose.detectors import backend_available, BACKENDS; "
            "[backend_available(k) for k in BACKENDS]; "
            f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         timeout=120, env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"imported at start: {out.stdout.strip()}"


def test_detectors_package_exports():
    for name in ("BACKENDS", "BackendSpec", "BackendUnavailable", "Detector2D", "Person2D",
                 "backend_available", "create_detector", "list_backends"):
        assert hasattr(detectors, name)


# ------------------------------------------------------------------ MediaPipe, real model
@pytest.fixture(scope="module")
def mediapipe_model():
    pytest.importorskip("mediapipe")
    from poseboard.pose.mediapipe_backend import ensure_model

    try:
        return ensure_model("full")
    except RuntimeError as e:
        pytest.skip(f"MediaPipe model not available: {e}")


def test_mediapipe_detector_on_a_real_image(person_image, mediapipe_model):
    h, w = person_image.shape[:2]
    det = create_detector("mediapipe", model="full", num_poses=2)
    try:
        assert det.key == "mediapipe" and det.provides_3d and det.format.key == "mediapipe33"
        assert det.options == {"model": "full", "num_poses": 2}
        people = det.detect(person_image, 0.0, "cam0")
    finally:
        det.close()
    assert len(people) == 1
    p = people[0]
    assert p.keypoints.shape == (33, 2) and p.scores.shape == (33,)
    assert p.keypoints_3d.shape == (33, 3) and np.all(np.isfinite(p.keypoints_3d))
    fmt = FORMATS["mediapipe33"]
    vis = p.scores > 0.5
    assert vis.sum() >= 20
    inside = (p.keypoints[:, 0] > -5) & (p.keypoints[:, 0] < w + 5) & (p.keypoints[:, 1] > -5) \
        & (p.keypoints[:, 1] < h + 5)
    assert inside[vis].all()
    nose, la, ra = (p.keypoints[fmt.index(n)] for n in ("nose", "left_ankle", "right_ankle"))
    assert nose[1] < la[1] and nose[1] < ra[1]  # head above the feet in the image
    x1, y1, x2, y2 = p.bbox
    assert (x2 - x1) > 0.3 * w and (y2 - y1) > 0.4 * h and 0 < p.score <= 1
    # metric 3D skeleton: shoulder width 0.2-0.6 m
    sw = np.linalg.norm(p.keypoints_3d[fmt.index("left_shoulder")]
                        - p.keypoints_3d[fmt.index("right_shoulder")])
    assert 0.2 < sw < 0.6


def test_mediapipe_single_camera_3d_on_a_real_image(person_image, mediapipe_model):
    h, w = person_image.shape[:2]
    cams = {"cam0": approximate_calibration("cam0", w, h)}
    est = MultiViewEstimator(create_detector("mediapipe"), min_score=0.5)
    try:
        pose = est.process({"cam0": (1.0, person_image)}, cams)
    finally:
        est.close()
    assert pose.mode == MODE_SINGLE_VIEW_3D and pose.format_key == "mediapipe33"
    assert pose.views_used == ["cam0"] and pose.reproj_error_px < 25
    ok = np.all(np.isfinite(pose.keypoints), axis=1)
    assert ok.all() and np.all(pose.keypoints[:, 2] > 0)  # in front of the camera

    from poseboard.pose.mediapipe_backend import MediaPipePose

    legacy = MediaPipePose("full")
    try:
        p2, world = legacy.detect("cam0", 0.0, person_image)
        assert p2.keypoints.shape == (33, 2) and world.shape == (33, 3)
        pose2 = legacy.process({"cam0": (2.0, person_image)}, cams)
    finally:
        legacy.close()
    assert pose2.mode == MODE_SINGLE_VIEW_3D and pose2.t == 2.0
    assert np.nanmax(np.linalg.norm(pose2.keypoints - pose.keypoints, axis=1)) < 0.2
