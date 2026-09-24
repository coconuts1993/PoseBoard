"""The ``movenet`` backend (MoveNet Lightning / Thunder through ONNX Runtime).

The first part runs everywhere: onnxruntime is replaced by a stub session, so the model file
handling, input tensors (dtype, layout, RGB, letterbox), the crop region, the mapping back to
image pixels, scores and multi-pose outputs are checked without the library. The second part
runs the real models (downloaded from GitHub, SHA-256 checked) on a real image when onnxruntime
is installed; otherwise it is skipped with the reason.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from poseboard.calibration import approximate_calibration
from poseboard.pose import mediapipe_backend
from poseboard.pose.base import MODE_2D_ONLY
from poseboard.pose.detectors import backend_available, create_detector
from poseboard.pose.detectors import movenet_det as mn
from poseboard.pose.formats import COCO17
from poseboard.pose.multiview import MultiViewEstimator
from tests.conftest import cache_dir

ROOT = Path(__file__).resolve().parents[1]

# Keypoint order of the MoveNet outputs, from the TF Hub model card / tutorial KEYPOINT_DICT
# ("nose, left eye, right eye, left ear, right ear, left shoulder, ..., right ankle").
MOVENET_ORDER = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                 "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                 "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")


def test_keypoint_order_is_coco17():
    assert MOVENET_ORDER == COCO17.names
    assert mn.MoveNetDetector.format is COCO17


def test_module_import_loads_no_model_library():
    code = ("import sys, poseboard.pose.detectors.movenet_det; "
            "print(','.join(m for m in ('onnxruntime', 'torch', 'tensorflow') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


# ------------------------------------------------------------------ onnxruntime stub
@pytest.fixture
def fake_ort(monkeypatch, tmp_path):
    """Replace onnxruntime by a stub (also where it is installed). ``state.output(x)`` gives
    the model output for input tensor x; ``state.inputs`` records the inputs."""
    state = SimpleNamespace(type="tensor(int32)", shape=[1, 192, 192, 3], inputs=[],
                            providers=["CPUExecutionProvider"], sessions=[],
                            output=lambda x: np.zeros((1, 1, 17, 3), np.float32))

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path, self.providers = path, providers
            state.sessions.append(self)

        def get_inputs(self):
            return [SimpleNamespace(name="input", type=state.type, shape=list(state.shape))]

        def get_outputs(self):
            return [SimpleNamespace(name="output_0", type="tensor(float)", shape=[1, 1, 17, 3])]

        def run(self, names, feeds):
            (x,) = feeds.values()
            state.inputs.append(np.array(x))
            return [np.asarray(state.output(x), np.float32)]

    m = types.ModuleType("onnxruntime")
    m.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", None)
    m.InferenceSession = FakeSession
    m.SessionOptions = lambda: SimpleNamespace()
    m.get_available_providers = lambda: list(state.providers)
    monkeypatch.setitem(sys.modules, "onnxruntime", m)
    monkeypatch.setattr(mediapipe_backend, "MODEL_DIR", tmp_path / "models")
    state.model = tmp_path / "movenet.onnx"
    state.model.write_bytes(b"fake onnx")
    return state


def single_output(kp_orig, scores, region, size=(192, 192)):
    """A [1, 1, 17, 3] output that decodes to ``kp_orig`` (17, 2) original pixels when the
    network saw ``region`` (x0, y0, width, height)."""
    x0, y0, rw, rh = region
    out = np.zeros((1, 1, 17, 3), np.float32)
    out[0, 0, :, 0] = (kp_orig[:, 1] - y0) / rh
    out[0, 0, :, 1] = (kp_orig[:, 0] - x0) / rw
    out[0, 0, :, 2] = scores
    return out


def standing(h, w, rng=None):
    """17 COCO keypoints of an upright person facing the camera, in pixels."""
    cx, top, bottom = w * 0.5, h * 0.15, h * 0.9
    ys = {"nose": 0.0, "eye": -0.02, "ear": 0.0, "shoulder": 0.2, "elbow": 0.38, "wrist": 0.52,
          "hip": 0.55, "knee": 0.77, "ankle": 1.0}
    dx = {"nose": 0.0, "eye": 0.02, "ear": 0.04, "shoulder": 0.1, "elbow": 0.12, "wrist": 0.13,
          "hip": 0.07, "knee": 0.07, "ankle": 0.07}
    pts = []
    for name in COCO17.names:
        part = name.split("_")[-1]
        side = 1 if name.startswith("left") else (-1 if name.startswith("right") else 0)
        pts.append((cx + side * dx[part] * h, top + ys[part] * (bottom - top)))
    kp = np.array(pts)
    if rng is not None:
        kp += rng.normal(0, 1.0, kp.shape)
    return kp


def test_model_files_and_download(fake_ort, tmp_path, monkeypatch):
    for m, url in mn.MODEL_URLS.items():
        assert url.startswith("https://raw.githubusercontent.com/")
        assert mn.MOVENET_COMMIT in url and url.endswith(mn.MODEL_FILES[m][0])
    with pytest.raises(FileNotFoundError, match="not found"):
        mn.resolve_model("lightning", tmp_path / "nope.onnx")
    assert mn.resolve_model("thunder", fake_ort.model) == fake_ort.model
    with pytest.raises(ValueError, match="lightning"):
        mn.resolve_model("heavy")
    folder = tmp_path / "models"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mn.resolve_model("lightning", folder=folder, download=False)

    payload = b"onnx bytes"
    calls = []

    class Resp:
        def __init__(self, data):
            self.data = data

        def read(self, n=-1):
            out, self.data = (self.data, b"") if n < 0 else (self.data[:n], self.data[n:])
            return out

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mn.urllib.request, "urlopen",
                        lambda url, timeout=None: calls.append(url) or Resp(payload))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch") as e:
        mn.resolve_model("lightning", folder=folder)
    assert calls == [mn.MODEL_URLS["lightning"]] and "model_path" in str(e.value)
    assert not list(folder.glob("*"))
    name = mn.MODEL_FILES["lightning"][0]
    monkeypatch.setitem(mn.MODEL_FILES, "lightning", (name, hashlib.sha256(payload).hexdigest()))
    path = mn.resolve_model("lightning", folder=folder)
    assert path == folder / name and path.read_bytes() == payload
    assert mn.resolve_model("lightning", folder=folder) == path and len(calls) == 2


def test_providers():
    cpu = ["CPUExecutionProvider"]
    gpu = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert mn.resolve_providers("auto", cpu) == cpu
    assert mn.resolve_providers(None, gpu) == gpu
    assert mn.resolve_providers("cpu", gpu) == cpu
    assert mn.resolve_providers("cuda", gpu) == gpu
    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        mn.resolve_providers("cuda", cpu)
    with pytest.raises(ValueError, match="device"):
        mn.resolve_providers("tpu", cpu)


def test_crop_and_resize_maps_continuous_coordinates():
    img = np.zeros((300, 500, 3), np.uint8)
    img[140:150, 300:310] = 255  # a 10 x 10 square centered on (305, 145)
    for region in [(0.0, -100.0, 500.0, 500.0), (250.0, 100.0, 100.0, 100.0),
                   (150.0, 0.0, 300.0, 300.0)]:
        out = cv2.cvtColor(mn.crop_and_resize(img, region, (192, 192)), cv2.COLOR_BGR2GRAY)
        ys, xs = np.nonzero(out > 60)
        w = out[ys, xs].astype(float)
        cx, cy = (xs * w).sum() / w.sum() + 0.5, (ys * w).sum() / w.sum() + 0.5
        x0, y0, rw, rh = region
        assert cx == pytest.approx((305 - x0) * 192 / rw, abs=0.6), region
        assert cy == pytest.approx((145 - y0) * 192 / rh, abs=0.6), region
    # outside the image is black
    out = mn.crop_and_resize(np.full((100, 200, 3), 200, np.uint8), (0, -50, 200, 200), (192, 192))
    assert out[:40].max() == 0 and out[-40:].max() == 0 and out[60:130].min() == 200


def test_input_tensor_letterbox_and_coordinates(fake_ort):
    h, w = 480, 640
    truth = standing(h, w, np.random.default_rng(0))
    region = mn.init_region(h, w)
    assert region == (0.0, -80.0, 640.0, 640.0)
    fake_ort.output = lambda x: single_output(truth, np.full(17, 0.8), region)
    det = create_detector("movenet", model_path=str(fake_ort.model), smart_crop=False)
    assert det.format is COCO17 and det.key == "movenet" and det.input_hw == (192, 192)
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = 255  # pure blue (BGR) -> the network gets RGB (0, 0, 255)
    (p,) = det.detect(img, 0.0, "cam0")
    np.testing.assert_allclose(p.keypoints, truth, atol=1e-3)
    np.testing.assert_allclose(p.scores, 0.8, atol=1e-6)
    assert p.score == pytest.approx(0.8)
    np.testing.assert_allclose(p.bbox, [truth[:, 0].min(), truth[:, 1].min(),
                                        truth[:, 0].max(), truth[:, 1].max()], atol=1e-3)
    x = fake_ort.inputs[-1]
    assert x.dtype == np.int32 and x.shape == (1, 192, 192, 3)
    band = 192 * 80 // 640  # 24 rows of padding above and below
    assert x[0, :band - 1].max() == 0 and x[0, -band + 1:].max() == 0
    mid = x[0, band + 2:-band - 2]
    assert np.all(mid[..., 2] == 255) and np.all(mid[..., :2] == 0)


@pytest.mark.parametrize("typ,shape,dtype", [
    ("tensor(float)", [1, 256, 256, 3], np.float32),
    ("tensor(uint8)", [1, 192, 192, 3], np.uint8),
    ("tensor(int32)", [1, 3, 192, 192], np.int32),
    ("tensor(int32)", ["batch", "h", "w", 3], np.int32),
])
def test_model_input_variants(fake_ort, typ, shape, dtype):
    fake_ort.type, fake_ort.shape = typ, shape
    det = create_detector("movenet", model="thunder", model_path=str(fake_ort.model))
    det.detect(np.full((100, 100, 3), 7, np.uint8), 0.0, "cam0")
    x = fake_ort.inputs[-1]
    assert x.dtype == dtype
    size = shape[2] if isinstance(shape[2], int) and shape[1] != 3 else 256
    if shape[1] == 3:
        assert x.shape == (1, 3, 192, 192)
    else:
        assert x.shape == (1, size, size, 3)
    assert x.max() == 7  # values stay 0..255 (no normalization)
    fake_ort.type = "tensor(string)"
    with pytest.raises(ValueError, match="input type"):
        create_detector("movenet", model_path=str(fake_ort.model))


def test_smart_crop_follows_the_person(fake_ort):
    h, w = 720, 1280
    truth = standing(h, w) * 0.5 + (300, 200)  # a small person left of center
    det = create_detector("movenet", model_path=str(fake_ort.model))
    assert det.smart_crop
    img = np.zeros((h, w, 3), np.uint8)
    region = mn.init_region(h, w)
    fake_ort.output = lambda x: single_output(truth, np.full(17, 0.9), region)
    (p,) = det.detect(img, 0.0, "cam0")
    np.testing.assert_allclose(p.keypoints, truth, atol=1e-3)
    nxt = det._regions["cam0"][:4]
    assert nxt == pytest.approx(mn.next_region(truth, np.full(17, 0.9), h, w))
    x0, y0, rw, rh = nxt
    hips = truth[[11, 12]].mean(axis=0)
    assert rw == rh < 0.6 * h and (x0 + rw / 2, y0 + rh / 2) == pytest.approx(tuple(hips))
    # the second frame uses that crop; the output is relative to it
    fake_ort.output = lambda x: single_output(truth, np.full(17, 0.9), nxt)
    (p2,) = det.detect(img, 0.04, "cam0")
    np.testing.assert_allclose(p2.keypoints, truth, atol=1e-3)
    # other cameras are independent; time going back or a lost person resets the crop
    fake_ort.output = lambda x: single_output(truth, np.full(17, 0.9), region)
    (p3,) = det.detect(img, 0.04, "cam1")
    np.testing.assert_allclose(p3.keypoints, truth, atol=1e-3)
    (p4,) = det.detect(img, 0.0, "cam0")  # t went back: full frame again
    np.testing.assert_allclose(p4.keypoints, truth, atol=1e-3)
    fake_ort.output = lambda x: single_output(truth, np.full(17, 0.05), region)
    assert det.detect(img, 0.1, "cam0") == []
    assert "cam0" not in det._regions
    det.reset()
    assert det._regions == {}


def test_next_region_rules():
    h, w = 480, 640
    kp = standing(h, w)
    sc = np.full(17, 0.9)
    no_torso = sc.copy()
    no_torso[[5, 6]] = 0.1
    assert mn.next_region(kp, no_torso, h, w) == mn.init_region(h, w)
    _, _, side, side2 = mn.next_region(kp, sc, h, w)
    assert side == side2
    d = np.abs(kp - kp[[11, 12]].mean(axis=0))
    half = max(1.9 * d[[5, 6, 11, 12]].max(), 1.2 * d.max())
    half = min(half, max(kp[11:13, 0].mean(), w - kp[11:13, 0].mean(), kp[11:13, 1].mean(),
                         h - kp[11:13, 1].mean()))
    assert side == pytest.approx(2 * half)
    big = kp * 3 - (w, h)  # a person larger than the image: whole image
    assert mn.next_region(big, sc, h, w) == mn.init_region(h, w)


def test_scores_empty_scenes_and_bad_outputs(fake_ort):
    h, w = 200, 200
    truth = standing(h, w)
    sc = np.full(17, 0.9)
    sc[0], sc[1], sc[2] = 1.7, np.nan, -0.2
    region = mn.init_region(h, w)
    fake_ort.output = lambda x: single_output(truth, sc, region)
    det = create_detector("movenet", model_path=str(fake_ort.model))
    (p,) = det.detect(np.zeros((h, w, 3), np.uint8), 0.0, "cam0")
    assert p.scores[0] == 1.0 and p.scores[1] == 0.0 and p.scores[2] == 0.0
    assert np.all((p.scores >= 0) & (p.scores <= 1))
    low = np.full(17, 0.1)
    low[:2] = 0.5  # two keypoints are not a person
    fake_ort.output = lambda x: single_output(truth, low, region)
    assert det.detect(np.zeros((h, w, 3), np.uint8), 1.0, "cam0") == []
    assert det.detect(np.zeros((0, 0, 3), np.uint8), 1.0, "cam0") == []
    assert det.detect(None, 1.0, "cam0") == []
    fake_ort.output = lambda x: np.zeros((1, 17, 2))
    with pytest.raises(ValueError, match="output shape"):
        det.detect(np.zeros((h, w, 3), np.uint8), 2.0, "cam0")
    fake_ort.output = lambda x: single_output(truth, np.full(17, 0.9), region)
    assert len(det.detect(np.zeros((h, w), np.uint8), 3.0, "cam0")) == 1  # grayscale
    assert len(det.detect(np.zeros((h, w, 4), np.uint8), 4.0, "cam0")) == 1  # BGRA
    det.close()
    with pytest.raises(RuntimeError, match="closed"):
        det.detect(np.zeros((h, w, 3), np.uint8), 5.0, "cam0")


def test_multipose_outputs(fake_ort):
    h, w = 400, 600
    region = mn.init_region(h, w)
    x0, y0, rw, rh = region
    a, b = standing(h, w) * 0.5, standing(h, w) * 0.5 + (300, 150)
    out = np.zeros((1, 6, 56), np.float32)
    for i, (kp, s) in enumerate([(a, 0.8), (b, 0.6)]):
        out[0, i, :51] = single_output(kp, np.full(17, 0.7), region).reshape(-1)
        box = [(kp[:, 1].min() - y0) / rh, (kp[:, 0].min() - x0) / rw,
               (kp[:, 1].max() - y0) / rh, (kp[:, 0].max() - x0) / rw]
        out[0, i, 51:] = box + [s]
    out[0, 2, :51] = single_output(a, np.full(17, 0.7), region).reshape(-1)
    out[0, 2, 55] = 0.1  # below min_person_score
    fake_ort.output = lambda x: out
    det = create_detector("movenet", model_path=str(fake_ort.model))
    people = det.detect(np.zeros((h, w, 3), np.uint8), 0.0, "cam0")
    assert len(people) == 2
    for p, kp, s in zip(people, (a, b), (0.8, 0.6)):
        np.testing.assert_allclose(p.keypoints, kp, atol=1e-3)
        assert p.score == pytest.approx(s)
        np.testing.assert_allclose(p.bbox, [kp[:, 0].min(), kp[:, 1].min(), kp[:, 0].max(),
                                            kp[:, 1].max()], atol=1e-3)
    assert "cam0" not in det._regions  # no crop tracking for multi-pose models


def test_registry_and_pipeline(fake_ort):
    ok, why = backend_available("movenet")
    assert ok, why
    h, w = 480, 640
    truth = standing(h, w)
    fake_ort.output = lambda x: single_output(truth, np.full(17, 0.9), mn.init_region(h, w))
    fake_ort.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    det = create_detector("movenet", model_path=str(fake_ort.model))
    assert det.options["model"] == "lightning" and det.options["model_path"]
    assert fake_ort.sessions[-1].providers[0] == "CUDAExecutionProvider"
    info = det.info()
    assert info["backend"] == "movenet" and info["keypoint_format"] == "coco17"
    assert info["pose2sim_model"] == "COCO_17" and info["license"] == "Apache-2.0"
    est = MultiViewEstimator(det)
    cams = {"cam0": approximate_calibration("cam0", w, h)}
    pose = est.process({"cam0": (1.0, np.zeros((h, w, 3), np.uint8))}, cams)
    assert pose.mode == MODE_2D_ONLY and pose.format_key == "coco17"
    np.testing.assert_allclose(pose.per_camera_2d["cam0"].keypoints, truth, atol=1e-3)
    est.close()


# ------------------------------------------------------------------ real models
def _real_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module", params=["lightning", "thunder"])
def movenet_real(request):
    ok, why = backend_available("movenet")
    if not ok or not _real_onnxruntime():
        pytest.skip(why if not ok else "onnxruntime cannot be imported")
    try:
        mn.resolve_model(request.param, folder=cache_dir())
    except RuntimeError as e:
        pytest.skip(f"MoveNet model not available: {e}")
    det = create_detector("movenet", model=request.param, device="auto",
                          models_folder=cache_dir())
    yield det
    det.close()


@pytest.fixture(scope="module")
def mediapipe_points(person_image_path):
    """MediaPipe's keypoints (pixels) on the test image, as the reference."""
    pytest.importorskip("mediapipe")
    try:
        mediapipe_backend.ensure_model("full")
    except RuntimeError as e:
        pytest.skip(f"MediaPipe model not available: {e}")
    det = create_detector("mediapipe", model="full")
    try:
        people = det.detect(cv2.imread(str(person_image_path)), 0.0, "ref")
    finally:
        det.close()
    assert len(people) == 1
    return {n: people[0].keypoints[det.format.index(n)] for n in det.format.names}


LIMBS = ("shoulder", "hip", "knee", "ankle")


def test_real_movenet_on_person_image(movenet_real, person_image):
    h, w = person_image.shape[:2]
    movenet_real.reset()
    people = movenet_real.detect(person_image, 0.0, "cam0")
    assert len(people) == 1
    p = people[0]
    assert p.keypoints.shape == (17, 2) and np.all((p.scores >= 0) & (p.scores <= 1))
    kp = {n: p.keypoints[COCO17.index(n)] for n in COCO17.names}
    for side in ("left", "right"):
        assert all(p.scores[COCO17.index(f"{side}_{j}")] > 0.3 for j in LIMBS)
        ys = [kp[f"{side}_{j}"][1] for j in LIMBS]
        assert ys == sorted(ys), (side, ys)  # shoulder above hip above knee above ankle
    # the subject faces the camera: their left side is on the image right
    assert kp["left_shoulder"][0] > kp["right_shoulder"][0]
    assert kp["left_hip"][0] > kp["right_hip"][0]
    body = np.array([kp[f"{s}_{j}"] for s in ("left", "right") for j in LIMBS])
    assert np.all((body >= 0) & (body < [w, h]))


def test_real_movenet_agrees_with_mediapipe(movenet_real, person_image, mediapipe_points):
    mp = mediapipe_points
    assert mp["left_shoulder"][0] > mp["right_shoulder"][0]  # same left/right convention
    tol = 0.08 * np.hypot(*person_image.shape[:2])
    movenet_real.reset()
    for t in (0.0, 0.04):  # full frame, then the smart crop around the person
        (p,) = movenet_real.detect(person_image, t, "cam0")
        for side in ("left", "right"):
            for j in LIMBS:
                name = f"{side}_{j}"
                d = np.linalg.norm(p.keypoints[COCO17.index(name)] - mp[name])
                assert d < tol, f"t={t} {name}: {d:.1f} px from MediaPipe (limit {tol:.1f})"
    assert "cam0" in movenet_real._regions


def test_real_movenet_returns_original_image_pixels(movenet_real, person_image):
    """Letterboxing (other aspect ratio) and resizing are undone: same body, same pixels."""
    movenet_real.reset()
    (base,) = movenet_real.detect(person_image, 0.0, "a")
    idx = [COCO17.index(f"{s}_{j}") for s in ("left", "right") for j in LIMBS]
    tol = 0.03 * np.hypot(*person_image.shape[:2])
    padded = cv2.copyMakeBorder(person_image, 90, 150, 240, 0, cv2.BORDER_CONSTANT,
                                value=(0, 0, 0))
    (pp,) = movenet_real.detect(padded, 0.0, "b")
    err = np.linalg.norm(pp.keypoints[idx] - (240.0, 90.0) - base.keypoints[idx], axis=1)
    assert np.all(err < tol), err
    half = cv2.resize(person_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    (ph,) = movenet_real.detect(half, 0.0, "c")
    err = np.linalg.norm(ph.keypoints[idx] * 2.0 - base.keypoints[idx], axis=1)
    assert np.all(err < tol), err


def test_real_movenet_empty_image(movenet_real):
    movenet_real.reset()
    assert movenet_real.detect(np.zeros((480, 640, 3), np.uint8), 0.0, "cam0") == []
