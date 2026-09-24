"""The ``yolo_pose`` backend (Ultralytics YOLO11 / YOLOv8 pose).

The first part runs everywhere: ultralytics is replaced by a stub, so weight handling, the
keypoint order, coordinates, scores, boxes and empty inputs are checked without the library.
The second part runs the real models on a real image when ultralytics is installed and the
weights can be downloaded (GitHub); otherwise it is skipped with the reason.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from poseboard.calibration import approximate_calibration
from poseboard.pose import mediapipe_backend
from poseboard.pose.base import MODE_2D_ONLY
from poseboard.pose.detectors import backend_available, create_detector
from poseboard.pose.detectors import ultralytics_det as ud
from poseboard.pose.formats import COCO17, HALPE26
from poseboard.pose.multiview import MultiViewEstimator
from tests.conftest import cache_dir

ROOT = Path(__file__).resolve().parents[1]

# Keypoint order of the Ultralytics COCO pose weights, copied from
# ultralytics/cfg/datasets/coco-pose.yaml (kpt_names); checked against the installed file below.
ULTRALYTICS_COCO_ORDER = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


# ------------------------------------------------------------------ ultralytics stub
class FakeTensor:
    """Mimics a torch tensor on a GPU: only .detach().cpu().numpy() gives the values."""

    def __init__(self, a):
        self._a = np.asarray(a, np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a


def make_result(xy, conf="ones", boxes=None, box_conf=None):
    xy = np.asarray(xy, np.float32).reshape(-1, np.shape(xy)[-2], 2)
    if isinstance(conf, str):
        conf = np.ones(xy.shape[:2])
    kp = SimpleNamespace(xy=FakeTensor(xy), conf=None if conf is None else FakeTensor(conf))
    bx = None
    if boxes is not None:
        bx = SimpleNamespace(xyxy=FakeTensor(boxes),
                             conf=None if box_conf is None else FakeTensor(box_conf))
    return SimpleNamespace(keypoints=kp, boxes=bx)


@pytest.fixture
def fake_ul(monkeypatch, tmp_path):
    """Replace the ultralytics package by a stub (also in an environment that has it)."""
    state = SimpleNamespace(task="pose", kpt_shape=[17, 3], kpt_names=None, downloads=[],
                            download_fails=False, yolos=[], events=SimpleNamespace(enabled=True),
                            results=lambda img: [])

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path, self.task_arg, self.task = path, task, state.task
            self.model = SimpleNamespace(kpt_shape=state.kpt_shape, kpt_names=state.kpt_names)
            self.calls = []
            state.yolos.append(self)

        def predict(self, img, **kw):
            self.calls.append((img, kw))
            return state.results(img)

    def attempt_download_asset(file, **kw):
        state.downloads.append(str(file))
        if state.download_fails:
            raise ConnectionError("Download failure. Environment may be offline.")
        Path(file).write_bytes(b"fake weights")
        return str(file)

    def module(name, is_pkg=False, **attrs):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=is_pkg)
        if is_pkg:
            m.__path__ = []
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    module("ultralytics", True, YOLO=FakeYOLO, __version__="0-fake")
    module("ultralytics.utils", True)
    module("ultralytics.utils.downloads", attempt_download_asset=attempt_download_asset)
    module("ultralytics.utils.events", events=state.events)
    monkeypatch.setattr(mediapipe_backend, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(ud, "_cuda_available", lambda: False)
    state.models = tmp_path / "models"
    return state


def _raw_person(k=17, x0=100.0, y0=50.0):
    """Distinct coordinates per keypoint index."""
    j = np.arange(k, dtype=np.float64)
    return np.stack([x0 + 7.0 * j, y0 + 11.0 * j], axis=1)


# ------------------------------------------------------------------ stubbed library
def test_module_import_loads_no_model_library():
    code = ("import sys, poseboard.pose.detectors.ultralytics_det; "
            "print(','.join(m for m in ('torch', 'ultralytics', 'torchvision', 'onnxruntime') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_weights_names_and_versions():
    assert ud.weights_name("n", "11") == "yolo11n-pose.pt"
    assert ud.weights_name("S", "8") == "yolov8s-pose.pt"
    assert ud.weights_name("m", "yolov8") == "yolov8m-pose.pt"
    assert ud.weights_name("x", "yolo11") == "yolo11x-pose.pt"
    assert ud.weights_name("l", "26") == "yolo26l-pose.pt"
    assert ud.weights_name("yolov8m-pose", "11") == "yolov8m-pose.pt"
    assert ud.weights_name("yolo11l-pose.pt") == "yolo11l-pose.pt"
    assert ud.weights_name("my_models/custom.pt") is None
    assert ud.weights_name("yolo11n.pt") is None  # a detection model is not a pose model
    with pytest.raises(ValueError, match="version"):
        ud.weights_name("n", "5")


def test_resolve_device(monkeypatch):
    monkeypatch.setattr(ud, "_cuda_available", lambda: True)
    assert ud.resolve_device("auto") == "cuda:0" and ud.resolve_device(None) == "cuda:0"
    assert ud.resolve_device("cuda") == "cuda:0" and ud.resolve_device("cuda:1") == "cuda:1"
    assert ud.resolve_device("CPU") == "cpu"
    monkeypatch.setattr(ud, "_cuda_available", lambda: False)
    assert ud.resolve_device("auto") == "cpu" and ud.resolve_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        ud.resolve_device("cuda")
    with pytest.raises(RuntimeError, match="CUDA"):
        ud.resolve_device("0")


def test_weights_are_downloaded_into_the_models_folder(fake_ul, tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    det = ud.create(model="s", version="8", device="cpu")
    target = fake_ul.models / "yolov8s-pose.pt"
    assert fake_ul.downloads == [str(target)] and target.is_file()
    assert det.weights_path == target and fake_ul.yolos[-1].path == str(target)
    assert fake_ul.yolos[-1].task_arg == "pose"
    assert list(cwd.iterdir()) == []  # nothing lands in the working directory
    info = det.info()
    assert info["backend"] == "yolo_pose" and info["weights"] == "yolov8s-pose.pt"
    assert info["license"] == "AGPL-3.0" and info["keypoint_format"] == "coco17"
    assert info["device_used"] == "cpu" and det.options["version"] == "8"
    assert fake_ul.events.enabled is False  # ultralytics usage analytics off in this process
    ud.create(model="s", version="yolov8", device="cpu")
    assert len(fake_ul.downloads) == 1  # already there: no second download


def test_download_failure_explains_the_manual_fix(fake_ul):
    fake_ul.download_fails = True
    with pytest.raises(RuntimeError) as e:
        ud.create(model="m", version="11", device="cpu")
    msg = str(e.value)
    assert ud.RELEASES_URL in msg and str(fake_ul.models / "yolo11m-pose.pt") in msg
    assert not (fake_ul.models / "yolo11m-pose.pt").exists()


def test_custom_model_file(fake_ul, tmp_path):
    custom = tmp_path / "custom-pose.pt"
    custom.write_bytes(b"x")
    det = ud.create(model=str(custom), device="cpu")
    assert det.weights_path == custom.resolve() and fake_ul.downloads == []
    with pytest.raises(FileNotFoundError, match="not found"):
        ud.create(model=str(tmp_path / "missing.pt"), device="cpu")


def test_keypoint_order_is_coco17(fake_ul):
    raw = _raw_person()
    fake_ul.results = lambda img: [make_result(raw[None], boxes=[[90, 40, 300, 300]],
                                               box_conf=[0.9])]
    det = ud.create(device="cpu")
    assert det.format is COCO17 and det.key == "yolo_pose" and not det.provides_3d
    (p,) = det.detect(np.zeros((480, 640, 3), np.uint8), 0.0, "cam0")
    for i, name in enumerate(ULTRALYTICS_COCO_ORDER):
        np.testing.assert_allclose(p.keypoints[COCO17.index(name)], raw[i])
    assert ULTRALYTICS_COCO_ORDER == COCO17.names


def test_installed_ultralytics_metadata_matches_coco17():
    """The order and left/right pairs in the installed ultralytics' COCO pose dataset file."""
    try:
        spec = importlib.util.find_spec("ultralytics")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("ultralytics is not installed")
    yaml = pytest.importorskip("yaml")
    path = Path(next(iter(spec.submodule_search_locations))) / "cfg" / "datasets" / "coco-pose.yaml"
    if not path.is_file():
        pytest.skip(f"{path} not found in this ultralytics version")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert tuple(data["kpt_names"][0]) == ULTRALYTICS_COCO_ORDER == COCO17.names
    assert list(data["kpt_shape"]) == [17, 3]
    for i, j in enumerate(data["flip_idx"]):  # flipping swaps exactly left_* <-> right_*
        a, b = COCO17.names[i], COCO17.names[j]
        assert a.replace("left_", "right_") == b or b.replace("left_", "right_") == a or a == b


def test_keypoints_are_mapped_by_name_when_the_model_has_names(fake_ul):
    names = list(reversed(ULTRALYTICS_COCO_ORDER))
    names[names.index("left_shoulder")] = "Left Shoulder"  # names are normalized
    fake_ul.kpt_names = {0: names}
    raw = _raw_person()
    fake_ul.results = lambda img: [make_result(raw[None])]
    det = ud.create(device="cpu")
    (p,) = det.detect(np.zeros((100, 100, 3), np.uint8), 0.0, "cam0")
    np.testing.assert_allclose(p.keypoints, raw[::-1])
    assert p.keypoints[COCO17.index("left_shoulder")][0] == raw[names.index("Left Shoulder")][0]


def test_coordinates_scores_and_boxes_of_all_persons(fake_ul):
    h, w = 1080, 1920  # larger than the 640 network input: ultralytics returns original pixels
    a = _raw_person(x0=1500.0, y0=300.0)
    a[3] = 0.0  # (0, 0) = not predicted
    b = _raw_person(x0=200.0, y0=600.0)
    conf = np.full((2, 17), 0.8)
    conf[0, 4] = np.nan
    conf[0, 5] = 1.3
    conf[1, 16] = -0.2
    boxes = [[1490, 290, 1640, 490], [190, 590, 340, 790]]
    fake_ul.results = lambda img: [make_result(np.stack([a, b]), conf, boxes, [0.95, 0.6])]
    det = ud.create(device="cpu", imgsz=640)
    img = np.full((h, w, 3), 7, np.uint8)
    people = det.detect(img, 1.5, "cam1")
    sent, kw = fake_ul.yolos[-1].calls[-1]
    assert sent.shape == (h, w, 3) and sent.dtype == np.uint8 and np.all(sent == 7)
    assert kw["imgsz"] == 640 and kw["device"] == "cpu" and kw["verbose"] is False
    assert len(people) == 2  # every person; the subject is chosen later
    pa, pb = people
    assert np.all(np.isnan(pa.keypoints[3])) and pa.scores[3] == 0.0
    keep = np.arange(17) != 3
    np.testing.assert_allclose(pa.keypoints[keep], a[keep])  # no rescaling
    np.testing.assert_allclose(pb.keypoints, b)
    assert pa.scores[4] == 0.0 and pa.scores[5] == 1.0 and pb.scores[16] == 0.0
    assert np.all((pa.scores >= 0) & (pa.scores <= 1)) and pa.scores[0] == pytest.approx(0.8)
    np.testing.assert_allclose(pa.bbox, boxes[0])
    np.testing.assert_allclose(pb.bbox, boxes[1])
    assert pa.score == pytest.approx(0.95) and pb.score == pytest.approx(0.6)


def test_models_without_keypoint_confidence_or_boxes(fake_ul):
    raw = _raw_person()
    raw[0] = 0.0
    fake_ul.results = lambda img: [make_result(raw[None], conf=None, boxes=None)]
    det = ud.create(device="cpu")
    (p,) = det.detect(np.zeros((480, 640, 3), np.uint8), 0.0, "cam0")
    assert p.scores[0] == 0.0 and np.all(p.scores[1:] == 1.0) and np.all(np.isnan(p.keypoints[0]))
    np.testing.assert_allclose(p.bbox, [raw[1:, 0].min(), raw[1:, 1].min(),
                                        raw[1:, 0].max(), raw[1:, 1].max()])
    assert p.score == 1.0


def test_empty_images_and_empty_results(fake_ul):
    det = ud.create(device="cpu")
    yolo = fake_ul.yolos[-1]
    assert det.detect(None, 0.0, "c") == []
    assert det.detect(np.zeros((0, 0, 3), np.uint8), 0.0, "c") == []
    assert det.detect(np.zeros((0, 640, 3), np.uint8), 0.0, "c") == []
    assert yolo.calls == []  # the model is not run on an empty image
    img = np.zeros((48, 64, 3), np.uint8)
    for res in ([], [make_result(np.zeros((0, 17, 2)), conf=np.zeros((0, 17)),
                                 boxes=np.zeros((0, 4)), box_conf=np.zeros(0))],
                [SimpleNamespace(keypoints=None, boxes=None)]):
        fake_ul.results = lambda img, res=res: res
        assert det.detect(img, 0.0, "c") == []
    assert len(yolo.calls) == 3


def test_grayscale_and_bgra_images_become_bgr(fake_ul):
    det = ud.create(device="cpu")
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    det.detect(gray, 0.0, "c")
    sent = fake_ul.yolos[-1].calls[-1][0]
    assert sent.shape == (3, 4, 3) and np.all(sent[..., 0] == gray) and np.all(sent[..., 2] == gray)
    bgra = np.dstack([np.full((3, 4), v, np.uint8) for v in (1, 2, 3, 255)])
    det.detect(bgra, 0.0, "c")
    sent = fake_ul.yolos[-1].calls[-1][0]
    assert sent.shape == (3, 4, 3) and list(sent[0, 0]) == [1, 2, 3]


def test_predict_options(fake_ul):
    det = ud.create(model="x", version="11", device="cpu", conf=0.4, iou=0.5, imgsz=320,
                    max_det=5, half=True)
    det.detect(np.zeros((10, 10, 3), np.uint8), 0.0, "c")
    kw = fake_ul.yolos[-1].calls[-1][1]
    assert kw == {"conf": 0.4, "iou": 0.5, "imgsz": 320, "max_det": 5, "device": "cpu",
                  "half": False, "verbose": False}  # no FP16 on the CPU
    assert det.options == {"model": "x", "version": "11", "device": "cpu", "conf": 0.4,
                           "iou": 0.5, "imgsz": 320, "max_det": 5, "half": True,
                           "keypoint_format": "coco17"}


def test_models_that_do_not_fit(fake_ul):
    fake_ul.task = "detect"
    with pytest.raises(ValueError, match="not a pose model"):
        ud.create(device="cpu")
    fake_ul.task = "pose"
    fake_ul.kpt_shape = [21, 3]
    with pytest.raises(ValueError, match="keypoint_format"):
        ud.create(device="cpu")
    fake_ul.kpt_shape = [26, 3]
    raw = _raw_person(26)
    fake_ul.results = lambda img: [make_result(raw[None])]
    det = ud.create(device="cpu", keypoint_format="halpe26")
    assert det.format is HALPE26 and det.info()["pose2sim_model"] == "HALPE_26"
    (p,) = det.detect(np.zeros((480, 640, 3), np.uint8), 0.0, "c")
    assert p.keypoints.shape == (26, 2)


def test_exported_models_get_their_layout_from_the_first_result(fake_ul):
    fake_ul.kpt_shape = None  # e.g. ONNX: unknown until the predictor has run
    det = ud.create(device="cpu")
    img = np.zeros((48, 64, 3), np.uint8)
    fake_ul.results = lambda img: [make_result(np.zeros((0, 17, 2)))]
    assert det.detect(img, 0.0, "c") == []
    fake_ul.results = lambda img: [make_result(_raw_person(17)[None])]
    assert len(det.detect(img, 0.0, "c")) == 1
    det2 = ud.create(device="cpu")
    fake_ul.results = lambda img: [make_result(_raw_person(21)[None])]
    with pytest.raises(ValueError, match="21 keypoints"):
        det2.detect(img, 0.0, "c")


def test_registry_and_pipeline_with_the_stub(fake_ul):
    assert backend_available("yolo_pose") == (True, "available")
    raw = _raw_person()
    fake_ul.results = lambda img: [make_result(raw[None], boxes=[[90, 40, 300, 300]],
                                               box_conf=[0.9])]
    det = create_detector("yolo_pose", device="cpu")
    assert isinstance(det, ud.YoloPoseDetector)
    assert det.options["model"] == "n" and det.options["version"] == "11"
    assert det.weights_path.name == "yolo11n-pose.pt"
    est = MultiViewEstimator(det)
    cams = {"cam0": approximate_calibration("cam0", 640, 480)}
    pose = est.process({"cam0": (2.0, np.zeros((480, 640, 3), np.uint8))}, cams)
    assert pose.mode == MODE_2D_ONLY and pose.format_key == "coco17"
    np.testing.assert_allclose(pose.per_camera_2d["cam0"].keypoints, raw)
    est.close()
    with pytest.raises(RuntimeError, match="closed"):
        det.detect(np.zeros((4, 4, 3), np.uint8), 0.0, "cam0")


# ------------------------------------------------------------------ real models
@pytest.fixture(scope="module", params=["11", "8"], ids=["yolo11n", "yolov8n"])
def yolo_real(request):
    ok, why = backend_available("yolo_pose")
    if not ok:
        pytest.skip(why)
    try:
        ud.resolve_weights("n", request.param, weights_dir=cache_dir())
    except RuntimeError as e:
        pytest.skip(f"YOLO pose weights not available: {e}")
    det = create_detector("yolo_pose", model="n", version=request.param, device="auto",
                          weights_dir=cache_dir())
    yield det
    det.close()


@pytest.fixture(scope="module")
def mediapipe_points(person_image_path):
    """MediaPipe's keypoints (pixels) on the test image, as the reference."""
    import cv2

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


def test_real_yolo_pose_on_person_image(yolo_real, person_image):
    h, w = person_image.shape[:2]
    people = yolo_real.detect(person_image, 0.0, "cam0")
    assert len(people) == 1
    p = people[0]
    assert p.keypoints.shape == (17, 2) and p.scores.shape == (17,)
    assert np.all((p.scores >= 0) & (p.scores <= 1)) and 0.5 < p.score <= 1
    kp = {n: p.keypoints[COCO17.index(n)] for n in COCO17.names}
    for side in ("left", "right"):
        assert all(p.scores[COCO17.index(f"{side}_{j}")] > 0.5 for j in LIMBS)
        ys = [kp[f"{side}_{j}"][1] for j in LIMBS]
        assert ys == sorted(ys), (side, ys)  # shoulder above hip above knee above ankle
    # the subject faces the camera: their left side is on the image right
    assert kp["left_shoulder"][0] > kp["right_shoulder"][0]
    assert kp["left_hip"][0] > kp["right_hip"][0]
    body = np.array([kp[f"{s}_{j}"] for s in ("left", "right") for j in LIMBS])
    assert np.all((body >= 0) & (body < [w, h]))
    x1, y1, x2, y2 = p.bbox
    assert np.all((body[:, 0] >= x1 - 5) & (body[:, 0] <= x2 + 5))
    assert np.all((body[:, 1] >= y1 - 5) & (body[:, 1] <= y2 + 5))


def test_real_yolo_pose_agrees_with_mediapipe(yolo_real, person_image, mediapipe_points):
    mp = mediapipe_points
    assert mp["left_shoulder"][0] > mp["right_shoulder"][0]  # same left/right convention
    (p,) = yolo_real.detect(person_image, 0.0, "cam0")
    tol = 0.08 * np.hypot(*person_image.shape[:2])
    for side in ("left", "right"):
        for j in LIMBS:
            name = f"{side}_{j}"
            d = np.linalg.norm(p.keypoints[COCO17.index(name)] - mp[name])
            assert d < tol, f"{name}: {d:.1f} px from MediaPipe (limit {tol:.1f})"


def test_real_yolo_pose_returns_original_image_pixels(yolo_real, person_image):
    """Letterboxing (other aspect ratio) and resizing are undone: same body, same pixels."""
    import cv2

    (base,) = yolo_real.detect(person_image, 0.0, "cam0")
    ok = base.scores > 0.5
    tol = 0.03 * np.hypot(*person_image.shape[:2])
    padded = cv2.copyMakeBorder(person_image, 90, 150, 240, 0, cv2.BORDER_CONSTANT,
                                value=(0, 0, 0))
    (pp,) = yolo_real.detect(padded, 0.0, "cam0")
    err = np.linalg.norm(pp.keypoints - (240.0, 90.0) - base.keypoints, axis=1)
    assert np.all(err[ok] < tol), err
    half = cv2.resize(person_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    (ph,) = yolo_real.detect(half, 0.0, "cam0")
    err = np.linalg.norm(ph.keypoints * 2.0 - base.keypoints, axis=1)
    assert np.all(err[ok] < tol), err


def test_real_yolo_pose_in_the_pipeline(yolo_real, person_image):
    h, w = person_image.shape[:2]
    est = MultiViewEstimator(yolo_real)  # not closed here: the detector is shared (fixture)
    cams = {"cam0": approximate_calibration("cam0", w, h)}
    pose = est.process({"cam0": (1.0, person_image)}, cams)
    assert pose.mode == MODE_2D_ONLY and pose.format_key == "coco17"
    p2 = pose.per_camera_2d["cam0"]
    assert p2.keypoints.shape == (17, 2) and np.nanmin(p2.scores) >= 0
    assert np.isfinite(p2.keypoints[COCO17.index("left_ankle")]).all()
