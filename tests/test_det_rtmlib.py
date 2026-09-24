"""rtmlib backends (poseboard/pose/detectors/rtmlib_det.py).

* Always: stub models (keypoint order, score normalization, missing keypoints, boxes, empty
  images, the RTMPose3D depth decoding and metric skeleton, single-view 3D through
  ``MultiViewEstimator``) and the factories / registry with a fake ``rtmlib`` module.
* rtmlib installed: rtmlib's own model classes with fake ONNX sessions (the crop / letterbox
  mapping back to image pixels and the colour order are checked by a round trip through
  rtmlib's real pre- and post-processing), and rtmlib's keypoint metadata.
* rtmlib + GitHub reachable: rtmlib's YOLOX with the real YOLOX-tiny ONNX (GitHub release).
* rtmlib + model hosts reachable (e.g. the Windows CI runner): every backend with its real
  models on the test photo, compared with MediaPipe. Skipped with the reason when the weights
  cannot be downloaded. ``POSEBOARD_RTMLIB_SMOKE`` = comma-separated keys (default: all),
  ``none`` to skip; ``POSEBOARD_RTMLIB_SMOKE_MODE`` = model size (default ``lightweight``;
  rtmpose3d always uses ``balanced``, its only mode).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import types
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from poseboard.calibration import approximate_calibration
from poseboard.geometry import project_to_image
from poseboard.pose.base import MODE_2D_ONLY, MODE_SINGLE_VIEW_3D
from poseboard.pose.detectors import BACKENDS, backend_available, create_detector
from poseboard.pose.detectors import rtmlib_det as rd
from poseboard.pose.detectors.base import Detector2D
from poseboard.pose.formats import COCO17, FORMATS, HALPE26, WHOLEBODY133
from poseboard.pose.multiview import MultiViewEstimator, single_view_lift
from tests.conftest import download_cached
from tests.fakes import standing_person

RTMLIB_KEYS = ("rtmpose", "rtmpose_halpe26", "rtmw_wholebody", "rtmo", "vitpose_onnx",
               "rtmpose3d")
JOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee",
          "left_ankle", "right_ankle")
BLUE = (255, 0, 0)  # BGR


def _has(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


needs_rtmlib = pytest.mark.skipif(not (_has("rtmlib") and _has("onnxruntime")),
                                  reason="rtmlib / onnxruntime not installed")


# ------------------------------------------------------------------ stub models
class StubDet:
    """Person detector returning fixed boxes (or ``(boxes, classes)``)."""

    def __init__(self, out):
        self.out = out
        self.images = []

    def __call__(self, image):
        self.images.append(image)
        return self.out


class StubPose:
    """Top-down pose model: ``fn(box) -> (keypoints (K, 2), scores (K,))`` per box."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, image, bboxes=()):
        self.calls.append((image, [list(b) for b in bboxes]))
        out = [self.fn(np.asarray(b, float)) for b in bboxes]
        return (np.stack([o[0] for o in out]).astype(np.float32),
                np.stack([o[1] for o in out]).astype(np.float32))


def _grid_keypoints(box, k):
    """k distinct points inside ``box`` (their index is recoverable from the position)."""
    x1, y1, x2, y2 = box
    i = np.arange(k)
    return np.column_stack([x1 + (x2 - x1) * (i + 1) / (k + 1), y1 + (y2 - y1) * 0.5 + i])


def test_library_orders_match_the_formats():
    assert rd.index_map(rd.RTMLIB_COCO17, COCO17).tolist() == list(range(17))
    assert rd.index_map(rd.RTMLIB_HALPE26, HALPE26).tolist() == list(range(26))
    assert rd.index_map(rd.RTMLIB_COCO133, WHOLEBODY133).tolist() == list(range(133))
    for kind in rd.KINDS.values():
        assert kind.fmt.key == BACKENDS[kind.key].keypoint_format
        assert kind.provides_3d == BACKENDS[kind.key].provides_3d
    assert rd.canonical_name("face-67") == "face_67"
    assert rd.canonical_name("left_hand_root") == "left_hand_0"
    assert rd.canonical_name("left_thumb1") == "left_hand_1"
    assert rd.canonical_name("right_forefinger2") == "right_hand_6"
    assert rd.canonical_name("right_pinky_finger4") == "right_hand_20"
    assert rd.canonical_name("left_heel") == "left_heel"
    # a library layout in another order is permuted into the format order
    perm = rd.index_map(tuple(reversed(rd.RTMLIB_COCO17)), COCO17)
    assert perm.tolist() == list(range(16, -1, -1))
    with pytest.raises(ValueError):
        rd.index_map(rd.RTMLIB_COCO17[:-1], COCO17)
    with pytest.raises(ValueError):
        rd.index_map(rd.RTMLIB_COCO17, HALPE26)


def test_rtmlib_metadata_matches_our_orders():
    pytest.importorskip("rtmlib")
    from rtmlib.visualization.skeleton import coco17, coco133, halpe26

    for meta, ours in ((coco17, rd.RTMLIB_COCO17), (halpe26, rd.RTMLIB_HALPE26),
                       (coco133, rd.RTMLIB_COCO133)):
        info = meta["keypoint_info"]
        names = tuple(info[i]["name"] for i in sorted(info))
        assert [info[i]["id"] for i in sorted(info)] == list(range(len(info)))
        assert names == ours
        for i in sorted(info):  # left/right swaps point to the mirrored name
            n, swap = info[i]["name"], info[i]["swap"]
            if n.startswith("left_"):
                assert swap == "right_" + n[5:]


def test_topdown_pipeline_maps_order_scores_and_boxes():
    h, w = 120, 200
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = 200  # BGR blue: the pose model must get RGB (blue in channel 2)
    boxes = np.array([[20, 10, 80, 110], [150, -30, 260, 100], [5, 5, 6, 6]], float)
    raw = np.linspace(-0.2, 1.6, 17)  # <= 0: not found; > 1: clipped
    raw[3] = np.nan

    def fn(box):
        return _grid_keypoints(box, 17)[::-1], raw[::-1]  # library order = reversed COCO

    det, pose = StubDet(boxes), StubPose(fn)
    d = rd.RTMLibDetector("rtmpose", pose, det, lib_names=tuple(reversed(rd.RTMLIB_COCO17)))
    assert isinstance(d, Detector2D) and d.key == "rtmpose" and d.format is COCO17
    assert not d.provides_3d and d.label == BACKENDS["rtmpose"].label
    people = d.detect(img, 0.0, "cam0")
    assert len(people) == 2  # the 1-px box is dropped
    (image, sent), = pose.calls
    assert image[0, 0].tolist() == [0, 0, 200]  # RGB
    assert np.allclose(sent, boxes[:2])  # unclipped boxes go to the pose model
    assert det.images[0][0, 0].tolist() == [200, 0, 0]  # YOLOX gets BGR
    for p, box in zip(people, boxes[:2]):
        exp = _grid_keypoints(box, 17)
        ok = raw > 0
        assert np.allclose(p.keypoints[ok], exp[ok])
        assert np.all(np.isnan(p.keypoints[~ok]))
        assert np.allclose(p.scores, np.clip(np.nan_to_num(raw), 0, 1))
        assert p.scores.max() == 1.0 and p.scores.min() == 0.0
        assert np.isclose(p.score, p.scores.mean())
        assert p.keypoints_3d is None
    assert np.allclose(people[1].bbox, [150, 0, 200, 100])  # clipped to the image
    assert d.info()["keypoint_format"] == "coco17" and d.info()["rgb_input"] is True


def test_score_scale_and_multiclass_detector():
    boxes = np.array([[10, 10, 60, 90], [70, 10, 120, 90]], float)
    det = StubDet((boxes, np.array([0, 2])))  # a person and a car (COCO class 2)
    pose = StubPose(lambda b: (_grid_keypoints(b, 26), np.full(26, 3.0)))
    d = rd.RTMLibDetector("rtmpose_halpe26", pose, det, score_scale=4.0, rgb_input=False)
    img = np.full((100, 130, 3), 7, np.uint8)
    (p,) = d.detect(img, 0.0, "c")
    assert p.keypoints.shape == (26, 2) and np.allclose(p.scores, 0.75)
    assert np.allclose(p.bbox, boxes[0]) and pose.calls[0][0] is img  # BGR as given
    with pytest.raises(ValueError):
        rd.RTMLibDetector("rtmpose", pose, det, score_scale=0)


def test_empty_images_and_no_person_give_no_people():
    pose = StubPose(lambda b: (_grid_keypoints(b, 17), np.ones(17)))
    d = rd.RTMLibDetector("rtmpose", pose, StubDet(np.array([])))
    assert d.detect(np.zeros((480, 640, 3), np.uint8), 0.0, "c") == []
    assert pose.calls == []  # rtmlib would run the pose model on the whole image
    assert d.detect(None, 0.0, "c") == []
    assert d.detect(np.zeros((0, 0, 3), np.uint8), 0.0, "c") == []
    d.det_model = StubDet(np.array([[1, 1, 30, 40]], float))
    gray = np.zeros((50, 50), np.uint8)
    assert len(d.detect(gray, 0.0, "c")) == 1  # grayscale is accepted
    bgra = np.zeros((50, 50, 4), np.uint8)
    assert len(d.detect(bgra, 0.0, "c")) == 1
    d.close()
    with pytest.raises(RuntimeError, match="closed"):
        d.detect(gray, 0.0, "c")
    with pytest.raises(ValueError):
        rd.RTMLibDetector("rtmpose", pose, None)
    with pytest.raises(ValueError):
        rd.RTMLibDetector("rtmo", pose, StubDet(np.array([])))


def test_one_stage_rtmo_stub():
    calls = []

    def rtmo(image):
        calls.append(image)
        if len(calls) == 1:  # rtmlib's placeholder when nothing passes NMS
            return np.zeros((1, 17, 2), np.float32), np.zeros((1, 17), np.float32)
        kp = np.stack([_grid_keypoints([10, 10, 50, 90], 17), _grid_keypoints([60, 5, 90, 60], 17)])
        sc = np.full((2, 17), 0.8)
        sc[1, :5] = 0.1
        return kp, sc

    d = rd.RTMLibDetector("rtmo", rtmo)
    img = np.zeros((100, 100, 3), np.uint8)
    img[..., 0] = 9
    assert d.detect(img, 0.0, "c") == []
    people = d.detect(img, 0.1, "c")
    assert len(people) == 2 and calls[1][0, 0].tolist() == [9, 0, 0]  # RTMO gets BGR
    kp0 = _grid_keypoints([10, 10, 50, 90], 17)
    assert np.allclose(people[0].bbox, [kp0[:, 0].min(), kp0[:, 1].min(), kp0[:, 0].max(),
                                        kp0[:, 1].max()])
    kp1 = _grid_keypoints([60, 5, 90, 60], 17)[5:]  # the box uses the confident points only
    assert np.isclose(people[1].bbox[0], kp1[:, 0].min())


# ------------------------------------------------------------------ RTMPose3D, stubbed
W3, H3 = 1000, 700
CAM = approximate_calibration("cam0", W3, H3)
# world (Z up) -> camera (x right, y down, z forward), camera at (0, -3, 1) looking along +Y
R_WC = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
C_W = np.array([0.0, -3.0, 1.0])


def _person_cam(yaw=30.0):
    world = standing_person("wholebody133", yaw_deg=yaw)
    return (world - C_W) @ R_WC.T


def _project(pc):
    uv = pc[:, :2] / pc[:, 2:3]
    return uv * [CAM.K[0, 0], CAM.K[1, 1]] + [CAM.K[0, 2], CAM.K[1, 2]]


class StubPose3d:
    """rtmlib ``RTMPose3d``-like model for a known camera-frame skeleton."""

    z_range = rd.RTMW3D_Z_RANGE

    def __init__(self, pc, scores=None):
        self.pc = pc
        self.scores = np.full(len(pc), 0.9) if scores is None else scores

    def __call__(self, image, bboxes=()):
        pc = self.pc
        uv = _project(pc)
        root = pc[[11, 12], 2].mean()
        zs = ((pc[:, 2] - root) / self.z_range + 1) * rd.RTMW3D_Z_BINS / 2
        zs = np.round(zs * 2) / 2  # SimCC bins of 0.5 input pixel
        simcc = np.column_stack([uv * 0.3, zs])  # x/y in crop pixels: not used
        rtmlib_kp = np.column_stack([uv * 0.3, (zs / 192 - 1) * self.z_range])  # rtmlib's z
        n = len(bboxes)
        rep = lambda a: np.repeat(a[None], n, axis=0)  # noqa: E731
        return rep(rtmlib_kp), rep(self.scores), rep(simcc), rep(uv)


def _detector3d(pc, **kw):
    uv = _project(pc)
    box = np.array([uv[:, 0].min() - 20, uv[:, 1].min() - 20, uv[:, 0].max() + 20,
                    uv[:, 1].max() + 20])
    return rd.RTMPose3DDetector("rtmpose3d", StubPose3d(pc, **kw), StubDet(box[None]))


def test_rtmpose3d_depth_decoding():
    d = _detector3d(_person_cam())
    assert d.provides_3d and d.format is WHOLEBODY133
    assert d.z_bins == 288 and np.isclose(d.z_range, 2.1744869)
    assert np.allclose(d.depth_m([0, 144, 288]), [-2.1744869, 0, 2.1744869])


@pytest.mark.parametrize("yaw", [0.0, 30.0, -60.0])
def test_rtmpose3d_metric_skeleton(yaw):
    pc = _person_cam(yaw)
    d = _detector3d(pc)
    (p,) = d.detect(np.zeros((H3, W3, 3), np.uint8), 0.0, "cam0")
    assert np.allclose(p.keypoints, _project(pc))
    truth = pc - pc[[11, 12]].mean(axis=0)
    err = np.linalg.norm(p.keypoints_3d - truth, axis=1)
    body = [WHOLEBODY133.index(n) for n in JOINTS]
    assert err[body].max() < 0.08 and np.median(err) < 0.05, err[body]
    ls, rs = p.keypoints_3d[5], p.keypoints_3d[6]
    assert 0.25 < np.linalg.norm(ls - rs) < 0.5
    assert p.keypoints_3d[15, 1] > p.keypoints_3d[11, 1]  # ankles below the hips (y down)


def test_rtmpose3d_missing_scale_or_hips_gives_no_3d():
    pc = _person_cam()
    sc = np.full(133, 0.9)
    sc[5:17] = 0.1  # no confident limb or trunk segment
    (p,) = _detector3d(pc, scores=sc).detect(np.zeros((H3, W3, 3), np.uint8), 0.0, "c")
    assert p.keypoints_3d is None
    sc = np.full(133, 0.9)
    sc[11] = 0.0  # left hip not found
    (p,) = _detector3d(pc, scores=sc).detect(np.zeros((H3, W3, 3), np.uint8), 0.0, "c")
    assert p.keypoints_3d is None and np.isnan(p.keypoints[11]).all()
    sc = np.full(133, 0.9)
    sc[100] = -1.0  # a missing hand point is NaN in 2D and 3D
    (p,) = _detector3d(pc, scores=sc).detect(np.zeros((H3, W3, 3), np.uint8), 0.0, "c")
    assert np.isnan(p.keypoints_3d[100]).all() and np.isfinite(p.keypoints_3d[99]).all()
    with pytest.raises(ValueError):
        rd.RTMPose3DDetector("rtmpose3d", StubPose3d(pc), StubDet(np.array([])), body_height=170)


def test_rtmpose3d_single_camera_3d_via_multiview():
    pc = _person_cam(20.0)
    est = MultiViewEstimator(_detector3d(pc), min_score=0.3)
    pose = est.process({"cam0": (1.0, np.zeros((H3, W3, 3), np.uint8))}, {"cam0": CAM})
    assert pose.mode == MODE_SINGLE_VIEW_3D and pose.format_key == "wholebody133"
    assert pose.reproj_error_px < 3.0
    body = [WHOLEBODY133.index(n) for n in JOINTS]
    err = np.linalg.norm(pose.keypoints[body] - pc[body], axis=1)
    assert err.max() < 0.15, err  # metric scale from a nominal 1.70 m body
    assert pose.keypoints[5, 0] > pose.keypoints[6, 0]  # left shoulder on the image right
    # a 2D-only rtmlib backend with one camera gives 2D only
    d2 = rd.RTMLibDetector("rtmpose", StubPose(lambda b: (_grid_keypoints(b, 17), np.ones(17))),
                           StubDet(np.array([[10, 10, 90, 200]], float)))
    pose2 = MultiViewEstimator(d2).process({"cam0": (1.0, np.zeros((H3, W3, 3), np.uint8))},
                                           {"cam0": CAM})
    assert pose2.mode == MODE_2D_ONLY and pose2.per_camera_2d["cam0"].keypoints.shape == (17, 2)


# ------------------------------------------------------------------ factories (fake rtmlib)
class _FakeTool:
    def __init__(self, onnx_model, model_input_size=None, **kw):
        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.kw = kw
        self.score_thr, self.nms_thr = 0.7, 0.45


def _fake_rtmlib():
    mod = types.ModuleType("rtmlib")
    mod.__spec__ = importlib.machinery.ModuleSpec("rtmlib", None)

    def table(tag, pose_size, det=True, modes=rd.MODES):
        return {m: dict(({"det": f"https://h/{tag}-{m}-det.zip", "det_input_size": (416, 416)}
                         if det else {}),
                        pose=f"https://h/{tag}-{m}-pose.zip", pose_input_size=pose_size)
                for m in modes}

    for name in ("YOLOX", "RTMPose", "RTMO", "ViTPose", "RTMPose3d"):
        setattr(mod, name, type(name, (_FakeTool,), {}))
    mod.RTMPose3d.z_range = 2.0
    mod.Body = type("Body", (), {"MODE": table("body", (192, 256)),
                                 "RTMO_MODE": table("rtmo", (640, 640), det=False)})
    mod.BodyWithFeet = type("BodyWithFeet", (), {"MODE": table("feet", (192, 256))})
    mod.Wholebody = type("Wholebody", (), {"MODE": table("wb", (288, 384))})
    mod.Wholebody3d = type("Wholebody3d", (), {"MODE": table("wb3d", (288, 384),
                                                             modes=("balanced",))})
    return mod


@pytest.fixture
def fake_rtmlib(monkeypatch, tmp_path):
    mod = _fake_rtmlib()
    monkeypatch.setitem(sys.modules, "rtmlib", mod)
    ort = types.ModuleType("onnxruntime")
    ort.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", None)
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    downloads = []

    def download(url):
        downloads.append(url)
        if "fail" in url:
            raise OSError("HTTP Error 403: Forbidden")
        return str(tmp_path / url.rsplit("/", 1)[-1].replace(".zip", ".onnx"))

    monkeypatch.setattr(rd, "_download", download)
    mod.downloads = downloads
    mod.ort = ort
    return mod


@pytest.mark.parametrize("key,solution,table,pose_cls", [
    ("rtmpose", "body", "MODE", "RTMPose"), ("rtmpose_halpe26", "feet", "MODE", "RTMPose"),
    ("rtmw_wholebody", "wb", "MODE", "RTMPose"), ("rtmo", "rtmo", "RTMO_MODE", "RTMO"),
    ("vitpose_onnx", None, None, "ViTPose"), ("rtmpose3d", "wb3d", "MODE", "RTMPose3d")])
def test_factories_use_rtmlibs_model_tables(fake_rtmlib, key, solution, table, pose_cls):
    mode = "balanced" if key == "rtmpose3d" else "lightweight"
    factory = getattr(rd, BACKENDS[key].factory)
    d = factory(mode=mode, device="auto")
    assert isinstance(d, rd.RTMLibDetector) and d.key == key
    assert d.format.key == BACKENDS[key].keypoint_format and d.device == "cpu"
    assert type(d.pose_model).__name__ == pose_cls
    assert d.pose_model.kw == {"to_openpose": False, "backend": "onnxruntime", "device": "cpu"}
    if key == "vitpose_onnx":
        assert d.pose_model.onnx_model.endswith("vitpose-s-coco.onnx")
        assert fake_rtmlib.downloads[-1] == rd.VITPOSE_ONNX["lightweight"][0]
        assert d.pose_model.model_input_size == (192, 256)
    else:
        assert d.pose_model.onnx_model.endswith(f"{solution}-{mode}-pose.onnx")
    if key == "rtmo":
        assert d.det_model is None and d.one_stage and not d.rgb_input
    else:
        det_tag = "body" if key == "vitpose_onnx" else solution
        assert d.det_model.onnx_model.endswith(f"{det_tag}-{mode}-det.onnx")
        assert d.det_model.model_input_size == (416, 416) and d.rgb_input
    expected = {"mode": mode, "device": "auto", "backend": "onnxruntime"}
    if key == "rtmpose3d":
        expected["body_height"] = 1.70
    assert d.options == expected
    assert set(d.info()["model_files"]) == ({"pose"} if key == "rtmo" else {"det", "pose"})
    assert d.provides_3d == (key == "rtmpose3d")
    if key == "rtmpose3d":
        assert d.z_range == 2.0 and d.body_height == 1.70


def test_factory_options_and_errors(fake_rtmlib, monkeypatch, tmp_path):
    d = rd.create_rtmpose(mode="performance", device="cuda", backend="onnxruntime",
                          det_score_thr=0.5, nms_thr=0.6)
    assert d.device == "cpu"  # no CUDA provider: falls back to the CPU
    assert d.det_model.score_thr == 0.5 and d.det_model.nms_thr == 0.6
    assert d.options["det_score_thr"] == 0.5 and d.options["device"] == "cuda"
    fake_rtmlib.ort.get_available_providers = lambda: ["CUDAExecutionProvider",
                                                       "CPUExecutionProvider"]
    assert rd.create_rtmo(device="auto", det_score_thr=0.3).device == "cuda"
    assert rd.create_rtmo(det_score_thr=0.3).pose_model.score_thr == 0.3
    assert rd.resolve_device("cpu") == "cpu" and rd.resolve_device("auto", "openvino") == "cpu"
    with pytest.raises(ValueError, match="mode must be one of"):
        rd.create_rtmpose(mode="huge")
    with pytest.raises(ValueError, match="balanced"):
        rd.create_rtmpose3d(mode="lightweight")
    with pytest.raises(ValueError, match="backend"):
        rd.create_rtmpose(backend="tensorflow")
    with pytest.raises(TypeError):
        rd.create_rtmpose(body_height=1.8)
    with pytest.raises(TypeError):
        rd.create_rtmpose(unknown_option=1)
    with pytest.raises(rd.ModelUnavailable, match="403"):
        rd.create_rtmpose(pose_model="https://h/fail.onnx")
    with pytest.raises(rd.ModelUnavailable, match="not found"):
        rd.create_rtmpose(det_model=str(tmp_path / "missing.onnx"))
    # a local mmdeploy zip is unpacked next to itself
    zpath = tmp_path / "rtmpose-custom.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("rtmpose-custom/end2end.onnx", b"onnx")
    d = rd.create_rtmpose_halpe26(pose_model=str(zpath), pose_input_size=(288, 384),
                                  rgb_input=False)
    assert d.pose_model.onnx_model == str(tmp_path / "rtmpose-custom.onnx")
    assert (tmp_path / "rtmpose-custom.onnx").read_bytes() == b"onnx"
    assert d.pose_model.model_input_size == (288, 384) and not d.rgb_input
    assert d.options["pose_model"] == str(zpath)
    d3 = rd.create_rtmpose3d(body_height=1.8)
    assert d3.body_height == 1.8 and d3.options["body_height"] == 1.8


def test_factory_without_rtmlib(monkeypatch):
    monkeypatch.setitem(sys.modules, "rtmlib", None)  # import fails
    with pytest.raises(RuntimeError, match="pip install rtmlib"):
        rd.create_rtmpose()


def test_registry_creates_the_rtmlib_backends(fake_rtmlib):
    for key in RTMLIB_KEYS:
        ok, why = backend_available(key)
        assert ok, why
        d = create_detector(key)  # the registry defaults are valid factory arguments
        assert d.key == key and d.options["mode"] == BACKENDS[key].defaults["mode"]
        assert d.format is FORMATS[BACKENDS[key].keypoint_format]
    d = create_detector("rtmpose_halpe26", mode="lightweight", device="cpu")
    assert d.pose_model.onnx_model.endswith("feet-lightweight-pose.onnx")


# ------------------------------------------------------------------ rtmlib classes, fake ONNX
class FakeSession:
    """ONNX Runtime session double: ``fn(input) -> list of outputs``."""

    def __init__(self, fn, in_shape, out_shapes):
        self.fn, self.in_shape, self.out_shapes = fn, in_shape, out_shapes
        self.inputs = []

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=list(self.in_shape))]

    def get_outputs(self):
        return [SimpleNamespace(name=f"out{i}", shape=list(s)) for i, s in enumerate(self.out_shapes)]

    def run(self, names, feed):
        x = feed["input"]
        self.inputs.append(x)
        return self.fn(x)


def _tool(cls, session, **attrs):
    """An rtmlib model object without loading an ONNX file."""
    obj = cls.__new__(cls)
    obj.onnx_model, obj.backend, obj.device, obj.session = "fake.onnx", "onnxruntime", "cpu", session
    obj.to_openpose = False
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _dot_centroid(channel, thr):
    """Centroid (x, y) of the pixels of ``channel`` above ``thr``."""
    ys, xs = np.nonzero(channel > thr)
    wts = channel[ys, xs] - thr
    return np.array([np.sum(xs * wts), np.sum(ys * wts)]) / np.sum(wts)


def _dots_image(h, w, dots):
    img = np.zeros((h, w, 3), np.uint8)
    for x, y in dots:
        cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), BLUE, -1)
    return img


# RGB-normalized blue channel: dot (255 - 103.53) / 57.375, background -103.53 / 57.375
BLUE_THR = 0.0


@needs_rtmlib
def test_rtmlib_rtmpose_crops_map_back_to_image_pixels():
    import rtmlib

    K = 17
    raw = np.linspace(0.05, 1.3, K)
    raw[4] = 0.0  # not found

    def fn(x):
        c = _dot_centroid(x[0, 2], BLUE_THR)  # blue in channel 2 = RGB input
        sx, sy = np.zeros((1, K, 192 * 2), np.float32), np.zeros((1, K, 256 * 2), np.float32)
        bx, by = int(round(c[0] * 2)), int(round(c[1] * 2))
        sx[0, :, bx] = raw
        sy[0, :, by] = raw
        return [sx, sy]

    sess = FakeSession(fn, (1, 3, 256, 192), [(1, K, 384), (1, K, 512)])
    pose = _tool(rtmlib.RTMPose, sess, model_input_size=(192, 256),
                 mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
    dots = [(97, 143), (512, 260)]
    boxes = np.array([[40, 60, 160, 300], [430, 150, 600, 330]], float)
    d = rd.RTMLibDetector("rtmpose", pose, StubDet(boxes))
    people = d.detect(_dots_image(400, 700, dots), 0.0, "cam0")
    assert len(people) == 2 and len(sess.inputs) == 2
    for p, dot in zip(people, dots):
        ok = raw > 0
        err = np.linalg.norm(p.keypoints[ok] - dot, axis=1)
        assert err.max() < 1.5, err
        assert np.isnan(p.keypoints[4]).all() and p.scores[4] == 0
        assert np.allclose(p.scores[ok], np.clip(raw[ok], 0, 1), atol=1e-6)


@needs_rtmlib
def test_rtmlib_rtmpose3d_crops_and_depth():
    import rtmlib

    K = 133
    zbin = np.linspace(40, 520, K).round()  # SimCC z bins (0..576)

    def fn(x):
        c = _dot_centroid(x[0, 2], BLUE_THR)
        sx = np.zeros((1, K, 576), np.float32)
        sy = np.zeros((1, K, 768), np.float32)
        sz = np.zeros((1, K, 576), np.float32)
        sx[0, :, int(round(c[0] * 2))] = 0.8
        sy[0, :, int(round(c[1] * 2))] = 0.9
        sz[0, np.arange(K), zbin.astype(int)] = 1.0
        return [sx, sy, sz]

    sess = FakeSession(fn, (1, 3, 384, 288), [(1, K, 576), (1, K, 768), (1, K, 576)])
    pose = _tool(rtmlib.RTMPose3d, sess, model_input_size=(288, 384), z_range=2.1744869,
                 mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
    d = rd.RTMPose3DDetector("rtmpose3d", pose, StubDet(np.array([[300, 100, 520, 460]], float)))
    assert d.z_bins == 288  # from the ONNX output length
    (p,) = d.detect(_dots_image(500, 800, [(401, 222)]), 0.0, "c")
    assert np.linalg.norm(p.keypoints - [401, 222], axis=1).max() < 1.5
    assert np.allclose(p.scores, 0.8)  # rtmlib: min of the x and y maxima
    assert np.allclose(d.depth_m(zbin / 2), (zbin / 2 / 144 - 1) * 2.1744869)
    assert p.keypoints_3d is None  # every point at one pixel: no scale


@needs_rtmlib
def test_rtmlib_vitpose_heatmaps_map_back_to_image_pixels():
    import rtmlib

    K = 17

    def fn(x):
        c = _dot_centroid(x[0, 2], BLUE_THR)  # crop pixels (192 x 256 input)
        hx, hy = c[0] * 47 / 192, c[1] * 63 / 256  # heatmap 48 x 64 (rtmlib: / (W - 1))
        yy, xx = np.mgrid[0:64, 0:48]
        blob = np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * 2.0 ** 2)).astype(np.float32)
        return [np.repeat((blob * 0.9)[None, None], K, axis=1)]

    sess = FakeSession(fn, (1, 3, 256, 192), [(1, K, 64, 48)])
    pose = _tool(rtmlib.ViTPose, sess, model_input_size=(192, 256),
                 mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
    d = rd.RTMLibDetector("vitpose_onnx", pose, StubDet(np.array([[100, 80, 260, 400]], float)))
    (p,) = d.detect(_dots_image(480, 640, [(171, 255)]), 0.0, "c")
    assert np.linalg.norm(p.keypoints - [171, 255], axis=1).max() < 3.0
    assert np.allclose(p.scores, 0.9, atol=0.05)


@needs_rtmlib
def test_rtmlib_rtmo_letterbox_maps_back_to_image_pixels():
    import rtmlib

    K = 17

    def fn(x):
        c = _dot_centroid(x[0, 0], 128.0)  # BGR, unnormalized: blue in channel 0
        det = np.array([[[c[0] - 40, c[1] - 90, c[0] + 40, c[1] + 90, 0.9],
                         [10, 10, 50, 50, 0.2]]], np.float32)  # the 2nd is below score_thr
        kp = np.zeros((1, 2, K, 3), np.float32)
        kp[0, :, :, :2] = c
        kp[0, :, :, 2] = 0.85
        return [det, kp]

    sess = FakeSession(fn, (1, 3, 640, 640), [(1, 2, 5), (1, 2, K, 3)])
    pose = _tool(rtmlib.RTMO, sess, model_input_size=(640, 640), mean=None, std=None,
                 nms_thr=0.45, score_thr=0.7)
    d = rd.RTMLibDetector("rtmo", pose)
    (p,) = d.detect(_dots_image(600, 1000, [(733, 412)]), 0.0, "c")
    assert np.linalg.norm(p.keypoints - [733, 412], axis=1).max() < 2.5
    assert np.allclose(p.scores, 0.85)
    sess.fn = lambda x: [np.array([[[1, 1, 5, 5, 0.1]]], np.float32),
                         np.zeros((1, 1, K, 3), np.float32)]
    assert d.detect(_dots_image(600, 1000, [(733, 412)]), 0.0, "c") == []


# ------------------------------------------------------------------ real YOLOX (GitHub)
YOLOX_TINY_URL = ("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
                  "yolox_tiny.onnx")


@needs_rtmlib
def test_real_yolox_person_detector_on_the_photo(person_image):
    import rtmlib

    path = download_cached(YOLOX_TINY_URL, "yolox_tiny.onnx", timeout=120)
    det = rtmlib.YOLOX(str(path), model_input_size=(416, 416), det_mode="multiclass",
                       backend="onnxruntime", device="cpu")
    pose = StubPose(lambda b: (_grid_keypoints(b, 17), np.ones(17)))
    d = rd.RTMLibDetector("rtmpose", pose, det)
    people = d.detect(person_image, 0.0, "cam0")
    assert len(people) == 1
    x1, y1, x2, y2 = people[0].bbox
    # the person spans about x 250-760, y 235-650 (MediaPipe's keypoints lie inside)
    assert x1 < 300 and x2 > 700 and y1 < 280 and y2 > 620
    assert pose.calls[0][0][0, 0].tolist() == person_image[0, 0, ::-1].tolist()  # RGB
    assert d.detect(np.zeros_like(person_image), 0.0, "cam0") == []


# ------------------------------------------------------------------ real models (smoke)
def _smoke_keys():
    sel = os.environ.get("POSEBOARD_RTMLIB_SMOKE", "all").strip().lower()
    if sel in ("", "all"):
        return set(RTMLIB_KEYS)
    return {k.strip() for k in sel.split(",")} & set(RTMLIB_KEYS)


@pytest.fixture(scope="module")
def mediapipe_reference(person_image_path):
    """MediaPipe's shoulders/hips/knees/ankles on the photo (None if MediaPipe is unusable)."""
    if not _has("mediapipe"):
        return None
    img = cv2.imread(str(person_image_path))
    try:
        det = create_detector("mediapipe", model="full")
    except Exception:  # noqa: BLE001  (model download failed, ...)
        return None
    try:
        people = det.detect(img, 0.0, "ref")
    finally:
        det.close()
    if len(people) != 1:
        return None
    p, fmt = people[0], FORMATS["mediapipe33"]
    return {n: p.keypoints[fmt.index(n)] for n in JOINTS}


@needs_rtmlib
@pytest.mark.parametrize("key", RTMLIB_KEYS)
def test_real_model_on_the_photo(key, person_image, mediapipe_reference):
    if key not in _smoke_keys():
        pytest.skip(f"{key} not selected by POSEBOARD_RTMLIB_SMOKE")
    ok, why = backend_available(key)
    if not ok:
        pytest.skip(why)
    mode = "balanced" if key == "rtmpose3d" else os.environ.get(
        "POSEBOARD_RTMLIB_SMOKE_MODE", "lightweight")
    try:
        det = create_detector(key, mode=mode, device="cpu")
    except rd.ModelUnavailable as e:
        pytest.skip(f"{key}: the model weights cannot be downloaded here: {e}")
    try:
        people = det.detect(person_image, 0.0, "cam0")
        fmt = det.format
    finally:
        det.close()
    h, w = person_image.shape[:2]
    assert len(people) == 1, [p.bbox for p in people]
    p = people[0]
    assert p.keypoints.shape == (len(fmt), 2) and p.scores.shape == (len(fmt),)
    assert np.all((p.scores >= 0) & (p.scores <= 1)) and 0 < p.score <= 1
    xy = {n: p.keypoints[fmt.index(n)] for n in JOINTS}
    sc = {n: p.scores[fmt.index(n)] for n in JOINTS}
    assert min(sc.values()) > 0.3, sc
    # the subject faces the camera: his left shoulder is on the image right
    assert xy["left_shoulder"][0] > xy["right_shoulder"][0]
    for side in ("left", "right"):
        ys = [xy[f"{side}_{j}"][1] for j in ("shoulder", "hip", "knee", "ankle")]
        assert ys == sorted(ys), (side, ys)
    if key == "rtmpose3d":
        k3 = p.keypoints_3d
        body = [fmt.index(n) for n in JOINTS]
        assert k3 is not None and np.all(np.isfinite(k3[body]))
        assert 0.2 < np.linalg.norm(k3[fmt.index("left_shoulder")]
                                    - k3[fmt.index("right_shoulder")]) < 0.6
        assert k3[fmt.index("left_shoulder"), 0] > k3[fmt.index("right_shoulder"), 0]
        assert k3[fmt.index("left_ankle"), 1] > k3[fmt.index("left_hip"), 1]  # y down
        # placed with one (approximately calibrated) camera like MultiViewEstimator does
        cam = approximate_calibration("cam0", w, h)
        pts = single_view_lift(cam, p, k3, 0.3)
        assert pts is not None and np.all(pts[body, 2] > 0)
        reproj = np.linalg.norm(project_to_image(cam, pts[body]) - p.keypoints[body], axis=1)
        assert reproj.mean() < 25, reproj
    if mediapipe_reference is None:
        pytest.skip("MediaPipe reference unavailable: the comparison with MediaPipe was not run")
    tol = 0.08 * np.hypot(w, h)
    dist = {n: float(np.linalg.norm(xy[n] - mediapipe_reference[n])) for n in JOINTS}
    assert max(dist.values()) < tol, dist
