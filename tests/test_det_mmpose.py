"""The ``mmpose`` backend (MMPose 2D models through ``MMPoseInferencer``).

MMPose is not a PoseBoard dependency. The first part runs everywhere with ``mmpose.apis``
replaced by a stub inferencer that returns results in MMPose 1.x's format (keypoint layout,
name mapping, scores, boxes, empty inputs). The second part runs a real MMPose model on a real
image when MMPose is installed and its checkpoints can be downloaded; otherwise it is skipped
with the reason.
"""

from __future__ import annotations

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
from poseboard.pose.detectors import BACKENDS, backend_available, create_detector
from poseboard.pose.detectors import mmpose_det as md
from poseboard.pose.formats import COCO17, HALPE26, WHOLEBODY133
from poseboard.pose.multiview import MultiViewEstimator

ROOT = Path(__file__).resolve().parents[1]

# Keypoint names of MMPose's dataset meta files (v1.3.2), transcribed from
# configs/_base_/datasets/coco.py, halpe26.py and coco_wholebody.py (keypoint_info, by id).
MMPOSE_COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
               "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
               "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
MMPOSE_HALPE26 = MMPOSE_COCO + ["head", "neck", "hip", "left_big_toe", "right_big_toe",
                                "left_small_toe", "right_small_toe", "left_heel", "right_heel"]
_HAND = ["hand_root", "thumb1", "thumb2", "thumb3", "thumb4", "forefinger1", "forefinger2",
         "forefinger3", "forefinger4", "middle_finger1", "middle_finger2", "middle_finger3",
         "middle_finger4", "ring_finger1", "ring_finger2", "ring_finger3", "ring_finger4",
         "pinky_finger1", "pinky_finger2", "pinky_finger3", "pinky_finger4"]
MMPOSE_WHOLEBODY = (MMPOSE_COCO + ["left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
                                   "right_small_toe", "right_heel"]
                    + [f"face-{i}" for i in range(68)] + ["left_" + n for n in _HAND]
                    + ["right_" + n for n in _HAND])


def _meta(names):
    return {"dataset_name": "test", "num_keypoints": len(names),
            "keypoint_id2name": dict(enumerate(names)),
            "keypoint_name2id": {n: i for i, n in enumerate(names)}}


def test_mmpose_names_map_to_the_registered_formats():
    assert [md.canonical_name(n) for n in MMPOSE_COCO] == list(COCO17.names)
    assert [md.canonical_name(n) for n in MMPOSE_HALPE26] == list(HALPE26.names)
    assert len(MMPOSE_WHOLEBODY) == 133
    assert [md.canonical_name(n) for n in MMPOSE_WHOLEBODY] == list(WHOLEBODY133.names)
    assert md.canonical_name("right_forefinger2") == "right_hand_6"
    assert md.canonical_name("left_pinky_finger4") == "left_hand_20"


def test_module_import_loads_no_model_library():
    code = ("import sys, poseboard.pose.detectors.mmpose_det; "
            "print(','.join(m for m in ('torch', 'mmpose', 'mmcv', 'mmengine', 'onnxruntime') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_format_and_order():
    assert md.format_for(17) is COCO17 and md.format_for(26) is HALPE26
    assert md.format_for(133) is WHOLEBODY133
    assert md.format_for(133, MMPOSE_WHOLEBODY) is WHOLEBODY133
    with pytest.raises(ValueError, match="21 keypoints"):
        md.format_for(21)
    with pytest.raises(ValueError, match="has 17 keypoints"):
        md.format_for(26, keypoint_format="coco17")
    assert md.format_for(17, keypoint_format="coco17") is COCO17
    assert md.keypoint_order(COCO17, MMPOSE_COCO) is None
    assert md.keypoint_order(WHOLEBODY133, MMPOSE_WHOLEBODY) is None
    swapped = MMPOSE_COCO[:5] + [MMPOSE_COCO[6], MMPOSE_COCO[5]] + MMPOSE_COCO[7:]
    order = md.keypoint_order(COCO17, swapped)
    assert order[5] == 6 and order[6] == 5 and order[:5] == [0, 1, 2, 3, 4]
    assert md.keypoint_order(COCO17, [f"kp{i}" for i in range(17)]) is None  # unknown names


# ------------------------------------------------------------------ mmpose stub
@pytest.fixture
def fake_mmpose(monkeypatch):
    """Replace mmpose / mmengine / mmcv by stubs (also where they are installed)."""
    state = SimpleNamespace(meta=_meta(MMPOSE_COCO), calls=[], init=[], init_error=None,
                            instances=lambda img: [])

    class FakeInferencer:
        def __init__(self, pose2d=None, pose2d_weights=None, pose3d=None, pose3d_weights=None,
                     device=None, scope="mmpose", det_model=None, det_weights=None,
                     det_cat_ids=None, show_progress=False):
            state.init.append({"pose2d": pose2d, "pose2d_weights": pose2d_weights,
                               "device": device, "det_model": det_model,
                               "det_weights": det_weights, "det_cat_ids": det_cat_ids})
            if state.init_error is not None:
                raise state.init_error
            model = SimpleNamespace(dataset_meta=state.meta)
            self.inferencer = SimpleNamespace(model=model)

        def __call__(self, inputs, return_datasamples=False, batch_size=1, out_dir=None, **kw):
            state.calls.append((np.array(inputs), kw))

            def gen():
                yield {"visualization": [], "predictions": [state.instances(inputs)]}

            return gen()

    def module(name, is_pkg=False, **attrs):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=is_pkg)
        if is_pkg:
            m.__path__ = []
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    module("mmpose", True, __version__="1.3.2-fake")
    module("mmpose.apis", MMPoseInferencer=FakeInferencer)
    module("mmengine", True)
    module("mmcv", True)
    monkeypatch.setattr(md, "_cuda_available", lambda: False)
    return state


def inst(kp, scores, bbox=None, bbox_score=None):
    """One instance as MMPose's split_instances makes it (bbox is a 1-tuple)."""
    d = {"keypoints": np.asarray(kp, float).tolist(), "keypoint_scores": list(scores)}
    if bbox is not None:
        d["bbox"] = (list(bbox),)
    if bbox_score is not None:
        d["bbox_score"] = np.float32(bbox_score)
    return d


def _kp(k, x0=100.0, y0=50.0):
    j = np.arange(k, dtype=float)
    return np.stack([x0 + 7.0 * j, y0 + 11.0 * j], axis=1)


def test_detect_maps_all_people(fake_mmpose):
    a, b = _kp(17), _kp(17, 400, 80)
    sa, sb = np.linspace(0.3, 0.95, 17), np.full(17, 0.6)
    fake_mmpose.instances = lambda img: [inst(a, sa, (90, 40, 220, 240), 0.91),
                                         inst(b, sb, (390, 70, 520, 270), 0.55)]
    det = create_detector("mmpose")
    assert det.format is COCO17 and det.key == "mmpose" and not det.provides_3d
    assert fake_mmpose.init[-1]["pose2d"] == md.DEFAULT_MODEL
    assert fake_mmpose.init[-1]["device"] == "cpu"
    img = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    people = det.detect(img, 0.0, "cam0")
    assert len(people) == 2
    np.testing.assert_allclose(people[0].keypoints, a)
    np.testing.assert_allclose(people[0].scores, sa)
    np.testing.assert_allclose(people[0].bbox, (90, 40, 220, 240))
    assert people[0].score == pytest.approx(0.91) and people[1].score == pytest.approx(0.55)
    np.testing.assert_allclose(people[1].keypoints, b)
    passed, kw = fake_mmpose.calls[-1]
    np.testing.assert_array_equal(passed, img)  # BGR array, unchanged (MMPose convention)
    assert kw["return_vis"] is False and kw["show"] is False
    assert kw["bbox_thr"] == 0.3 and kw["nms_thr"] == 0.3


def test_keypoints_are_mapped_by_name(fake_mmpose):
    names = list(MMPOSE_COCO)
    names[5], names[6] = names[6], names[5]  # a model listing right_shoulder first
    fake_mmpose.meta = _meta(names)
    raw = _kp(17)
    fake_mmpose.instances = lambda img: [inst(raw, np.full(17, 0.8))]
    det = create_detector("mmpose")
    (p,) = det.detect(np.zeros((10, 10, 3), np.uint8), 0.0, "cam0")
    np.testing.assert_allclose(p.keypoints[COCO17.index("left_shoulder")], raw[6])
    np.testing.assert_allclose(p.keypoints[COCO17.index("right_shoulder")], raw[5])
    np.testing.assert_allclose(p.keypoints[:5], raw[:5])


@pytest.mark.parametrize("names,fmt", [(MMPOSE_HALPE26, HALPE26),
                                       (MMPOSE_WHOLEBODY, WHOLEBODY133)])
def test_halpe_and_wholebody_models(fake_mmpose, names, fmt):
    fake_mmpose.meta = _meta(names)
    raw = _kp(len(names))
    fake_mmpose.instances = lambda img: [inst(raw, np.full(len(names), 0.7), (0, 0, 9, 9), 0.9)]
    det = create_detector("mmpose", model="wholebody")
    assert det.format is fmt and det.options["keypoint_format"] == fmt.key
    (p,) = det.detect(np.zeros((10, 10, 3), np.uint8), 0.0, "cam0")
    np.testing.assert_allclose(p.keypoints, raw)
    if fmt is WHOLEBODY133:
        i = MMPOSE_WHOLEBODY.index("left_middle_finger2")
        assert fmt.names[i] == "left_hand_10"


def test_layout_from_the_first_result_without_metadata(fake_mmpose, caplog):
    fake_mmpose.meta = None
    raw = _kp(26)
    fake_mmpose.instances = lambda img: [inst(raw, np.full(26, 0.7))]
    det = create_detector("mmpose")
    assert det.format is COCO17  # unknown until the first result
    (p,) = det.detect(np.zeros((10, 10, 3), np.uint8), 0.0, "cam0")
    assert det.format is HALPE26 and p.keypoints.shape == (26, 2)
    assert "26 keypoints" in caplog.text
    forced = create_detector("mmpose", keypoint_format="halpe26")
    assert forced.format is HALPE26


def test_scores_boxes_and_missing_keypoints(fake_mmpose):
    raw = _kp(17)
    raw[3] = np.nan
    sc = np.full(17, 0.5)
    sc[0], sc[1], sc[2] = 1.3, 0.0, np.nan
    fake_mmpose.instances = lambda img: [inst(raw, sc)]
    det = create_detector("mmpose")
    (p,) = det.detect(np.zeros((10, 10, 3), np.uint8), 0.0, "cam0")
    assert p.scores[0] == 1.0
    for i in (1, 2, 3):
        assert np.isnan(p.keypoints[i]).all() and p.scores[i] == 0
    assert p.score == pytest.approx(np.mean([1.0] + [0.5] * 13))  # mean of the found ones
    ok = np.isfinite(p.keypoints[:, 0])
    np.testing.assert_allclose(p.bbox, [raw[ok, 0].min(), raw[ok, 1].min(), raw[ok, 0].max(),
                                        raw[ok, 1].max()])
    half = create_detector("mmpose", score_scale=2.0)
    (q,) = half.detect(np.zeros((10, 10, 3), np.uint8), 0.0, "cam0")
    assert q.scores[0] == pytest.approx(0.65) and q.scores[5] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="score_scale"):
        create_detector("mmpose", score_scale=0)


def test_empty_inputs_and_results(fake_mmpose):
    det = create_detector("mmpose")
    assert det.detect(np.zeros((0, 0, 3), np.uint8), 0.0, "cam0") == []
    assert det.detect(None, 0.0, "cam0") == []
    assert fake_mmpose.calls == []
    assert det.detect(np.zeros((10, 10, 3), np.uint8), 0.0, "cam0") == []
    fake_mmpose.instances = lambda img: [{"keypoints": [], "keypoint_scores": []}]
    assert det.detect(np.zeros((10, 10, 3), np.uint8), 0.0, "cam0") == []
    fake_mmpose.instances = lambda img: [inst(_kp(17), np.full(17, 0.9))]
    assert len(det.detect(np.zeros((10, 10), np.uint8), 0.0, "cam0")) == 1  # grayscale
    assert fake_mmpose.calls[-1][0].shape == (10, 10, 3)
    fake_mmpose.instances = lambda img: [inst(_kp(26), np.full(26, 0.9))]
    with pytest.raises(ValueError, match="26 keypoints"):
        det.detect(np.zeros((10, 10, 3), np.uint8), 0.0, "cam0")
    det.close()
    with pytest.raises(RuntimeError, match="closed"):
        det.detect(np.zeros((10, 10, 3), np.uint8), 0.0, "cam0")


def test_unsupported_models_and_load_errors(fake_mmpose):
    fake_mmpose.meta = _meta([f"hand_{i}" for i in range(21)])
    with pytest.raises(ValueError, match="21 keypoints"):
        create_detector("mmpose", model="hand")
    fake_mmpose.meta = _meta(MMPOSE_COCO)
    fake_mmpose.init_error = ConnectionError("download.openmmlab.com unreachable")
    with pytest.raises(RuntimeError, match="MMPose could not load model 'human'") as e:
        create_detector("mmpose", model="human")
    assert "weights" in str(e.value)


def test_factory_options(fake_mmpose, monkeypatch):
    det = md.create(pose2d="rtmpose-l", pose2d_weights="w.pth", det_model="rtmdet-m",
                    det_weights="d.pth", det_cat_ids=[0], device="cpu", bbox_thr=0.5)
    assert fake_mmpose.init[-1] == {"pose2d": "rtmpose-l", "pose2d_weights": "w.pth",
                                    "device": "cpu", "det_model": "rtmdet-m",
                                    "det_weights": "d.pth", "det_cat_ids": [0]}
    assert det.options["model"] == "rtmpose-l" and det.bbox_thr == 0.5
    info = det.info()
    assert info["backend"] == "mmpose" and info["device_used"] == "cpu"
    assert info["pose2sim_model"] == "COCO_17"
    monkeypatch.setattr(md, "_cuda_available", lambda: True)
    create_detector("mmpose", device="auto")
    assert fake_mmpose.init[-1]["device"] == "cuda:0"


def test_device_resolution(monkeypatch):
    monkeypatch.setattr(md, "_cuda_available", lambda: False)
    assert md.resolve_device("auto") == "cpu" and md.resolve_device(None) == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        md.resolve_device("cuda")
    monkeypatch.setattr(md, "_cuda_available", lambda: True)
    assert md.resolve_device("auto") == "cuda:0" and md.resolve_device("cuda:1") == "cuda:1"
    assert md.resolve_device("CPU") == "cpu"


def test_registry_and_pipeline(fake_mmpose):
    ok, why = backend_available("mmpose")
    assert ok, why
    assert set(BACKENDS["mmpose"].defaults) == {"model", "device"}
    raw = _kp(17)
    fake_mmpose.instances = lambda img: [inst(raw, np.full(17, 0.9), (90, 40, 220, 240), 0.9)]
    est = MultiViewEstimator(create_detector("mmpose"))
    cams = {"cam0": approximate_calibration("cam0", 640, 480)}
    pose = est.process({"cam0": (1.0, np.zeros((480, 640, 3), np.uint8))}, cams)
    assert pose.mode == MODE_2D_ONLY and pose.format_key == "coco17"
    np.testing.assert_allclose(pose.per_camera_2d["cam0"].keypoints, raw)
    est.close()


# ------------------------------------------------------------------ real MMPose
@pytest.fixture(scope="module")
def mmpose_real():
    ok, why = backend_available("mmpose")
    if not ok:
        pytest.skip(why)
    try:
        det = create_detector("mmpose", device="cpu")
    except RuntimeError as e:
        if "MMPose could not load model" in str(e):
            pytest.skip(f"MMPose model not available: {e}")
        raise
    yield det
    det.close()


@pytest.fixture(scope="module")
def mediapipe_points(person_image_path):
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


def test_real_mmpose_on_person_image(mmpose_real, person_image, mediapipe_points):
    h, w = person_image.shape[:2]
    fmt = mmpose_real.format
    people = [p for p in mmpose_real.detect(person_image, 0.0, "cam0") if p.score > 0.5]
    assert len(people) == 1
    p = people[0]
    assert np.all((p.scores >= 0) & (p.scores <= 1))
    kp = {n: p.keypoints[fmt.index(n)] for n in fmt.names}
    for side in ("left", "right"):
        ys = [kp[f"{side}_{j}"][1] for j in LIMBS]
        assert ys == sorted(ys), (side, ys)
    assert kp["left_shoulder"][0] > kp["right_shoulder"][0]
    mp = mediapipe_points
    tol = 0.08 * np.hypot(h, w)
    for side in ("left", "right"):
        for j in LIMBS:
            name = f"{side}_{j}"
            d = np.linalg.norm(kp[name] - mp[name])
            assert d < tol, f"{name}: {d:.1f} px from MediaPipe (limit {tol:.1f})"
