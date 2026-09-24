"""The ``keypoint_rcnn`` backend (torchvision Keypoint R-CNN ResNet-50-FPN).

Part 1 runs everywhere. It uses plain numpy arrays for the output mapping: keypoint order,
sigmoid scores, the person filter and empty inputs.

Part 2 needs torch and torchvision. It uses randomly initialised models, since the real weights
are not needed:

* ``forward`` is replaced to check the image passed in and the mapping of the outputs.
* The region proposal network and the heads are replaced so that the real torchvision
  postprocess rescales known coordinates back to the original image.
* The architecture is checked against the official pretrained builder.
* The weight files are resolved, loaded and their errors reported.

Part 3 runs the real pretrained model on a real image when torch/torchvision are installed and
the weights can be obtained (download.pytorch.org, or a local copy). Otherwise it is skipped
with the reason.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from poseboard.pose.detectors import torchvision_det as td
from poseboard.pose.formats import COCO17

ROOT = Path(__file__).resolve().parents[1]
N_KP = 17
LS, RS = COCO17.index("left_shoulder"), COCO17.index("right_shoulder")


def _output(n=1, seed=0, **over):
    """A torchvision-like inference result (numpy) with ``n`` persons."""
    rng = np.random.default_rng(seed)
    kps = np.zeros((n, N_KP, 3))
    kps[:, :, 0] = rng.uniform(0, 640, (n, N_KP))
    kps[:, :, 1] = rng.uniform(0, 480, (n, N_KP))
    kps[:, :, 2] = 1.0
    out = {"boxes": np.tile([10.0, 20.0, 300.0, 400.0], (n, 1)) + np.arange(n)[:, None],
           "labels": np.ones(n, np.int64),
           "scores": np.linspace(0.99, 0.6, n) if n else np.zeros(0),
           "keypoints": kps,
           "keypoints_scores": rng.normal(3.0, 4.0, (n, N_KP))}
    out.update(over)
    return out


# ================================================================== part 1: no torch needed
def test_importing_the_module_imports_no_torch():
    code = ("import sys; import poseboard.pose.detectors.torchvision_det; "
            "print(','.join(m for m in ('torch', 'torchvision') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert "torch" not in out.stdout


def test_keypoint_scores_are_the_sigmoid_of_the_logits():
    x = np.array([-20.0, -2.0, -0.85, 0.0, 2.0, 20.0])
    np.testing.assert_allclose(td.keypoint_scores(x), 1.0 / (1.0 + np.exp(-x)), atol=1e-12)
    np.testing.assert_allclose(td.keypoint_scores([0.0, 2.0, -2.0]), [0.5, 0.8808, 0.1192],
                               atol=1e-4)
    with np.errstate(all="raise"):  # no overflow for huge logits
        s = td.keypoint_scores([-1e6, 1e6, np.inf, -np.inf, np.nan])
    np.testing.assert_array_equal(s, [0.0, 1.0, 1.0, 0.0, 0.0])
    s = td.keypoint_scores(np.linspace(-50, 50, 101))
    assert np.all((s >= 0) & (s <= 1)) and np.all(np.diff(s) >= 0)


def test_keypoint_order_is_coco17_and_mapped_by_name():
    assert td.TORCHVISION_KEYPOINT_NAMES == COCO17.names
    np.testing.assert_array_equal(td.keypoint_index_map(td.TORCHVISION_KEYPOINT_NAMES),
                                  np.arange(N_KP))
    rev = td.TORCHVISION_KEYPOINT_NAMES[::-1]
    np.testing.assert_array_equal(td.keypoint_index_map(rev), np.arange(N_KP)[::-1])
    m = td.keypoint_index_map(("nose", "left_eye"))
    assert m[0] == 0 and m[1] == 1 and np.all(m[2:] == -1)


def test_people_from_output_maps_keypoints_scores_and_boxes():
    out = _output(2)
    people = td.people_from_output(out, score_threshold=0.5)
    assert len(people) == 2
    for i, p in enumerate(people):
        assert p.keypoints.shape == (N_KP, 2) and p.scores.shape == (N_KP,)
        np.testing.assert_allclose(p.keypoints, out["keypoints"][i, :, :2])
        np.testing.assert_allclose(p.scores, 1 / (1 + np.exp(-out["keypoints_scores"][i])))
        np.testing.assert_allclose(p.bbox, out["boxes"][i])
        assert p.score == pytest.approx(out["scores"][i]) and p.keypoints_3d is None
    assert people[0].score > people[1].score


def test_people_from_output_follows_the_model_names_not_the_position():
    """A model whose keypoints come in another order (here: left and right swapped) must still
    give the COCO-17 order with the person's left in the left_* slots."""
    swapped = tuple(n.replace("left_", "TMP_").replace("right_", "left_").replace("TMP_", "right_")
                    for n in td.TORCHVISION_KEYPOINT_NAMES)
    out = _output(1)
    p = td.people_from_output(out, names=swapped)[0]
    src = out["keypoints"][0, :, :2]
    np.testing.assert_allclose(p.keypoints[LS], src[RS])  # model slot "left" is at index 6
    np.testing.assert_allclose(p.keypoints[RS], src[LS])
    np.testing.assert_allclose(p.keypoints[0], src[0])  # nose unchanged
    # a model without some keypoints: NaN / score 0 there
    p = td.people_from_output({"keypoints": out["keypoints"][:, :5],
                               "keypoints_scores": out["keypoints_scores"][:, :5]},
                              names=td.TORCHVISION_KEYPOINT_NAMES[:5])[0]
    assert np.all(np.isfinite(p.keypoints[:5])) and np.all(np.isnan(p.keypoints[5:]))
    assert np.all(p.scores[5:] == 0) and p.bbox is None and p.score == 1.0
    with pytest.raises(ValueError, match="keypoints"):
        td.people_from_output({"keypoints": out["keypoints"][:, :5]})


def test_people_from_output_filters_and_sorts():
    out = _output(4, scores=np.array([0.3, 0.97, 0.8, 0.95]),
                  labels=np.array([1, 1, 1, 2]))  # label 2 is not a person
    out["keypoints"][2, 3, :2] = np.nan  # a broken keypoint
    out["keypoints_scores"][2, 4] = np.nan  # a broken score
    out["boxes"][1, 0] = np.nan  # a broken box
    people = td.people_from_output(out, score_threshold=0.5)
    assert [round(p.score, 2) for p in people] == [0.97, 0.8]
    assert people[0].bbox is None and people[0].box() is not None
    p = people[1]
    assert np.all(np.isnan(p.keypoints[3])) and p.scores[3] == 0
    assert np.all(np.isfinite(p.keypoints[4])) and p.scores[4] == 0
    assert len(td.people_from_output(out, score_threshold=0.0)) == 3
    assert len(td.people_from_output(out, score_threshold=0.0, person_label=None)) == 4
    # without keypoint scores every point counts as seen
    del out["keypoints_scores"]
    assert np.all(td.people_from_output(out, score_threshold=0.9)[0].scores == 1.0)


def test_people_from_output_empty():
    assert td.people_from_output({}) == []
    assert td.people_from_output(None) == []
    assert td.people_from_output(_output(0)) == []
    assert td.people_from_output({"boxes": np.zeros((0, 4)),
                                  "keypoints": np.zeros((0, N_KP, 3))}) == []


# ================================================================== part 2: torch, random weights
@pytest.fixture(scope="module")
def tv():
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models.detection import keypointrcnn_resnet50_fpn

    torch.manual_seed(0)
    return torch, torchvision, keypointrcnn_resnet50_fpn


@pytest.fixture(scope="module")
def random_model(tv):
    """An untrained Keypoint R-CNN (weights=None, weights_backbone=None: nothing is
    downloaded), small input size so it runs fast on a CPU."""
    _torch, _tvis, build = tv
    return build(weights=None, weights_backbone=None, min_size=400, max_size=1000).eval()


@pytest.fixture(scope="module")
def checkpoint(tv, random_model, tmp_path_factory):
    """The untrained model's weights saved like the official file (FrozenBatchNorm2d has no
    num_batches_tracked buffers), under the official file name."""
    torch = tv[0]
    sd = {k: v.clone() for k, v in random_model.state_dict().items()
          if not k.endswith("num_batches_tracked")}
    path = tmp_path_factory.mktemp("kprcnn") / td.WEIGHTS_FILE
    torch.save(sd, path)
    return path, sd


@pytest.fixture
def fake_detector(tv, random_model, monkeypatch):
    """A ``KeypointRCNNDetector`` on the untrained model (``load_model`` stubbed)."""
    monkeypatch.setattr(td, "load_model", lambda *a, **k: (
        random_model, td.TORCHVISION_KEYPOINT_NAMES, Path("untrained")))
    det = td.create(device="cpu", score_threshold=0.5)
    yield det
    det.close()


def _torch_output(torch, out: dict) -> dict:
    types = {"labels": torch.int64}
    return {k: torch.as_tensor(np.asarray(v), dtype=types.get(k, torch.float32))
            for k, v in out.items()}


def test_detector_passes_an_rgb_float_image_and_maps_the_output(tv, fake_detector,
                                                                monkeypatch):
    torch = tv[0]
    det = fake_detector
    assert det.key == "keypoint_rcnn" and det.format is COCO17 and not det.provides_3d
    assert det.device == "cpu" and det.options["score_threshold"] == 0.5
    info = det.info()
    assert info["keypoint_format"] == "coco17" and info["pose2sim_model"] == "COCO_17"
    assert "sigmoid" in info["keypoint_score"]

    seen = []
    out = _output(3, scores=np.array([0.9, 0.4, 0.7]))
    out["keypoints"][..., 0] += 0.25  # sub-pixel values must survive

    def fake_forward(images, targets=None):
        seen.append(images)
        return [_torch_output(torch, out)]

    monkeypatch.setattr(det.model, "forward", fake_forward)
    img = np.zeros((48, 64, 3), np.uint8)
    img[..., 0] = 255  # blue in BGR
    img[..., 1] = 51
    people = det.detect(img, 0.0, "cam0")
    (x,) = seen[0]
    assert tuple(x.shape) == (3, 48, 64) and x.dtype == torch.float32
    assert str(x.device) == "cpu"
    # RGB order, 0..1
    assert float(x[0].max()) == 0.0 and float(x[1].mean()) == pytest.approx(0.2)
    assert float(x[2].min()) == 1.0
    assert [round(p.score, 2) for p in people] == [0.9, 0.7]
    np.testing.assert_allclose(people[1].keypoints, out["keypoints"][2, :, :2], atol=1e-4)
    np.testing.assert_allclose(people[1].scores, td.keypoint_scores(out["keypoints_scores"][2]),
                               atol=1e-6)
    np.testing.assert_allclose(people[0].bbox, out["boxes"][0], atol=1e-4)

    # gray and BGRA images work; empty images return [] without running the model
    det.detect(np.full((48, 64), 128, np.uint8), 0.1, "cam0")
    bgra = np.dstack([img, np.full((48, 64), 7, np.uint8)])
    det.detect(bgra, 0.2, "cam0")
    assert len(seen) == 3 and tuple(seen[1][0].shape) == (3, 48, 64)
    assert float(seen[1][0][0].mean()) == pytest.approx(128 / 255)
    assert float(seen[2][0][2].min()) == 1.0  # the alpha channel is dropped
    for empty in (np.zeros((0, 0, 3), np.uint8), np.zeros((1, 64, 3), np.uint8), None):
        assert det.detect(empty, 0.3, "cam0") == []
    assert len(seen) == 3


def test_no_detection_gives_an_empty_list(tv, fake_detector, monkeypatch):
    torch = tv[0]
    monkeypatch.setattr(fake_detector.model, "forward", lambda images, targets=None: [
        _torch_output(torch, _output(0))])
    assert fake_detector.detect(np.zeros((40, 50, 3), np.uint8), 0.0, "c") == []


def test_real_postprocess_maps_keypoints_back_to_the_original_image(tv, fake_detector,
                                                                    monkeypatch):
    """The RPN and the ROI heads are stubbed to give known detections in the RESIZED network
    image. torchvision's own transform (resize to min_size=400) and postprocess must return
    them in original image pixels."""
    torch = tv[0]
    model = fake_detector.model
    seen = {}

    def rpn(images, features, targets=None):
        seen["sizes"] = list(images.image_sizes)
        return [torch.zeros((1, 4))], {}

    kp_net = np.zeros((1, N_KP, 3), np.float32)
    kp_net[0, :, 0] = np.linspace(40, 560, N_KP)
    kp_net[0, :, 1] = np.linspace(360, 20, N_KP)
    kp_net[0, :, 2] = 1
    logits = np.linspace(-4, 12, N_KP, dtype=np.float32)[None]

    def heads(features, proposals, image_shapes, targets=None):
        seen["shapes"] = list(image_shapes)
        return [{"boxes": torch.tensor([[20.0, 10.0, 580.0, 390.0]]),
                 "labels": torch.tensor([1]), "scores": torch.tensor([0.93]),
                 "keypoints": torch.from_numpy(kp_net.copy()),
                 "keypoints_scores": torch.from_numpy(logits.copy())}], {}

    monkeypatch.setattr(model.rpn, "forward", rpn)
    monkeypatch.setattr(model.roi_heads, "forward", heads)
    img = np.zeros((200, 300, 3), np.uint8)  # H=200, W=300 -> resized x2 to 400 x 600
    people = fake_detector.detect(img, 0.0, "cam0")
    assert seen["sizes"] == [(400, 600)] and seen["shapes"] == [(400, 600)]
    assert len(people) == 1
    p = people[0]
    np.testing.assert_allclose(p.keypoints, kp_net[0, :, :2] / 2.0, atol=1e-3)
    np.testing.assert_allclose(p.bbox, [10.0, 5.0, 290.0, 195.0], atol=1e-3)
    np.testing.assert_allclose(p.scores, td.keypoint_scores(logits[0]), atol=1e-6)
    assert p.score == pytest.approx(0.93)


def test_2d_only_pose_with_the_multiview_estimator(tv, fake_detector, monkeypatch):
    """One camera + a 2D-only backend: no 3D, but the 2D skeleton of the camera is kept."""
    from poseboard.calibration import approximate_calibration
    from poseboard.pose.base import MODE_2D_ONLY
    from poseboard.pose.multiview import MultiViewEstimator

    torch = tv[0]
    out = _output(1, keypoints_scores=np.full((1, N_KP), 5.0))
    monkeypatch.setattr(fake_detector.model, "forward",
                        lambda images, targets=None: [_torch_output(torch, out)])
    est = MultiViewEstimator(fake_detector)
    pose = est.process({"cam0": (1.0, np.zeros((480, 640, 3), np.uint8))},
                       {"cam0": approximate_calibration("cam0", 640, 480)})
    assert pose.mode == MODE_2D_ONLY and pose.format_key == "coco17"
    p2 = pose.per_camera_2d["cam0"]
    np.testing.assert_allclose(p2.keypoints, out["keypoints"][0, :, :2], atol=1e-3)
    assert np.all(p2.scores > 0.99)


def test_resolve_device(tv, monkeypatch):
    torch = tv[0]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert td.resolve_device("auto") == "cpu" and td.resolve_device(None) == "cpu"
    assert td.resolve_device("CPU") == "cpu"
    for d in ("cuda", "gpu", "cuda:0"):
        with pytest.raises(RuntimeError, match="CUDA"):
            td.resolve_device(d)
    with pytest.raises(ValueError, match="unknown device"):
        td.resolve_device("bogus")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert td.resolve_device("auto") == "cuda" and td.resolve_device("cuda") == "cuda"
    assert td.resolve_device("cuda:0") == "cuda:0"
    with pytest.raises(RuntimeError, match="no CUDA device"):
        td.resolve_device("cuda:3")
    with pytest.raises(ValueError, match="score_threshold"):
        td.KeypointRCNNDetector(device="cpu", score_threshold=2.0)


def test_architecture_matches_the_official_pretrained_builder(tv, checkpoint, monkeypatch):
    """``load_model`` must build exactly what ``keypointrcnn_resnet50_fpn(weights=DEFAULT)``
    builds: same modules (FrozenBatchNorm2d with eps 0), same state dict keys, same outputs.
    The official builder is fed our weights instead of downloading them."""
    torch, _torchvision, build = tv
    from torchvision.models import _api
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
    from torchvision.ops.misc import FrozenBatchNorm2d

    path, sd = checkpoint
    assert KeypointRCNN_ResNet50_FPN_Weights.DEFAULT.url.endswith("/" + td.WEIGHTS_FILE)
    assert tuple(KeypointRCNN_ResNet50_FPN_Weights.DEFAULT.meta["keypoint_names"]) \
        == td.TORCHVISION_KEYPOINT_NAMES == COCO17.names
    urls = []

    def fake_load(url, *a, **k):
        urls.append(url)
        return dict(sd)  # load_state_dict copies the values; FrozenBN only drops keys

    monkeypatch.setattr(_api, "load_state_dict_from_url", fake_load)
    kw = {"box_score_thresh": 0.0, "min_size": 320, "max_size": 480, "box_detections_per_img": 7}
    official = build(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT, **kw).eval()
    assert urls == [KeypointRCNN_ResNet50_FPN_Weights.DEFAULT.url]
    ours, names, used = td.load_model(path, score_threshold=0.0, min_size=320, max_size=480,
                                      max_people=7)
    assert used == path and names == COCO17.names and not ours.training
    assert [type(m) for m in ours.modules()] == [type(m) for m in official.modules()]
    assert list(ours.state_dict()) == list(official.state_dict())
    eps = {m.eps for m in ours.modules() if isinstance(m, FrozenBatchNorm2d)}
    assert eps == {m.eps for m in official.modules() if isinstance(m, FrozenBatchNorm2d)} == {0.0}
    x = [torch.rand(3, 240, 320, generator=torch.Generator().manual_seed(1))]
    with torch.inference_mode():
        a, b = ours(x)[0], official(x)[0]
    assert len(a["boxes"]) == len(b["boxes"])
    for k in ("boxes", "scores", "labels", "keypoints", "keypoints_scores"):
        assert torch.equal(a[k], b[k]), k


def test_weights_files_are_found_loaded_and_errors_are_clear(tv, checkpoint, tmp_path,
                                                             monkeypatch):
    torch = tv[0]
    from torchvision.ops.misc import FrozenBatchNorm2d

    from poseboard.pose import mediapipe_backend

    path, sd = checkpoint
    hub = tmp_path / "torch_home"
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setenv("TORCH_HOME", str(hub))
    monkeypatch.setattr(mediapipe_backend, "MODEL_DIR", models)
    assert td.torch_hub_checkpoints_dir() == hub / "hub" / "checkpoints"
    assert td.find_weights() is None and td.find_weights("") is None

    # download failure (offline / 403): a clear error, no partial file, no silent fallback
    calls = []

    def no_network(url, dst, hash_prefix=None, progress=True):
        calls.append((url, dst, hash_prefix))
        raise urllib.error.HTTPError(url, 403, "Forbidden", None, None)

    monkeypatch.setattr(torch.hub, "download_url_to_file", no_network)
    with pytest.raises(td.WeightsUnavailable) as e:
        td.download_weights()
    msg = str(e.value)
    assert td.DEFAULT_WEIGHTS_URL in msg and "weights_path" in msg and "403" in msg
    assert str(hub / "hub" / "checkpoints" / td.WEIGHTS_FILE) in msg
    assert str(models / td.WEIGHTS_FILE) in msg
    assert calls == [(td.DEFAULT_WEIGHTS_URL, str(hub / "hub" / "checkpoints" / td.WEIGHTS_FILE),
                      "fc266e95")]
    assert not any((hub / "hub" / "checkpoints").iterdir())
    with pytest.raises(td.WeightsUnavailable):
        td.create(device="cpu")  # the registry factory path
    with pytest.raises(td.WeightsUnavailable, match="not found"):
        td.load_model(download=False)
    with pytest.raises(FileNotFoundError, match="nope.pth"):
        td.create(device="cpu", weights_path=str(tmp_path / "nope.pth"))

    # the official file in the models folder or in the torch hub cache is used, no download
    n_calls = len(calls)
    for folder in (models, hub / "hub" / "checkpoints"):
        target = folder / td.WEIGHTS_FILE
        target.symlink_to(path)
        assert td.find_weights() == target
        if folder != models:  # download_weights only looks at (and fills) the hub cache
            assert td.download_weights() == target
        target.unlink()
    assert len(calls) == n_calls

    # a user file with another name: loaded; not the official hash -> default eps (1e-5)
    other = tmp_path / "my_keypoint_rcnn.pth"
    torch.save({"model": {"module." + k: v for k, v in sd.items()}, "epoch": 3}, other)
    loaded = td.load_state_dict(other)
    assert list(loaded) == list(sd) and all(torch.equal(loaded[k], sd[k]) for k in sd)
    det = td.create(device="cpu", weights_path=str(other), score_threshold=0.8, min_size=256,
                    max_size=512, max_people=3)
    try:
        assert det.weights_file == other.resolve() and det.options["weights_path"] == str(other)
        assert det.model.roi_heads.score_thresh == 0.8
        assert det.model.roi_heads.detections_per_img == 3
        assert det.model.transform.min_size == (256,) and det.model.transform.max_size == 512
        eps = {m.eps for m in det.model.modules() if isinstance(m, FrozenBatchNorm2d)}
        assert eps == {FrozenBatchNorm2d(4).eps}
    finally:
        det.close()
    assert det.model is None
    with pytest.raises(RuntimeError, match="closed"):
        det.detect(np.zeros((8, 8, 3), np.uint8), 0.0, "c")

    # wrong files
    bad = tmp_path / "bad.pth"
    torch.save({td._KPS_WEIGHT: torch.zeros(512, 5, 4, 4),
                td._CLS_WEIGHT: torch.zeros(2, 1024)}, bad)
    with pytest.raises(ValueError, match="5 keypoints"):
        td.load_model(bad)
    torch.save({"conv.weight": torch.zeros(1)}, bad)
    with pytest.raises(ValueError, match="not a Keypoint R-CNN"):
        td.load_model(bad)
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="Cannot read"):
        td.load_model(bad)
    partial = {k: v for k, v in sd.items() if not k.startswith("roi_heads.box_head")}
    torch.save(partial, bad)
    with pytest.raises(ValueError, match="does not match"):
        td.load_model(bad)


# ================================================================== part 3: real model
# MediaPipe Pose (full) on the same image, pixels (x, y); used when MediaPipe cannot run.
MEDIAPIPE_REFERENCE = {
    "left_shoulder": (544.6, 321.4), "right_shoulder": (449.7, 325.0),
    "left_hip": (522.1, 471.9), "right_hip": (464.5, 465.9),
    "left_knee": (611.5, 549.3), "right_knee": (356.6, 491.7),
    "left_ankle": (701.2, 616.1), "right_ankle": (351.2, 607.0),
}
JOINTS = tuple(MEDIAPIPE_REFERENCE)


def _mediapipe_keypoints(image) -> dict:
    try:
        from poseboard.pose.detectors import backend_available, create_detector
        from poseboard.pose.mediapipe_backend import ensure_model

        if not backend_available("mediapipe")[0]:
            raise RuntimeError("mediapipe not installed")
        ensure_model("full")
        det = create_detector("mediapipe", model="full")
        try:
            people = det.detect(image, 0.0, "ref")
        finally:
            det.close()
        if len(people) != 1:
            raise RuntimeError(f"MediaPipe found {len(people)} persons")
        return {n: people[0].keypoints[det.format.index(n)] for n in JOINTS}
    except Exception as e:  # noqa: BLE001  (no mediapipe / model: use the recorded values)
        print(f"MediaPipe reference not computed ({e}); using the recorded values")
        return {n: np.array(v) for n, v in MEDIAPIPE_REFERENCE.items()}


@pytest.fixture(scope="module")
def real_detector():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from poseboard.pose.detectors import backend_available, create_detector

    ok, why = backend_available("keypoint_rcnn")
    if not ok:
        pytest.skip(why)
    try:
        det = create_detector("keypoint_rcnn", device="auto")
    except td.WeightsUnavailable as e:
        pytest.skip(f"Keypoint R-CNN weights not available: {str(e).splitlines()[0]}")
    yield det
    det.close()


def test_real_model_on_a_real_image(real_detector, person_image):
    det = real_detector
    h, w = person_image.shape[:2]
    people = det.detect(person_image, 0.0, "cam0")
    assert len(people) == 1, [p.score for p in people]
    p = people[0]
    assert p.keypoints.shape == (N_KP, 2) and p.scores.shape == (N_KP,)
    assert 0.5 <= p.score <= 1.0 and np.all((p.scores >= 0) & (p.scores <= 1))
    kp = {n: p.keypoints[COCO17.index(n)] for n in JOINTS}
    sc = {n: p.scores[COCO17.index(n)] for n in JOINTS}
    assert all(np.all(np.isfinite(v)) for v in kp.values())
    assert all(s > 0.5 for s in sc.values()), sc  # clearly visible joints: logit > 0
    # the subject faces the camera: their left side is on the image right
    assert kp["left_shoulder"][0] > kp["right_shoulder"][0]
    for side in ("left", "right"):  # shoulders above hips above knees above ankles
        ys = [kp[f"{side}_{j}"][1] for j in ("shoulder", "hip", "knee", "ankle")]
        assert ys == sorted(ys), (side, ys)
    x1, y1, x2, y2 = p.bbox
    assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
    ref = _mediapipe_keypoints(person_image)
    tol = 0.08 * float(np.hypot(w, h))
    dist = {n: float(np.linalg.norm(kp[n] - ref[n])) for n in JOINTS}
    assert max(dist.values()) < tol, dist
