"""The ``vitpose_hf`` backend (ViTPose / ViTPose++ through Hugging Face transformers).

Part 1 runs everywhere (numpy + OpenCV only): keypoint names and order, boxes, the crop, the
heatmap decoding back to ORIGINAL image pixels, scores, flip test, person boxes and empty inputs.
The model and the person detector are replaced by numpy fakes through
``VitPoseHFDetector.from_components``.

Part 2 needs torch + transformers. It uses tiny randomly initialised ViTPose / ViTPose++ /
RT-DETR models saved to local folders (huggingface.co is not needed), runs the real loading
code, and compares the pre- and post-processing with transformers' own functions. The YOLO
person detector test needs ultralytics and its GitHub weights.

Part 3 runs the real pretrained models on a real image when torch + transformers are installed
and the weights can be obtained (huggingface.co); otherwise it is skipped with the reason.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from poseboard.pose.detectors import BACKENDS, backend_available, create_detector
from poseboard.pose.detectors import vitpose_hf_det as vd
from poseboard.pose.formats import COCO17, HALPE26, WHOLEBODY133
from tests.conftest import cache_dir

ROOT = Path(__file__).resolve().parents[1]
IN_H, IN_W = vd.INPUT_SIZE
HM_H, HM_W = IN_H // 4, IN_W // 4  # the heatmaps of the official models: 64 x 48


def _gauss(hx, hy, peak=0.9, sigma=2.0, h=HM_H, w=HM_W):
    yy, xx = np.mgrid[0:h, 0:w]
    return (peak * np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * sigma ** 2))).astype(
        np.float32)


def _heatmap_to_image(hx, hy, center, size, h=HM_H, w=HM_W):
    """Expected image position of heatmap position (hx, hy) for a padded box."""
    return np.array([hx * size[0] / (w - 1) + center[0] - size[0] / 2,
                     hy * size[1] / (h - 1) + center[1] - size[1] / 2])


# ================================================================== part 1: no torch needed
def test_importing_the_module_imports_no_heavy_library():
    code = ("import sys; import poseboard.pose.detectors.vitpose_hf_det; "
            "print(','.join(m for m in ('torch', 'transformers', 'ultralytics', 'scipy') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_registry_entry_and_availability():
    import importlib.util

    spec = BACKENDS["vitpose_hf"]
    assert spec.module == vd.__name__ and callable(getattr(vd, spec.factory))
    assert spec.keypoint_format == "coco17" and not spec.provides_3d
    assert spec.defaults["model"] == vd.DEFAULT_MODEL
    ok, why = backend_available("vitpose_hf")
    have = all(importlib.util.find_spec(m) is not None for m in ("transformers", "torch"))
    assert ok == have, why
    if not ok:
        assert "pip install" in why


def test_official_labels_are_the_coco17_order():
    # config.id2label of the usyd-community checkpoints (convert_vitpose_to_hf.py)
    id2label = {str(i): n for i, n in enumerate(vd.HF_COCO_LABELS)}
    names = vd.model_keypoint_names(id2label, 17)
    assert names == COCO17.names
    fmt, idx = vd.resolve_format(names)
    assert fmt is COCO17 and idx.tolist() == list(range(17))
    # the person's own left: "L_Shoulder" is channel 5 = COCO17 left_shoulder
    assert vd.HF_COCO_LABELS[5] == "L_Shoulder" and COCO17.names[5] == "left_shoulder"
    assert vd.HF_COCO_LABELS[16] == "R_Ankle" and COCO17.names[16] == "right_ankle"


@pytest.mark.parametrize("label, name", [
    ("L_Shoulder", "left_shoulder"), ("R_Eye", "right_eye"), ("Nose", "nose"),
    ("R_Big_Toe", "right_big_toe"), ("LeftKnee", "left_knee"), ("right ankle", "right_ankle"),
    ("left_hand_3", "left_hand_3"), ("LABEL_7", "label_7"), ("face-12", "face_12"),
])
def test_canonical_keypoint_name(label, name):
    assert vd.canonical_keypoint_name(label) == name


def test_resolve_format_by_name_generic_and_errors():
    rng = np.random.default_rng(3)
    perm = rng.permutation(17)
    shuffled = tuple(COCO17.names[p] for p in perm)
    fmt, idx = vd.resolve_format(shuffled)
    assert fmt is COCO17
    assert [shuffled[i] for i in idx] == list(COCO17.names)
    # generic labels: the format with the same number of points, in the model's order
    for fmt_ in (COCO17, HALPE26, WHOLEBODY133):
        generic = tuple(f"label_{i}" for i in range(len(fmt_)))
        f, idx = vd.resolve_format(generic)
        assert f is fmt_ and idx.tolist() == list(range(len(fmt_)))
    # a Halpe-26 model is Halpe-26, not COCO-17 (which is a subset)
    assert vd.resolve_format(HALPE26.names)[0] is HALPE26
    # explicit format: COCO-17 out of a Halpe-26 model, by name
    f, idx = vd.resolve_format(HALPE26.names, "coco17")
    assert f is COCO17 and idx.tolist() == list(range(17))
    with pytest.raises(ValueError, match="match none"):
        vd.resolve_format(("a", "b", "c"))
    with pytest.raises(KeyError):
        vd.resolve_format(COCO17.names, "no_such_format")


def test_flip_permutation():
    perm = vd.flip_permutation(COCO17.names)
    assert perm[0] == 0  # nose
    for left in ("eye", "ear", "shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        i, j = COCO17.index(f"left_{left}"), COCO17.index(f"right_{left}")
        assert perm[i] == j and perm[j] == i
    assert sorted(perm.tolist()) == list(range(17))
    with pytest.raises(ValueError, match="mirror"):
        vd.flip_permutation(WHOLEBODY133.names)  # face_0 .. face_67: no names for the mirror
    with pytest.raises(ValueError, match="has no"):
        vd.flip_permutation(("nose", "left_eye"))


def test_flip_back_swaps_sides_and_mirrors():
    hm = np.zeros((1, 3, 4, 5), np.float32)
    hm[0, 1, 2, 0] = 1.0  # channel 1 (left_x) at column 0 of the mirrored crop
    back = vd.flip_back(hm, np.array([0, 2, 1]))
    assert back[0, 2, 2, 4] == 1.0 and back.sum() == 1.0  # -> right_x, last column


def test_box_to_center_and_size():
    # tall box: widened to the 192:256 input aspect ratio, then padded by 1.25
    c, s = vd.box_to_center_and_size([10, 20, 30, 100])
    np.testing.assert_allclose(c, [25, 70])
    np.testing.assert_allclose(s, [75 * 1.25, 100 * 1.25])
    # wide box: the height grows
    c, s = vd.box_to_center_and_size([0, 0, 200, 100])
    np.testing.assert_allclose(c, [100, 50])
    np.testing.assert_allclose(s, [200 * 1.25, 200 / 0.75 * 1.25])
    np.testing.assert_allclose(vd.xyxy_to_xywh([[10, 20, 40, 120]]), [[10, 20, 30, 100]])


def test_warp_matrix_maps_the_padded_box_onto_the_input_pixel_centers():
    c, s = np.array([320.5, 240.25]), np.array([150.0, 200.0])
    m = vd.warp_matrix(c, s)
    tl = m @ [c[0] - s[0] / 2, c[1] - s[1] / 2, 1.0]
    br = m @ [c[0] + s[0] / 2, c[1] + s[1] / 2, 1.0]
    np.testing.assert_allclose(tl, [0, 0], atol=1e-9)
    np.testing.assert_allclose(br, [IN_W - 1, IN_H - 1], atol=1e-9)


def test_crop_and_normalize():
    img = np.zeros((100, 120, 3), np.uint8)
    img[40:60, 50:70] = (255, 128, 0)
    c, s = vd.box_to_center_and_size(vd.xyxy_to_xywh([[40, 30, 80, 70]])[0])
    crop = vd.crop_person(img, c, s)
    assert crop.shape == (IN_H, IN_W, 3) and crop.dtype == np.uint8
    assert crop[IN_H // 2, IN_W // 2].tolist() == [255, 128, 0]  # the box center
    assert crop[0, 0].tolist() == [0, 0, 0]
    px = vd.normalize_crops([crop, crop])
    assert px.shape == (2, 3, IN_H, IN_W) and px.dtype == np.float32
    mid = px[0, :, IN_H // 2, IN_W // 2]
    np.testing.assert_allclose(mid, (np.array([1.0, 128 / 255, 0.0]) - vd.IMAGENET_MEAN)
                               / vd.IMAGENET_STD, rtol=1e-5)


def test_to_rgb():
    bgr = np.zeros((4, 5, 3), np.uint8)
    bgr[..., 0] = 10  # blue
    rgb = vd.to_rgb(bgr)
    assert rgb[0, 0].tolist() == [0, 0, 10] and rgb.flags.c_contiguous
    assert vd.to_rgb(np.full((4, 5), 7, np.uint8))[0, 0].tolist() == [7, 7, 7]
    bgra = np.zeros((4, 5, 4), np.uint8)
    bgra[..., 2] = 99
    assert vd.to_rgb(bgra)[0, 0].tolist() == [99, 0, 0]
    assert vd.to_rgb(np.full((4, 5, 3), 300.0))[0, 0].tolist() == [255, 255, 255]
    for bad in (None, np.zeros((0, 5, 3), np.uint8), np.zeros((1, 1, 3), np.uint8),
                np.zeros(5, np.uint8), np.zeros((4, 5, 2), np.uint8)):
        assert vd.to_rgb(bad) is None


def test_heatmap_maxima_and_dark_subpixel_refinement():
    true = [(10.3, 20.6), (30.8, 50.2), (5.5, 7.45)]
    hm = np.stack([_gauss(x, y) for x, y in true])[None]
    coords, raw = vd.heatmap_maxima(hm)
    np.testing.assert_allclose(coords[0], np.round(true), atol=1)
    assert np.all(raw > 0.8)
    refined = vd.dark_refine(coords, hm)
    np.testing.assert_allclose(refined[0], true, atol=0.05)
    # nothing predicted: maximum <= 0 -> -1, unchanged by the refinement
    flat = np.full((1, 2, HM_H, HM_W), -0.1, np.float32)
    flat[0, 1] = np.nan
    coords, raw = vd.heatmap_maxima(flat)
    assert np.all(coords == -1)
    assert np.all(vd.dark_refine(coords, flat) == -1)
    # a constant (flat) positive heatmap cannot move the point by more than 1 heatmap pixel
    const = np.full((1, 1, HM_H, HM_W), 0.5, np.float32)
    coords, _ = vd.heatmap_maxima(const)
    assert np.all(np.abs(vd.dark_refine(coords, const) - coords) <= 1.0)


def test_keypoints_from_heatmaps_are_in_original_image_pixels():
    centers = np.array([[400.0, 300.0], [55.5, 700.25]])
    sizes = np.array([[150.0, 200.0], [300.0, 400.0]])
    pos = [[(4.0, 5.0), (HM_W - 5.0, HM_H - 4.0), (20.3, 33.7)],
           [(12.6, 40.1), (40.0, 6.0), (23.5, 31.5)]]
    hm = np.stack([np.stack([_gauss(x, y, sigma=1.5) for x, y in p]) for p in pos])
    kp, raw = vd.keypoints_from_heatmaps(hm, centers, sizes)
    assert kp.shape == (2, 3, 2) and raw.shape == (2, 3)
    for b in range(2):
        for k, (x, y) in enumerate(pos[b]):
            np.testing.assert_allclose(kp[b, k], _heatmap_to_image(x, y, centers[b], sizes[b]),
                                       atol=0.05 * sizes[b][0] / (HM_W - 1))
    # the mapping is affine: extrapolated, heatmap pixel (0, 0) is the padded box's top-left
    # corner and (W-1, H-1) its bottom-right corner
    (x0, y0), (x1, y1) = pos[0][0], pos[0][1]
    per_px = (kp[0, 1] - kp[0, 0]) / np.array([x1 - x0, y1 - y0])
    np.testing.assert_allclose(per_px, sizes[0] / [HM_W - 1, HM_H - 1], rtol=1e-3)
    np.testing.assert_allclose(kp[0, 0] - per_px * [x0, y0], centers[0] - sizes[0] / 2,
                               atol=0.1)


def test_keypoint_scores_are_clipped_heatmap_maxima():
    np.testing.assert_allclose(vd.keypoint_scores([-0.2, 0.0, 0.5, 1.2, np.nan, np.inf]),
                               [0, 0, 0.5, 1, 0, 0])


def test_clean_person_boxes():
    boxes = [[10, 10, 110, 210], [12, 11, 111, 209],  # duplicate of the first
             [-50, -20, 60, 80],  # partly outside: clipped
             [300, 300, 301, 400],  # 1 px wide: dropped
             [np.nan, 0, 10, 10],  # not finite: dropped
             [200, 50, 260, 150], [400, 10, 500, 60]]
    scores = [0.8, 0.7, 0.95, 0.9, 0.99, 1.4, 0.5]
    b, s = vd.clean_person_boxes(boxes, scores, (240, 320), max_people=3)
    np.testing.assert_allclose(s, [1.0, 0.95, 0.8])
    np.testing.assert_allclose(b, [[200, 50, 260, 150], [0, 0, 60, 80], [10, 10, 110, 210]])
    b, s = vd.clean_person_boxes(np.zeros((0, 4)), [], (10, 10))
    assert b.shape == (0, 4) and s.shape == (0,)
    with pytest.raises(ValueError):
        vd.clean_person_boxes([[0, 0, 5, 5]], [0.5, 0.6], (10, 10))


# ------------------------------------------------------------------ the detector with fakes
def _blob_image(h, w, dots, sigma=2.5):
    """Black BGR image with a bright (red) Gaussian blob at each (x, y)."""
    yy, xx = np.mgrid[0:h, 0:w]
    red = np.zeros((h, w))
    for x, y in dots:
        red += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 2] = np.clip(red * 255, 0, 255).astype(np.uint8)
    return img


class DotModel:
    """A fake ViTPose: every heatmap channel peaks where the red blob is in the crop."""

    def __init__(self, k=17):
        self.k = k
        self.calls = []

    def __call__(self, px):
        self.calls.append(px.copy())
        out = np.zeros((len(px), self.k, HM_H, HM_W), np.float32)
        for b in range(len(px)):
            red = px[b, 0] * vd.IMAGENET_STD[0] + vd.IMAGENET_MEAN[0]  # undo the normalization
            wts = np.where(red > 0.05, red, 0.0)
            ys, xs = np.mgrid[0:IN_H, 0:IN_W]
            cx, cy = (xs * wts).sum() / wts.sum(), (ys * wts).sum() / wts.sum()
            out[b] = _gauss(cx * (HM_W - 1) / (IN_W - 1), cy * (HM_H - 1) / (IN_H - 1))
        return out


class FixedPersons:
    def __init__(self, boxes, scores):
        self.boxes, self.scores = np.asarray(boxes, float), np.asarray(scores, float)
        self.calls = []

    def __call__(self, rgb):
        self.calls.append(rgb.shape)
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
        return self.boxes.copy(), self.scores.copy()


def test_detector_returns_keypoints_in_original_image_pixels():
    h, w = 480, 800
    dots = [(150.3, 200.7), (600.6, 300.2), (780.2, 20.4)]
    img = _blob_image(h, w, dots)
    boxes = [[100, 120, 220, 300],  # tall
             [520, 250, 700, 360],  # wide
             [740, -30, 830, 60]]  # partly outside the image: clipped, crop has black borders
    persons = FixedPersons(boxes, [0.7, 0.9, 0.8])
    model = DotModel()
    det = vd.VitPoseHFDetector.from_components(model, persons, batch_size=2)
    people = det.detect(img, 1.0, "cam0")
    assert len(people) == 3 and len(model.calls) == 2  # 3 crops in batches of 2
    assert [p.score for p in people] == [0.9, 0.8, 0.7]  # best person first
    order = [1, 2, 0]
    for p, i in zip(people, order):
        assert p.keypoints.shape == (17, 2) and p.scores.shape == (17,)
        err = np.linalg.norm(p.keypoints - np.array(dots[i]), axis=1)
        assert err.max() < 1.0, (i, err.max())
        # the heatmap value at the argmax pixel, next to the 0.9 peak
        assert np.all((p.scores > 0.8) & (p.scores <= 0.9))
    np.testing.assert_allclose(people[1].bbox, [740, 0, 800, 60])
    np.testing.assert_allclose(people[2].bbox, boxes[0])
    assert det.format is COCO17 and det.info()["keypoint_format"] == "coco17"


def test_detector_maps_heatmap_channels_by_label_name():
    rng = np.random.default_rng(7)
    perm = rng.permutation(17)
    labels = [vd.HF_COCO_LABELS[p] for p in perm]  # channel c is labelled HF label perm[c]

    def model(px):
        return np.stack([np.stack([_gauss(2 + 2.5 * c, 10 + c) for c in range(17)])] * len(px))

    box = [100.0, 50.0, 250.0, 350.0]
    det = vd.VitPoseHFDetector.from_components(model, FixedPersons([box], [1.0]),
                                               keypoint_names=labels)
    (p,) = det.detect(np.zeros((400, 400, 3), np.uint8), 0.0, "c")
    c, s = vd.box_to_center_and_size(vd.xyxy_to_xywh([box])[0])
    for ch in range(17):
        name = vd.canonical_keypoint_name(vd.HF_COCO_LABELS[perm[ch]])
        np.testing.assert_allclose(p.keypoints[COCO17.index(name)],
                                   _heatmap_to_image(2 + 2.5 * ch, 10 + ch, c, s), atol=0.5)


def test_detector_scores_and_missing_keypoints():
    peaks = np.full(17, 0.9)
    peaks[:6] = [1.3, 0.4, 0.0, -0.5, np.nan, 0.05]

    def model(px):
        hm = np.stack([_gauss(20, 30, peak=1.0 if np.isnan(v) else v) for v in peaks])
        hm[4] = np.nan  # a broken channel
        return np.stack([hm] * len(px))

    det = vd.VitPoseHFDetector.from_components(model, FixedPersons([[0, 0, 90, 120]], [0.6]))
    (p,) = det.detect(np.zeros((120, 160, 3), np.uint8), 0.0, "c")
    np.testing.assert_allclose(p.scores[:6], [1.0, 0.4, 0.0, 0.0, 0.0, 0.05], atol=1e-6)
    assert np.all(np.isnan(p.keypoints[2:5])) and np.all(np.isfinite(p.keypoints[[0, 1, 5]]))
    assert np.all(np.isfinite(p.keypoints[6:])) and p.score == 0.6


def test_flip_test_runs_the_mirrored_crop_and_keeps_left_and_right():
    """A mirror-consistent fake: the person's left = the rightmost blob in the crop. With the
    flip test the model also gets the mirrored crop and the result stays the same."""
    img = _blob_image(300, 400, [(150.0, 150.0), (250.0, 160.0), (200.0, 100.0)])
    left = [i for i, n in enumerate(COCO17.names) if n.startswith("left_")]
    right = [i for i, n in enumerate(COCO17.names) if n.startswith("right_")]

    def model(px):
        out = np.zeros((len(px), 17, HM_H, HM_W), np.float32)
        for b in range(len(px)):
            import cv2

            red = (px[b, 0] * vd.IMAGENET_STD[0] + vd.IMAGENET_MEAN[0] > 0.5).astype(np.uint8)
            _n, _lab, _st, cents = cv2.connectedComponentsWithStats(red)
            cents = cents[1:]  # without the background
            cents = cents[np.argsort(cents[:, 0])]  # left to right in the crop
            to_hm = np.array([(HM_W - 1) / (IN_W - 1), (HM_H - 1) / (IN_H - 1)])
            lo, mid, hi = (c * to_hm for c in cents)
            for i in range(17):
                pt = hi if i in left else lo if i in right else mid
                out[b, i] = _gauss(*pt)
        return out

    calls = []

    def counting(px):
        calls.append(px.copy())
        return model(px)

    persons = FixedPersons([[100, 50, 300, 250]], [1.0])
    plain = vd.VitPoseHFDetector.from_components(model, persons).detect(img, 0.0, "c")[0]
    det = vd.VitPoseHFDetector.from_components(counting, persons, flip_test=True)
    flipped = det.detect(img, 0.0, "c")[0]
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1], calls[0][..., ::-1])  # the mirrored crop
    np.testing.assert_allclose(flipped.keypoints, plain.keypoints, atol=0.3)
    # generic labels (LABEL_0 ...): the mirror pairs come from the COCO-17 names
    generic = vd.VitPoseHFDetector.from_components(
        model, persons, flip_test=True, keypoint_names=[f"LABEL_{i}" for i in range(17)])
    np.testing.assert_array_equal(generic._flip_perm, vd.flip_permutation(COCO17.names))
    np.testing.assert_allclose(generic.detect(img, 0.0, "c")[0].keypoints, plain.keypoints,
                               atol=0.3)
    assert plain.keypoints[COCO17.index("left_shoulder"), 0] > 240  # image right
    assert plain.keypoints[COCO17.index("right_shoulder"), 0] < 160


def test_detector_empty_inputs_and_close():
    model = DotModel()
    det = vd.VitPoseHFDetector.from_components(model, FixedPersons(np.zeros((0, 4)), []))
    assert det.detect(np.zeros((50, 60, 3), np.uint8), 0.0, "c") == []  # nobody
    for bad in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((1, 1, 3), np.uint8)):
        assert det.detect(bad, 0.0, "c") == []
    assert model.calls == []
    det2 = vd.VitPoseHFDetector.from_components(DotModel(), None)  # whole image = one box
    (p,) = det2.detect(_blob_image(100, 80, [(40.0, 60.0)]), 0.0, "c")
    np.testing.assert_allclose(p.bbox, [0, 0, 80, 100])
    np.testing.assert_allclose(p.keypoints[0], [40, 60], atol=0.5)
    assert det2.estimate(np.zeros((10, 10, 3), np.uint8), np.zeros((0, 4))) == []
    with pytest.raises(ValueError, match="box scores"):
        det2.estimate(np.zeros((10, 10, 3), np.uint8), [[0, 0, 5, 5]], [0.5, 0.5])
    det.close()
    det.close()
    with pytest.raises(RuntimeError, match="closed"):
        det.detect(np.zeros((50, 60, 3), np.uint8), 0.0, "c")


def test_detector_rejects_wrong_heatmaps_and_bad_options():
    det = vd.VitPoseHFDetector.from_components(lambda px: np.zeros((len(px), 5, 8, 6)),
                                               FixedPersons([[0, 0, 20, 20]], [1.0]))
    with pytest.raises(ValueError, match="heatmaps"):
        det.detect(np.zeros((30, 30, 3), np.uint8), 0.0, "c")
    with pytest.raises(ValueError, match="person_detector"):
        vd.VitPoseHFDetector(person_detector="faster-rcnn")


def test_with_the_multiview_estimator_gives_a_2d_only_pose():
    from poseboard.calibration import approximate_calibration
    from poseboard.pose.base import MODE_2D_ONLY
    from poseboard.pose.multiview import MultiViewEstimator

    img = _blob_image(240, 320, [(160.0, 120.0)])
    det = vd.VitPoseHFDetector.from_components(DotModel(), FixedPersons([[100, 40, 220, 200]],
                                                                        [0.9]))
    est = MultiViewEstimator(det)
    cam = approximate_calibration("cam0", 320, 240)
    pose = est.process({"cam0": (1.0, img)}, {"cam0": cam})
    assert pose is not None and pose.mode == MODE_2D_ONLY and pose.format_key == "coco17"
    np.testing.assert_allclose(pose.per_camera_2d["cam0"].keypoints[5], [160, 120], atol=1.0)
    est.close()


# ================================================================== part 2: torch + transformers
@pytest.fixture(scope="module")
def hf():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "VitPoseForPoseEstimation"):
        pytest.skip(f"transformers {transformers.__version__} has no ViTPose (needs >= 4.48)")
    return SimpleNamespace(torch=torch, transformers=transformers)


def _tiny_vitpose(hf, labels=None, num_experts=1, seed=0):
    t = hf.transformers
    hf.torch.manual_seed(seed)
    bb = t.VitPoseBackboneConfig(image_size=[IN_H, IN_W], patch_size=[16, 16], hidden_size=32,
                                 num_hidden_layers=1, num_attention_heads=2, mlp_ratio=2,
                                 out_indices=[1], num_experts=num_experts,
                                 part_features=8 if num_experts > 1 else 0)
    kw = {}
    if labels is not None:
        kw = {"id2label": dict(enumerate(labels)),
              "label2id": {n: i for i, n in enumerate(labels)}}
    cfg = t.VitPoseConfig(backbone_config=bb, num_labels=17, **kw)
    return t.VitPoseForPoseEstimation(cfg).eval()


def _tiny_rtdetr(hf, seed=0):
    t = hf.transformers
    hf.torch.manual_seed(seed)
    rb = t.RTDetrResNetConfig(embedding_size=8, hidden_sizes=[8, 8, 16, 16], depths=[1, 1, 1, 1],
                              layer_type="basic", out_features=["stage2", "stage3", "stage4"])
    labels = {0: "person", 1: "bicycle", 2: "car"}
    cfg = t.RTDetrConfig(backbone_config=rb, encoder_hidden_dim=16, encoder_in_channels=[8, 16, 16],
                         d_model=16, num_queries=10, decoder_layers=1, encoder_layers=1,
                         decoder_ffn_dim=16, encoder_ffn_dim=16, num_feature_levels=3,
                         decoder_attention_heads=2, encoder_attention_heads=2,
                         decoder_in_channels=[16, 16, 16], num_labels=3, id2label=labels,
                         label2id={v: k for k, v in labels.items()})
    return t.RTDetrForObjectDetection(cfg).eval()


@pytest.fixture(scope="module")
def tiny_folders(hf, tmp_path_factory):
    """Local model folders (config + weights + preprocessor config), like the Hub ones."""
    t = hf.transformers
    root = tmp_path_factory.mktemp("hf_models")
    folders = {}
    for name, model in (("vitpose", _tiny_vitpose(hf, vd.HF_COCO_LABELS)),
                        ("vitpose_plus", _tiny_vitpose(hf, vd.HF_COCO_LABELS, num_experts=3)),
                        ("vitpose_generic", _tiny_vitpose(hf))):
        folders[name] = root / name
        model.save_pretrained(folders[name])
    t.VitPoseImageProcessor().save_pretrained(folders["vitpose"])
    t.VitPoseImageProcessor(image_mean=[0.5, 0.5, 0.5], image_std=[0.25, 0.25, 0.25]) \
        .save_pretrained(folders["vitpose_plus"])  # custom normalization: must be read
    folders["rtdetr"] = root / "rtdetr"
    _tiny_rtdetr(hf).save_pretrained(folders["rtdetr"])
    t.RTDetrImageProcessor().save_pretrained(folders["rtdetr"])
    return folders


def test_hf_model_wrapper_loads_a_local_folder(hf, tiny_folders):
    m = vd.HFVitPoseModel(str(tiny_folders["vitpose"]), "cpu", local_files_only=True)
    assert m.names == COCO17.names and m.input_size == (IN_H, IN_W)
    assert m.num_experts == 1 and m.dataset_index is None
    assert m.mean == vd.IMAGENET_MEAN and m.std == vd.IMAGENET_STD
    hm = m(np.zeros((3, 3, IN_H, IN_W), np.float32))
    assert hm.shape == (3, 17, HM_H, HM_W) and hm.dtype == np.float32
    # no preprocessor_config.json: the ImageNet defaults; generic labels: COCO-17 order
    g = vd.HFVitPoseModel(str(tiny_folders["vitpose_generic"]), "cpu", local_files_only=True)
    assert g.names[0] == "label_0" and g.mean == vd.IMAGENET_MEAN
    assert vd.resolve_format(g.names)[0] is COCO17


def test_vitpose_plus_gets_the_dataset_index(hf, tiny_folders):
    folder = str(tiny_folders["vitpose_plus"])
    m = vd.HFVitPoseModel(folder, "cpu", local_files_only=True)
    assert m.num_experts == 3 and m.dataset_index == 0  # COCO expert by default
    assert m.mean == (0.5, 0.5, 0.5) and m.std == (0.25, 0.25, 0.25)
    x = np.random.default_rng(0).normal(size=(2, 3, IN_H, IN_W)).astype(np.float32)
    h0 = m(x)
    raw = m.model
    with hf.torch.inference_mode():
        ref0 = raw(pixel_values=hf.torch.from_numpy(x),
                   dataset_index=hf.torch.tensor([0, 0])).heatmaps.numpy()
        with pytest.raises(ValueError, match="dataset_index"):
            raw(pixel_values=hf.torch.from_numpy(x))  # why it must be passed
    np.testing.assert_allclose(h0, ref0, atol=1e-6)
    m2 = vd.HFVitPoseModel(folder, "cpu", dataset_index=2, local_files_only=True)
    assert not np.allclose(m2(x), h0)  # another expert
    with pytest.raises(ValueError, match="dataset_index"):
        vd.HFVitPoseModel(folder, "cpu", dataset_index=3, local_files_only=True)


def test_numpy_flip_back_equals_transformers_flip_pairs(hf):
    model = _tiny_vitpose(hf, vd.HF_COCO_LABELS)
    x = hf.torch.from_numpy(np.random.default_rng(1).normal(size=(2, 3, IN_H, IN_W))
                            .astype(np.float32))
    perm = vd.flip_permutation(COCO17.names)
    pairs = hf.torch.tensor([[i, int(perm[i])] for i in range(17) if perm[i] > i])
    with hf.torch.inference_mode():
        ref = model(x, flip_pairs=pairs).heatmaps.numpy()
        mine = vd.flip_back(model(x).heatmaps.numpy(), perm)
    np.testing.assert_allclose(mine, ref, atol=1e-6)


def _gaussian_filter_cv2(image, sigma, radius, axes=(0, 1)):
    """scipy.ndimage.gaussian_filter(mode="reflect") with OpenCV (for transformers without
    SciPy)."""
    import cv2

    k = 2 * int(radius[0]) + 1
    return cv2.GaussianBlur(np.asarray(image, np.float32), (k, k), sigma,
                            borderType=cv2.BORDER_REFLECT)


@pytest.fixture
def vitpose_ip(hf, monkeypatch):
    import importlib

    mod = importlib.import_module("transformers.models.vitpose.image_processing_vitpose")
    if importlib.util.find_spec("scipy") is None:
        monkeypatch.setattr(mod, "gaussian_filter", _gaussian_filter_cv2, raising=False)
    return mod


def test_geometry_equals_transformers_functions(vitpose_ip):
    ip = vitpose_ip
    rng = np.random.default_rng(5)
    for _ in range(20):
        x, y = rng.uniform(-50, 600, 2)
        w, h = rng.uniform(5, 400, 2)
        c, s = vd.box_to_center_and_size([x, y, w, h])
        c_ref, s_ref = ip.box_to_center_and_scale([x, y, w, h], image_width=IN_W,
                                                  image_height=IN_H)
        np.testing.assert_allclose(c, c_ref, rtol=1e-5)
        np.testing.assert_allclose(s, s_ref * 200.0, rtol=1e-5)
        m_ref = ip.get_warp_matrix(0, c_ref * 2.0, np.array((IN_W, IN_H)) - 1.0, s_ref * 200.0)
        np.testing.assert_allclose(vd.warp_matrix(c, s), m_ref, rtol=1e-4, atol=1e-3)
    hm = rng.normal(0, 0.3, (2, 17, HM_H, HM_W)).astype(np.float32)
    hm[0, 3] = -1.0  # no prediction
    coords, scores = vd.heatmap_maxima(hm)
    c_ref, s_ref = ip.get_keypoint_predictions(hm)
    np.testing.assert_array_equal(coords, c_ref)
    np.testing.assert_allclose(scores, s_ref[..., 0])


def test_decoding_equals_transformers_post_process(vitpose_ip):
    """Keypoints and scores equal ``post_process_pose_estimation`` (with the DARK blur of
    SciPy, or its OpenCV equivalent when SciPy is not installed)."""
    rng = np.random.default_rng(2)
    boxes_xywh = np.array([[300.0, 180.0, 460.0, 470.0], [10.5, 20.25, 110.0, 380.0],
                           [900.0, 600.0, 100.0, 67.0]])
    true = rng.uniform([3, 3], [HM_W - 4, HM_H - 4], (3, 17, 2))
    hm = np.stack([np.stack([_gauss(*p) for p in person]) for person in true])
    hm += rng.normal(0, 0.01, hm.shape).astype(np.float32)
    proc = vitpose_ip.VitPoseImageProcessor()
    import torch

    res = proc.post_process_pose_estimation(SimpleNamespace(heatmaps=torch.from_numpy(hm)),
                                            boxes=[boxes_xywh.tolist()])[0]
    cs = [vd.box_to_center_and_size(b) for b in boxes_xywh]
    kp, raw = vd.keypoints_from_heatmaps(hm, np.array([c for c, _ in cs]),
                                         np.array([s for _, s in cs]))
    for i, r in enumerate(res):
        np.testing.assert_allclose(kp[i], r["keypoints"].numpy(), atol=1e-3)
        np.testing.assert_allclose(raw[i], r["scores"].numpy(), atol=1e-6)
        assert r["labels"].tolist() == list(range(17))


def test_crop_equals_transformers_preprocess(vitpose_ip, person_image):
    """``pixel_values`` equal the transformers processor. It needs SciPy for its crop; the
    only differences allowed are the pixel rows / columns on the image border (SciPy and
    OpenCV interpolate the border differently; PoseBoard matches the original OpenCV code) and
    rounding (1 gray level)."""
    import importlib.util

    import cv2

    if importlib.util.find_spec("scipy") is None:
        pytest.skip("scipy is not installed (transformers' ViTPose crop needs it; PoseBoard's "
                    "does not)")
    rgb = vd.to_rgb(person_image)
    h, w = rgb.shape[:2]
    boxes_xywh = np.array([[300.0, 180.0, 460.0, 470.0], [10.5, 20.25, 110.0, 380.0],
                           [900.0, 600.0, 100.0, 67.0]])
    ref = vitpose_ip.VitPoseImageProcessor()(images=rgb, boxes=[boxes_xywh.tolist()],
                                             return_tensors="pt").pixel_values.numpy()
    cs = [vd.box_to_center_and_size(b) for b in boxes_xywh]
    mine = vd.normalize_crops([vd.crop_person(rgb, c, s) for c, s in cs])
    ys, xs = np.mgrid[0:IN_H, 0:IN_W]
    for i, (c, s) in enumerate(cs):
        inv = cv2.invertAffineTransform(vd.warp_matrix(c, s))
        x = inv[0, 0] * xs + inv[0, 2]
        y = inv[1, 1] * ys + inv[1, 2]
        interior = (x > 1.5) & (x < w - 2.5) & (y > 1.5) & (y < h - 2.5)
        diff = np.abs(ref[i] - mine[i]).max(axis=0)
        assert diff[interior].max() < 1.5 / 255 / min(vd.IMAGENET_STD), diff[interior].max()


def test_resolve_device(hf, monkeypatch):
    monkeypatch.setattr(hf.torch.cuda, "is_available", lambda: False)
    assert vd.resolve_device("auto") == "cpu" and vd.resolve_device(None) == "cpu"
    assert vd.resolve_device("CPU") == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        vd.resolve_device("cuda")
    with pytest.raises(ValueError, match="unknown device"):
        vd.resolve_device("tpu")
    monkeypatch.setattr(hf.torch.cuda, "is_available", lambda: True)
    assert vd.resolve_device("auto") == "cuda"


def test_missing_weights_raise_a_clear_error(hf, tmp_path):
    with pytest.raises(vd.WeightsUnavailable, match="huggingface.co") as e:
        vd.HFVitPoseModel("no-such-org/no-such-vitpose", "cpu", local_files_only=True,
                          cache_dir=str(tmp_path))
    assert "no-such-org/no-such-vitpose" in str(e.value) and "--local-dir" in str(e.value)
    with pytest.raises(vd.WeightsUnavailable, match="person detector"):
        vd.RTDetrPersonDetector("no-such-org/no-such-detr", "cpu", local_files_only=True,
                                cache_dir=str(tmp_path))
    with pytest.raises(vd.WeightsUnavailable, match="from the folder") as e:
        vd.HFVitPoseModel(str(tmp_path / "missing_model"), "cpu")
    assert "config.json" in str(e.value)


def test_person_label_ids():
    assert vd.person_label_ids({0: "person", 1: "car"}) == [0]
    assert vd.person_label_ids({"1": "bicycle", "3": "Person"}) == [3]
    assert vd.person_label_ids({0: "LABEL_0", 1: "LABEL_1"}) == [0]
    assert vd.person_label_ids(None) == [0]
    with pytest.raises(ValueError, match="person"):
        vd.person_label_ids({0: "cat", 1: "dog"})


def test_rtdetr_boxes_are_converted_to_original_pixels(hf, tiny_folders):
    torch = hf.torch
    det = vd.RTDetrPersonDetector(str(tiny_folders["rtdetr"]), "cpu", threshold=0.3,
                                  local_files_only=True)
    assert det.person_ids == [0]
    seen = []

    def fake_model(**inputs):
        seen.append(inputs["pixel_values"].shape)
        logits = torch.full((1, 4, 3), -10.0)
        logits[0, 0, 0] = 3.0  # person, p = 0.95
        logits[0, 1, 2] = 3.0  # car
        logits[0, 2, 0] = -2.0  # person, p = 0.12 < threshold
        logits[0, 3, 0] = 0.0  # person, p = 0.5
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.6], [0.2, 0.2, 0.1, 0.1],
                               [0.7, 0.7, 0.1, 0.1], [0.1, 0.25, 0.2, 0.5]]])  # cx, cy, w, h
        return SimpleNamespace(logits=logits, pred_boxes=boxes)

    det.model = fake_model
    rgb = np.zeros((480, 640, 3), np.uint8)
    boxes, scores = det(rgb)
    assert seen == [(1, 3, 640, 640)]  # RT-DETR resizes to 640 x 640 itself
    order = np.argsort(-scores)
    np.testing.assert_allclose(scores[order], [1 / (1 + np.exp(-3)), 0.5], rtol=1e-5)
    np.testing.assert_allclose(boxes[order], [[256, 96, 384, 384], [0, 0, 128, 240]], atol=1e-3)


def test_full_detector_from_local_folders(hf, tiny_folders, person_image):
    det = create_detector("vitpose_hf", model=str(tiny_folders["vitpose"]), device="cpu",
                          detector_model=str(tiny_folders["rtdetr"]), local_files_only=True,
                          detector_threshold=0.0, max_people=3)
    try:
        assert isinstance(det, vd.VitPoseHFDetector) and det.format is COCO17
        people = det.detect(person_image, 0.0, "cam0")
        assert 1 <= len(people) <= 3  # random weights: any boxes, at most max_people
        h, w = person_image.shape[:2]
        for p in people:
            assert p.keypoints.shape == (17, 2)
            assert np.all((p.scores >= 0) & (p.scores <= 1))
            x1, y1, x2, y2 = p.bbox
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
            c, s = vd.box_to_center_and_size([x1, y1, x2 - x1, y2 - y1])
            ok = np.all(np.isfinite(p.keypoints), axis=1)
            # inside the crop (the DARK step may add up to one heatmap pixel)
            assert np.all(np.abs(p.keypoints[ok] - c) <= s / 2 * (1 + 2.0 / (HM_W - 1)))
        info = det.info()
        assert info["backend"] == "vitpose_hf" and info["person_detector"] == "rtdetr"
        assert info["options"]["detector_model"] == str(tiny_folders["rtdetr"])
        assert info["device_used"] == "cpu" and info["pose2sim_model"] == "COCO_17"
    finally:
        det.close()


def test_yolo_person_detector_with_real_weights(hf, tiny_folders, person_image):
    """Real YOLO person boxes (weights from GitHub) + a tiny random ViTPose."""
    pytest.importorskip("ultralytics")
    try:
        vd.resolve_yolo_weights("yolo11n.pt", cache_dir())
    except vd.WeightsUnavailable as e:
        pytest.skip(f"YOLO weights not available: {str(e).splitlines()[0]}")
    det = create_detector("vitpose_hf", model=str(tiny_folders["vitpose"]), device="cpu",
                          person_detector="yolo", weights_dir=str(cache_dir()),
                          local_files_only=True)
    try:
        people = det.detect(person_image, 0.0, "cam0")
        assert len(people) == 1
        x1, y1, x2, y2 = people[0].bbox
        # the box holds MediaPipe's shoulders .. ankles of the subject
        for x, y in MEDIAPIPE_REFERENCE.values():
            assert x1 - 5 <= x <= x2 + 5 and y1 - 5 <= y <= y2 + 5
        assert people[0].score > 0.5
        assert "AGPL" in det.info()["license"]
    finally:
        det.close()
    with pytest.raises(FileNotFoundError):
        vd.resolve_yolo_weights("no_such_weights.pt", cache_dir())


# ================================================================== part 3: real models
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


@pytest.fixture(scope="module", params=["rtdetr", "yolo"])
def real_detector(request):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    ok, why = backend_available("vitpose_hf")
    if not ok:
        pytest.skip(why)
    kw = {}
    if request.param == "yolo":
        pytest.importorskip("ultralytics")
        kw["weights_dir"] = str(cache_dir())
    try:
        det = create_detector("vitpose_hf", device="auto", person_detector=request.param, **kw)
    except vd.WeightsUnavailable as e:
        pytest.skip(f"ViTPose / person detector weights not available: "
                    f"{str(e).splitlines()[0]}")
    yield det
    det.close()


def test_real_model_on_a_real_image(real_detector, person_image):
    det = real_detector
    h, w = person_image.shape[:2]
    people = det.detect(person_image, 0.0, "cam0")
    assert len(people) == 1, [(p.score, p.bbox) for p in people]
    p = people[0]
    assert p.keypoints.shape == (17, 2) and p.scores.shape == (17,)
    assert 0.5 <= p.score <= 1.0 and np.all((p.scores >= 0) & (p.scores <= 1))
    kp = {n: p.keypoints[COCO17.index(n)] for n in JOINTS}
    sc = {n: p.scores[COCO17.index(n)] for n in JOINTS}
    assert all(np.all(np.isfinite(v)) for v in kp.values())
    assert all(s > 0.3 for s in sc.values()), sc
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
