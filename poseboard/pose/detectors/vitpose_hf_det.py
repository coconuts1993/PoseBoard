"""ViTPose / ViTPose++ through Hugging Face ``transformers`` as a ``Detector2D`` (backend key
``vitpose_hf``).

ViTPose is a top-down model: a person detector finds the people, then ViTPose estimates the
17 COCO keypoints of each person inside a crop around its box. ``detect`` returns every person
found; the subject on the board is picked later by ``poseboard.pose.subject.select_subject``.

License: ViTPose, the ``usyd-community`` checkpoints, RT-DETR (``PekingU/...``) and
``transformers`` are Apache-2.0. The optional ``person_detector="yolo"`` uses Ultralytics YOLO,
which is **AGPL-3.0** (see ``ultralytics_det``).

Models
------
* ``model``: a Hugging Face model id or a local folder saved with ``save_pretrained``
  (``config.json`` + weights, optionally ``preprocessor_config.json``). Official checkpoints:
  ``usyd-community/vitpose-base-simple`` (default), ``usyd-community/vitpose-base``,
  ``usyd-community/vitpose-plus-{small,base,large,huge}`` (ViTPose++).
* ViTPose++ backbones are mixture-of-experts models trained on 6 datasets. ``forward`` needs a
  ``dataset_index`` then (``transformers/models/vitpose_backbone/modeling_vitpose_backbone.py``,
  ``VitPoseBackboneLayer.forward``, lines 302-308 in transformers 5.17). The Hugging Face
  checkpoints only keep the COCO head (17 keypoints), and the conversion script uses
  ``dataset_index = 0`` for every checkpoint (``convert_vitpose_to_hf.py`` in the transformers
  GitHub repository, line 272). PoseBoard therefore passes ``dataset_index=0`` (COCO) for these
  models unless the ``dataset_index`` option says otherwise; it is ignored for single-expert
  models.
* ``person_detector``:

  * ``"rtdetr"`` (default): ``PekingU/rtdetr_r50vd_coco_o365`` through
    ``AutoModelForObjectDetection``, as in the transformers ViTPose documentation. Any
    transformers object-detection model with a ``person`` label can be given as
    ``detector_model`` (Hub id or local folder).
  * ``"yolo"``: Ultralytics YOLO detection weights (``detector_model``, default ``yolo11n.pt``,
    downloaded from the Ultralytics GitHub release assets into the PoseBoard models folder or
    ``weights_dir``). Needs ``pip install ultralytics`` (AGPL-3.0).
  * ``"none"``: the whole image is one person box (single person filling the view).

* Weights are downloaded by ``from_pretrained`` into the Hugging Face cache
  (``~/.cache/huggingface/hub``, or ``cache_dir``) on first use. Without access to
  huggingface.co, ``WeightsUnavailable`` (a RuntimeError) explains how to copy the model folders
  by hand. ``local_files_only=True`` (or ``HF_HUB_OFFLINE=1``) never touches the network.

Pre- and post-processing
------------------------
transformers' ``VitPoseImageProcessor`` needs SciPy for its crop (``scipy_warp_affine``) and for
the DARK refinement (``gaussian_filter``), and SciPy is not a PoseBoard dependency. This module
therefore implements the same steps with OpenCV and numpy, following
``transformers/models/vitpose/image_processing_vitpose.py`` (transformers 5.17) line by line.
The tests compare them with the transformers functions.

1. Boxes. The person detector gives (x1, y1, x2, y2) in original image pixels. ViTPose works
   with COCO boxes (x, y, width, height) with ``width = x2 - x1``, as in the transformers
   ViTPose documentation. ``box_to_center_and_scale`` (lines 68-109) takes the box center,
   widens the shorter side to the model's input aspect ratio (192:256) and pads the box by
   1.25. Here the padded size is kept in pixels (transformers stores it divided by
   ``normalize_factor`` = 200 and multiplies it back, lines 106 and 303, so this is the same).
2. Crop. ``affine_transform`` (lines 386-401) warps the padded box onto the 192 x 256 input
   with ``get_warp_matrix`` (lines 112-146, the "unbiased" UDP transform: the box edges map to
   the centers of the first and last input pixels). The same matrix is applied with
   ``cv2.warpAffine`` (bilinear, black border), which transformers' ``scipy_warp_affine``
   emulates (lines 149-172). RGB, scaled to 0..1 and normalized with the ImageNet mean / std
   (lines 343-348, or the values in the model's ``preprocessor_config.json``).
3. Heatmaps. ``get_keypoint_predictions`` (lines 175-205): the argmax of each heatmap; the
   score is the heatmap maximum, and a maximum <= 0 means "no prediction" (line 204).
   ``post_dark_unbiased_data_processing`` (lines 208-265): Gaussian blur (sigma 0.8, kernel
   11 = ``kernel_size`` of ``post_process_pose_estimation``), log, then one Newton step on the
   log heatmap for the sub-pixel position. The Newton step is limited to 1 heatmap pixel (a
   real peak is always within 0.5 px of its argmax; only flat or broken heatmaps hit the
   limit, where transformers would return a far-away point).
4. Back to the image. ``transform_preds`` (lines 268-313): heatmap pixel (0 .. W-1) maps
   linearly onto the padded box (left edge .. right edge). So the keypoints are in ORIGINAL
   image pixels; nothing else is rescaled.

``post_process_pose_estimation`` itself is not used. Its ``target_sizes`` argument reads
(width, height) although it documents (height, width) (lines 501-504), and its ``"bbox"``
output is the center and scale, not a box (lines 512-518). ``Person2D.bbox`` is the person
detector's box instead.

Keypoint order and scores
-------------------------
* The keypoints are mapped BY NAME from the model's ``config.id2label``. The official
  checkpoints use ``Nose, L_Eye, R_Eye, L_Ear, R_Ear, L_Shoulder, R_Shoulder, ...,
  R_Ankle`` (``convert_vitpose_to_hf.py``, ``get_config``, lines 129-147; ``L_`` / ``R_`` are
  the person's own left / right, as in COCO), i.e. ``FORMATS["coco17"]`` index for index.
  ``L_Shoulder`` becomes ``left_shoulder``. A model with generic labels (``LABEL_0`` ...) is
  taken in the order of the format with the same number of points (17: COCO-17).
* Scores: ViTPose is trained with an MSE loss on Gaussian target heatmaps whose peak is 1, so
  the heatmap maximum is a confidence of about 0 (nothing) .. 1 (sure); the official
  checkpoints give about 0.85-0.9 for clearly visible joints (``convert_vitpose_to_hf.py``,
  lines 309-373). It is not a calibrated probability. PoseBoard clips it to 0..1: values
  slightly above 1 become 1, a maximum <= 0 or NaN gives a missing keypoint (NaN, score 0).
* Person score: the person detector's confidence (RT-DETR: sigmoid class probability; YOLO:
  box confidence), already 0..1.

Only numpy, OpenCV and PoseBoard modules are imported with this module; torch, transformers and
ultralytics are imported when the detector is created.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from poseboard.pose.detectors.base import Detector2D, Person2D
from poseboard.pose.formats import COCO17, FORMATS, KeypointFormat, get_format

__all__ = [
    "DEFAULT_MODEL", "DEFAULT_RTDETR", "DEFAULT_YOLO", "HF_COCO_LABELS", "IMAGENET_MEAN",
    "IMAGENET_STD", "PERSON_DETECTORS", "VITPOSE_MODELS", "HFVitPoseModel",
    "RTDetrPersonDetector", "VitPoseHFDetector", "WeightsUnavailable", "YoloPersonDetector",
    "box_to_center_and_size", "canonical_keypoint_name", "clean_person_boxes", "create",
    "crop_person", "dark_refine", "flip_back", "flip_permutation", "heatmap_maxima",
    "keypoint_index_map", "keypoint_scores", "keypoints_from_heatmaps", "model_keypoint_names",
    "normalize_crops", "people_from_keypoints", "person_label_ids", "resolve_device",
    "resolve_format", "resolve_yolo_weights", "to_rgb", "warp_matrix", "xyxy_to_xywh",
]

log = logging.getLogger(__name__)

DEFAULT_MODEL = "usyd-community/vitpose-base-simple"
VITPOSE_MODELS = (
    "usyd-community/vitpose-base-simple", "usyd-community/vitpose-base",
    "usyd-community/vitpose-plus-small", "usyd-community/vitpose-plus-base",
    "usyd-community/vitpose-plus-large", "usyd-community/vitpose-plus-huge",
)
DEFAULT_RTDETR = "PekingU/rtdetr_r50vd_coco_o365"
DEFAULT_YOLO = "yolo11n.pt"
PERSON_DETECTORS = ("rtdetr", "yolo", "none")
HF_URL = "https://huggingface.co"

# config.id2label of the official checkpoints (convert_vitpose_to_hf.py, get_config, lines
# 129-147 in the transformers GitHub repository): the COCO-17 order.
HF_COCO_LABELS = (
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder", "R_Shoulder", "L_Elbow",
    "R_Elbow", "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
)
# VitPoseImageProcessor defaults (image_processing_vitpose.py, lines 343-349)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = (256, 192)  # (height, width) of the official models
PADDING_FACTOR = 1.25  # box_to_center_and_scale(padding_factor=1.25)
DARK_KERNEL = 11  # post_process_pose_estimation(kernel_size=11)
DARK_SIGMA = 0.8  # post_dark_unbiased_data_processing: gaussian_filter(sigma=0.8)
MAX_DARK_STEP = 1.0  # heatmap pixels (see the module docstring)


class WeightsUnavailable(RuntimeError):
    """A model is not on this computer and cannot be downloaded."""


# ================================================================== keypoint names / format
def canonical_keypoint_name(label) -> str:
    """"L_Shoulder" -> "left_shoulder", "R_Big_Toe" -> "right_big_toe", "Nose" -> "nose",
    "LeftKnee" -> "left_knee"."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(label).strip())
    s = re.sub(r"[\s\-\.]+", "_", s.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    s = re.sub(r"^l_", "left_", s)
    s = re.sub(r"^r_", "right_", s)
    return s


def model_keypoint_names(id2label: dict | Sequence | None, num_labels: int) -> tuple[str, ...]:
    """Canonical keypoint names of the model's heatmap channels 0 .. num_labels-1."""
    if isinstance(id2label, dict):
        lut = {int(k): v for k, v in id2label.items()}
        labels = [lut.get(i, f"LABEL_{i}") for i in range(num_labels)]
    elif id2label is not None:
        labels = list(id2label)
        labels += [f"LABEL_{i}" for i in range(len(labels), num_labels)]
    else:
        labels = [f"LABEL_{i}" for i in range(num_labels)]
    return tuple(canonical_keypoint_name(x) for x in labels[:num_labels])


def keypoint_index_map(names: Sequence[str], fmt: KeypointFormat) -> np.ndarray:
    """For each keypoint of ``fmt``: its channel in the model output (by name), -1 if the model
    does not have it."""
    names = list(names)
    return np.array([names.index(n) if n in names else -1 for n in fmt.names], np.int64)


def resolve_format(names: Sequence[str], keypoint_format: str | KeypointFormat | None = None
                   ) -> tuple[KeypointFormat, np.ndarray]:
    """(format, index map) for a model with the canonical channel ``names``.

    * A format all of whose names the model has (the one with the same number of points
      first): mapped by name.
    * Generic names (``label_0`` ...) or an explicit ``keypoint_format`` with the same number
      of points: taken in the model's order (logged).
    * Otherwise ValueError.
    """
    names = tuple(names)
    k = len(names)
    cands = [get_format(keypoint_format)] if keypoint_format else list(FORMATS.values())
    have = set(names)
    for exact in (True, False):
        for fmt in cands:
            if set(fmt.names) <= have and (len(fmt) == k or not exact):
                return fmt, keypoint_index_map(names, fmt)
    generic = all(re.fullmatch(r"label_\d+", n) for n in names)
    if generic or keypoint_format:
        for fmt in cands:
            if len(fmt) == k:
                log.warning("ViTPose model keypoint labels %s do not name the %s keypoints; "
                            "assuming the %s order", names[:3] + ("...",), fmt.key, fmt.key)
                return fmt, np.arange(k, dtype=np.int64)
    raise ValueError(
        f"The ViTPose model has {k} keypoints ({', '.join(names[:5])}, ...) that match none of "
        f"the PoseBoard formats ({', '.join(FORMATS)}); pass keypoint_format=<format key>.")


def flip_permutation(names: Sequence[str]) -> np.ndarray:
    """``perm[i]`` = channel of the mirror image of channel ``i`` (left <-> right; central
    points map to themselves). ValueError when a point's mirror cannot be derived from the
    names (e.g. the numbered face points of COCO-WholeBody)."""
    names = list(names)
    perm = np.arange(len(names))
    for i, n in enumerate(names):
        if n.startswith("left_"):
            partner = "right_" + n[5:]
        elif n.startswith("right_"):
            partner = "left_" + n[6:]
        elif re.search(r"\d", n):
            raise ValueError(f"flip_test: cannot tell the mirror keypoint of {n!r}")
        else:
            continue
        if partner not in names:
            raise ValueError(f"flip_test: keypoint {n!r} has no {partner!r}")
        perm[i] = names.index(partner)
    return perm


# ================================================================== image and boxes
def to_rgb(image) -> np.ndarray | None:
    """BGR / BGRA / gray image -> contiguous RGB uint8; None for an empty or tiny image."""
    import cv2

    if image is None:
        return None
    img = np.asarray(image)
    if img.ndim not in (2, 3) or img.size == 0 or min(img.shape[:2]) < 2:
        return None
    if img.dtype != np.uint8:
        img = np.clip(np.nan_to_num(img.astype(np.float64)), 0, 255).astype(np.uint8)
    if img.ndim == 2 or img.shape[2] == 1:
        return cv2.cvtColor(img.reshape(img.shape[:2]), cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    if img.shape[2] != 3:
        return None
    return np.ascontiguousarray(img[:, :, ::-1])


def xyxy_to_xywh(boxes) -> np.ndarray:
    """(N, 4) x1, y1, x2, y2 -> COCO x, y, width, height (width = x2 - x1, as the transformers
    ViTPose documentation converts the RT-DETR boxes)."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4).copy()
    b[:, 2:] -= b[:, :2]
    return b


def box_to_center_and_size(box_xywh, input_size: tuple[int, int] = INPUT_SIZE,
                           padding: float = PADDING_FACTOR) -> tuple[np.ndarray, np.ndarray]:
    """COCO box -> (center (2,), padded size (2,) = width, height in pixels).

    ``box_to_center_and_scale`` of transformers (image_processing_vitpose.py, lines 68-109)
    with ``image_width, image_height`` = the model input size: the shorter side is widened to
    the input aspect ratio, then both are multiplied by ``padding``. transformers returns
    ``size / normalize_factor``; this returns the size itself.
    """
    x, y, w, h = (float(v) for v in np.asarray(box_xywh, np.float64).reshape(-1)[:4])
    in_h, in_w = input_size
    aspect = in_w / in_h
    center = np.array([x + w * 0.5, y + h * 0.5])
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    return center, np.array([w, h]) * padding


def warp_matrix(center, size, input_size: tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    """2x3 affine matrix from image pixels to model input pixels (no rotation).

    ``get_warp_matrix(0, center * 2, (in_w - 1, in_h - 1), size)`` of transformers (lines
    112-146, called on lines 395-397): the padded box edges map to the centers of the first
    and last input pixels (UDP)."""
    in_h, in_w = input_size
    sx = (in_w - 1.0) / float(size[0])
    sy = (in_h - 1.0) / float(size[1])
    return np.array([[sx, 0.0, sx * (-float(center[0]) + 0.5 * float(size[0]))],
                     [0.0, sy, sy * (-float(center[1]) + 0.5 * float(size[1]))]])


def crop_person(image_rgb: np.ndarray, center, size,
                input_size: tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    """The padded person box warped onto the model input (in_h, in_w, 3), bilinear, black
    outside the image (``cv2.warpAffine``, which transformers' ``scipy_warp_affine``
    emulates)."""
    import cv2

    in_h, in_w = input_size
    m = warp_matrix(center, size, input_size)
    return cv2.warpAffine(image_rgb, m, (int(in_w), int(in_h)), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def normalize_crops(crops: Sequence[np.ndarray], mean=IMAGENET_MEAN, std=IMAGENET_STD,
                    rescale: float = 1.0 / 255.0) -> np.ndarray:
    """RGB uint8 crops -> ``pixel_values`` (B, 3, H, W) float32: ``(x * rescale - mean) /
    std``."""
    x = np.stack([np.asarray(c) for c in crops]).astype(np.float32) * np.float32(rescale)
    x -= np.asarray(mean, np.float32).reshape(1, 1, 1, 3)
    x /= np.asarray(std, np.float32).reshape(1, 1, 1, 3)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def clean_person_boxes(boxes, scores, image_size: tuple[int, int], max_people: int = 10,
                       nms_iou: float = 0.7, min_size: float = 2.0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Person boxes (x1, y1, x2, y2) clipped to the image (``image_size`` = (h, w)), without
    non-finite or tiny boxes and duplicates (IoU > ``nms_iou``), best score first, at most
    ``max_people``. Scores are clipped to 0..1."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    s = np.asarray(scores, np.float64).reshape(-1)
    if len(s) != len(b):
        raise ValueError(f"{len(b)} boxes but {len(s)} scores")
    h, w = image_size
    ok = np.all(np.isfinite(b), axis=1) & np.isfinite(s)
    b, s = b[ok], np.clip(s[ok], 0.0, 1.0)
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, float(w))
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, float(h))
    ok = ((b[:, 2] - b[:, 0]) >= min_size) & ((b[:, 3] - b[:, 1]) >= min_size)
    b, s = b[ok], s[ok]
    order = np.argsort(-s, kind="stable")
    keep: list[int] = []
    for i in order:
        if len(keep) >= max(0, int(max_people)):
            break
        if all(_iou(b[i], b[j]) <= nms_iou for j in keep):
            keep.append(int(i))
    return b[keep].reshape(-1, 4), s[keep]


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


# ================================================================== heatmaps -> keypoints
def heatmap_maxima(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(coords (B, K, 2) integer heatmap x, y of the maximum; -1 where the maximum is <= 0 or
    not finite, scores (B, K) = the maximum). ``get_keypoint_predictions`` of transformers
    (lines 175-205)."""
    hm = np.asarray(heatmaps, np.float64)
    if hm.ndim != 4:
        raise ValueError(f"heatmaps must be (batch, keypoints, height, width), not {hm.shape}")
    b, k, _h, w = hm.shape
    flat = np.where(np.isfinite(hm), hm, -np.inf).reshape(b, k, -1)
    idx = np.argmax(flat, axis=2)
    scores = np.take_along_axis(flat, idx[..., None], axis=2)[..., 0]
    coords = np.stack([idx % w, idx // w], axis=-1).astype(np.float64)
    coords[~(scores > 0.0)] = -1.0
    return coords, scores


def dark_refine(coords: np.ndarray, heatmaps: np.ndarray, kernel: int = DARK_KERNEL,
                sigma: float = DARK_SIGMA, max_step: float = MAX_DARK_STEP) -> np.ndarray:
    """Sub-pixel refinement of the integer maxima ``coords`` (B, K, 2): DARK / UDP,
    ``post_dark_unbiased_data_processing`` of transformers (lines 208-265). The Gaussian blur
    uses ``cv2.GaussianBlur`` with the same kernel and the same border (scipy "reflect" =
    ``cv2.BORDER_REFLECT``). Points with coords -1 are returned unchanged."""
    import cv2

    hm = np.ascontiguousarray(heatmaps, np.float32)
    b, k, h, w = hm.shape
    ksize = 2 * ((int(kernel) - 1) // 2) + 1
    blurred = np.empty_like(hm)
    for i in range(b):
        for j in range(k):
            blurred[i, j] = cv2.GaussianBlur(hm[i, j], (ksize, ksize), sigma,
                                             borderType=cv2.BORDER_REFLECT)
    logh = np.log(np.clip(np.nan_to_num(blurred, nan=0.001), 0.001, 50.0)).astype(np.float64)
    pad = np.pad(logh, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    out = np.array(coords, np.float64, copy=True)
    valid = np.all(out >= 0, axis=-1)
    x = np.where(valid, np.clip(out[..., 0], 0, w - 1), 0).astype(np.int64) + 1
    y = np.where(valid, np.clip(out[..., 1], 0, h - 1), 0).astype(np.int64) + 1
    bi, ki = np.indices((b, k))

    def at(dy, dx):
        return pad[bi, ki, y + dy, x + dx]

    i_ = at(0, 0)
    ix1, ix1_ = at(0, 1), at(0, -1)
    iy1, iy1_ = at(1, 0), at(-1, 0)
    ix1y1, ix1_y1_ = at(1, 1), at(-1, -1)
    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    eps = float(np.finfo(np.float32).eps)  # hessian + eps * I, as transformers
    a, d = dxx + eps, dyy + eps
    det = a * d - dxy * dxy
    with np.errstate(divide="ignore", invalid="ignore"):
        step_x = (d * dx - dxy * dy) / det
        step_y = (-dxy * dx + a * dy) / det
    step = np.stack([step_x, step_y], axis=-1)
    step = np.where(np.isfinite(step), np.clip(step, -max_step, max_step), 0.0)
    out[valid] -= step[valid]
    return out


def keypoints_from_heatmaps(heatmaps: np.ndarray, centers: np.ndarray, sizes: np.ndarray,
                            kernel: int = DARK_KERNEL) -> tuple[np.ndarray, np.ndarray]:
    """(keypoints (B, K, 2) in ORIGINAL image pixels, NaN where the model predicts nothing;
    raw scores (B, K) = heatmap maxima).

    ``centers`` / ``sizes`` (B, 2): the padded boxes of the crops (``box_to_center_and_size``).
    Heatmap pixel 0 .. W-1 maps onto the padded box from its left to its right edge
    (``transform_preds`` of transformers, lines 268-313).
    """
    hm = np.asarray(heatmaps, np.float64)
    b, _k, h, w = hm.shape
    coords, raw = heatmap_maxima(hm)
    coords = dark_refine(coords, hm, kernel=kernel)
    c = np.asarray(centers, np.float64).reshape(b, 1, 2)
    s = np.asarray(sizes, np.float64).reshape(b, 1, 2)
    kp = np.empty_like(coords)
    kp[..., 0] = coords[..., 0] * s[..., 0] / (w - 1.0) + c[..., 0] - s[..., 0] * 0.5
    kp[..., 1] = coords[..., 1] * s[..., 1] / (h - 1.0) + c[..., 1] - s[..., 1] * 0.5
    missing = ~(raw > 0.0) | ~np.all(np.isfinite(kp), axis=-1)
    kp[missing] = np.nan
    return kp, raw


def keypoint_scores(raw) -> np.ndarray:
    """Heatmap maxima -> 0..1 (clipped; NaN / <= 0 -> 0). See the module docstring."""
    r = np.asarray(raw, np.float64)
    return np.where(np.isfinite(r), np.clip(r, 0.0, 1.0), 0.0)


def flip_back(heatmaps_flipped: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Heatmaps of the horizontally flipped crop -> heatmaps of the crop: swap left / right
    channels and flip horizontally (``flip_back`` in transformers' modeling_vitpose.py, lines
    78-117, for gaussian heatmaps)."""
    return np.asarray(heatmaps_flipped)[:, np.asarray(perm)][..., ::-1]


def people_from_keypoints(keypoints: np.ndarray, raw_scores: np.ndarray, boxes_xyxy,
                          box_scores, index_map: np.ndarray,
                          fmt: KeypointFormat = COCO17) -> list[Person2D]:
    """One ``Person2D`` per box, in ``fmt`` order (``index_map[i]`` = model channel of
    keypoint ``i`` of ``fmt``, -1: missing)."""
    kp_all = np.asarray(keypoints, np.float64)
    sc_all = keypoint_scores(raw_scores)
    boxes = np.asarray(boxes_xyxy, np.float64).reshape(-1, 4)
    bscore = np.asarray(box_scores, np.float64).reshape(-1)
    src = np.asarray(index_map, np.int64)
    has = src >= 0
    people = []
    for i in range(len(kp_all)):
        kp = np.full((len(fmt), 2), np.nan)
        sc = np.zeros(len(fmt))
        kp[has] = kp_all[i, src[has]]
        sc[has] = sc_all[i, src[has]]
        bad = ~np.all(np.isfinite(kp), axis=1)
        kp[bad] = np.nan
        sc[bad] = 0.0
        people.append(Person2D(kp, sc, bbox=boxes[i].copy(),
                               score=float(np.clip(np.nan_to_num(bscore[i]), 0.0, 1.0))))
    return people


# ================================================================== torch helpers
def resolve_device(device: str | None = "auto") -> str:
    """torch device string: "auto" -> "cuda" if PyTorch sees a CUDA GPU, else "cpu";
    "cuda" / "gpu" / "cuda:N" -> RuntimeError without that GPU; "mps" checked; "cpu"."""
    import torch

    d = "auto" if device is None else str(device).strip().lower()
    if d in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d in ("cuda", "gpu") or d.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"ViTPose: device {device!r} needs a CUDA GPU, but PyTorch has no CUDA "
                "(torch.cuda.is_available() is False). Install a CUDA build of PyTorch or use "
                "device 'cpu' / 'auto'.")
        if d.startswith("cuda:"):
            idx = d.split(":", 1)[1]
            if not idx.isdigit() or int(idx) >= torch.cuda.device_count():
                raise RuntimeError(f"ViTPose: no CUDA device {device!r} "
                                   f"({torch.cuda.device_count()} GPU(s) found)")
            return d
        return "cuda"
    if d == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("ViTPose: device 'mps' is not available on this computer")
        return d
    if d == "cpu":
        return d
    raise ValueError(f"ViTPose: unknown device {device!r} (use auto, cpu or cuda)")


def _to_numpy(x) -> np.ndarray:
    """torch tensor (any device / dtype) or array-like -> float64 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "float") and hasattr(x, "cpu"):
        x = x.float().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, np.float64)


def _is_download_error(e: BaseException) -> bool:
    """True for errors that mean "the files are not here and cannot be fetched"."""
    if isinstance(e, (OSError, ConnectionError, TimeoutError)):
        return True
    root = type(e).__module__.split(".", 1)[0]
    return root in ("huggingface_hub", "requests", "httpx", "httpcore", "urllib3")


def _unavailable(what: str, repo: str, e: BaseException, cache_dir) -> WeightsUnavailable:
    first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    where = f"{cache_dir}" if cache_dir else "the Hugging Face cache (~/.cache/huggingface/hub)"
    return WeightsUnavailable(
        f"Cannot load the {what} {repo!r} ({type(e).__name__}: {first}).\n"
        f"It is downloaded from {HF_URL}/{repo} on first use into {where}. Without access to "
        f"huggingface.co, download the model folder on another computer (for example with "
        f"`huggingface-cli download {repo} --local-dir <folder>`) and give that folder as the "
        f"model option, or set HF_ENDPOINT to a Hugging Face mirror.")


def _pair(v) -> tuple[int, int]:
    if isinstance(v, (int, float)):
        return int(v), int(v)
    v = list(v)
    return int(v[0]), int(v[1])


class HFVitPoseModel:
    """``VitPoseForPoseEstimation`` + the values of its image processor. ``__call__`` maps
    ``pixel_values`` (B, 3, H, W) float32 numpy -> heatmaps (B, K, h, w) float32 numpy."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu",
                 dataset_index: int | None = None, cache_dir: str | None = None,
                 local_files_only: bool = False):
        import torch
        from transformers import VitPoseForPoseEstimation

        self.model_id = str(model)
        self.device = device
        kw = {"cache_dir": cache_dir or None, "local_files_only": bool(local_files_only)}
        try:
            net = VitPoseForPoseEstimation.from_pretrained(self.model_id, **kw)
        except Exception as e:
            if _is_download_error(e):
                raise _unavailable("ViTPose model", self.model_id, e, cache_dir) from e
            raise
        self.model = net.float().to(device).eval()
        self.model.requires_grad_(False)
        cfg = self.model.config
        bb = cfg.backbone_config
        self.input_size = _pair(getattr(bb, "image_size", INPUT_SIZE))  # (height, width)
        self.num_experts = int(getattr(bb, "num_experts", 1) or 1)
        self.names = model_keypoint_names(getattr(cfg, "id2label", None), int(cfg.num_labels))
        if self.num_experts > 1:
            idx = 0 if dataset_index is None else int(dataset_index)
            if not 0 <= idx < self.num_experts:
                raise ValueError(f"ViTPose: dataset_index must be 0..{self.num_experts - 1}, "
                                 f"not {dataset_index!r}")
            self.dataset_index: int | None = idx
        else:
            if dataset_index not in (None, 0):
                log.info("ViTPose: %s has one expert; dataset_index %s ignored", self.model_id,
                         dataset_index)
            self.dataset_index = None
        self.mean, self.std, self.rescale = IMAGENET_MEAN, IMAGENET_STD, 1.0 / 255.0
        try:  # mean / std of the checkpoint (preprocessor_config.json); defaults otherwise
            from transformers import AutoImageProcessor

            proc = AutoImageProcessor.from_pretrained(self.model_id, **kw)
            if getattr(proc, "do_normalize", True):
                self.mean = tuple(float(v) for v in getattr(proc, "image_mean", self.mean))
                self.std = tuple(float(v) for v in getattr(proc, "image_std", self.std))
            else:
                self.mean, self.std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
            if getattr(proc, "do_rescale", True):
                self.rescale = float(getattr(proc, "rescale_factor", self.rescale))
            else:
                self.rescale = 1.0
        except Exception as e:  # noqa: BLE001  (no preprocessor_config.json: the defaults)
            log.debug("ViTPose image processor of %s not loaded (%s); ImageNet defaults",
                      self.model_id, e)
        self._torch = torch

    def __call__(self, pixel_values: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(pixel_values, np.float32)).to(self.device)
        kw = {}
        if self.dataset_index is not None:
            kw["dataset_index"] = torch.full((x.shape[0],), self.dataset_index,
                                             dtype=torch.long, device=x.device)
        with torch.inference_mode():
            out = self.model(pixel_values=x, **kw)
        return out.heatmaps.float().cpu().numpy()


# ================================================================== person detectors
def person_label_ids(id2label: dict | None) -> list[int]:
    """Class ids labelled "person" (case-insensitive). Generic labels (``LABEL_0`` ...): [0]
    (COCO). ValueError when the detector has no person class."""
    lut = {int(k): str(v) for k, v in (id2label or {}).items()}
    ids = sorted(i for i, v in lut.items() if v.strip().lower() in ("person", "people", "human"))
    if ids:
        return ids
    if not lut or all(re.fullmatch(r"(?i)label_\d+", v) for v in lut.values()):
        return [0]
    raise ValueError("the person detector has no 'person' class "
                     f"(labels: {', '.join(list(lut.values())[:8])}, ...)")


class RTDetrPersonDetector:
    """transformers object detector (default RT-DETR R50, COCO + Objects365): ``__call__``
    maps an RGB image to (boxes (N, 4) x1, y1, x2, y2 in its pixels, scores (N,)) of the
    persons with a score >= ``threshold``.

    ``post_process_object_detection`` with ``target_sizes=[(height, width)]`` scales the
    relative (cx, cy, w, h) boxes to absolute x1, y1, x2, y2 of the original image
    (transformers/models/rt_detr/image_processing_rt_detr.py, lines 511-523 in 5.17)."""

    def __init__(self, model: str = DEFAULT_RTDETR, device: str = "cpu",
                 threshold: float = 0.3, cache_dir: str | None = None,
                 local_files_only: bool = False):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.model_id = str(model)
        self.device = device
        self.threshold = float(threshold)
        kw = {"cache_dir": cache_dir or None, "local_files_only": bool(local_files_only)}
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id, **kw)
            net = AutoModelForObjectDetection.from_pretrained(self.model_id, **kw)
        except Exception as e:
            if _is_download_error(e):
                raise _unavailable("person detector", self.model_id, e, cache_dir) from e
            raise
        self.model = net.float().to(device).eval()
        self.model.requires_grad_(False)
        self.person_ids = person_label_ids(getattr(self.model.config, "id2label", None))
        self._torch = torch

    def __call__(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        h, w = image_rgb.shape[:2]
        inputs = self.processor(images=image_rgb, return_tensors="pt",
                                input_data_format="channels_last")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.inference_mode():
            out = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            out, threshold=self.threshold, target_sizes=[(int(h), int(w))])[0]
        keep = np.isin(_to_numpy(res["labels"]).astype(np.int64).reshape(-1), self.person_ids)
        boxes = _to_numpy(res["boxes"]).reshape(-1, 4)[keep]
        return boxes, _to_numpy(res["scores"]).reshape(-1)[keep]


# official Ultralytics detection weights (release asset names)
_YOLO_NAME = re.compile(r"^(?:(?:yolo11|yolov8|yolo12|yolo26)[nsmlx]|yolov5[nsmlx]u|yolov9[tsmce]|"
                        r"yolov10[nsmblx])\.pt$", re.IGNORECASE)


def resolve_yolo_weights(model: str = DEFAULT_YOLO, weights_dir: str | Path | None = None,
                         download: bool = True) -> Path:
    """Local path of YOLO detection weights: an official file name (``yolo11n.pt``,
    ``yolov8s.pt``, ...) in ``weights_dir`` (default: the PoseBoard models folder), downloaded
    from the Ultralytics GitHub release assets if missing; anything else must be an existing
    file."""
    from poseboard.pose.detectors import ultralytics_det as ud

    folder = Path(weights_dir) if weights_dir else ud.models_dir()
    name = str(model).strip()
    if not _YOLO_NAME.match(name):
        p = Path(name).expanduser()
        for cand in (p, folder / p):
            if cand.is_file():
                return cand.resolve()
        raise FileNotFoundError(f"YOLO person detector weights {model!r} not found (use an "
                                "official name such as yolo11n.pt or the path of a file)")
    target = folder / name
    if target.is_file():
        return target
    if not download:
        raise FileNotFoundError(f"{target} does not exist")
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        target.parent.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(target))
        if not target.is_file():
            raise RuntimeError("not found in the Ultralytics release assets")
    except Exception as e:  # offline, proxy, disk
        raise WeightsUnavailable(
            f"Cannot download the YOLO person detector {name} ({e}). Download it from "
            f"{ud.RELEASES_URL} and save it as {target}") from e
    return target


class YoloPersonDetector:
    """Ultralytics YOLO detection model used only for its person boxes (AGPL-3.0).
    ``Results.boxes.xyxy`` is already in original image pixels (ultralytics scales the boxes
    back from the letterboxed input in ``DetectionPredictor.postprocess``)."""

    def __init__(self, model: str = DEFAULT_YOLO, device: str = "cpu", threshold: float = 0.3,
                 max_det: int = 20, weights_dir: str | Path | None = None):
        from ultralytics import YOLO

        from poseboard.pose.detectors import ultralytics_det as ud

        ud._disable_analytics()  # no usage events (see ultralytics_det)
        self.weights_path = resolve_yolo_weights(model, weights_dir)
        self.yolo = YOLO(str(self.weights_path))
        names = getattr(self.yolo, "names", None) or {}
        self.person_ids = person_label_ids(names if isinstance(names, dict)
                                           else dict(enumerate(names)))
        self.threshold = float(threshold)
        self.max_det = int(max_det)
        self.device = "cuda:0" if device == "cuda" else device  # ultralytics device string

    def __call__(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])  # ultralytics expects BGR arrays
        res = self.yolo.predict(bgr, classes=self.person_ids, conf=self.threshold,
                                device=self.device, max_det=self.max_det, verbose=False)
        boxes = getattr(res[0], "boxes", None) if res else None
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return np.zeros((0, 4)), np.zeros(0)
        xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
        conf = getattr(boxes, "conf", None)
        conf = np.ones(len(xyxy)) if conf is None else _to_numpy(conf).reshape(-1)
        return xyxy, conf


def _whole_image(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_rgb.shape[:2]
    return np.array([[0.0, 0.0, float(w), float(h)]]), np.ones(1)


# ================================================================== detector
PoseFn = Callable[[np.ndarray], np.ndarray]
PersonFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


class VitPoseHFDetector(Detector2D):
    """Person detector + ViTPose (transformers). See the module docstring."""

    key = "vitpose_hf"
    label = "ViTPose (Hugging Face transformers)"
    format = COCO17
    provides_3d = False

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto",
                 person_detector: str = "rtdetr", detector_model: str | None = None,
                 detector_threshold: float = 0.3, max_people: int = 10, flip_test: bool = False,
                 dataset_index: int | None = None, keypoint_format: str | None = None,
                 batch_size: int = 8, cache_dir: str | None = None,
                 weights_dir: str | None = None, local_files_only: bool = False):
        super().__init__()
        pd = str(person_detector or "rtdetr").strip().lower()
        if pd not in PERSON_DETECTORS:
            raise ValueError(f"person_detector must be one of {', '.join(PERSON_DETECTORS)}, "
                             f"not {person_detector!r}")
        self.device = resolve_device(device)
        pose = HFVitPoseModel(model, self.device, dataset_index=dataset_index,
                              cache_dir=cache_dir, local_files_only=local_files_only)
        if pd == "rtdetr":
            det_model = detector_model or DEFAULT_RTDETR
            persons: PersonFn = RTDetrPersonDetector(
                det_model, self.device, threshold=detector_threshold, cache_dir=cache_dir,
                local_files_only=local_files_only)
        elif pd == "yolo":
            det_model = detector_model or DEFAULT_YOLO
            persons = YoloPersonDetector(det_model, self.device, threshold=detector_threshold,
                                         max_det=max(20, int(max_people)),
                                         weights_dir=weights_dir)
        else:
            det_model = ""
            persons = _whole_image
        self._setup(pose, persons, names=pose.names, input_size=pose.input_size,
                    mean=pose.mean, std=pose.std, rescale=pose.rescale,
                    keypoint_format=keypoint_format, flip_test=flip_test,
                    batch_size=batch_size, max_people=max_people)
        self.person_detector = pd
        self.detector_model = str(det_model)
        self.model_id = pose.model_id
        self.dataset_index = pose.dataset_index
        short = self.model_id.rstrip("/").replace("\\", "/").rsplit("/", 1)[-1]
        self.label = f"ViTPose ({short}, transformers)"
        self.options = {"model": str(model), "device": str(device), "person_detector": pd,
                        "detector_model": self.detector_model,
                        "detector_threshold": float(detector_threshold),
                        "max_people": int(max_people), "flip_test": bool(flip_test),
                        "dataset_index": pose.dataset_index,
                        "keypoint_format": self.format.key, "batch_size": int(batch_size)}

    @classmethod
    def from_components(cls, pose_model: PoseFn, person_detector: PersonFn | None, *,
                        keypoint_names: Sequence[str] = HF_COCO_LABELS,
                        input_size: tuple[int, int] = INPUT_SIZE, mean=IMAGENET_MEAN,
                        std=IMAGENET_STD, rescale: float = 1.0 / 255.0,
                        keypoint_format: str | None = None, flip_test: bool = False,
                        batch_size: int = 8, max_people: int = 10) -> VitPoseHFDetector:
        """A detector around any heatmap model and person detector (tests, custom models).
        ``pose_model``: pixel_values (B, 3, H, W) float32 -> heatmaps (B, K, h, w);
        ``person_detector``: RGB image -> (boxes x1, y1, x2, y2 (N, 4), scores (N,)); None:
        the whole image. ``keypoint_names``: the model's channel labels."""
        self = cls.__new__(cls)
        Detector2D.__init__(self)
        self.device = "cpu"
        names = tuple(canonical_keypoint_name(n) for n in keypoint_names)
        self._setup(pose_model, person_detector or _whole_image, names=names,
                    input_size=_pair(input_size), mean=mean, std=std, rescale=rescale,
                    keypoint_format=keypoint_format, flip_test=flip_test,
                    batch_size=batch_size, max_people=max_people)
        self.person_detector = "custom" if person_detector else "none"
        self.detector_model = ""
        self.model_id = "custom"
        self.dataset_index = None
        self.options = {"keypoint_format": self.format.key, "flip_test": bool(flip_test),
                        "max_people": int(max_people), "batch_size": int(batch_size)}
        return self

    def _setup(self, pose: PoseFn, persons: PersonFn, *, names, input_size, mean, std,
               rescale, keypoint_format, flip_test, batch_size, max_people) -> None:
        self._pose: PoseFn | None = pose
        self._persons: PersonFn | None = persons
        self.keypoint_names_model = tuple(names)
        self.format, self._index_map = resolve_format(self.keypoint_names_model,
                                                      keypoint_format)
        self.input_size = _pair(input_size)
        self.mean, self.std, self.rescale = tuple(mean), tuple(std), float(rescale)
        self.flip_test = bool(flip_test)
        self._flip_perm = flip_permutation(self.keypoint_names_model) if flip_test else None
        self.batch_size = max(1, int(batch_size))
        self.max_people = max(1, int(max_people))

    # ------------------------------------------------------------------ inference
    def detect(self, image_bgr: np.ndarray, t: float, cam_name: str) -> list[Person2D]:
        if self._pose is None or self._persons is None:
            raise RuntimeError("ViTPose detector is closed")
        rgb = to_rgb(image_bgr)
        if rgb is None:
            return []
        boxes, scores = self._persons(rgb)
        boxes, scores = clean_person_boxes(boxes, scores, rgb.shape[:2], self.max_people)
        if not len(boxes):
            return []
        return self.estimate(rgb, boxes, scores)

    def estimate(self, image_rgb: np.ndarray, boxes_xyxy, box_scores=None) -> list[Person2D]:
        """ViTPose on given person boxes (x1, y1, x2, y2, original pixels) of an RGB image."""
        if self._pose is None:
            raise RuntimeError("ViTPose detector is closed")
        boxes = np.asarray(boxes_xyxy, np.float64).reshape(-1, 4)
        if not len(boxes):
            return []
        scores = np.ones(len(boxes)) if box_scores is None else \
            np.asarray(box_scores, np.float64).reshape(-1)
        if len(scores) != len(boxes):
            raise ValueError(f"{len(boxes)} boxes but {len(scores)} box scores")
        cs = [box_to_center_and_size(b, self.input_size) for b in xyxy_to_xywh(boxes)]
        centers = np.array([c for c, _ in cs])
        sizes = np.array([s for _, s in cs])
        heatmaps = []
        for i in range(0, len(boxes), self.batch_size):
            crops = [crop_person(image_rgb, c, s, self.input_size)
                     for c, s in zip(centers[i:i + self.batch_size],
                                     sizes[i:i + self.batch_size])]
            px = normalize_crops(crops, self.mean, self.std, self.rescale)
            hm = np.asarray(self._pose(px), np.float32)
            if self._flip_perm is not None:
                hm_f = np.asarray(self._pose(np.ascontiguousarray(px[..., ::-1])), np.float32)
                hm = 0.5 * (hm + flip_back(hm_f, self._flip_perm))
            heatmaps.append(hm)
        hm = np.concatenate(heatmaps, axis=0)
        if hm.shape[:2] != (len(boxes), len(self.keypoint_names_model)):
            raise ValueError(f"ViTPose returned heatmaps {hm.shape} for {len(boxes)} boxes and "
                             f"{len(self.keypoint_names_model)} keypoints")
        kp, raw = keypoints_from_heatmaps(hm, centers, sizes)
        return people_from_keypoints(kp, raw, boxes, scores, self._index_map, self.format)

    def close(self) -> None:
        was_open = self._pose is not None
        self._pose = None
        self._persons = None
        if was_open and str(getattr(self, "device", "cpu")).startswith("cuda"):
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                log.debug("torch.cuda.empty_cache: %s", e)

    def info(self) -> dict:
        d = super().info()
        d.update(device_used=self.device, model_id=self.model_id,
                 person_detector=self.person_detector, detector_model=self.detector_model,
                 flip_test=self.flip_test, dataset_index=self.dataset_index,
                 keypoint_score="heatmap maximum clipped to 0..1",
                 license="Apache-2.0" + (" (person detector: AGPL-3.0)"
                                         if self.person_detector == "yolo" else ""))
        return d


def create(model: str = DEFAULT_MODEL, device: str = "auto", person_detector: str = "rtdetr",
           **kwargs) -> VitPoseHFDetector:
    """Factory of the ``vitpose_hf`` backend.

    * ``model``: Hugging Face id or local folder of a ViTPose checkpoint (default
      ``usyd-community/vitpose-base-simple``; ViTPose++: ``usyd-community/vitpose-plus-*``).
    * ``device``: auto / cpu / cuda.
    * ``person_detector``: "rtdetr" (default, transformers RT-DETR), "yolo" (Ultralytics,
      AGPL-3.0) or "none" (whole image = one person).
    * Extra options: ``detector_model`` (RT-DETR id / folder or YOLO weights),
      ``detector_threshold`` (0.3), ``max_people`` (10), ``flip_test`` (False; also runs the
      mirrored crop and averages, as the official evaluation; twice as slow),
      ``dataset_index`` (ViTPose++ expert, default 0 = COCO), ``keypoint_format`` (custom
      models), ``batch_size`` (8 crops per forward pass), ``cache_dir`` (Hugging Face
      download folder), ``weights_dir`` (YOLO weights folder), ``local_files_only``.
    """
    return VitPoseHFDetector(model=model, device=device, person_detector=person_detector,
                             **kwargs)
