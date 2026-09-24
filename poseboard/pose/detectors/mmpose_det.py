"""MMPose 2D models through ``mmpose.apis.MMPoseInferencer``, as a ``Detector2D`` (backend key
``mmpose``).

Install (not part of PoseBoard's requirements; needs PyTorch)::

    pip install -U openmim && mim install mmengine "mmcv>=2.0.1" mmdet mmpose

Options:

* ``model`` (alias ``pose2d``): anything ``MMPoseInferencer(pose2d=...)`` accepts - a model
  alias (``"human"`` = RTMPose-m body, ``"rtmpose-l"``, ``"wholebody"``, ...), a config name
  from the MMPose model zoo (default ``td-hm_hrnet-w32_8xb64-210e_coco-256x192``, HRNet-W32
  COCO) or a config file path; ``weights`` (alias ``pose2d_weights``): a checkpoint path or URL
  (default: the model zoo checkpoint).
* ``det_model`` / ``det_weights`` / ``det_cat_ids``: the person detector of top-down models
  (default: MMPose's default for the model's dataset, e.g. RTMDet for people);
  ``bbox_thr`` (0.3) and ``nms_thr`` (0.3) its thresholds.
* ``device``: "auto" (``cuda:0`` when PyTorch sees a GPU, else ``cpu``), "cpu", "cuda", ...

MMPose downloads the configs' checkpoints from download.openmmlab.com on first use (into the
PyTorch hub cache); without access to that host pass local ``weights`` / ``det_weights``.

Result mapping (MMPose 1.x): ``next(inferencer(image_bgr))["predictions"][0]`` is the list of
instances of the image, made by ``mmpose/structures/utils.py`` ``split_instances`` (v1.3.2,
lines 116-136): ``keypoints`` (K x [x, y]) - already in ORIGINAL image pixels (top-down
models map them back from the person crop in ``TopdownPoseEstimator.add_pred_to_datasample``)
-, ``keypoint_scores`` (K), and for detected persons ``bbox`` (a 1-tuple holding
[x1, y1, x2, y2], note the trailing comma in line 131) and ``bbox_score``. The image is passed
as a BGR array, which is MMPose's convention for arrays (``Pose2DInferencer.preprocess_single``
v1.3.2 line 155; the model's data preprocessor converts to RGB).

Keypoint layout: from the model's ``dataset_meta`` (``keypoint_id2name``) when available, else
from the number of keypoints: 17 -> ``coco17`` (MMPose ``configs/_base_/datasets/coco.py``),
26 -> ``halpe26`` (``halpe26.py``), 133 -> ``wholebody133`` (``coco_wholebody.py``). The body
and foot names in those files are exactly PoseBoard's names (nose, left_eye, ...,
left_big_toe, ...; "left" = the person's left), so keypoints are mapped by name; the
whole-body face points ``face-0..67`` and hand points ``left_hand_root``, ``left_thumb1..4``,
``left_forefinger1..4``, ``left_middle_finger1..4``, ``left_ring_finger1..4``,
``left_pinky_finger1..4`` become ``face_0..67`` / ``left_hand_0..20`` (same for the right
hand). ``keypoint_format`` forces a layout (same count, model order).

Scores: MMPose's ``keypoint_scores`` are the heatmap maxima (heatmap heads such as HRNet: the
target Gaussians have peak 1) or the SimCC maxima (RTMPose, same scale); they are not
calibrated probabilities. They are mapped with ``clip(score / score_scale, 0, 1)``
(``score_scale`` = 1 by default, so MMPose's usual thresholds such as 0.3 keep their meaning).
``Person2D.score`` = ``bbox_score`` when present (top-down), else the mean keypoint score.
Keypoints with a score <= 0 or non-finite coordinates are NaN.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from poseboard.pose.detectors.base import Detector2D, Person2D, keypoint_bbox
from poseboard.pose.formats import FORMATS, KeypointFormat, get_format

__all__ = ["MMPoseDetector", "canonical_name", "create", "format_for", "keypoint_order",
           "people_from_instances", "resolve_device"]

log = logging.getLogger(__name__)

DEFAULT_MODEL = "td-hm_hrnet-w32_8xb64-210e_coco-256x192"
_BY_COUNT = {17: "coco17", 26: "halpe26", 133: "wholebody133"}
_FINGERS = ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger")


def canonical_name(name) -> str:
    """MMPose keypoint name -> PoseBoard name (``face-3`` -> ``face_3``, ``left_hand_root`` ->
    ``left_hand_0``, ``right_forefinger2`` -> ``right_hand_6``, others unchanged)."""
    n = re.sub(r"[\s\-]+", "_", str(name).strip().lower())
    m = re.fullmatch(r"(left|right)_hand_root", n)
    if m:
        return f"{m.group(1)}_hand_0"
    m = re.fullmatch(r"(left|right)_(" + "|".join(_FINGERS) + r")(\d)", n)
    if m:
        return f"{m.group(1)}_hand_{1 + 4 * _FINGERS.index(m.group(2)) + int(m.group(3)) - 1}"
    return n


def _meta_names(meta) -> list[str] | None:
    if not isinstance(meta, dict):
        return None
    id2name = meta.get("keypoint_id2name")
    if isinstance(id2name, dict) and id2name:
        try:
            return [str(id2name[i]) for i in range(len(id2name))]
        except KeyError:
            return None
    info = meta.get("keypoint_info")
    if isinstance(info, dict) and info:
        try:
            return [str(v["name"]) for _, v in sorted(info.items(), key=lambda kv: int(kv[0]))]
        except (KeyError, TypeError, ValueError):
            return None
    return None


def format_for(n_keypoints: int, names: list[str] | None = None,
               keypoint_format: str | KeypointFormat | None = None) -> KeypointFormat:
    """The PoseBoard format of a model with ``n_keypoints`` (and optional keypoint names)."""
    if keypoint_format:
        fmt = get_format(keypoint_format)
        if len(fmt) != n_keypoints:
            raise ValueError(f"keypoint_format {fmt.key} has {len(fmt)} keypoints, the MMPose "
                             f"model gives {n_keypoints}")
        return fmt
    if names:
        canon = {canonical_name(x) for x in names}
        for fmt in FORMATS.values():
            if len(fmt) == n_keypoints and set(fmt.names) <= canon:
                return fmt
    key = _BY_COUNT.get(int(n_keypoints))
    if key is None:
        raise ValueError(
            f"The MMPose model gives {n_keypoints} keypoints; PoseBoard supports body models "
            "with COCO-17, Halpe-26 or COCO-WholeBody-133 keypoints (or pass keypoint_format)")
    return FORMATS[key]


def keypoint_order(fmt: KeypointFormat, names: list[str] | None) -> list[int] | None:
    """Indices into the model output for the keypoints of ``fmt`` (None: model order). Names
    are matched after ``canonical_name``; ValueError when they contradict the format."""
    if not names:
        return None
    canon = [canonical_name(x) for x in names]
    if len(canon) != len(fmt):
        raise ValueError(f"{len(canon)} keypoint names for the {len(fmt)}-point {fmt.key} "
                         "format")
    if set(fmt.names) <= set(canon):
        order = [canon.index(x) for x in fmt.names]
    else:  # e.g. a custom dataset with other names: trust the count, keep the order
        log.warning("MMPose keypoint names do not match the %s format; assuming its order",
                    fmt.key)
        return None
    return None if order == list(range(len(fmt))) else order


def _array(x, shape_tail: tuple[int, ...]) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, np.float64).reshape((-1,) + shape_tail)


def people_from_instances(instances, n_keypoints: int, order: list[int] | None = None,
                          score_scale: float = 1.0) -> list[Person2D]:
    """``Person2D`` list from MMPose's per-image instance dicts (see the module docstring)."""
    people = []
    for inst in instances or []:
        if not isinstance(inst, dict) or inst.get("keypoints") is None:
            continue
        kp = _array(inst["keypoints"], (2,))
        if len(kp) == 0:
            continue
        if len(kp) != n_keypoints:
            raise ValueError(f"MMPose returned {len(kp)} keypoints, expected {n_keypoints}")
        raw = inst.get("keypoint_scores")
        raw = np.ones(len(kp)) if raw is None else _array(raw, ()).reshape(-1)
        sc = np.clip(np.nan_to_num(raw / score_scale, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        missing = ~np.all(np.isfinite(kp), axis=1) | (sc <= 0)
        kp = np.where(missing[:, None], np.nan, kp)
        sc = np.where(missing, 0.0, sc)
        if order is not None:
            kp, sc = kp[order], sc[order]
        box = inst.get("bbox")
        if box is not None:
            b = np.asarray(box, np.float64).reshape(-1)
            box = b[:4] if b.size >= 4 and np.all(np.isfinite(b[:4])) else None
        if box is None:
            box = keypoint_bbox(kp, sc)
        bs = inst.get("bbox_score")
        if bs is not None and np.size(bs) >= 1 and np.isfinite(np.asarray(bs, float).ravel()[0]):
            score = float(np.clip(np.asarray(bs, float).ravel()[0], 0.0, 1.0))
        else:
            vis = sc[sc > 0]
            score = float(vis.mean()) if len(vis) else 0.0
        people.append(Person2D(kp, sc, bbox=box, score=score))
    return people


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def resolve_device(device: str | None = "auto") -> str:
    """PyTorch device: auto -> "cuda:0" if available else "cpu"; "cuda" -> "cuda:0"
    (RuntimeError without CUDA); others are passed through."""
    d = str(device or "auto").strip().lower()
    if d in ("", "auto"):
        return "cuda:0" if _cuda_available() else "cpu"
    if d in ("cuda", "gpu") or d.startswith("cuda:"):
        if not _cuda_available():
            raise RuntimeError("MMPose: device 'cuda' requested, but PyTorch has no CUDA "
                               "(torch.cuda.is_available() is False); use 'cpu' or 'auto'")
        return "cuda:0" if d in ("cuda", "gpu") else d
    return d


def _as_bgr(image) -> np.ndarray | None:
    if image is None:
        return None
    img = np.asarray(image)
    if img.ndim < 2 or img.shape[0] == 0 or img.shape[1] == 0:
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        img = np.repeat(img.reshape(img.shape[0], img.shape[1], 1), 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(img)


class MMPoseDetector(Detector2D):
    key = "mmpose"
    label = "MMPose"
    format = FORMATS["coco17"]
    provides_3d = False

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto",
                 weights: str | None = None, det_model: str | None = None,
                 det_weights: str | None = None, det_cat_ids: int | list | None = None,
                 bbox_thr: float = 0.3, nms_thr: float = 0.3,
                 keypoint_format: str | None = None, score_scale: float = 1.0):
        super().__init__()
        from mmpose.apis import MMPoseInferencer  # heavy (PyTorch): only when selected

        self.device = resolve_device(device)
        self.score_scale = float(score_scale)
        if not self.score_scale > 0:
            raise ValueError("score_scale must be > 0")
        self.bbox_thr, self.nms_thr = float(bbox_thr), float(nms_thr)
        try:
            self._inferencer = MMPoseInferencer(
                pose2d=model, pose2d_weights=weights or None, device=self.device,
                det_model=det_model or None, det_weights=det_weights or None,
                det_cat_ids=det_cat_ids)
        except Exception as e:
            raise RuntimeError(
                f"MMPose could not load model {model!r} ({type(e).__name__}: {e}). MMPose "
                "downloads checkpoints from download.openmmlab.com; offline, pass local "
                "'weights' (and 'det_weights') files.") from e
        meta = getattr(getattr(getattr(self._inferencer, "inferencer", None), "model", None),
                       "dataset_meta", None)
        names = _meta_names(meta)
        n = len(names) if names else None
        if n is None and isinstance(meta, dict) and meta.get("num_keypoints"):
            n = int(meta["num_keypoints"])
        self._forced = keypoint_format
        self._layout_known = False
        self._order: list[int] | None = None
        self._n: int | None = None
        if n:
            self._set_layout(int(n), names)
        elif keypoint_format:
            self.format = get_format(keypoint_format)
        self.label = f"MMPose ({model})"
        self.options = {"model": str(model), "device": str(device), "weights": weights or "",
                        "det_model": det_model or "", "det_weights": det_weights or "",
                        "bbox_thr": self.bbox_thr, "nms_thr": self.nms_thr,
                        "keypoint_format": self.format.key, "score_scale": self.score_scale}

    def _set_layout(self, n: int, names: list[str] | None) -> None:
        self.format = format_for(n, names, self._forced)
        self._order = keypoint_order(self.format, names) if not self._forced else None
        self._n = n
        self._layout_known = True
        if self.options:  # the layout became known after the first result
            self.options["keypoint_format"] = self.format.key

    def detect(self, image_bgr: np.ndarray, t: float, cam_name: str) -> list[Person2D]:
        if self._inferencer is None:
            raise RuntimeError("MMPose detector is closed")
        img = _as_bgr(image_bgr)
        if img is None:
            return []
        result = next(iter(self._inferencer(img, return_vis=False, show=False,
                                            bbox_thr=self.bbox_thr, nms_thr=self.nms_thr)))
        preds = result.get("predictions") if isinstance(result, dict) else None
        instances = preds[0] if preds else []
        if not instances:
            return []
        if not self._layout_known:  # no dataset_meta: the first result gives the layout
            first = next((i for i in instances if isinstance(i, dict) and i.get("keypoints")
                          is not None), None)
            if first is None:
                return []
            n = len(_array(first["keypoints"], (2,)))
            fmt_before = self.format
            self._set_layout(n, None)
            if self.format is not fmt_before:
                log.warning("MMPose: the model gives %d keypoints (%s), not %s", n,
                            self.format.key, fmt_before.key)
        return people_from_instances(instances, self._n, self._order, self.score_scale)

    def close(self) -> None:
        was_open, self._inferencer = self._inferencer is not None, None
        if was_open and self.device.startswith("cuda"):
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                log.debug("torch.cuda.empty_cache: %s", e)

    def info(self) -> dict:
        d = super().info()
        d.update({"device_used": self.device, "license": "Apache-2.0 (MMPose; check the "
                  "license of the model weights)"})
        return d


def create(model: str = DEFAULT_MODEL, device: str = "auto", pose2d: str | None = None,
           pose2d_weights: str | None = None, **kwargs) -> MMPoseDetector:
    """Factory of the ``mmpose`` backend. ``model`` (alias ``pose2d``): an MMPose alias
    ("human", "rtmpose-l", ...), model-zoo config name or config file; ``device``:
    auto/cpu/cuda. Extra options: weights (alias pose2d_weights), det_model, det_weights,
    det_cat_ids, bbox_thr (0.3), nms_thr (0.3), keypoint_format, score_scale (1.0)."""
    if pose2d:
        model = pose2d
    if pose2d_weights and not kwargs.get("weights"):
        kwargs["weights"] = pose2d_weights
    return MMPoseDetector(model=model, device=device, **kwargs)
