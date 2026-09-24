"""Ultralytics YOLO pose models (YOLO11, YOLOv8; also YOLO26) as a ``Detector2D``
(backend key ``yolo_pose``).

LICENSE: the ``ultralytics`` package and its pretrained YOLO weights are licensed under the
**AGPL-3.0** (https://ultralytics.com/license). PoseBoard imports ultralytics only when this
backend is selected and does not ship it. If you distribute software, or offer a network
service, that uses this backend, you must meet the AGPL-3.0 terms (or buy an Ultralytics
Enterprise license). Check this before commercial use.

A one-stage multi-person model: ``detect`` returns every person found. The subject on the
board is picked later by ``poseboard.pose.subject.select_subject``.

Weights: ``<family><size>-pose.pt`` (e.g. ``yolo11n-pose.pt``, ``yolov8s-pose.pt``) are looked
up in, and downloaded on first use into, the PoseBoard models folder
(``poseboard.pose.mediapipe_backend.MODEL_DIR``, by default ``models/`` in the PoseBoard
folder, next to the MediaPipe model), never into the current working directory. The files come from the
Ultralytics GitHub release assets (https://github.com/ultralytics/assets/releases); without
internet, download the file there and put it in that folder. ``model`` may also be the path
of your own ``.pt`` pose model or an exported one (ONNX, TensorRT, ...).

How the ultralytics results are mapped to ``Person2D``:

* keypoints: ``Results.keypoints.xy`` (N, K, 2). ultralytics already maps them from the
  letterboxed network input back to ORIGINAL image pixels
  (``ultralytics/models/yolo/pose/predict.py``, ``PosePredictor.construct_result``:
  ``ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)``, lines 63-65 in 8.4), so
  nothing is rescaled here. A point at exactly (0, 0) means "not predicted" (older ultralytics
  versions zero points with a confidence < 0.5, and invisible points are labelled (0, 0) in
  the training data); it becomes NaN with score 0.
* scores: ``Results.keypoints.conf`` is the sigmoid of the visibility logit
  (``ultralytics/nn/modules/head.py``, ``Pose.kpts_decode``, lines 597-610 in 8.4), i.e.
  already a probability in 0..1; it is only clipped to 0..1 (NaN -> 0). It is None for models
  trained with ``kpt_shape: [K, 2]`` (no visibility): every predicted point then gets 1.0.
* person: ``Results.boxes.xyxy`` / ``Results.boxes.conf`` -> ``Person2D.bbox`` / ``.score``.
* keypoint order: the official pose weights are trained on COCO-Keypoints in the COCO-17 order
  (``ultralytics/cfg/datasets/coco-pose.yaml``, ``kpt_names`` lines 26-44: nose, left_eye,
  right_eye, left_ear, right_ear, left_shoulder, ..., right_ankle; ``flip_idx`` on line 19
  swaps exactly those left/right pairs), which is ``FORMATS["coco17"]`` index for index. A
  model that carries its own ``kpt_names`` metadata (stored by recent ultralytics versions
  when training) is mapped by name instead.

Privacy: ultralytics sends anonymous usage events (Google Analytics) after predictions when
its ``sync`` setting is on. PoseBoard switches that off for its own process when it loads
this backend; the user's ultralytics settings file is not changed.

Note: importing ultralytics sets ``OMP_NUM_THREADS=1`` for the process if it is not set yet
(ultralytics/__init__.py).
"""

from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path

import numpy as np

from poseboard.pose.detectors.base import Detector2D, Person2D, keypoint_bbox
from poseboard.pose.formats import COCO17, KeypointFormat, get_format

__all__ = ["FAMILIES", "MIN_ULTRALYTICS", "RELEASES_URL", "SIZES", "YoloPoseDetector",
           "check_ultralytics_version", "create", "models_dir",
           "people_from_result", "resolve_device", "resolve_weights", "weights_name"]

log = logging.getLogger(__name__)

SIZES = ("n", "s", "m", "l", "x")
# version option -> file name prefix of the official pose weights
FAMILIES = {"11": "yolo11", "8": "yolov8", "26": "yolo26"}
_VERSION_ALIASES = {
    "11": "11", "v11": "11", "yolo11": "11", "yolov11": "11",
    "8": "8", "v8": "8", "yolov8": "8", "yolo8": "8",
    "26": "26", "v26": "26", "yolo26": "26", "yolov26": "26",
}
# "yolo11n", "yolo11n-pose", "yolo11n-pose.pt" (but not "yolo11n.pt": that is a detection model)
_NAME_RE = re.compile(r"^(yolo11|yolov8|yolo26)([nsmlx])(?:-pose(?:\.pt)?)?$", re.IGNORECASE)
RELEASES_URL = "https://github.com/ultralytics/assets/releases"
# First ultralytics release that knows each family (older ones cannot load its weights)
MIN_ULTRALYTICS = {"yolo26": (8, 4)}


# ------------------------------------------------------------------ weights and device
def models_dir() -> Path:
    """The PoseBoard models folder (shared with the MediaPipe model files)."""
    from poseboard.pose import mediapipe_backend as mpb  # MODEL_DIR is patched in the frozen app

    return Path(mpb.MODEL_DIR)


def check_ultralytics_version(name: str | None, installed: str | None = None) -> None:
    """Raise RuntimeError when the installed ultralytics is too old for the weights ``name``
    (e.g. YOLO26 needs ultralytics >= 8.4); unknown version strings are not checked."""
    if not name:
        return
    need = next((v for fam, v in MIN_ULTRALYTICS.items() if name.lower().startswith(fam)), None)
    if need is None:
        return
    if installed is None:
        import ultralytics

        installed = str(getattr(ultralytics, "__version__", ""))
    m = re.match(r"^(\d+)\.(\d+)", str(installed).strip())
    if m and (int(m.group(1)), int(m.group(2))) < need:
        fam = name.split("-")[0].rstrip("nsmlx").upper()
        raise RuntimeError(f"{fam} needs ultralytics>={need[0]}.{need[1]} (installed: "
                           f"{installed}): pip install -U ultralytics")


def _version_key(version) -> str:
    key = _VERSION_ALIASES.get(str(version).strip().lower())
    if key is None:
        raise ValueError(f"YOLO pose version must be one of {', '.join(FAMILIES)} "
                         f"(YOLO11, YOLOv8, YOLO26), not {version!r}")
    return key


def weights_name(model: str = "n", version: str = "11") -> str | None:
    """File name of an official pose checkpoint: ("n", "11") -> "yolo11n-pose.pt",
    ("s", "8") -> "yolov8s-pose.pt"; a name such as "yolov8m-pose" is normalized. None if
    ``model`` is not a size or an official name (then it is a file path)."""
    m = str(model).strip()
    if m.lower() in SIZES:
        return f"{FAMILIES[_version_key(version)]}{m.lower()}-pose.pt"
    hit = _NAME_RE.match(Path(m).name) if Path(m).name == m else None
    if hit:
        return f"{hit.group(1).lower()}{hit.group(2).lower()}-pose.pt"
    return None


def resolve_weights(model: str = "n", version: str = "11", weights_dir: str | Path | None = None,
                    download: bool = True) -> Path:
    """Local path of the weights for ``model`` / ``version``.

    * a size ("n", "s", "m", "l", "x") or an official file name: ``<weights_dir>/<name>``
      (default: the PoseBoard models folder), downloaded from the Ultralytics GitHub release
      assets if missing;
    * anything else: the path of an existing model file (``.pt``, ``.onnx``, ...).

    Raises RuntimeError (download failed, with manual instructions) or FileNotFoundError.
    """
    folder = Path(weights_dir) if weights_dir else models_dir()
    name = weights_name(model, version)
    if name is None:
        p = Path(str(model).strip()).expanduser()
        for cand in (p, folder / p):
            if cand.is_file():
                return cand.resolve()
        raise FileNotFoundError(
            f"YOLO pose model {model!r} not found. Use a size ({', '.join(SIZES)}), an official "
            f"name such as yolo11n-pose.pt, or the path of an existing model file.")
    target = folder / name
    if target.is_file():
        return target
    if not download:
        raise FileNotFoundError(f"{target} does not exist")
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        target.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s into %s", name, target.parent)
        attempt_download_asset(str(target))  # downloads to exactly this path (not the CWD)
        if not target.is_file():
            raise RuntimeError("ultralytics did not find it in its release assets")
    except Exception as e:  # offline, proxy, disk, ...
        raise RuntimeError(
            f"Cannot download the YOLO pose weights {name} ({e}).\nDownload {name} on any "
            f"computer from\n  {RELEASES_URL}\nand save it as\n  {target}") from e
    return target


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def resolve_device(device: str | None = "auto") -> str:
    """ultralytics device string: "auto" -> "cuda:0" if PyTorch sees a CUDA GPU, else "cpu";
    "cuda"/"gpu" -> "cuda:0" (RuntimeError without CUDA); others ("cpu", "cuda:1", "0",
    "mps") are passed through."""
    d = "auto" if device is None else str(device).strip().lower()
    if d in ("", "auto"):
        return "cuda:0" if _cuda_available() else "cpu"
    if d in ("cuda", "gpu") or d.startswith("cuda:") or d.isdigit():
        if not _cuda_available():
            raise RuntimeError(
                f"YOLO pose: device {device!r} needs a CUDA GPU, but PyTorch has no CUDA "
                "(torch.cuda.is_available() is False). Install a CUDA build of PyTorch or use "
                "device 'cpu' / 'auto'.")
        return "cuda:0" if d in ("cuda", "gpu") else d
    return d


def _disable_analytics() -> None:
    """Switch off ultralytics' anonymous usage events for this process (see module docstring);
    the module moved between versions (8.4: utils.events, before: hub.utils)."""
    for mod_name in ("ultralytics.utils.events", "ultralytics.hub.utils"):
        try:
            ev = getattr(importlib.import_module(mod_name), "events", None)
        except Exception as e:  # noqa: BLE001  (module not in this ultralytics version)
            log.debug("ultralytics analytics: %s: %s", mod_name, e)
            continue
        if ev is not None and hasattr(ev, "enabled"):
            ev.enabled = False
            return


# ------------------------------------------------------------------ result conversion
def _to_numpy(x) -> np.ndarray:
    """torch tensor (any device) or array-like -> float64 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, np.float64)


def people_from_result(result, order: np.ndarray | list[int] | None = None) -> list[Person2D]:
    """All persons of one ultralytics ``Results`` (see the module docstring for the mapping).
    ``order[i]`` = index in the model output of keypoint ``i`` of the target format (None:
    same order)."""
    kps = getattr(result, "keypoints", None)
    if kps is None or getattr(kps, "xy", None) is None:
        return []
    xy = _to_numpy(kps.xy)
    if xy.size == 0:
        return []
    xy = xy.reshape(-1, xy.shape[-2], 2)
    n, k = xy.shape[:2]
    conf = getattr(kps, "conf", None)
    sc = np.ones((n, k)) if conf is None else _to_numpy(conf).reshape(n, k)
    sc = np.clip(np.nan_to_num(sc, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    missing = ~np.all(np.isfinite(xy), axis=-1) | np.all(xy == 0.0, axis=-1)
    xy = np.where(missing[..., None], np.nan, xy)
    sc = np.where(missing, 0.0, sc)
    if order is not None:
        order = np.asarray(order, int)
        xy, sc = xy[:, order], sc[:, order]

    boxes = getattr(result, "boxes", None)
    bb = bconf = None
    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
        b = _to_numpy(boxes.xyxy).reshape(-1, 4)
        if len(b) == n:
            bb = b
            if getattr(boxes, "conf", None) is not None:
                bconf = np.clip(np.nan_to_num(_to_numpy(boxes.conf).reshape(-1), nan=0.0), 0, 1)
    people = []
    for i in range(n):
        box = bb[i] if bb is not None else keypoint_bbox(xy[i], sc[i])
        if bconf is not None and len(bconf) == n:
            score = float(bconf[i])
        else:
            vis = sc[i][sc[i] > 0]
            score = float(vis.mean()) if len(vis) else 0.0
        people.append(Person2D(xy[i], sc[i], bbox=box, score=score))
    return people


def _canon(name) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def _model_attr(yolo, name):
    """``name`` of the loaded network (.pt: the PoseModel; exported: the AutoBackend once the
    predictor is set up); None if unknown."""
    for obj in (getattr(yolo, "model", None),
                getattr(getattr(yolo, "predictor", None), "model", None)):
        val = getattr(obj, name, None) if obj is not None and not isinstance(obj, str) else None
        if val is not None:
            return val
    return None


def _keypoint_order(n_model: int, model_names, fmt: KeypointFormat) -> list[int] | None:
    """Indices into the model output for the keypoints of ``fmt`` (None: identical order).
    Raises ValueError when the model's keypoints do not fit ``fmt``."""
    names = model_names
    if isinstance(names, dict):  # ultralytics stores {class_id: [names]}
        names = names.get(0, next(iter(names.values()), None)) if names else None
    if names is not None:
        names = [_canon(x) for x in names]
        if len(names) == n_model and set(fmt.names) <= set(names):
            order = [names.index(x) for x in fmt.names]
            return None if order == list(range(n_model)) else order
        if n_model == len(fmt):
            log.warning("YOLO pose model keypoint names %s do not match the %s format; "
                        "assuming the %s order", names, fmt.key, fmt.key)
    if n_model != len(fmt):
        raise ValueError(
            f"The YOLO pose model gives {n_model} keypoints, but the {fmt.label or fmt.key} "
            f"format has {len(fmt)}. Pass keypoint_format=<one of the PoseBoard formats with "
            f"{n_model} points in the model's order> for a custom model.")
    return None


# ------------------------------------------------------------------ detector
class YoloPoseDetector(Detector2D):
    key = "yolo_pose"
    label = "YOLO pose (Ultralytics)"
    format = COCO17
    provides_3d = False

    def __init__(self, model: str = "n", version: str = "11", device: str = "auto",
                 conf: float = 0.25, iou: float = 0.7, imgsz: int | tuple[int, int] = 640,
                 max_det: int = 20, half: bool = False, keypoint_format: str | None = None,
                 weights_dir: str | Path | None = None):
        super().__init__()
        from ultralytics import YOLO  # heavy (PyTorch): only when this backend is used

        _disable_analytics()
        check_ultralytics_version(weights_name(model, version))  # before any download
        self.device = resolve_device(device)
        self.weights_path = resolve_weights(model, version, weights_dir)
        self.format = get_format(keypoint_format or "coco17")
        self._yolo = YOLO(str(self.weights_path), task="pose")
        task = getattr(self._yolo, "task", "pose")
        if task != "pose":
            raise ValueError(f"{self.weights_path.name} is a {task!r} model, not a pose model")
        self._order: list[int] | None = None
        self._layout_known = False
        shape = _model_attr(self._yolo, "kpt_shape")
        if shape is not None:
            self._set_layout(int(shape[0]))
        stem = self.weights_path.stem
        self.label = f"YOLO pose ({stem}, Ultralytics)"
        official = weights_name(model, version) is not None
        self.options = {"model": str(model), "version": _version_key(version) if official
                        else str(version), "device": str(device), "conf": float(conf),
                        "iou": float(iou), "imgsz": imgsz, "max_det": int(max_det),
                        "half": bool(half), "keypoint_format": self.format.key}
        self._predict_kw = {"conf": float(conf), "iou": float(iou), "imgsz": imgsz,
                            "max_det": int(max_det), "device": self.device,
                            "half": bool(half) and self.device != "cpu",
                            "verbose": False}

    def _set_layout(self, n_model: int) -> None:
        self._order = _keypoint_order(n_model, _model_attr(self._yolo, "kpt_names"), self.format)
        self._layout_known = True

    def detect(self, image_bgr: np.ndarray, t: float, cam_name: str) -> list[Person2D]:
        if self._yolo is None:
            raise RuntimeError("YOLO pose detector is closed")
        img = _as_bgr(image_bgr)
        if img is None:
            return []
        results = self._yolo.predict(img, **self._predict_kw)
        if not results:
            return []
        r = results[0]
        if not self._layout_known:  # exported models: the layout is known after the first run
            xy = getattr(getattr(r, "keypoints", None), "xy", None)
            if xy is None or _to_numpy(xy).size == 0:
                return []
            self._set_layout(int(_to_numpy(xy).shape[-2]))
        return people_from_result(r, self._order)

    def close(self) -> None:
        was_open, self._yolo = self._yolo is not None, None
        if was_open and self.device.startswith("cuda"):
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                log.debug("torch.cuda.empty_cache: %s", e)

    def info(self) -> dict:
        d = super().info()
        d.update({"weights": self.weights_path.name, "device_used": self.device,
                  "license": "AGPL-3.0"})
        return d


def _as_bgr(image) -> np.ndarray | None:
    """3-channel uint8 BGR image, or None for an empty image."""
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
        img = img[:, :, :3]  # BGRA -> BGR
    return np.ascontiguousarray(img)


def create(model: str = "n", version: str = "11", device: str = "auto",
           **kwargs) -> YoloPoseDetector:
    """Factory of the ``yolo_pose`` backend. ``model``: size n/s/m/l/x (or an official file
    name, or the path of a pose model file); ``version``: "11" (YOLO11), "8" (YOLOv8) or "26"
    (YOLO26); ``device``: auto/cpu/cuda. Extra keyword options: conf (person confidence
    threshold, 0.25), iou (NMS, 0.7), imgsz (network input size, 640), max_det (20), half
    (FP16 on CUDA), keypoint_format (for custom models), weights_dir (default: the PoseBoard
    models folder)."""
    return YoloPoseDetector(model=model, version=version, device=device, **kwargs)
