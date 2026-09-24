"""Keypoint R-CNN (torchvision, ResNet-50-FPN, COCO) as a ``Detector2D`` (backend key
``keypoint_rcnn``).

A two-stage multi-person model: a Faster R-CNN person detector plus a keypoint head.
``detect`` returns every person above ``score_threshold``. The subject on the board is picked
later by ``poseboard.pose.subject.select_subject``. The model is accurate but slow without a
GPU: expect about 1-3 s per image on a CPU at the default ``min_size`` of 800.

License: torchvision is BSD-3-Clause. The pretrained weights are trained on COCO.

Weights (``KeypointRCNN_ResNet50_FPN_Weights.DEFAULT`` = ``COCO_V1``, file
``keypointrcnn_resnet50_fpn_coco-fc266e95.pth``, 226 MB) are taken from the first of these that
exists:

1. ``weights_path``: a state dict saved with ``torch.save(model.state_dict(), path)``. A
   torchvision training checkpoint (``{"model": state_dict, ...}``) also works.
2. ``<PoseBoard models folder>/keypointrcnn_resnet50_fpn_coco-fc266e95.pth``. This is the
   folder of the MediaPipe and YOLO models, ``models/`` by default.
3. The PyTorch hub cache, ``<TORCH_HOME>/hub/checkpoints/``. ``TORCH_HOME`` defaults to
   ``~/.cache/torch``, i.e. ``%USERPROFILE%\\.cache\\torch`` on Windows. If the file is not
   there, it is downloaded from https://download.pytorch.org/models/ into this folder on first
   use. The SHA-256 prefix in the file name is checked, as torchvision itself does.

Without internet, download the file on any computer and save it in folder 2 or 3, or pass its
path as ``weights_path``. When the download fails, ``WeightsUnavailable`` (a RuntimeError)
says exactly this. Every file is loaded with ``torch.load(..., weights_only=True)``, so no
pickled code is ever executed.

How the torchvision output is mapped to ``Person2D``
(``torchvision/models/detection/keypoint_rcnn.py`` docstring of ``keypointrcnn_resnet50_fpn``,
torchvision 0.29):

* keypoints: ``output["keypoints"]`` (N, 17, 3) is already in ORIGINAL image pixels.
  ``GeneralizedRCNNTransform.postprocess`` undoes the internal resize with
  ``resize_keypoints(keypoints, im_s, o_im_s)``
  (``torchvision/models/detection/transform.py``, lines 273-276), so nothing is rescaled here.
  The third column ("visibility") is always 1 (``roi_heads.py``, ``heatmaps_to_keypoints``,
  line 304) and is ignored.
* scores: ``output["keypoints_scores"]`` (N, 17) are NOT probabilities. Each is the value of
  the keypoint's heatmap logit at its maximum (``roi_heads.py``, ``heatmaps_to_keypoints``,
  line 305). The head is trained with a softmax cross-entropy over the heatmap positions
  (``keypointrcnn_loss``, line 337), so the value is unbounded and not calibrated. PoseBoard
  maps it to 0..1 with the logistic sigmoid ``1 / (1 + exp(-logit))``:

  * logit 0 -> 0.5; 2 -> 0.88; -2 -> 0.12; -0.85 -> 0.3 (PoseBoard's default ``min_score``).
  * Clearly visible joints have large positive logits, so a score of about 1.
  * Occluded or guessed joints have logits near or below 0.
  * A NaN logit gives score 0.
* person: ``output["boxes"]`` (x1, y1, x2, y2, original pixels) -> ``Person2D.bbox``.
  ``output["scores"]`` (softmax class probability, already 0..1) -> ``Person2D.score``. Only
  label 1 ("person", ``_COCO_PERSON_CATEGORIES`` in ``torchvision/models/_meta.py``) is kept.
* keypoint order: the weights' ``meta["keypoint_names"]`` is ``_COCO_PERSON_KEYPOINT_NAMES``
  (``torchvision/models/_meta.py``, lines 1107-1125 in 0.29: nose, left_eye, right_eye,
  left_ear, right_ear, left_shoulder, ..., right_ankle). That is ``FORMATS["coco17"]`` index for
  index, with left/right as the person's own sides. The mapping is still done by name
  (``keypoint_index_map``), so a different order could not silently swap sides.

Only the heavy libraries (torch, torchvision) are imported inside the factory / detector, never
when this module is imported.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np

from poseboard.pose.detectors.base import Detector2D, Person2D
from poseboard.pose.formats import COCO17, KeypointFormat

__all__ = ["DEFAULT_WEIGHTS_URL", "PERSON_LABEL", "TORCHVISION_KEYPOINT_NAMES", "WEIGHTS_FILE",
           "KeypointRCNNDetector", "WeightsUnavailable", "create", "download_weights",
           "find_weights", "keypoint_index_map", "keypoint_scores", "load_model",
           "load_state_dict", "people_from_output", "resolve_device",
           "torch_hub_checkpoints_dir"]

log = logging.getLogger(__name__)

# KeypointRCNN_ResNet50_FPN_Weights.COCO_V1 (== DEFAULT) and COCO_LEGACY
# (torchvision/models/detection/keypoint_rcnn.py, lines 319-359 in 0.29). At run time the URL
# is read from torchvision's enum; these constants are for messages and the models folder.
WEIGHTS_FILE = "keypointrcnn_resnet50_fpn_coco-fc266e95.pth"
DEFAULT_WEIGHTS_URL = "https://download.pytorch.org/models/" + WEIGHTS_FILE
_COCO_V1_SHA256_PREFIX = "fc266e95"
_COCO_LEGACY_SHA256_PREFIX = "9f466800"

# torchvision/models/_meta.py, _COCO_PERSON_KEYPOINT_NAMES (lines 1107-1125 in torchvision
# 0.29), the ``keypoint_names`` meta of both Keypoint R-CNN weights.
TORCHVISION_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
PERSON_LABEL = 1  # _COCO_PERSON_CATEGORIES = ["no person", "person"]

_HASH_RE = re.compile(r"-([a-f0-9]{8,})\.")  # the same convention as torch.hub.HASH_REGEX
_KPS_WEIGHT = "roi_heads.keypoint_predictor.kps_score_lowres.weight"
_CLS_WEIGHT = "roi_heads.box_predictor.cls_score.weight"


class WeightsUnavailable(RuntimeError):
    """The pretrained weights are not on this computer and cannot be downloaded."""


# ------------------------------------------------------------------ output conversion
def keypoint_scores(logits) -> np.ndarray:
    """Keypoint R-CNN heatmap logits -> 0..1 with the logistic sigmoid (NaN -> 0).

    ``0.5 * (1 + tanh(x / 2))`` equals ``1 / (1 + exp(-x))``. Written this way it does not
    overflow for large |x|: +inf -> 1, -inf -> 0.
    """
    x = _to_numpy(logits)
    with np.errstate(invalid="ignore"):
        s = 0.5 * (1.0 + np.tanh(0.5 * x))
    return np.where(np.isfinite(s), np.clip(s, 0.0, 1.0), 0.0)


def keypoint_index_map(names, fmt: KeypointFormat = COCO17) -> np.ndarray:
    """For each keypoint of ``fmt``: its index in the model's ``names``, or -1 if the model does
    not have it. Mapping by name keeps left and right exact whatever the model's order is."""
    names = [str(n) for n in names]
    return np.array([names.index(n) if n in names else -1 for n in fmt.names], np.int64)


def _to_numpy(x) -> np.ndarray:
    """torch tensor (any device) or array-like -> float64 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, np.float64)


def people_from_output(output: dict, *, score_threshold: float = 0.0,
                       names=TORCHVISION_KEYPOINT_NAMES, fmt: KeypointFormat = COCO17,
                       person_label: int | None = PERSON_LABEL) -> list[Person2D]:
    """One torchvision Keypoint R-CNN inference result (the dict of one image) ->
    ``Person2D`` list, highest person score first.

    ``output`` holds ``boxes`` (N, 4), ``scores`` (N,), ``labels`` (N,), ``keypoints``
    (N, K, 3) and ``keypoints_scores`` (N, K) logits. Tensors or numpy arrays are both
    accepted. The coordinates must already be in original image pixels, which torchvision's
    postprocess does. Persons whose score is below ``score_threshold``, or whose label is not
    ``person_label``, are dropped. ``person_label=None`` keeps every label. A keypoint of
    ``fmt`` that the model lacks, or that has a non-finite position, becomes NaN with score 0.
    """
    kps = output.get("keypoints") if output else None
    if kps is None:
        return []
    kps = _to_numpy(kps)
    if kps.size == 0:
        return []
    kps = kps.reshape(kps.shape[0], -1, kps.shape[-1])
    n, k = kps.shape[:2]
    boxes = output.get("boxes")
    boxes = _to_numpy(boxes).reshape(-1, 4) if boxes is not None else None
    pscore = output.get("scores")
    pscore = _to_numpy(pscore).reshape(-1) if pscore is not None else np.ones(n)
    labels = output.get("labels")
    labels = _to_numpy(labels).reshape(-1) if labels is not None else None
    logits = output.get("keypoints_scores")
    # Without keypoint scores every predicted point counts as seen (score 1).
    kscore = keypoint_scores(_to_numpy(logits).reshape(n, k)) if logits is not None \
        else np.ones((n, k))

    src = keypoint_index_map(names, fmt)
    if np.any(src >= k):
        raise ValueError(f"Keypoint R-CNN output has {k} keypoints, but the names list has "
                         f"{len(names)}")
    has = src >= 0
    people: list[Person2D] = []
    for i in np.argsort(-np.nan_to_num(pscore, nan=-np.inf), kind="stable"):
        s = float(pscore[i])
        if not np.isfinite(s) or s < score_threshold:
            continue
        if person_label is not None and labels is not None and int(labels[i]) != person_label:
            continue
        kp = np.full((len(fmt), 2), np.nan)
        sc = np.zeros(len(fmt))
        kp[has] = kps[i, src[has], :2]
        sc[has] = kscore[i, src[has]]
        bad = ~np.all(np.isfinite(kp), axis=1)
        kp[bad] = np.nan
        sc[bad] = 0.0
        box = None
        if boxes is not None and i < len(boxes) and np.all(np.isfinite(boxes[i])):
            box = boxes[i].copy()
        people.append(Person2D(kp, sc, bbox=box, score=float(np.clip(s, 0.0, 1.0))))
    return people


# ------------------------------------------------------------------ device and weights
def resolve_device(device: str | None = "auto") -> str:
    """torch device string.

    * ``"auto"`` (or empty / None) -> ``"cuda"`` if PyTorch sees a CUDA GPU, else ``"cpu"``.
    * ``"cuda"`` / ``"gpu"`` / ``"cuda:N"`` -> RuntimeError when CUDA (or GPU N) is missing.
    * Anything else (``"cpu"``, ``"mps"``, ...) is checked by ``torch.device``.
    """
    import torch

    d = "auto" if device is None else str(device).strip().lower()
    if d in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d in ("cuda", "gpu") or d.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Keypoint R-CNN: device {device!r} needs a CUDA GPU, but PyTorch has no CUDA "
                "(torch.cuda.is_available() is False). Install a CUDA build of PyTorch or use "
                "device 'cpu' / 'auto'.")
        if d.startswith("cuda:"):
            idx = d.split(":", 1)[1]
            if not idx.isdigit() or int(idx) >= torch.cuda.device_count():
                raise RuntimeError(f"Keypoint R-CNN: no CUDA device {device!r} "
                                   f"({torch.cuda.device_count()} GPU(s) found)")
            return d
        return "cuda"
    if d == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("Keypoint R-CNN: device 'mps' is not available on this computer")
        return d
    try:
        return str(torch.device(d))
    except (RuntimeError, TypeError, ValueError) as e:
        raise ValueError(f"Keypoint R-CNN: unknown device {device!r} (use auto, cpu or cuda)") \
            from e


def _models_dir() -> Path:
    # imported here: MODEL_DIR is patched in the frozen app
    from poseboard.pose import mediapipe_backend as mpb

    return Path(mpb.MODEL_DIR)


def torch_hub_checkpoints_dir() -> Path:
    """``<torch.hub.get_dir()>/checkpoints``, i.e. ``$TORCH_HOME/hub/checkpoints``, where
    torchvision caches the pretrained weights."""
    import torch.hub

    return Path(torch.hub.get_dir()) / "checkpoints"


def _default_weights():
    """(url, file name, keypoint names) of torchvision's default Keypoint R-CNN weights."""
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights

    w = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    url = w.url
    names = tuple(w.meta.get("keypoint_names", TORCHVISION_KEYPOINT_NAMES))
    return url, url.rsplit("/", 1)[-1], names


def find_weights(weights_path: str | Path | None = None) -> Path | None:
    """Local weights file.

    * ``weights_path``, if given. FileNotFoundError when it does not exist.
    * Otherwise the official file in the PoseBoard models folder or in the PyTorch hub cache.
    * None when none of these exist; ``download_weights`` then fetches it.
    """
    if weights_path not in (None, ""):
        p = Path(str(weights_path)).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Keypoint R-CNN weights file not found: {p}")
        return p.resolve()
    try:
        _url, name, _names = _default_weights()
    except Exception:  # noqa: BLE001  (torchvision too old / not installed: use our constant)
        name = WEIGHTS_FILE
    for folder in (_models_dir(), torch_hub_checkpoints_dir()):
        cand = folder / name
        if cand.is_file() and cand.stat().st_size > 0:
            return cand
    return None


def download_weights(dest_dir: str | Path | None = None, progress: bool = False) -> Path:
    """Download torchvision's default Keypoint R-CNN weights into ``dest_dir``. The default is
    the PyTorch hub cache, ``$TORCH_HOME/hub/checkpoints``, the same file torchvision itself
    would use. The SHA-256 prefix in the file name is checked. An existing file is returned
    unchanged.

    Raises ``WeightsUnavailable`` with manual instructions when the download fails.
    """
    import torch.hub

    url, name, _names = _default_weights()
    folder = Path(dest_dir) if dest_dir else torch_hub_checkpoints_dir()
    target = folder / name
    if target.is_file() and target.stat().st_size > 0:
        return target
    m = _HASH_RE.search(name)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s into %s", url, folder)
        # Writes to a temporary file and moves it into place only when the hash matches.
        torch.hub.download_url_to_file(url, str(target), m.group(1) if m else None,
                                       progress=progress)
    except Exception as e:  # offline, proxy, 403, disk full, hash mismatch
        raise WeightsUnavailable(
            f"Cannot download the Keypoint R-CNN weights {name} ({type(e).__name__}: {e}).\n"
            f"Download\n  {url}\non any computer and save it as\n  {target}\nor as\n  "
            f"{_models_dir() / name}\nor pass the file as the backend option weights_path.") from e
    return target


def load_state_dict(path: str | Path) -> dict:
    """The state dict in ``path``, loaded with ``weights_only=True``.

    Accepts a plain ``model.state_dict()``, a torchvision training checkpoint
    (``{"model": ...}``) or ``{"state_dict": ...}``. A DataParallel ``module.`` prefix is
    removed.
    """
    import torch

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as e:
        raise ValueError(f"Cannot read Keypoint R-CNN weights {path}: {e}. Expected a state "
                         "dict saved with torch.save(model.state_dict(), path).") from e
    for key in ("model", "state_dict"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"{path} does not contain a Keypoint R-CNN state dict")
    if all(str(k).startswith("module.") for k in obj):
        obj = {str(k)[len("module."):]: v for k, v in obj.items()}
    return obj


def _sha256_prefix(path: Path, n: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _frozen_bn_eps(path: Path) -> float | None:
    """Epsilon of the frozen batch norms for the weights in ``path``.

    torchvision's builder sets 0.0 for the COCO_V1 weights (``overwrite_eps(model, 0.0)``,
    keypoint_rcnn.py line 474), because they were trained before pytorch/vision#2933. It keeps
    the default (1e-5) for COCO_LEGACY and for any other checkpoint. The weights are recognised
    by the SHA-256 prefix in their file name, or else by hashing the file. Returns None to keep
    the default.
    """
    m = _HASH_RE.search(path.name)
    prefix = m.group(1)[:8] if m else None
    if prefix not in (_COCO_V1_SHA256_PREFIX, _COCO_LEGACY_SHA256_PREFIX):
        try:
            prefix = _sha256_prefix(path)
        except OSError:
            prefix = None
    return 0.0 if prefix == _COCO_V1_SHA256_PREFIX else None


def load_model(weights_path: str | Path | None = None, *, download: bool = True,
               score_threshold: float = 0.5, max_people: int = 20, min_size: int = 800,
               max_size: int = 1333):
    """(model, keypoint names, weights file). The model is the Keypoint R-CNN ResNet-50-FPN
    in eval mode on the CPU, with the weights from ``find_weights`` or ``download_weights``.

    The architecture is the one ``keypointrcnn_resnet50_fpn(weights=...)`` builds for
    pretrained weights: a ResNet-50 with ``FrozenBatchNorm2d``, an FPN with 3 trainable
    layers, and ``KeypointRCNN``. The number of classes and keypoints is read from the
    checkpoint, which is loaded strictly.
    """
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
    from torchvision.ops.misc import FrozenBatchNorm2d

    path = find_weights(weights_path)
    if path is None:
        if not download:
            raise WeightsUnavailable(f"Keypoint R-CNN weights {WEIGHTS_FILE} not found")
        path = download_weights()
    sd = load_state_dict(path)
    if _KPS_WEIGHT not in sd or _CLS_WEIGHT not in sd:
        raise ValueError(f"{path} is not a Keypoint R-CNN checkpoint ({_KPS_WEIGHT} missing)")
    # ConvTranspose2d weight: (in_channels, out_channels = keypoints, kh, kw)
    num_keypoints = int(sd[_KPS_WEIGHT].shape[1])
    num_classes = int(sd[_CLS_WEIGHT].shape[0])
    try:
        names = _default_weights()[2]
    except Exception:  # noqa: BLE001
        names = TORCHVISION_KEYPOINT_NAMES
    if num_keypoints != len(names):
        raise ValueError(
            f"{path} predicts {num_keypoints} keypoints; the keypoint_rcnn backend needs a "
            f"COCO-17 model ({len(names)} keypoints)")

    backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None,
                                   norm_layer=FrozenBatchNorm2d, trainable_layers=3)
    model = KeypointRCNN(backbone, num_classes=num_classes, num_keypoints=num_keypoints,
                         box_score_thresh=float(score_threshold),
                         box_detections_per_img=max(1, int(max_people)),
                         min_size=int(min_size), max_size=int(max_size))
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise ValueError(f"{path} does not match the Keypoint R-CNN ResNet-50-FPN "
                         f"architecture: {e}") from e
    eps = _frozen_bn_eps(Path(path))
    if eps is not None:
        for mod in model.modules():
            if isinstance(mod, FrozenBatchNorm2d):
                mod.eps = eps
    model.eval()
    model.requires_grad_(False)
    return model, tuple(names), Path(path)


# ------------------------------------------------------------------ detector
class KeypointRCNNDetector(Detector2D):
    """torchvision Keypoint R-CNN ResNet-50-FPN (COCO-17, multi-person). See the module
    docstring for the weights, the score mapping and the keypoint order."""

    key = "keypoint_rcnn"
    label = "Keypoint R-CNN (torchvision)"
    format = COCO17
    provides_3d = False

    def __init__(self, device: str = "auto", score_threshold: float = 0.5,
                 weights_path: str | Path | None = "", min_size: int = 800,
                 max_size: int = 1333, max_people: int = 20, download: bool = True):
        super().__init__()
        import torch  # noqa: F401  (fail here, clearly, if torch is missing)

        self.score_threshold = float(score_threshold)
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError(f"score_threshold must be in 0..1, not {score_threshold!r}")
        self.device = resolve_device(device)
        self.model, self.keypoint_names, self.weights_file = load_model(
            weights_path, download=download, score_threshold=self.score_threshold,
            max_people=max_people, min_size=min_size, max_size=max_size)
        self.model.to(self.device)
        self.options = {"device": device, "score_threshold": self.score_threshold,
                        "weights_path": str(weights_path or ""), "min_size": int(min_size),
                        "max_size": int(max_size), "max_people": int(max_people)}

    def info(self) -> dict:
        d = super().info()
        d.update(device=self.device, weights_file=str(self.weights_file),
                 keypoint_score="sigmoid(heatmap logit)")
        return d

    @staticmethod
    def _to_rgb(image) -> np.ndarray | None:
        """BGR / BGRA / gray image -> contiguous RGB uint8 (None if empty)."""
        import cv2

        img = np.asarray(image) if image is not None else None
        if img is None or img.ndim not in (2, 3) or img.size == 0 or min(img.shape[:2]) < 2:
            return None
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2 or img.shape[2] == 1:
            return cv2.cvtColor(img.reshape(img.shape[:2]), cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return np.ascontiguousarray(img[:, :, 2::-1])

    def detect(self, image_bgr: np.ndarray, t: float, cam_name: str) -> list[Person2D]:
        import torch

        if self.model is None:
            raise RuntimeError("Keypoint R-CNN detector is closed")
        rgb = self._to_rgb(image_bgr)
        if rgb is None:
            return []
        # (3, H, W) float 0..1 (the model normalizes and resizes internally)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div_(255.0)
        with torch.inference_mode():
            out = self.model([x])[0]
        return people_from_output(out, score_threshold=self.score_threshold,
                                  names=self.keypoint_names, fmt=self.format)

    def close(self) -> None:
        if self.model is None:
            return
        self.model = None
        if str(self.device).startswith("cuda"):
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                log.debug("torch.cuda.empty_cache: %s", e)


def create(device: str = "auto", score_threshold: float = 0.5, weights_path: str = "",
           min_size: int = 800, max_size: int = 1333, max_people: int = 20,
           **kwargs) -> KeypointRCNNDetector:
    """Factory of the ``keypoint_rcnn`` backend.

    * ``device``: auto / cpu / cuda.
    * ``score_threshold``: minimum person score, 0..1.
    * ``weights_path``: optional local weights file. Empty: the models folder, the PyTorch hub
      cache, or a download.
    * ``min_size`` / ``max_size``: the internal resize. A smaller ``min_size`` is faster and
      less accurate.
    * ``max_people``: the most persons returned per image.
    """
    return KeypointRCNNDetector(device=device, score_threshold=score_threshold,
                                weights_path=weights_path, min_size=min_size, max_size=max_size,
                                max_people=max_people, **kwargs)
