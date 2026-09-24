"""rtmlib pose backends (RTMPose, RTMW / DWPose, RTMO, ViTPose, RTMPose3D) as ``Detector2D``.

`rtmlib <https://github.com/Tau-J/rtmlib>`_ (Apache-2.0) runs the OpenMMLab pose models as
ONNX files, without mmcv / mmpose. One module serves six registry keys:

==================  ===============================================  ============  ===
key                 models for ``mode="balanced"``                   format        3D
==================  ===============================================  ============  ===
rtmpose             YOLOX-m + RTMPose-m (rtmlib ``Body``)            coco17        no
rtmpose_halpe26     YOLOX-m + RTMPose-m Halpe-26 (``BodyWithFeet``)  halpe26       no
rtmw_wholebody      YOLOX-m + RTMW-x (``Wholebody``)                 wholebody133  no
rtmo                RTMO-m, one stage (``Body`` ``RTMO_MODE``)       coco17        no
vitpose_onnx        YOLOX-m + ViTPose-B (easy_ViTPose ONNX)          coco17        no
rtmpose3d           YOLOX-m + RTMW3D-x (``Wholebody3d``)             wholebody133  yes
==================  ===============================================  ============  ===

``mode`` is ``lightweight`` / ``balanced`` / ``performance`` (RTMPose3D: ``balanced`` only);
the model files of each mode are taken from rtmlib's own solution classes
(``rtmlib.Body.MODE`` ...), so they follow the installed rtmlib version. ViTPose has no rtmlib
solution: ``lightweight`` / ``balanced`` / ``performance`` = ViTPose-S / B / L (COCO,
easy_ViTPose ONNX files on huggingface.co) with the YOLOX detector of the same ``Body`` mode.

**Pipeline.** Top-down models: YOLOX finds the person boxes (BGR input, as the mmdet YOLOX
models expect), then the pose model runs once per box; an image without a person box gives
``[]`` (rtmlib's own solutions would run the pose model on the whole image instead). RTMO
detects persons and keypoints in one pass. rtmlib maps all keypoints back to ORIGINAL image
pixels itself (top-down crop: ``rtmlib/tools/pose_estimation/rtmpose.py`` ``postprocess``;
RTMO letterbox: ``rtmo.py`` ``postprocess``); keypoints the model did not find (score <= 0)
are NaN. Every detected person is returned; the subject is picked later
(``poseboard.pose.subject``).

**Colour order.** rtmlib feeds the image unchanged (BGR from OpenCV) to every model, but the
top-down pose models were trained on RGB (mmpose configs, e.g.
``rtmpose-m_8xb256-420e_body8-256x192.py``, ``rtmw-x_8xb320-270e_cocktail14-384x288.py``,
``rtmw3d-l_8xb64_cocktail14-384x288.py``: ``data_preprocessor.bgr_to_rgb=True``; easy_ViTPose
normalizes RGB images) and rtmlib's normalization constants are the RGB ImageNet means. This
module therefore gives those models an RGB image (``rgb_input=True``, the default; False
reproduces rtmlib's behaviour). YOLOX and RTMO (no ``bgr_to_rgb`` in their configs) get BGR.

**Scores** (``Person2D.scores``, 0..1). RTMPose / RTMW / RTMW3D (SimCC heads) report the
maximum of the SimCC x/y logits (rtmlib ``get_simcc_maximum``: mean of x and y; 3D: minimum).
These are not probabilities: the heads are trained with a KL loss against Gaussian targets
of peak 1 (``normalize=False``), so a well localized keypoint scores about 0.6-1.0 and
occasionally a little more. ViTPose reports heatmap maxima (targets of peak 1), RTMO sigmoid
visibilities (0..1). All are mapped with ``clip(raw / score_scale, 0, 1)``, ``score_scale=1``
by default: the scale is kept, so rtmlib's / Pose2Sim's usual thresholds (0.3-0.5) keep their
meaning, and scores > 1 become 1. ``Person2D.score`` is the mean score of the 17 COCO body
keypoints (YOLOX does not return its box scores).

**3D (rtmpose3d).** RTMW3D predicts, per keypoint, image x/y and the depth relative to the hip
midpoint (``root_index=(11, 12)``) in meters, positive away from the camera (mmpose
``projects/rtmpose3d``: ``SimCC3DLabel.encode`` subtracts the root depth of the camera-space
keypoints; ``TopdownPoseEstimator3D`` adds it back and back-projects x/y). The depth is decoded
here from rtmlib's ``keypoints_simcc`` as ``(z_simcc / (D / 2) - 1) * z_range`` with the depth
bins ``D = 288`` of the model config (``input_size=(288, 384, 288)``) and ``z_range =
2.1744869`` m. (rtmlib 0.0.16 divides by the input *height* 384 instead of 288, which shrinks
and shifts the depths; its ``keypoints[..., 2]`` is therefore not used.) mmpose turns image
x/y into meters with an assumed camera (f = 1145 px, root depth 5.14 m), which is metric only
for images framed like Human3.6M. Here the image offsets from the hip midpoint are scaled by
one meters-per-pixel factor chosen so that the upper arms, forearms, thighs, shanks and trunk
sides get their typical lengths for a body of ``body_height`` (default 1.70 m; segment
lengths as fractions of height from Winter, *Biomechanics and Motor Control of Human
Movement*, 4th ed., Fig. 4.1), using the predicted depth of each segment (median over the
confident segments). ``Person2D.keypoints_3d`` is thus a metric skeleton in camera-aligned
axes (x right, y down, z away from the camera; a right-handed frame), centred between the hips,
which ``MultiViewEstimator`` places in the world with PnP when only one camera can be used.
Its depth (and so the distance from the camera) is approximate, like MediaPipe's.

**Model files.** rtmlib downloads the ONNX files on first use from download.openmmlab.com /
huggingface.co into ``~/.cache/rtmlib/hub/checkpoints`` (``$TORCH_HOME/hub/checkpoints`` or
``$XDG_CACHE_HOME/rtmlib/hub/checkpoints`` when set). A file put there by hand under the name
of its URL (the ``.zip``, or the ``.onnx`` with the same base name) is used without a download.
When the download fails the factories raise ``ModelUnavailable``; download the file elsewhere
and pass its local path (``.onnx``, or the mmdeploy ``.zip``) as ``pose_model=`` /
``det_model=`` (in the GUI: the **Pose model file** / **Person detector file** fields; empty =
the default model of the mode). Other factory options: ``backend`` ("onnxruntime", "opencv",
"openvino"), ``det_input_size`` / ``pose_input_size`` (taken from the ONNX file when a custom
model is given), ``det_mode`` (see below), ``det_score_thr`` / ``nms_thr`` (RTMO, and YOLOX
files without built-in NMS; for YOLOX files with built-in NMS, as rtmlib's mmdeploy files seem
to be, rtmlib applies a fixed 0.3 threshold), ``rgb_input``, ``score_scale`` and, for
rtmpose3d, ``body_height``.

**Person classes.** A YOLOX file without built-in NMS (ONNX output ``(1, N, 5 + classes)``,
e.g. the 80-class COCO YOLOX releases on GitHub) returns boxes of every class; rtmlib's default
``det_mode="human"`` would pass cars, benches or dogs on as persons. Such files are therefore
switched to ``det_mode="multiclass"`` automatically (read from the ONNX output shape), and only
class ``person_class`` (0) is kept. Files with built-in NMS (last dimension 5) keep "human".
"""

from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import dataclass

import cv2
import numpy as np

from poseboard.pose.detectors.base import Detector2D, Person2D, keypoint_bbox
from poseboard.pose.formats import (COCO17, COCO17_NAMES, HALPE26, WHOLEBODY133,
                                    KeypointFormat)

log = logging.getLogger(__name__)

__all__ = ["KINDS", "MODES", "ModelUnavailable", "RTMLibDetector", "RTMPose3DDetector",
           "VITPOSE_ONNX", "canonical_name", "create_rtmo", "create_rtmpose",
           "create_rtmpose3d", "create_rtmpose_halpe26", "create_rtmw_wholebody",
           "create_vitpose_onnx", "index_map", "keep_persons_only", "resolve_device"]

MODES = ("balanced", "performance", "lightweight")
RUNTIMES = ("onnxruntime", "opencv", "openvino")
INSTALL_HINT = "pip install rtmlib onnxruntime   (NVIDIA GPU: onnxruntime-gpu)"

# ------------------------------------------------------------------ keypoint orders of rtmlib
# The order of the models' outputs, as listed in rtmlib's own metadata (keypoint_info ids).
# rtmlib/visualization/skeleton/coco17.py, ids 0-16 (lines 4-81; left before right).
RTMLIB_COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
# rtmlib/visualization/skeleton/halpe26.py: ids 0-16 as COCO, then 17 head (line 106), 18 neck,
# 19 hip, 20 left_big_toe (line 124), 21 right_big_toe, 22 left_small_toe, 23 right_small_toe,
# 24 left_heel, 25 right_heel (line 154).
RTMLIB_HALPE26 = RTMLIB_COCO17 + (
    "head", "neck", "hip", "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
)
# rtmlib/visualization/skeleton/coco133.py: ids 0-16 as COCO, 17-22 feet (lines 47-69: left
# big toe, small toe, heel, then right), 23-90 "face-0".."face-67", 91 "left_hand_root"
# (line 207), 92-111 left thumb1-4, forefinger1-4, middle_finger1-4, ring_finger1-4,
# pinky_finger1-4, 112 "right_hand_root" (line 312), 113-132 the right fingers.
_FINGERS = ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger")


def _rtmlib_hand(side: str) -> tuple[str, ...]:
    return (f"{side}_hand_root",) + tuple(f"{side}_{f}{j}" for f in _FINGERS for j in range(1, 5))


RTMLIB_COCO133 = (RTMLIB_COCO17
                  + ("left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
                     "right_small_toe", "right_heel")
                  + tuple(f"face-{i}" for i in range(68))
                  + _rtmlib_hand("left") + _rtmlib_hand("right"))


def canonical_name(lib_name: str) -> str:
    """PoseBoard's name of an rtmlib keypoint name: ``face-3`` -> ``face_3``,
    ``left_hand_root`` -> ``left_hand_0``, ``left_forefinger2`` -> ``left_hand_6``
    (hand points 1-20 = thumb, forefinger, middle, ring, pinky, 4 joints each, root side
    first); other names are unchanged."""
    if lib_name.startswith("face-"):
        return "face_" + lib_name[5:]
    for side in ("left", "right"):
        prefix = side + "_"
        if lib_name == prefix + "hand_root":
            return f"{side}_hand_0"
        for fi, finger in enumerate(_FINGERS):
            head = prefix + finger
            rest = lib_name[len(head):]
            if lib_name.startswith(head) and rest.isdigit() and 1 <= int(rest) <= 4:
                return f"{side}_hand_{1 + 4 * fi + int(rest) - 1}"
    return lib_name


def index_map(lib_names: tuple[str, ...] | list[str], fmt: KeypointFormat) -> np.ndarray:
    """Indices ``idx`` so that ``library_output[idx]`` is in the order of ``fmt``. Raises
    ValueError unless the library layout has exactly the keypoints of ``fmt``."""
    canon = [canonical_name(n) for n in lib_names]
    pos = {n: i for i, n in enumerate(canon)}
    missing = [n for n in fmt.names if n not in pos]
    if missing or len(canon) != len(fmt.names) or len(pos) != len(canon):
        raise ValueError(f"rtmlib keypoint layout ({len(canon)} points) does not match format "
                         f"{fmt.key} ({len(fmt.names)} points); missing: {missing[:5]}")
    return np.array([pos[n] for n in fmt.names], dtype=np.intp)


# ------------------------------------------------------------------ backends
# ViTPose ONNX files of easy_ViTPose (COCO-17, input 192 x 256), the models rtmlib's README uses
# with its ``ViTPose`` class. Mode -> (URL, pose_input_size (w, h)).
_EASY_VITPOSE = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/"
VITPOSE_ONNX = {
    "lightweight": (_EASY_VITPOSE + "vitpose-s-coco.onnx", (192, 256)),
    "balanced": (_EASY_VITPOSE + "vitpose-b-coco.onnx", (192, 256)),
    "performance": (_EASY_VITPOSE + "vitpose-l-coco.onnx", (192, 256)),
}


@dataclass(frozen=True)
class _Kind:
    key: str
    label: str
    fmt: KeypointFormat
    lib_names: tuple[str, ...]
    pose_class: str  # rtmlib class of the pose model
    solution: str | None  # rtmlib solution class whose table gives the default model files
    table: str = "MODE"
    one_stage: bool = False  # RTMO: no separate person detector
    rgb: bool = True  # the pose model was trained on RGB images (see the module docstring)
    provides_3d: bool = False


KINDS: dict[str, _Kind] = {k.key: k for k in (
    _Kind("rtmpose", "RTMPose body (COCO-17, rtmlib)", COCO17, RTMLIB_COCO17, "RTMPose", "Body"),
    _Kind("rtmpose_halpe26", "RTMPose body + feet (Halpe-26, rtmlib)", HALPE26, RTMLIB_HALPE26,
          "RTMPose", "BodyWithFeet"),
    _Kind("rtmw_wholebody", "RTMW / DWPose whole body (133, rtmlib)", WHOLEBODY133,
          RTMLIB_COCO133, "RTMPose", "Wholebody"),
    _Kind("rtmo", "RTMO one-stage (COCO-17, rtmlib)", COCO17, RTMLIB_COCO17, "RTMO", "Body",
          table="RTMO_MODE", one_stage=True, rgb=False),
    _Kind("vitpose_onnx", "ViTPose (COCO-17, rtmlib ONNX)", COCO17, RTMLIB_COCO17, "ViTPose", None),
    _Kind("rtmpose3d", "RTMPose3D whole body 3D (rtmlib)", WHOLEBODY133, RTMLIB_COCO133,
          "RTMPose3d", "Wholebody3d", provides_3d=True),
)}

# RTMW3D decoding constants (mmpose projects/rtmpose3d: rtmw3d-l_8xb64_cocktail14-384x288.py
# ``codec.input_size=(288, 384, 288)``, ``simcc_split_ratio=2.0``; simcc_3d_label.py
# ``z_range = 2.1744869``).
RTMW3D_Z_BINS = 288.0
RTMW3D_Z_RANGE = 2.1744869
SIMCC_SPLIT_RATIO = 2.0

# Body segments (a, b, length / body height) used to scale RTMW3D x/y to meters (Winter,
# Biomechanics and Motor Control of Human Movement, Fig. 4.1: shoulder 0.818 H, elbow 0.630 H,
# wrist 0.485 H, hip 0.530 H, knee 0.285 H, ankle 0.039 H).
SEGMENTS = (
    ("left_shoulder", "left_elbow", 0.186), ("right_shoulder", "right_elbow", 0.186),
    ("left_elbow", "left_wrist", 0.146), ("right_elbow", "right_wrist", 0.146),
    ("left_hip", "left_knee", 0.245), ("right_hip", "right_knee", 0.245),
    ("left_knee", "left_ankle", 0.246), ("right_knee", "right_ankle", 0.246),
    ("left_shoulder", "left_hip", 0.288), ("right_shoulder", "right_hip", 0.288),
)
SEGMENT_MIN_SCORE = 0.3


class ModelUnavailable(RuntimeError):
    """A model file could not be downloaded (offline, blocked host, ...) or found."""


# ------------------------------------------------------------------ helpers
def _available_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return list(ort.get_available_providers())


def resolve_device(device: str | None, backend: str = "onnxruntime") -> str:
    """The rtmlib device for ``device``: "auto" = "cuda" when ONNX Runtime has the CUDA provider
    (onnxruntime-gpu), else "cpu". A requested "cuda" without that provider falls back to "cpu"
    with a warning."""
    d = str(device or "auto").strip().lower()
    if backend != "onnxruntime":
        return "cpu" if d == "auto" else d
    has_cuda = "CUDAExecutionProvider" in _available_providers()
    if d == "auto":
        return "cuda" if has_cuda else "cpu"
    if d.startswith("cuda") and not has_cuda:
        log.warning("rtmlib: device %r requested but ONNX Runtime has no CUDA provider "
                    "(install onnxruntime-gpu); using the CPU", d)
        return "cpu"
    return d


def _download(url: str) -> str:
    """Local path of the model at ``url`` (rtmlib's cached download; unzips mmdeploy zips)."""
    from rtmlib.tools.file import download_checkpoint

    return download_checkpoint(url)


def _is_url(src: str) -> bool:
    return str(src).lower().startswith(("http://", "https://"))


def _local_model(path: str) -> str:
    """``path`` itself, or the ``end2end.onnx`` extracted next to it from an mmdeploy zip."""
    if not path.lower().endswith(".zip"):
        return path
    out = os.path.splitext(path)[0] + ".onnx"
    if not os.path.isfile(out):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith("end2end.onnx")]
            if not names:
                raise ModelUnavailable(f"{path} contains no end2end.onnx")
            tmp = out + ".part"
            with z.open(names[0]) as src, open(tmp, "wb") as dst:
                dst.write(src.read())
            os.replace(tmp, out)
    return out


# GUI field (Record tab) of each factory option that takes a model file
_GUI_FIELDS = {"pose_model": "'Pose model file'", "det_model": "'Person detector file'"}


def _rtmlib_cache_dir() -> str:
    """rtmlib's download folder (rtmlib/tools/file.py ``download_checkpoint``)."""
    try:
        from rtmlib.tools.file import _get_rtmhub_dir

        return os.path.join(_get_rtmhub_dir(), "checkpoints")
    except Exception:  # noqa: BLE001  (other rtmlib version)
        home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"),
                                                    "rtmlib"))
        return os.path.join(os.path.expanduser(home), "hub", "checkpoints")


def _model_file(src: str, what: str, option: str, label: str) -> str:
    if _is_url(src):
        try:
            return _download(src)
        except Exception as e:  # noqa: BLE001  (HTTP errors, proxies, broken zips, ...)
            raise ModelUnavailable(
                f"{label}: cannot download the {what} model {src} ({type(e).__name__}: {e}). "
                "rtmlib downloads its models from download.openmmlab.com / huggingface.co; if "
                "these hosts are blocked, download the file elsewhere and select it as "
                f"{_GUI_FIELDS.get(option, option)} (Python: {option}=<path>), or copy it "
                f"unchanged into {_rtmlib_cache_dir()} (rtmlib's download cache).") from e
    path = os.path.expanduser(str(src))
    if not os.path.isfile(path):
        raise ModelUnavailable(f"{label}: {what} model file not found: {path}")
    return _local_model(path)


def _onnx_input_hw(model) -> tuple[int, int] | None:
    """(height, width) of the model's ONNX input, if the ONNX Runtime session tells it."""
    try:
        shape = model.session.get_inputs()[0].shape
        h, w = int(shape[2]), int(shape[3])
    except Exception:  # noqa: BLE001  (other runtime, dynamic axes, ...)
        return None
    return (h, w) if h > 0 and w > 0 else None


def _onnx_output_last_dim(model) -> int | None:
    """Last dimension of the model's first ONNX output, if the ONNX Runtime session tells it."""
    try:
        n = model.session.get_outputs()[0].shape[-1]
    except Exception:  # noqa: BLE001  (other runtime, no session, ...)
        return None
    return n if isinstance(n, int) and not isinstance(n, bool) else None  # symbolic: str


def keep_persons_only(det) -> str:
    """Make a YOLOX file without built-in NMS return class ids, so ``person_boxes`` keeps only
    persons: rtmlib's ``det_mode="human"`` returns the boxes of every class for such files (see
    the module docstring). Files with built-in NMS (output ``(1, N, 5)``) and runtimes that do
    not tell the output shape are left alone (rtmlib's "multiclass" mode fails on files with
    NMS). Returns the resulting ``det_mode``."""
    mode = getattr(det, "det_mode", "human")
    if mode == "human":
        n = _onnx_output_last_dim(det)
        if n is not None and n != 5:  # rtmlib: 4 or > 5 = no built-in NMS
            det.det_mode = mode = "multiclass"
    return mode


def _simcc_z_bins(model) -> float | None:
    """Depth bins D of an RTMW3D model from the length of its SimCC z output (D * 2)."""
    try:
        n = int(model.session.get_outputs()[2].shape[-1])
    except Exception:  # noqa: BLE001
        return None
    return n / SIMCC_SPLIT_RATIO if n > 0 else None


def _as_bgr8(image) -> np.ndarray | None:
    """A 3-channel uint8 BGR image, or None for an empty/missing image."""
    if image is None:
        return None
    img = np.asarray(image)
    if img.size == 0 or img.ndim not in (2, 3) or min(img.shape[:2]) < 2:
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2 or img.shape[2] == 1:
        return cv2.cvtColor(img.reshape(img.shape[:2]), cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img if img.shape[2] == 3 else None


def _registry_label(key: str, fallback: str) -> str:
    try:
        from poseboard.pose.detectors import BACKENDS

        return BACKENDS[key].label
    except Exception:  # noqa: BLE001
        return fallback


# ------------------------------------------------------------------ detectors
class RTMLibDetector(Detector2D):
    """A 2D rtmlib pose model (plus YOLOX person detector for the top-down models).

    ``pose_model`` / ``det_model`` are rtmlib model objects (``RTMPose``, ``ViTPose``, ``RTMO``,
    ``RTMPose3d`` / ``YOLOX``) or anything with the same call signature; the factories
    (``create_rtmpose`` ...) build them. ``det_model`` must be None for the one-stage RTMO and
    is required otherwise."""

    def __init__(self, key: str, pose_model, det_model=None, *, fmt: KeypointFormat | None = None,
                 lib_names: tuple[str, ...] | None = None, label: str | None = None,
                 rgb_input: bool | None = None, score_scale: float = 1.0, person_class: int = 0,
                 min_box_px: float = 4.0, device: str = "cpu",
                 model_files: dict | None = None):
        super().__init__()
        kind = KINDS[key]
        self.key = key
        self.label = label or _registry_label(key, kind.label)
        self.format = fmt or kind.fmt
        self.provides_3d = kind.provides_3d
        self.one_stage = kind.one_stage
        if self.one_stage and det_model is not None:
            raise ValueError(f"{self.label} is a one-stage model: no person detector needed")
        if not self.one_stage and det_model is None:
            raise ValueError(f"{self.label} needs a person detector (det_model)")
        self.pose_model = pose_model
        self.det_model = det_model
        self._index = index_map(lib_names or kind.lib_names, self.format)
        self._body = np.array([i for i in (self.format.find(n) for n in COCO17_NAMES)
                               if i is not None], dtype=np.intp)
        self.rgb_input = kind.rgb if rgb_input is None else bool(rgb_input)
        self.score_scale = float(score_scale)
        if not self.score_scale > 0:
            raise ValueError("score_scale must be > 0")
        self.person_class = int(person_class)
        self.min_box_px = float(min_box_px)
        self.device = device
        self.model_files = dict(model_files or {})

    # -------------------------------------------------------------- helpers
    def normalize_scores(self, raw: np.ndarray) -> np.ndarray:
        """Model scores -> 0..1: ``clip(raw / score_scale, 0, 1)`` (see the module docstring)."""
        raw = np.nan_to_num(np.asarray(raw, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(raw / self.score_scale, 0.0, 1.0)

    def person_boxes(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(boxes as given to the pose model, the same boxes clipped to the image), (N, 4)
        x1, y1, x2, y2 in pixels; boxes of other classes and boxes with less than
        ``min_box_px`` inside the image are dropped."""
        out = self.det_model(image_bgr)  # YOLOX: BGR, no normalization (mmdet YOLOX)
        if isinstance(out, tuple):  # det_mode="multiclass": (boxes, class ids)
            boxes, cls = out[0], np.asarray(out[1]).reshape(-1)
            boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
            if len(cls) == len(boxes):
                boxes = boxes[cls.astype(int) == self.person_class]
        else:
            boxes = np.asarray(out, np.float64).reshape(-1, 4)
        boxes = boxes[np.all(np.isfinite(boxes), axis=1)]
        h, w = image_bgr.shape[:2]
        clipped = boxes.copy()
        clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, w)
        clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, h)
        keep = ((clipped[:, 2] - clipped[:, 0] >= self.min_box_px)
                & (clipped[:, 3] - clipped[:, 1] >= self.min_box_px))
        return boxes[keep], clipped[keep]

    def _pose_image(self, image_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if self.rgb_input else image_bgr

    def _run(self, image_bgr: np.ndarray):
        """(pose model output, clipped boxes or None); None when no person box was found."""
        if self.one_stage:
            return self.pose_model(self._pose_image(image_bgr)), None
        boxes, clipped = self.person_boxes(image_bgr)
        if not len(boxes):
            return None
        out = self.pose_model(self._pose_image(image_bgr), bboxes=[b.tolist() for b in boxes])
        return out, clipped

    def _person(self, kp: np.ndarray, raw: np.ndarray, bbox: np.ndarray | None,
                keypoints_3d: np.ndarray | None = None) -> Person2D | None:
        """``Person2D`` from one person's keypoints (K, 2), raw scores (K,) and optional 3D
        skeleton (K, 3), all already in the order of ``self.format``."""
        kp = np.array(kp, np.float64).reshape(-1, 2)
        raw = np.asarray(raw, np.float64).reshape(-1)
        found = (raw > 0) & np.all(np.isfinite(kp), axis=1)
        if not found.any():  # e.g. RTMO's all-zero placeholder when nobody is detected
            return None
        kp[~found] = np.nan
        sc = self.normalize_scores(raw)
        sc[~found] = 0.0
        if bbox is None:
            bbox = keypoint_bbox(kp, sc, SEGMENT_MIN_SCORE)
            if bbox is None:
                bbox = keypoint_bbox(kp)
        body = sc[self._body] if len(self._body) else sc
        if keypoints_3d is not None:
            keypoints_3d = np.array(keypoints_3d, np.float64).reshape(-1, 3)
            keypoints_3d[~found] = np.nan
        return Person2D(kp, sc, bbox=bbox, score=float(body.mean()), keypoints_3d=keypoints_3d)

    def _people(self, out, boxes: np.ndarray | None) -> list[Person2D]:
        # rtmlib RTMPose / ViTPose / RTMO __call__ -> (keypoints (N, K, 2), scores (N, K)),
        # one row per box for the top-down models, in the order of the boxes
        kps, raw = np.asarray(out[0], np.float64), np.asarray(out[1], np.float64)
        if kps.ndim == 2:
            kps, raw = kps[None], raw[None]
        people = []
        for i in range(len(kps)):
            box = boxes[i] if boxes is not None and i < len(boxes) else None
            p = self._person(kps[i, :, :2][self._index], raw[i][self._index], box)
            if p is not None:
                people.append(p)
        return people

    # -------------------------------------------------------------- Detector2D
    def detect(self, image_bgr: np.ndarray, t: float, cam_name: str) -> list[Person2D]:
        if self.pose_model is None:
            raise RuntimeError(f"{self.label}: detector is closed")
        img = _as_bgr8(image_bgr)
        if img is None:
            return []
        res = self._run(img)
        if res is None:
            return []
        out, boxes = res
        return self._people(out, boxes)

    def close(self) -> None:
        self.pose_model = None
        self.det_model = None

    def info(self) -> dict:
        d = super().info()
        d.update(device_used=self.device, rgb_input=self.rgb_input, score_scale=self.score_scale,
                 model_files=dict(self.model_files))
        return d


class RTMPose3DDetector(RTMLibDetector):
    """RTMW3D (rtmlib ``RTMPose3d``): 2D keypoints plus a metric, hip-centred 3D skeleton in
    camera-aligned axes (``Person2D.keypoints_3d``; see the module docstring)."""

    def __init__(self, key: str = "rtmpose3d", pose_model=None, det_model=None, *,
                 body_height: float = 1.70, z_bins: float | None = None,
                 z_range: float | None = None, **kw):
        super().__init__(key, pose_model, det_model, **kw)
        self.body_height = float(body_height)
        if not 0.5 <= self.body_height <= 2.5:
            raise ValueError(f"body_height must be in meters (0.5-2.5), not {body_height!r}")
        zr = z_range if z_range is not None else getattr(pose_model, "z_range", None)
        self.z_range = float(zr if zr is not None else RTMW3D_Z_RANGE)
        self.z_bins = float(z_bins or _simcc_z_bins(pose_model) or RTMW3D_Z_BINS)
        self._hips = [self.format.index("left_hip"), self.format.index("right_hip")]
        self._segments = [(self.format.index(a), self.format.index(b), frac)
                          for a, b, frac in SEGMENTS]

    def depth_m(self, z_simcc: np.ndarray) -> np.ndarray:
        """Depth (meters, relative to the model's root, + away from the camera) from the SimCC
        z location (``keypoints_simcc[..., 2]``, in input pixels 0..D)."""
        return (np.asarray(z_simcc, np.float64) / (self.z_bins / 2.0) - 1.0) * self.z_range

    def meters_per_pixel(self, kp: np.ndarray, z: np.ndarray, scores: np.ndarray) -> float | None:
        """Scale of the image offsets (meters per pixel at the body) that gives the confident
        limb and trunk segments their typical lengths for ``body_height``; None if fewer than 2
        segments can be used."""
        est = []
        for a, b, frac in self._segments:
            if scores[a] < SEGMENT_MIN_SCORE or scores[b] < SEGMENT_MIN_SCORE:
                continue
            d_px = float(np.linalg.norm(kp[a] - kp[b]))
            dz = float(z[a] - z[b])
            if not (np.isfinite(d_px) and np.isfinite(dz)) or d_px < 2.0:
                continue
            L = frac * self.body_height
            # the segment's extent parallel to the image; at least 30 % of its length, so a
            # segment pointing at the camera (or a poor depth) cannot give a tiny scale
            est.append(np.sqrt(max(L * L - dz * dz, (0.3 * L) ** 2)) / d_px)
        if len(est) < 2:
            return None
        return float(np.median(est))

    def metric_3d(self, kp: np.ndarray, z: np.ndarray, scores: np.ndarray) -> np.ndarray | None:
        """(K, 3) meters, hip-centred, x right / y down / z away from the camera; None when the
        hips or the scale are unknown. ``kp`` pixels, ``z`` depths (m), in format order."""
        hips = self._hips
        if (np.any(np.asarray(scores)[hips] <= 0) or not np.all(np.isfinite(kp[hips]))
                or not np.all(np.isfinite(z[hips]))):
            return None
        m = self.meters_per_pixel(kp, z, scores)
        if m is None:
            return None
        root_uv = kp[hips].mean(axis=0)
        root_z = float(z[hips].mean())
        return np.column_stack([(kp - root_uv) * m, z - root_z])

    def _people(self, out, boxes: np.ndarray | None) -> list[Person2D]:
        # rtmlib RTMPose3d.__call__ -> (keypoints, scores, keypoints_simcc, keypoints_2d);
        # keypoints_2d (N, K, 2) are image pixels, keypoints_simcc[..., 2] the z location in
        # input pixels (0..D); rtmlib's keypoints[..., 2] is not used (see the module docstring)
        raw = np.asarray(out[1], np.float64)
        simcc = np.asarray(out[2], np.float64)
        kp2 = np.asarray(out[3], np.float64)
        if kp2.ndim == 2:
            raw, simcc, kp2 = raw[None], simcc[None], kp2[None]
        people = []
        for i in range(len(kp2)):
            box = boxes[i] if boxes is not None and i < len(boxes) else None
            kp = kp2[i, :, :2][self._index]
            r = raw[i][self._index]
            z = self.depth_m(simcc[i, :, 2][self._index])
            k3 = self.metric_3d(kp, z, self.normalize_scores(r))
            p = self._person(kp, r, box, k3)
            if p is not None:
                people.append(p)
        return people


# ------------------------------------------------------------------ factories
def _default_models(kind: _Kind, mode: str, rtmlib) -> tuple[str | None, tuple | None, str, tuple]:
    """(det URL, det_input_size, pose URL, pose_input_size) of ``mode`` from rtmlib's tables."""
    label = kind.label
    if kind.key == "vitpose_onnx":
        if mode not in VITPOSE_ONNX:
            raise ValueError(f"{label}: mode must be one of {', '.join(VITPOSE_ONNX)}, "
                             f"not {mode!r}")
        det = rtmlib.Body.MODE[mode]
        pose_url, pose_size = VITPOSE_ONNX[mode]
        return det["det"], tuple(det["det_input_size"]), pose_url, tuple(pose_size)
    table = getattr(getattr(rtmlib, kind.solution), kind.table)
    if mode not in table:
        raise ValueError(f"{label}: mode must be one of {', '.join(table)}, not {mode!r}")
    m = table[mode]
    det, det_size = (m["det"], tuple(m["det_input_size"])) if "det" in m else (None, None)
    return det, det_size, m["pose"], tuple(m["pose_input_size"])


def _build(cls, src: str, what: str, option: str, label: str, **kw):
    path = _model_file(src, what, option, label)
    try:
        return cls(path, **kw), path
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{label}: cannot load the {what} model {path}: {e}") from e


def _create(key: str, mode: str = "balanced", device: str = "auto",
            backend: str = "onnxruntime", *, det_model: str | None = None,
            det_input_size: tuple | None = None, det_mode: str = "human",
            pose_model: str | None = None, pose_input_size: tuple | None = None,
            det_score_thr: float | None = None, nms_thr: float | None = None,
            rgb_input: bool | None = None, score_scale: float = 1.0, person_class: int = 0,
            body_height: float | None = None) -> RTMLibDetector:
    kind = KINDS[key]
    label = _registry_label(key, kind.label)
    det_model = (str(det_model).strip() or None) if det_model else None  # "" = the default
    pose_model = (str(pose_model).strip() or None) if pose_model else None
    try:
        import rtmlib
    except ImportError as e:
        raise RuntimeError(f"{label} needs rtmlib: {INSTALL_HINT}") from e
    if backend not in RUNTIMES:
        raise ValueError(f"{label}: backend must be one of {', '.join(RUNTIMES)}, not {backend!r}")
    mode = str(mode or "balanced")
    def_det, def_det_size, def_pose, def_pose_size = _default_models(kind, mode, rtmlib)
    dev = resolve_device(device, backend)
    common = dict(backend=backend, device=dev)
    files = {}

    det = None
    if not kind.one_stage:
        det_kw = dict(common)
        if det_mode and det_mode != "human":
            det_kw["det_mode"] = det_mode
        size = tuple(det_input_size or def_det_size)
        det, files["det"] = _build(rtmlib.YOLOX, det_model or def_det, "person detector",
                                   "det_model", label, model_input_size=size, **det_kw)
        if det_model and det_input_size is None:  # a custom detector: use its ONNX input size
            hw = _onnx_input_hw(det)
            if hw is not None:
                det.model_input_size = hw  # YOLOX: (height, width)
        if det_score_thr is not None:
            det.score_thr = float(det_score_thr)
        if nms_thr is not None:
            det.nms_thr = float(nms_thr)
        keep_persons_only(det)

    pose_cls = getattr(rtmlib, kind.pose_class)
    size = tuple(pose_input_size or def_pose_size)
    pose, files["pose"] = _build(pose_cls, pose_model or def_pose, "pose", "pose_model", label,
                                 model_input_size=size, to_openpose=False, **common)
    if pose_model and pose_input_size is None:
        hw = _onnx_input_hw(pose)
        if hw is not None:
            # RTMO letterbox: (height, width); top-down crops: (width, height)
            pose.model_input_size = hw if kind.one_stage else (hw[1], hw[0])
    if kind.one_stage:
        if det_score_thr is not None:
            pose.score_thr = float(det_score_thr)
        if nms_thr is not None:
            pose.nms_thr = float(nms_thr)

    kw = dict(rgb_input=rgb_input, score_scale=score_scale, person_class=person_class,
              device=dev, model_files=files)
    if kind.provides_3d:
        detector = RTMPose3DDetector(key, pose, det, body_height=body_height or 1.70, **kw)
    else:
        if body_height is not None:
            raise TypeError(f"{label}: body_height is only used by rtmpose3d")
        detector = RTMLibDetector(key, pose, det, **kw)
    opts = {"mode": mode, "device": device, "backend": backend}
    for name, val in (("det_model", det_model), ("pose_model", pose_model),
                      ("det_input_size", det_input_size), ("pose_input_size", pose_input_size),
                      ("det_score_thr", det_score_thr), ("nms_thr", nms_thr),
                      ("rgb_input", rgb_input), ("body_height", body_height)):
        if val is not None:
            opts[name] = val
    if det_mode != "human":
        opts["det_mode"] = det_mode
    if score_scale != 1.0:
        opts["score_scale"] = score_scale
    detector.options = opts
    return detector


def create_rtmpose(mode: str = "balanced", device: str = "auto", backend: str = "onnxruntime",
                   **kw) -> RTMLibDetector:
    """YOLOX + RTMPose, COCO-17 (rtmlib ``Body``)."""
    return _create("rtmpose", mode, device, backend, **kw)


def create_rtmpose_halpe26(mode: str = "balanced", device: str = "auto",
                           backend: str = "onnxruntime", **kw) -> RTMLibDetector:
    """YOLOX + RTMPose Halpe-26: body with head, neck, mid-hip, heels and toes
    (rtmlib ``BodyWithFeet``)."""
    return _create("rtmpose_halpe26", mode, device, backend, **kw)


def create_rtmw_wholebody(mode: str = "balanced", device: str = "auto",
                          backend: str = "onnxruntime", **kw) -> RTMLibDetector:
    """YOLOX + RTMW (DWPose successor), COCO-WholeBody 133 (rtmlib ``Wholebody``)."""
    return _create("rtmw_wholebody", mode, device, backend, **kw)


def create_rtmo(mode: str = "balanced", device: str = "auto", backend: str = "onnxruntime",
                **kw) -> RTMLibDetector:
    """RTMO one-stage multi-person model, COCO-17 (rtmlib ``RTMO``)."""
    return _create("rtmo", mode, device, backend, **kw)


def create_vitpose_onnx(mode: str = "balanced", device: str = "auto",
                        backend: str = "onnxruntime", **kw) -> RTMLibDetector:
    """YOLOX + ViTPose-S/B/L (easy_ViTPose ONNX), COCO-17 (rtmlib ``ViTPose``)."""
    return _create("vitpose_onnx", mode, device, backend, **kw)


def create_rtmpose3d(mode: str = "balanced", device: str = "auto", backend: str = "onnxruntime",
                     body_height: float = 1.70, **kw) -> RTMPose3DDetector:
    """YOLOX + RTMW3D, COCO-WholeBody 133 with a metric 3D skeleton (rtmlib ``Wholebody3d``)."""
    return _create("rtmpose3d", mode, device, backend, body_height=body_height, **kw)
