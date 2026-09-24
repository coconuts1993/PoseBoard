"""Google MoveNet single-pose models (Lightning / Thunder) run with ONNX Runtime, as a
``Detector2D`` (backend key ``movenet``).

MoveNet is a very fast single-person model (Apache-2.0, Google; TF Hub / Kaggle
``google/movenet/singlepose/{lightning,thunder}/4``). The official releases are TensorFlow /
TFLite files; this backend runs ONNX exports of them:

* option ``model_path`` = your own ``.onnx`` file (e.g. exported with
  ``python -m tf2onnx.convert --saved-model <movenet saved model> --output movenet.onnx
  --opset 13`` from the Kaggle/TF Hub SavedModel), or
* empty ``model_path``: ``movenet_singlepose_<model>_4.onnx`` from the PoseBoard models folder
  (``poseboard.pose.mediapipe_backend.MODEL_DIR``); if it is missing it is downloaded from
  GitHub: the tf2onnx conversions of the TF Hub v4 models in Kazuhito00/MoveNet-Python-Example
  (Apache-2.0), pinned to commit ``MOVENET_COMMIT`` and checked by SHA-256 (``MODEL_FILES``).
  Without internet, download the file from ``MODEL_URLS[model]`` and save it in that folder.

Model input (read from the ONNX file): one image ``[1, H, W, 3]`` (or ``[1, 3, H, W]``), RGB,
values 0..255, as int32 (TF Hub SavedModel exports), uint8 (TFLite conversions) or float
(values still 0..255). H = W = 192 (Lightning) / 256 (Thunder); a dynamic size uses
``input_size`` (default by ``model``).

Preprocessing and coordinates: the network sees a square region of the ORIGINAL image,
resized to H x W (area-averaged when shrinking, then bilinear; parts outside the image are
black). At first (and whenever the subject is lost) the region is the whole image, centered
and padded to a square (as ``tf.image.resize_with_pad`` in the TF Hub tutorial). With
``smart_crop`` (default on) later frames of the same camera use the tutorial's "cropping
algorithm" (``determine_crop_region``: a square around the hips sized from the torso and body
extent of the previous frame's keypoints with score > 0.2), which helps when the person is
small in the image. The output ``[1, 1, 17, 3]`` holds ``(y, x, score)`` per keypoint with y/x
normalized to the network input; they are mapped back through the region to ORIGINAL image
pixels (continuous coordinates, like MediaPipe): ``x = x0 + x_n * region_width``.

Scores: the model's keypoint confidences are sigmoid outputs in 0..1 (model card: "prediction
confidence scores ..., also in the range [0.0, 1.0]"); they are only clipped. MoveNet always
outputs 17 locations, even without a person: fewer than ``min_keypoints`` (3) keypoints with a
score >= ``min_keypoint_score`` (0.2, the tutorial's MIN_CROP_KEYPOINT_SCORE) means no person.
``Person2D.score`` = mean keypoint score, ``bbox`` = box of the keypoints above that score.

Keypoint order: the model card lists nose, left eye, right eye, left ear, right ear, left
shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip,
left knee, right knee, left ankle, right ankle (the TF Hub tutorial's ``KEYPOINT_DICT``), i.e.
the COCO order = ``FORMATS["coco17"]`` index for index; left/right are the person's own sides.
MoveNet MultiPose files (output ``[1, 6, 56]``: 17 x (y, x, score) + box ymin, xmin, ymax, xmax,
score per person) are accepted too and return every person with a score >=
``min_person_score``.

Device: ``auto`` uses ONNX Runtime's CUDA provider when onnxruntime-gpu is installed, else the
CPU (Lightning runs at a few milliseconds per image on a CPU).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from poseboard.pose.detectors.base import Detector2D, Person2D, keypoint_bbox
from poseboard.pose.formats import COCO17

__all__ = ["INPUT_SIZE", "MODEL_FILES", "MODEL_URLS", "MOVENET_COMMIT", "MoveNetDetector",
           "create", "crop_and_resize", "init_region", "next_region", "parse_output",
           "resolve_model", "resolve_providers"]

log = logging.getLogger(__name__)

MOVENET_COMMIT = "515743a113e49b4f31677b1f6252c0041f237e30"
_BASE_URL = ("https://raw.githubusercontent.com/Kazuhito00/MoveNet-Python-Example/"
             f"{MOVENET_COMMIT}/onnx/")
# model -> (file name, SHA-256)
MODEL_FILES = {
    "lightning": ("movenet_singlepose_lightning_4.onnx",
                  "402dadf6a184171f293bc2ef4edf2f96c4b3e0ed6d3b5890198d036a3087b623"),
    "thunder": ("movenet_singlepose_thunder_4.onnx",
                "7fae4dc3cdd07ebd5535a1e9515c6baf945daeb60708158f03f915fc8e42eac6"),
}
MODEL_URLS = {m: _BASE_URL + f for m, (f, _) in MODEL_FILES.items()}
INPUT_SIZE = {"lightning": 192, "thunder": 256}
MIN_CROP_KEYPOINT_SCORE = 0.2  # TF Hub MoveNet tutorial
_TORSO = tuple(COCO17.index(n) for n in ("left_shoulder", "right_shoulder", "left_hip",
                                          "right_hip"))
_HIPS = (COCO17.index("left_hip"), COCO17.index("right_hip"))
_SHOULDERS = (COCO17.index("left_shoulder"), COCO17.index("right_shoulder"))
_DTYPES = {"tensor(int32)": np.int32, "tensor(uint8)": np.uint8, "tensor(int64)": np.int64,
           "tensor(float)": np.float32, "tensor(float16)": np.float16,
           "tensor(double)": np.float64}


def _model_name(model) -> str:
    m = str(model or "lightning").strip().lower()
    if m not in MODEL_FILES:
        raise ValueError(f"MoveNet model must be 'lightning' or 'thunder', not {model!r}")
    return m


# ------------------------------------------------------------------ model file
def models_dir() -> Path:
    from poseboard.pose import mediapipe_backend as mpb  # MODEL_DIR is patched in the frozen app

    return Path(mpb.MODEL_DIR)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, target: Path, sha256: str, timeout: float = 30.0) -> Path:
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s into %s", url, target.parent)
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 16)
        digest = _sha256(tmp)
        if digest != sha256:
            raise RuntimeError(f"SHA-256 mismatch ({digest})")
        tmp.replace(target)
    except Exception as e:  # noqa: BLE001  (offline, proxy, disk, checksum)
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"Cannot download the MoveNet model ({e}).\nDownload\n  {url}\nand save it as\n  "
            f"{target}\nor select a MoveNet .onnx file with option 'model_path'.") from e
    return target


def resolve_model(model: str = "lightning", model_path: str | Path = "",
                  folder: str | Path | None = None, download: bool = True) -> Path:
    """Path of the ONNX model: ``model_path`` if given (must exist), else
    ``<folder>/movenet_singlepose_<model>_4.onnx`` (default folder: the PoseBoard models
    folder), downloaded from GitHub if missing (RuntimeError with instructions on failure)."""
    if str(model_path or "").strip():
        p = Path(str(model_path)).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"MoveNet model file not found: {p}")
        return p
    m = _model_name(model)
    name, digest = MODEL_FILES[m]
    target = (Path(folder) if folder else models_dir()) / name
    if target.is_file():
        return target
    if not download:
        raise FileNotFoundError(f"{target} does not exist (download it from {MODEL_URLS[m]})")
    return _download(MODEL_URLS[m], target, digest)


def resolve_providers(device: str | None, available: list[str]) -> list[str]:
    """ONNX Runtime providers for ``device``: auto = CUDA if available else CPU; "cuda"
    without the CUDA provider raises RuntimeError."""
    d = str(device or "auto").strip().lower()
    has_cuda = "CUDAExecutionProvider" in available
    if d in ("", "auto"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"] if has_cuda else \
            ["CPUExecutionProvider"]
    if d in ("cuda", "gpu") or d.startswith("cuda:"):
        if not has_cuda:
            raise RuntimeError("MoveNet: device 'cuda' needs onnxruntime-gpu (ONNX Runtime has "
                               "no CUDAExecutionProvider); use device 'cpu' or 'auto'")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if d != "cpu":
        raise ValueError(f"device must be auto, cpu or cuda, not {device!r}")
    return ["CPUExecutionProvider"]


# ------------------------------------------------------------------ geometry
def init_region(h: int, w: int, aspect: float = 1.0) -> tuple[float, float, float, float]:
    """(x0, y0, width, height) of the region covering the whole image, centered, with
    width / height = ``aspect`` (the network input's aspect ratio)."""
    if w / h > aspect:
        rw, rh = float(w), w / aspect
    else:
        rw, rh = h * aspect, float(h)
    return (w - rw) / 2.0, (h - rh) / 2.0, rw, rh


def next_region(kp: np.ndarray, scores: np.ndarray, h: int,
                w: int) -> tuple[float, float, float, float]:
    """The TF Hub tutorial's ``determine_crop_region`` in pixels: a square centered on the hips
    from the torso and body extent of keypoints ``kp`` (17, 2) of the previous frame, or the
    whole image (``init_region``) when the torso is not visible."""
    sc = np.nan_to_num(np.asarray(scores, float), nan=0.0)
    kp = np.asarray(kp, float)
    torso_ok = (max(sc[i] for i in _HIPS) > MIN_CROP_KEYPOINT_SCORE
                and max(sc[i] for i in _SHOULDERS) > MIN_CROP_KEYPOINT_SCORE)
    if not torso_ok or not np.all(np.isfinite(kp[list(_TORSO)])):
        return init_region(h, w)
    cx, cy = kp[list(_HIPS)].mean(axis=0)
    d = np.abs(kp - (cx, cy))
    torso = d[list(_TORSO)].max(axis=0)
    ok = (sc >= MIN_CROP_KEYPOINT_SCORE) & np.all(np.isfinite(kp), axis=1)
    body = d[ok].max(axis=0) if ok.any() else np.zeros(2)
    half = max(torso[0] * 1.9, torso[1] * 1.9, body[1] * 1.2, body[0] * 1.2)
    half = min(half, max(cx, w - cx, cy, h - cy))
    if half > max(w, h) / 2.0 or not half > 0:
        return init_region(h, w)
    return cx - half, cy - half, 2.0 * half, 2.0 * half


def crop_and_resize(image: np.ndarray, region: tuple[float, float, float, float],
                    size_hw: tuple[int, int]) -> np.ndarray:
    """The ``region`` (x0, y0, width, height, pixels, may extend beyond the image: black)
    resized to ``size_hw``; continuous coordinates map exactly:
    ``u_out = (u - x0) * out_w / width``."""
    out_h, out_w = size_hw
    x0, y0, rw, rh = region
    h, w = image.shape[:2]
    fx = fy = 1.0
    k = min(out_w / rw, out_h / rh)
    if k < 0.9:  # anti-aliasing: shrink the whole image with area averaging first
        nw, nh = max(1, int(round(w * k))), max(1, int(round(h * k)))
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        fx, fy = nw / w, nh / h
    kx, ky = out_w / (rw * fx), out_h / (rh * fy)
    m = np.array([[kx, 0.0, (0.5 - x0 * fx) * kx - 0.5],
                  [0.0, ky, (0.5 - y0 * fy) * ky - 0.5]])
    return cv2.warpAffine(image, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def parse_output(out: np.ndarray, region: tuple[float, float, float, float]
                 ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None, float | None]]:
    """[(keypoints (17, 2) ORIGINAL pixels, scores (17,), box or None, person score or None)]
    from a MoveNet output: single pose ``[1, 1, 17, 3]`` or multi pose ``[1, N, 56]``."""
    a = np.asarray(out, np.float64)
    x0, y0, rw, rh = region
    if a.shape[-1] == 56:
        rows = a.reshape(-1, 56)
        kps, extra = rows[:, :51].reshape(-1, 17, 3), rows[:, 51:]
    elif a.size % 51 == 0 and a.shape[-1] == 3:
        kps, extra = a.reshape(-1, 17, 3), None
    else:
        raise ValueError(f"MoveNet: unexpected model output shape {a.shape} (expected "
                         "[1, 1, 17, 3] or [1, N, 56])")
    people = []
    for i, k in enumerate(kps):
        xy = np.stack([x0 + k[:, 1] * rw, y0 + k[:, 0] * rh], axis=1)
        sc = np.clip(np.nan_to_num(k[:, 2], nan=0.0), 0.0, 1.0)
        box = score = None
        if extra is not None:
            ymin, xmin, ymax, xmax, s = extra[i]
            box = np.array([x0 + xmin * rw, y0 + ymin * rh, x0 + xmax * rw, y0 + ymax * rh])
            score = float(np.clip(s, 0.0, 1.0))
        people.append((xy, sc, box, score))
    return people


def _as_bgr(image) -> np.ndarray | None:
    if image is None:
        return None
    img = np.asarray(image)
    if img.ndim < 2 or img.shape[0] == 0 or img.shape[1] == 0:
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img.reshape(img.shape[0], img.shape[1]), cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(img)


# ------------------------------------------------------------------ detector
class MoveNetDetector(Detector2D):
    key = "movenet"
    label = "MoveNet (ONNX Runtime)"
    format = COCO17
    provides_3d = False

    def __init__(self, model: str = "lightning", model_path: str = "", device: str = "auto",
                 smart_crop: bool = True, input_size: int | None = None,
                 min_keypoint_score: float = MIN_CROP_KEYPOINT_SCORE, min_keypoints: int = 3,
                 min_person_score: float = 0.2, threads: int = 0,
                 models_folder: str | Path | None = None, download: bool = True):
        super().__init__()
        import onnxruntime as ort  # only when this backend is used

        self.model = _model_name(model)
        self.model_path = resolve_model(self.model, model_path, models_folder, download)
        self.providers = resolve_providers(device, list(ort.get_available_providers()))
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if threads:
            so.intra_op_num_threads = int(threads)
        self._sess = ort.InferenceSession(str(self.model_path), sess_options=so,
                                          providers=self.providers)
        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        self._dtype = _DTYPES.get(str(inp.type))
        if self._dtype is None:
            raise ValueError(f"MoveNet: unsupported model input type {inp.type}")
        shape = list(inp.shape)
        if len(shape) != 4:
            raise ValueError(f"MoveNet: expected a 4-D image input, the model has {shape}")
        self._nchw = shape[1] == 3 and shape[3] != 3
        dims = shape[2:4] if self._nchw else shape[1:3]
        default = int(input_size or INPUT_SIZE.get(self.model, 192))
        self.input_hw = tuple(int(d) if isinstance(d, int) and d > 0 else default for d in dims)
        self.smart_crop = bool(smart_crop) and self.input_hw[0] == self.input_hw[1]
        self.min_keypoint_score = float(min_keypoint_score)
        self.min_keypoints = max(1, int(min_keypoints))
        self.min_person_score = float(min_person_score)
        self._regions: dict[str, tuple] = {}  # camera -> region for the next frame
        self._last_t: dict[str, float] = {}
        self.label = f"MoveNet {self.model} (ONNX Runtime)"
        self.options = {"model": self.model, "model_path": str(model_path or ""),
                        "device": str(device), "smart_crop": self.smart_crop,
                        "input_size": self.input_hw[0], "min_keypoint_score":
                        self.min_keypoint_score, "min_keypoints": self.min_keypoints}

    def _tensor(self, crop_rgb: np.ndarray) -> np.ndarray:
        x = crop_rgb.astype(self._dtype)[None]
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2)) if self._nchw else x

    def reset(self, cam_name: str | None = None) -> None:
        """Forget the crop region (all cameras or one)."""
        if cam_name is None:
            self._regions.clear()
            self._last_t.clear()
        else:
            self._regions.pop(cam_name, None)
            self._last_t.pop(cam_name, None)

    def detect(self, image_bgr: np.ndarray, t: float, cam_name: str) -> list[Person2D]:
        if self._sess is None:
            raise RuntimeError("MoveNet detector is closed")
        img = _as_bgr(image_bgr)
        if img is None:
            return []
        h, w = img.shape[:2]
        in_h, in_w = self.input_hw
        last = self._last_t.get(cam_name)
        if last is not None and t is not None and t < last:  # time went back: new sequence
            self._regions.pop(cam_name, None)
        if t is not None:
            self._last_t[cam_name] = float(t)
        region = self._regions.get(cam_name) if self.smart_crop else None
        if region is None or region[4:] != (h, w):
            region = init_region(h, w, in_w / in_h) + (h, w)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop = crop_and_resize(rgb, region[:4], (in_h, in_w))
        out = self._sess.run(None, {self._input_name: self._tensor(crop)})[0]
        parsed = parse_output(out, region[:4])
        people = []
        for kp, sc, box, pscore in parsed:
            good = sc >= self.min_keypoint_score
            if int(good.sum()) < self.min_keypoints:
                continue
            score = float(sc.mean()) if pscore is None else pscore
            if pscore is not None and pscore < self.min_person_score:
                continue
            if box is None:
                box = keypoint_bbox(kp, sc, self.min_keypoint_score)
            people.append(Person2D(kp, sc, bbox=box, score=score))
        if self.smart_crop:
            if len(parsed) == 1 and people:  # single-pose model: follow the person
                p = people[0]
                self._regions[cam_name] = next_region(p.keypoints, p.scores, h, w) + (h, w)
            else:
                self._regions.pop(cam_name, None)
        return people

    def close(self) -> None:
        self._sess = None
        self._regions.clear()

    def info(self) -> dict:
        d = super().info()
        d.update({"model_file": self.model_path.name, "providers": list(self.providers),
                  "input_hw": list(self.input_hw), "license": "Apache-2.0"})
        return d


def create(model: str = "lightning", model_path: str = "", **kwargs) -> MoveNetDetector:
    """Factory of the ``movenet`` backend. ``model``: "lightning" (192 px, fastest) or
    "thunder" (256 px, more accurate); ``model_path``: a MoveNet ``.onnx`` file (empty: the
    file in the models folder, downloaded from GitHub if needed). Extra options: device
    (auto/cpu/cuda), smart_crop (True), input_size (for models with a dynamic input size),
    min_keypoint_score (0.2), min_keypoints (3), min_person_score (0.2, multi-pose files),
    threads (ONNX Runtime intra-op threads, 0 = default), models_folder, download (True)."""
    return MoveNetDetector(model=model, model_path=model_path, **kwargs)
