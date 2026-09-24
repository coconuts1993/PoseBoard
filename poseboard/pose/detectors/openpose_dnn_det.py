"""OpenPose body models (BODY_25 or COCO-18, Caffe files) run with OpenCV DNN, as a
``Detector2D`` (backend key ``openpose_dnn``).

LICENSE: the OpenPose models are licensed by Carnegie Mellon University for **academic or
non-profit, non-commercial research use only** (OpenPose LICENSE,
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE). Commercial use
needs a license from CMU. PoseBoard does not ship or download the ``.caffemodel`` weights; you
supply them and are responsible for meeting that license.

Model files (two files per model; option ``model`` = ``body25`` or ``coco18``; ``model_type``
"BODY_25" / "COCO" is accepted as an alias):

=========  ==============================  ==============================  ========
model      prototxt (network definition)   caffemodel (weights)            size
=========  ==============================  ==============================  ========
body25     body_25/pose_deploy.prototxt    pose_iter_584000.caffemodel     ~105 MB
coco18     coco/pose_deploy_linevec.prototxt  pose_iter_440000.caffemodel  ~200 MB
=========  ==============================  ==============================  ========

* The prototxt files are in the OpenPose GitHub repository (``models/pose/...``). If option
  ``prototxt`` is empty, PoseBoard uses the standard file next to the caffemodel or in the
  models folder (``<models>/openpose/<body_25|coco>/``) and otherwise downloads it from
  GitHub, pinned to OpenPose commit ``OPENPOSE_COMMIT`` and checked by SHA-256.
* The caffemodel files are NOT downloaded automatically. The original CMU server
  (posefs1.perception.cs.cmu.edu, used by old ``getModels.sh`` scripts and many tutorials) is
  offline. Where to get them:

  - the current OpenPose ``models/getModels.sh`` / ``getModels.bat`` download from the
    official mirror ``http://vcl.snu.ac.kr/OpenPose/models/pose/<body_25|coco>/``;
  - OpenCV mirrors the COCO model as https://dl.opencv.org/models/openpose_pose_coco.caffemodel
    (SHA-1 ac7e97da66f3ab8169af2e601384c144e23a95c1, from opencv_extra
    ``testdata/dnn/download_models.py``);
  - community mirrors, e.g. the Hugging Face repositories ``camenduru/openpose`` or
    ``gaijingeek/openpose-models`` (not official: check the file size above).

  Put the file into ``<models>/openpose/<body_25|coco>/`` (``<models>`` = the PoseBoard models
  folder, see ``poseboard.pose.mediapipe_backend.MODEL_DIR``) or select it with option
  ``caffemodel``.

OpenCV 5 removed the Caffe importer (``cv2.dnn.readNetFromCaffe``). With OpenCV 4.x the files
are loaded with ``readNetFromCaffe``; otherwise (or with ``engine="onnx"``) this module converts
the Caffe network in memory to ONNX (``caffe_to_onnx``, pure Python: supports the layers used
by OpenPose - Convolution, ReLU, PReLU, Pooling, Concat - plus a few trivial ones) and loads it
with ``cv2.dnn.readNetFromONNX``. The weights are unchanged, so both paths give the same result.

Preprocessing (as OpenPose, ``src/openpose/core/cvMatToOpInput.cpp`` and
``src/openpose/utilities/openCv.cpp`` ``uCharCvMatToFloatPtr`` normalize == 1): the BGR image
is scaled (aspect ratio kept) to a height of ``input_size`` pixels, padded right/bottom with
black to a multiple of 16 pixels and normalized to ``value / 256 - 0.5``.

Decoding (SINGLE PERSON): the network outputs one heatmap per keypoint (plus a background
map) and part affinity fields (PAFs), at 1/8 of the input resolution. Only the heatmaps are
used: each keypoint is the global maximum of its heatmap inside the image area, refined to
sub-cell precision with a parabola through the log of the 3 x 3 neighbourhood. **The PAF-based
multi-person grouping of OpenPose is not implemented**: ``detect`` returns at most one person,
whose keypoints are the most confident detections of each body part, which is the most
confident (usually the largest/closest) person when one person dominates the view. With several
people in view keypoints may be taken from different people; use a multi-person backend
(RTMPose, YOLO pose, ...) there so that PoseBoard can pick the person on the board.

Coordinates: heatmap cell ``i`` covers input pixels ``[8 i, 8 i + 8)``; its centre is mapped
back through the resize to ORIGINAL image pixels (continuous coordinates, like the other
backends: ``x = (i + dx + 0.5) * stride / scale_x``).

Scores: the heatmap value at the peak. OpenPose regresses Gaussians of peak 1, so the value is
already roughly a confidence in 0..1; it is only clipped to 0..1. Keypoints whose peak is below
``min_keypoint_score`` (0.1, as in the OpenCV OpenPose sample) are NaN with score 0; fewer than
``min_keypoints`` (3) found keypoints means no person. ``Person2D.score`` = mean score of the
found keypoints.

Keypoint order: the heatmap channels are in OpenPose order, which is exactly
``FORMATS["body25"]`` / ``FORMATS["coco18"]`` index for index (OpenPose
``src/openpose/pose/poseParameters.cpp`` lines 7-33 ``POSE_BODY_25_BODY_PARTS``: 0 Nose, 1 Neck,
2 RShoulder, ..., 24 RHeel, 25 Background; lines 35-55 ``POSE_COCO_BODY_PARTS``: 0 Nose, ...,
17 LEar, 18 Background; "R"/"L" = the person's right/left). The final ``net_output`` Concat
puts the heatmaps (26 / 19 channels) before the PAFs (52 / 38 channels) in both deploy
prototxts (BODY_25: ``Mconv7_stage1_L1`` 26 + ``Mconv7_stage3_L2`` 52; COCO:
``Mconv7_stage6_L2`` 19 + ``Mconv7_stage6_L1`` 38).

Speed: the network is large (VGG-19 front end); expect roughly 0.5-3 s per image on a CPU at
``input_size`` 368. OpenCV pip wheels have no CUDA; ``device="cuda"`` needs an OpenCV built
with CUDA.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import shutil
import struct
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from poseboard.pose.detectors.base import Detector2D, Person2D, keypoint_bbox
from poseboard.pose.formats import BODY25, COCO18, KeypointFormat

__all__ = ["MODELS", "OPENPOSE_COMMIT", "OpenPoseDnnDetector", "OpenPoseModel", "caffe_to_onnx",
           "create", "decode_heatmaps", "load_net", "model_key", "models_dir", "net_output_channels",
           "parse_caffemodel", "parse_prototxt", "preprocess", "resolve_model_files"]

log = logging.getLogger(__name__)

# OpenPose GitHub commit the prototxt files are downloaded from (master, 2024).
OPENPOSE_COMMIT = "5c5d96523ef917bd30301245fdc8343937cae48d"
_RAW_URL = ("https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/"
            f"{OPENPOSE_COMMIT}/models/pose/")
LICENSE_NOTE = ("OpenPose models: CMU license for academic / non-profit non-commercial research "
                "use only")
_WHERE = ("The original CMU download server is offline. Get the file from the official mirror "
          "used by OpenPose's models/getModels.sh (http://vcl.snu.ac.kr/OpenPose/models/pose/"
          "{folder}/{caffemodel}){opencv_mirror}, or a community mirror (e.g. Hugging Face "
          "camenduru/openpose). " + LICENSE_NOTE + ".")


@dataclass(frozen=True)
class OpenPoseModel:
    key: str  # "body25" / "coco18" (FORMATS key)
    name: str  # OpenPose name ("BODY_25" / "COCO")
    format: KeypointFormat
    folder: str  # sub-folder in the OpenPose models/pose folder
    prototxt: str
    caffemodel: str
    prototxt_sha256: str
    caffemodel_mb: int
    n_pafs: int
    opencv_mirror: str = ""

    @property
    def n_parts(self) -> int:
        return len(self.format)

    @property
    def n_channels(self) -> int:
        """Channels of ``net_output``: heatmaps + background + PAFs."""
        return self.n_parts + 1 + self.n_pafs

    @property
    def prototxt_url(self) -> str:
        return f"{_RAW_URL}{self.folder}/{self.prototxt}"


MODELS: dict[str, OpenPoseModel] = {m.key: m for m in (
    OpenPoseModel("body25", "BODY_25", BODY25, "body_25", "pose_deploy.prototxt",
                  "pose_iter_584000.caffemodel",
                  "44d6ed3a5268d8d41ca59b3a040491277d876975c3234d82cf7ec0539b4b1f61", 105, 52),
    OpenPoseModel("coco18", "COCO", COCO18, "coco", "pose_deploy_linevec.prototxt",
                  "pose_iter_440000.caffemodel",
                  "17051b87f709aa094e09c5da7b78e9016a1f37b2b452ed1f190fe74cce70b1ad", 200, 38,
                  "https://dl.opencv.org/models/openpose_pose_coco.caffemodel"),
)}
_ALIASES = {"body25": "body25", "body_25": "body25", "coco": "coco18", "coco18": "coco18",
            "coco_18": "coco18"}


def model_key(model) -> str:
    """"body25" or "coco18" for any accepted spelling ("BODY_25", "COCO", "coco18", ...)."""
    k = _ALIASES.get(re.sub(r"[\s\-]+", "_", str(model).strip().lower()))
    if k is None:
        raise ValueError(f"OpenPose model must be 'body25' (BODY_25) or 'coco18' (COCO), "
                         f"not {model!r}")
    return k


# ------------------------------------------------------------------ model files
def models_dir() -> Path:
    """``<PoseBoard models folder>/openpose``."""
    from poseboard.pose import mediapipe_backend as mpb  # MODEL_DIR is patched in the frozen app

    return Path(mpb.MODEL_DIR) / "openpose"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_prototxt(spec: OpenPoseModel, target: Path, timeout: float = 30.0) -> Path:
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s into %s", spec.prototxt_url, target.parent)
        with urllib.request.urlopen(spec.prototxt_url, timeout=timeout) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 16)
        digest = _sha256(tmp)
        if digest != spec.prototxt_sha256:
            raise RuntimeError(f"SHA-256 mismatch ({digest})")
        tmp.replace(target)
    except Exception as e:  # noqa: BLE001  (offline, proxy, disk, checksum)
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"Cannot download the OpenPose {spec.name} network definition ({e}).\nDownload\n  "
            f"{spec.prototxt_url}\nand save it as\n  {target}\n(or select it with option "
            "'prototxt').") from e
    return target


def resolve_model_files(model: str = "body25", prototxt: str | Path = "",
                        caffemodel: str | Path = "", folder: str | Path | None = None,
                        download_prototxt: bool = True) -> tuple[Path, Path]:
    """(prototxt, caffemodel) paths for ``model`` (see the module docstring).

    Empty paths are looked up next to the other file and in ``folder`` (default:
    ``<models>/openpose/<body_25|coco>/``); a missing prototxt is downloaded from GitHub.
    Raises FileNotFoundError (with instructions) or RuntimeError (download failed)."""
    spec = MODELS[model_key(model)]
    base = Path(folder) if folder else models_dir() / spec.folder
    proto = Path(str(prototxt)).expanduser() if str(prototxt or "").strip() else None
    weights = Path(str(caffemodel)).expanduser() if str(caffemodel or "").strip() else None
    if weights is None:
        cands = ([proto.parent / spec.caffemodel] if proto else []) + [base / spec.caffemodel]
        weights = next((c for c in cands if c.is_file()), None)
        if weights is None:
            mirror = f", OpenCV's mirror {spec.opencv_mirror}" if spec.opencv_mirror else ""
            raise FileNotFoundError(
                f"OpenPose {spec.name} weights {spec.caffemodel} (~{spec.caffemodel_mb} MB) not "
                f"found. Select the file with option 'caffemodel' or put it into\n  {base}\n"
                + _WHERE.format(folder=spec.folder, caffemodel=spec.caffemodel,
                                opencv_mirror=mirror))
    elif not weights.is_file():
        raise FileNotFoundError(f"OpenPose caffemodel file not found: {weights}")
    if proto is None:
        cands = [weights.parent / spec.prototxt, base / spec.prototxt]
        proto = next((c for c in cands if c.is_file()), None)
        if proto is None:
            if not download_prototxt:
                raise FileNotFoundError(
                    f"OpenPose {spec.name} network definition {spec.prototxt} not found; "
                    f"download it from {spec.prototxt_url}")
            proto = _download_prototxt(spec, base / spec.prototxt)
    elif not proto.is_file():
        raise FileNotFoundError(f"OpenPose prototxt file not found: {proto}")
    return proto, weights


# ------------------------------------------------------------------ Caffe prototxt (text format)
_TOKEN_RE = re.compile(r"""\s*(?:(\#[^\n]*)|"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|([{}:<>\[\],;])|([^\s{}:<>\[\],;"'\#]+))""")


def _tokens(text: str):
    pos, n = 0, len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            if text[pos:].strip() == "":
                return
            raise ValueError(f"prototxt: cannot parse near {text[pos:pos + 40]!r}")
        pos = m.end()
        comment, dq, sq, sym, word = m.groups()
        if comment is not None:
            continue
        if dq is not None or sq is not None:
            yield "str", (dq if dq is not None else sq)
        elif sym is not None:
            yield "sym", sym
        elif word is not None:
            yield "word", word


def parse_prototxt(text: str) -> dict:
    """Protobuf text format -> nested dict ``{field: [values]}`` (messages are dicts, scalars
    are strings; quotes removed). Enough for Caffe network definitions."""
    toks = list(_tokens(text))
    pos = 0

    def message(end: str | None) -> dict:
        nonlocal pos
        msg: dict[str, list] = {}
        while pos < len(toks):
            kind, val = toks[pos]
            if kind == "sym" and val in (",", ";"):
                pos += 1
                continue
            if kind == "sym" and val in ("}", ">"):
                if end is None:
                    raise ValueError("prototxt: unexpected closing brace")
                pos += 1
                return msg
            if kind != "word":
                raise ValueError(f"prototxt: expected a field name, got {val!r}")
            name = val
            pos += 1
            if pos < len(toks) and toks[pos] == ("sym", ":"):
                pos += 1
            if pos >= len(toks):
                raise ValueError(f"prototxt: missing value of {name!r}")
            kind, val = toks[pos]
            if kind == "sym" and val in ("{", "<"):
                pos += 1
                msg.setdefault(name, []).append(message("}" if val == "{" else ">"))
            elif kind == "sym" and val == "[":
                pos += 1
                while pos < len(toks) and toks[pos] != ("sym", "]"):
                    if toks[pos] != ("sym", ","):
                        msg.setdefault(name, []).append(toks[pos][1])
                    pos += 1
                pos += 1
            elif kind in ("str", "word"):
                pos += 1
                msg.setdefault(name, []).append(val)
            else:
                raise ValueError(f"prototxt: bad value {val!r} for {name!r}")
        if end is not None:
            raise ValueError("prototxt: missing closing brace")
        return msg

    return message(None)


def _first(msg: dict, name: str, default=None):
    v = msg.get(name)
    return v[0] if v else default


def _ints(msg: dict, name: str) -> list[int]:
    return [int(float(x)) for x in msg.get(name, [])]


def _flag(value) -> bool:
    return str(value).strip().lower() in ("true", "1")


# ------------------------------------------------------------------ Caffe weights (binary)
def _varint(buf, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if b < 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("caffemodel: invalid varint")


def _fields(buf):
    """(field number, wire type, value) of a serialized protobuf message; length-delimited
    values are memoryview slices (no copy)."""
    buf = memoryview(buf)
    pos, n = 0, len(buf)
    while pos < n:
        key, pos = _varint(buf, pos)
        num, wt = key >> 3, key & 7
        if wt == 0:
            val, pos = _varint(buf, pos)
        elif wt == 1:
            val, pos = buf[pos:pos + 8], pos + 8
        elif wt == 2:
            ln, pos = _varint(buf, pos)
            val, pos = buf[pos:pos + ln], pos + ln
        elif wt == 5:
            val, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"caffemodel: unsupported protobuf wire type {wt}")
        if pos > n or num == 0:
            raise ValueError("caffemodel: truncated or not a Caffe model file")
        yield num, wt, val


def _packed_varints(buf) -> list[int]:
    out, pos, buf = [], 0, memoryview(buf)
    while pos < len(buf):
        v, pos = _varint(buf, pos)
        out.append(v)
    return out


def _blob(buf) -> np.ndarray:
    """caffe.BlobProto -> float32 array (shape from ``shape`` or the legacy num/channels/...)."""
    dims, legacy, data, ddata = None, {}, [], []
    for num, wt, val in _fields(buf):
        if num == 7:  # BlobShape shape
            dims = []
            for n2, wt2, v2 in _fields(val):
                if n2 == 1:
                    dims.extend(_packed_varints(v2) if wt2 == 2 else [v2])
        elif num == 5:  # repeated float data (packed)
            data.append(np.frombuffer(val, "<f4"))
        elif num == 8:  # repeated double double_data
            ddata.append(np.frombuffer(val, "<f8"))
        elif num in (1, 2, 3, 4) and wt == 0:
            legacy[num] = val
    parts = data or ddata
    arr = (np.concatenate(parts) if parts else np.zeros(0)).astype(np.float32)
    if dims is None:
        dims = [legacy.get(i, 1) for i in (1, 2, 3, 4)]
    if int(np.prod(dims)) != arr.size:
        raise ValueError(f"caffemodel: blob of shape {dims} has {arr.size} values")
    return arr.reshape(dims)


def parse_caffemodel(data: bytes | bytearray | memoryview) -> dict[str, list[np.ndarray]]:
    """Weights of a binary ``.caffemodel``: {layer name: [blobs]} (layers with blobs only;
    both the current ``layer`` and the legacy V1 ``layers`` messages are read)."""
    out: dict[str, list[np.ndarray]] = {}
    for num, wt, val in _fields(data):
        if wt != 2 or num not in (100, 2):  # LayerParameter layer = 100; V1 layers = 2
            continue
        name_f, blob_f = (1, 7) if num == 100 else (4, 6)
        name, blobs = None, []
        for n2, wt2, v2 in _fields(val):
            if n2 == name_f and wt2 == 2:
                name = bytes(v2).decode("utf-8", "replace")
            elif n2 == blob_f and wt2 == 2:
                blobs.append(_blob(v2))
        if name is not None and blobs:
            out[name] = blobs
    if not out:
        raise ValueError("caffemodel: no layer weights found (not a Caffe model file?)")
    return out


# ------------------------------------------------------------------ ONNX writer (protobuf)
def _enc_varint(n: int) -> bytes:
    n &= (1 << 64) - 1  # int64 two's complement
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _pb_int(num: int, value: int) -> bytes:
    return _enc_varint(num << 3) + _enc_varint(int(value))


def _pb_bytes(num: int, value: bytes) -> bytes:
    return _enc_varint((num << 3) | 2) + _enc_varint(len(value)) + value


def _pb_str(num: int, value: str) -> bytes:
    return _pb_bytes(num, value.encode("utf-8"))


def _pb_float(num: int, value: float) -> bytes:
    return _enc_varint((num << 3) | 5) + struct.pack("<f", float(value))


def _attr(name: str, value) -> bytes:
    # onnx.proto AttributeProto: name=1, f=2, i=3, ints=8, type=20 (FLOAT=1, INT=2, INTS=7)
    if isinstance(value, float):
        body = _pb_str(1, name) + _pb_float(2, value) + _pb_int(20, 1)
    elif isinstance(value, (list, tuple)):
        body = _pb_str(1, name) + b"".join(_pb_int(8, v) for v in value) + _pb_int(20, 7)
    else:
        body = _pb_str(1, name) + _pb_int(3, int(value)) + _pb_int(20, 2)
    return body


def _node(op: str, inputs: list[str], outputs: list[str], name: str, **attrs) -> bytes:
    # NodeProto: input=1, output=2, name=3, op_type=4, attribute=5
    body = (b"".join(_pb_str(1, i) for i in inputs) + b"".join(_pb_str(2, o) for o in outputs)
            + _pb_str(3, name) + _pb_str(4, op)
            + b"".join(_pb_bytes(5, _attr(k, v)) for k, v in attrs.items()))
    return _pb_bytes(1, body)  # GraphProto.node = 1


def _initializer(name: str, arr: np.ndarray) -> bytes:
    # TensorProto: dims=1, data_type=2 (FLOAT=1), name=8, raw_data=9 (little endian)
    a = np.ascontiguousarray(arr, dtype="<f4")
    body = (b"".join(_pb_int(1, d) for d in a.shape) + _pb_int(2, 1) + _pb_str(8, name)
            + _pb_bytes(9, a.tobytes()))
    return _pb_bytes(5, body)  # GraphProto.initializer = 5


def _value_info(num: int, name: str, dims: list) -> bytes:
    # ValueInfoProto: name=1, type=2 -> TypeProto.tensor_type=1 -> elem_type=1, shape=2 ->
    # TensorShapeProto.dim=1 -> Dimension dim_value=1 / dim_param=2
    shape = b"".join(_pb_bytes(1, _pb_int(1, d) if isinstance(d, int) else _pb_str(2, str(d)))
                     for d in dims)
    tensor = _pb_int(1, 1) + _pb_bytes(2, shape)
    return _pb_bytes(num, _pb_str(1, name) + _pb_bytes(2, _pb_bytes(1, tensor)))


# ------------------------------------------------------------------ Caffe -> ONNX
@dataclass
class _Graph:
    nodes: list[bytes] = field(default_factory=list)
    inits: list[bytes] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)  # Caffe blob -> current ONNX tensor
    count: int = 0

    def new(self, blob: str) -> str:
        self.count += 1
        return f"{blob}__{self.count}"


def _hw(p: dict, name: str, default: int) -> tuple[int, int]:
    """Caffe spatial parameter: ``name_h``/``name_w`` or repeated ``name`` (1 or 2 values)."""
    h, w = _first(p, f"{name}_h"), _first(p, f"{name}_w")
    if h is not None or w is not None:
        return int(float(h if h is not None else default)), int(float(w if w is not None else default))
    vals = _ints(p, name if name != "kernel" else "kernel_size")
    if not vals:
        return default, default
    return (vals[0], vals[0]) if len(vals) == 1 else (vals[0], vals[1])


def _net_inputs(net: dict, layers: list[dict]) -> list[tuple[str, list]]:
    names = net.get("input", [])
    dims = _ints(net, "input_dim")
    shapes = [_ints(s, "dim") for s in net.get("input_shape", [])]
    out = []
    for i, name in enumerate(names):
        shape = shapes[i] if i < len(shapes) else dims[4 * i:4 * i + 4]
        out.append((name, shape or [1, 3, 0, 0]))
    for lay in layers:
        if _first(lay, "type") == "Input":
            shape_msgs = _first(lay, "input_param", {}).get("shape", [])
            for j, top in enumerate(lay.get("top", [])):
                shape = _ints(shape_msgs[min(j, len(shape_msgs) - 1)], "dim") if shape_msgs else []
                out.append((top, shape or [1, 3, 0, 0]))
    return out


def _active(lay: dict) -> bool:
    """False for layers that only exist in the TRAIN phase."""
    for inc in lay.get("include", []):
        if str(_first(inc, "phase", "")).upper() == "TRAIN":
            return False
    return True


def net_output_channels(prototxt_text: str) -> dict[str, int | None]:
    """Channels of each output blob of a Caffe net (None when unknown), from the prototxt only
    (Convolution num_output, Concat sums, element-wise layers keep the channels)."""
    net = parse_prototxt(prototxt_text)
    layers = [lay for lay in net.get("layer", []) + net.get("layers", []) if _active(lay)]
    ch: dict[str, int | None] = {n: (s[1] if len(s) > 1 and s[1] else None)
                                 for n, s in _net_inputs(net, layers)}
    consumed: set[str] = set()
    for lay in layers:
        typ = str(_first(lay, "type", ""))
        bottoms, tops = lay.get("bottom", []), lay.get("top", [])
        consumed.update(b for b in bottoms if b not in tops)
        if typ == "Input":
            continue
        if typ == "Convolution":
            c = int(_first(_first(lay, "convolution_param", {}), "num_output", 0)) or None
        elif typ == "Concat":
            vals = [ch.get(b) for b in bottoms]
            c = sum(vals) if vals and all(v is not None for v in vals) else None
        else:
            c = ch.get(bottoms[0]) if bottoms else None
        for top in tops:
            ch[top] = c
            if top in bottoms:
                consumed.discard(top)
    return {b: c for b, c in ch.items() if b not in consumed}


def caffe_to_onnx(prototxt_text: str, weights: dict[str, list[np.ndarray]],
                  opset: int = 13) -> bytes:
    """Serialized ONNX model (float32, NCHW, dynamic height/width) of a Caffe network:
    ``prototxt_text`` (deploy definition) + ``weights`` from ``parse_caffemodel``.

    Supported layers: Input, Convolution (pad/stride/dilation/group), ReLU (with
    negative_slope), PReLU, Pooling (MAX/AVE, Caffe's rounding-up output size), Concat,
    Eltwise (SUM without coefficients/PROD/MAX), Sigmoid, TanH, Dropout and Split (no-ops).
    Raises ValueError for other layers or weights that do not fit the definition."""
    net = parse_prototxt(prototxt_text)
    layers = [lay for lay in net.get("layer", []) + net.get("layers", []) if _active(lay)]
    if net.get("layers") and not net.get("layer"):
        raise ValueError("prototxt: legacy V1 'layers' definitions are not supported")
    g = _Graph()
    inputs = _net_inputs(net, layers)
    if not inputs:
        raise ValueError("prototxt: the network declares no input")
    graph_inputs = []
    for name, shape in inputs:
        dims = [1, int(shape[1]) if len(shape) > 1 and shape[1] else 3, "height", "width"]
        graph_inputs.append(_value_info(11, name, dims))
        g.names[name] = name
    consumed: set[str] = set()
    produced: list[str] = []

    def src(blob: str, lname: str) -> str:
        if blob not in g.names:
            raise ValueError(f"prototxt: layer {lname!r} reads unknown blob {blob!r}")
        return g.names[blob]

    for lay in layers:
        lname = str(_first(lay, "name", "?"))
        typ = str(_first(lay, "type", ""))
        bottoms, tops = lay.get("bottom", []), lay.get("top", [])
        if typ == "Input":
            continue
        if typ in ("Silence",):
            consumed.update(bottoms)
            continue
        ins = [src(b, lname) for b in bottoms]
        consumed.update(bottoms)
        blobs = weights.get(lname, [])
        if typ in ("Split", "Dropout"):
            for top in tops:
                g.names[top] = ins[0]
                produced.append(top)
            continue
        if len(tops) != 1:
            raise ValueError(f"Caffe layer {lname!r} ({typ}) must have one top")
        out = g.new(tops[0])
        if typ == "Convolution":
            p = _first(lay, "convolution_param", {})
            n_out = int(_first(p, "num_output", 0))
            kh, kw = _hw(p, "kernel", 0)
            sh, sw = _hw(p, "stride", 1)
            ph, pw = _hw(p, "pad", 0)
            dil = _ints(p, "dilation") or [1]
            dh, dw = (dil[0], dil[0]) if len(dil) == 1 else dil[:2]
            group = int(_first(p, "group", 1))
            bias = _flag(_first(p, "bias_term", "true"))
            if not blobs:
                raise ValueError(f"caffemodel has no weights for layer {lname!r}")
            w = blobs[0]
            if w.ndim != 4:
                w = w.reshape(n_out, -1, kh, kw)
            if w.shape[0] != n_out or w.shape[2:] != (kh, kw):
                raise ValueError(f"layer {lname!r}: weights {w.shape} do not fit num_output "
                                 f"{n_out} and kernel {kh}x{kw} (caffemodel and prototxt of "
                                 "different models?)")
            wn = f"{lname}__W"
            g.inits.append(_initializer(wn, w))
            conv_in = [ins[0], wn]
            if bias and len(blobs) > 1:
                bn = f"{lname}__B"
                g.inits.append(_initializer(bn, blobs[1].reshape(-1)))
                conv_in.append(bn)
            g.nodes.append(_node("Conv", conv_in, [out], lname, kernel_shape=[kh, kw],
                                 strides=[sh, sw], pads=[ph, pw, ph, pw],
                                 dilations=[dh, dw], group=group))
        elif typ == "ReLU":
            slope = float(_first(_first(lay, "relu_param", {}), "negative_slope", 0.0))
            if slope:
                g.nodes.append(_node("LeakyRelu", ins[:1], [out], lname, alpha=slope))
            else:
                g.nodes.append(_node("Relu", ins[:1], [out], lname))
        elif typ == "PReLU":
            if not blobs:
                raise ValueError(f"caffemodel has no weights for layer {lname!r}")
            shared = _flag(_first(_first(lay, "prelu_param", {}), "channel_shared", "false"))
            s = blobs[0].reshape(-1)
            sn = f"{lname}__slope"
            g.inits.append(_initializer(sn, s.reshape(1) if shared else s.reshape(-1, 1, 1)))
            g.nodes.append(_node("PRelu", [ins[0], sn], [out], lname))
        elif typ == "Pooling":
            p = _first(lay, "pooling_param", {})
            mode = str(_first(p, "pool", "MAX")).upper()
            if mode not in ("MAX", "AVE", "0", "1"):
                raise ValueError(f"layer {lname!r}: pooling {mode} is not supported")
            op = "MaxPool" if mode in ("MAX", "0") else "AveragePool"
            if _flag(_first(p, "global_pooling", "false")):
                g.nodes.append(_node("Global" + op, ins[:1], [out], lname))
            else:
                kh, kw = _hw(p, "kernel", 1)
                sh, sw = _hw(p, "stride", 1)
                ph, pw = _hw(p, "pad", 0)
                extra = {"count_include_pad": 1} if op == "AveragePool" else {}
                g.nodes.append(_node(op, ins[:1], [out], lname, kernel_shape=[kh, kw],
                                     strides=[sh, sw], pads=[ph, pw, ph, pw], ceil_mode=1,
                                     **extra))
        elif typ == "Concat":
            p = _first(lay, "concat_param", {})
            axis = int(_first(p, "axis", _first(p, "concat_dim", 1)))
            g.nodes.append(_node("Concat", ins, [out], lname, axis=axis))
        elif typ == "Eltwise":
            p = _first(lay, "eltwise_param", {})
            op = {"SUM": "Sum", "PROD": "Mul", "MAX": "Max", "0": "Mul", "1": "Sum",
                  "2": "Max"}.get(str(_first(p, "operation", "SUM")).upper())
            coeff = [float(c) for c in p.get("coeff", [])]
            if op is None or any(c != 1.0 for c in coeff) or (op == "Mul" and len(ins) != 2):
                raise ValueError(f"layer {lname!r}: this Eltwise variant is not supported")
            g.nodes.append(_node(op, ins, [out], lname))
        elif typ in ("Sigmoid", "TanH"):
            g.nodes.append(_node("Sigmoid" if typ == "Sigmoid" else "Tanh", ins[:1], [out], lname))
        else:
            raise ValueError(f"Caffe layer type {typ!r} (layer {lname!r}) is not supported by "
                             "the Caffe-to-ONNX converter")
        g.names[tops[0]] = out
        produced.append(tops[0])
        if tops[0] in bottoms:
            consumed.discard(tops[0])

    outputs = [b for b in dict.fromkeys(produced) if b not in consumed]
    if not outputs:
        raise ValueError("prototxt: the network has no output")
    graph_outputs = []
    for b in outputs:
        g.nodes.append(_node("Identity", [g.names[b]], [b], f"{b}__output"))
        graph_outputs.append(_value_info(12, b, [1, "channels", "out_height", "out_width"]))
    # GraphProto: node=1, name=2, initializer=5, input=11, output=12
    graph = (b"".join(g.nodes) + _pb_str(2, str(_first(net, "name", "caffe_net")))
             + b"".join(g.inits) + b"".join(graph_inputs) + b"".join(graph_outputs))
    # ModelProto: ir_version=1, producer_name=2, graph=7, opset_import=8 (domain=1, version=2)
    return (_pb_int(1, 7) + _pb_str(2, "poseboard-caffe2onnx") + _pb_bytes(7, graph)
            + _pb_bytes(8, _pb_str(1, "") + _pb_int(2, opset)))


# ------------------------------------------------------------------ OpenCV network
def _cuda_devices() -> int:
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:  # noqa: BLE001  (OpenCV without the cuda module)
        return 0


def resolve_device(device: str | None = "auto") -> str:
    """"cuda" when requested/auto and OpenCV has a CUDA device, else "cpu"; an explicit
    "cuda" without CUDA raises RuntimeError (pip OpenCV wheels have no CUDA)."""
    d = str(device or "auto").strip().lower()
    if d in ("", "auto"):
        return "cuda" if _cuda_devices() > 0 else "cpu"
    if d in ("cuda", "gpu") or d.startswith("cuda:"):
        if _cuda_devices() <= 0:
            raise RuntimeError(
                "OpenPose (OpenCV DNN): device 'cuda' needs an OpenCV build with CUDA; the pip "
                "OpenCV packages have none. Use device 'cpu' or 'auto'.")
        return "cuda"
    if d != "cpu":
        raise ValueError(f"device must be auto, cpu or cuda, not {device!r}")
    return "cpu"


def load_net(prototxt: str | Path, caffemodel: str | Path, engine: str = "auto",
             device: str = "cpu"):
    """(cv2.dnn.Net, engine used: "caffe" or "onnx"). ``engine``: "auto" (Caffe importer when
    this OpenCV has it, else in-memory ONNX conversion), "caffe" or "onnx"."""
    e = str(engine or "auto").strip().lower()
    if e not in ("auto", "caffe", "onnx"):
        raise ValueError(f"engine must be auto, caffe or onnx, not {engine!r}")
    has_caffe = hasattr(cv2.dnn, "readNetFromCaffe")
    if e == "caffe" and not has_caffe:
        raise RuntimeError(f"OpenCV {cv2.__version__} has no Caffe importer (removed in OpenCV "
                           "5); use engine 'onnx' or 'auto'")
    if e == "caffe" or (e == "auto" and has_caffe):
        net, used = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel)), "caffe"
    else:
        text = Path(prototxt).read_text(encoding="utf-8", errors="replace")
        onnx = caffe_to_onnx(text, parse_caffemodel(Path(caffemodel).read_bytes()))
        buf = np.frombuffer(onnx, np.uint8)
        classic = getattr(cv2.dnn, "ENGINE_CLASSIC", None)
        if device == "cuda" and classic is not None:  # the new OpenCV 5 engine is CPU only
            net = cv2.dnn.readNetFromONNX(buf, classic)
        else:
            net = cv2.dnn.readNetFromONNX(buf)
        used = "onnx"
    if device == "cuda":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net, used


# ------------------------------------------------------------------ pre/post-processing
def preprocess(image_bgr: np.ndarray, input_size: int = 368, max_width_factor: float = 3.0,
               multiple: int = 16) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """(blob (1, 3, H, W) float32, (scale_x, scale_y), (content_h, content_w)).

    The image is scaled to a height of ``input_size`` (width limited to ``max_width_factor *
    input_size``), placed top-left, padded with black to multiples of ``multiple`` and
    normalized as OpenPose: ``value / 256 - 0.5``. ``scale_*`` = net pixels per image pixel."""
    h, w = image_bgr.shape[:2]
    s = min(input_size / h, max_width_factor * input_size / w)
    new_w, new_h = max(1, int(round(w * s))), max(1, int(round(h * s)))
    net_w = int(math.ceil(new_w / multiple) * multiple)
    net_h = int(math.ceil(new_h / multiple) * multiple)
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)
    padded = np.zeros((net_h, net_w, 3), np.uint8)
    padded[:new_h, :new_w] = resized
    blob = padded.astype(np.float32) * (1.0 / 256.0) - 0.5
    return (np.ascontiguousarray(blob.transpose(2, 0, 1)[None]),
            (new_w / w, new_h / h), (new_h, new_w))


def _subcell(left: float, center: float, right: float) -> float:
    """Offset (-0.5..0.5) of the maximum of a parabola through 3 samples; on the log values
    when all are positive (exact for a Gaussian peak)."""
    if left > 1e-6 and center > 1e-6 and right > 1e-6:
        left, center, right = math.log(left), math.log(center), math.log(right)
    den = left - 2.0 * center + right
    if den >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / den, -0.5, 0.5))


def decode_heatmaps(heatmaps: np.ndarray, n_parts: int, net_hw: tuple[int, int],
                    scale_xy: tuple[float, float], content_hw: tuple[int, int] | None = None,
                    min_score: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Single-person decoding: keypoints (K, 2) in ORIGINAL image pixels (NaN if the peak is
    below ``min_score``) and scores (K,) from the net output ``heatmaps`` (C >= K, h, w).

    ``net_hw``: network input size; ``scale_xy``: net pixels per image pixel;
    ``content_hw``: the image area inside the (padded) net input (default: all of it)."""
    hm = np.asarray(heatmaps, np.float32)
    if hm.ndim == 4:
        hm = hm[0]
    c, oh, ow = hm.shape
    if c < n_parts:
        raise ValueError(f"the network gives {c} maps, expected at least {n_parts}")
    net_h, net_w = net_hw
    stride_x, stride_y = net_w / ow, net_h / oh
    ch, cw = content_hw if content_hw is not None else (net_h, net_w)
    vh = min(oh, max(1, int(math.ceil(ch / stride_y))))
    vw = min(ow, max(1, int(math.ceil(cw / stride_x))))
    kp = np.full((n_parts, 2), np.nan)
    sc = np.zeros(n_parts)
    for k in range(n_parts):
        m = hm[k, :vh, :vw]
        idx = int(np.argmax(m))
        iy, ix = divmod(idx, vw)
        val = float(m[iy, ix])
        if not np.isfinite(val) or val < min_score:
            continue
        dx = _subcell(m[iy, ix - 1], val, m[iy, ix + 1]) if 0 < ix < vw - 1 else 0.0
        dy = _subcell(m[iy - 1, ix], val, m[iy + 1, ix]) if 0 < iy < vh - 1 else 0.0
        kp[k, 0] = (ix + dx + 0.5) * stride_x / scale_xy[0]
        kp[k, 1] = (iy + dy + 0.5) * stride_y / scale_xy[1]
        sc[k] = min(1.0, val)
    return kp, sc


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
        img = cv2.cvtColor(img.reshape(img.shape[0], img.shape[1]), cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(img)


# ------------------------------------------------------------------ detector
class OpenPoseDnnDetector(Detector2D):
    key = "openpose_dnn"
    label = "OpenPose (OpenCV DNN)"
    format = BODY25
    provides_3d = False

    def __init__(self, model: str = "body25", input_size: int = 368, prototxt: str = "",
                 caffemodel: str = "", device: str = "auto", engine: str = "auto",
                 min_keypoint_score: float = 0.1, min_keypoints: int = 3,
                 max_width_factor: float = 3.0, models_folder: str | Path | None = None,
                 download_prototxt: bool = True):
        super().__init__()
        requested = model_key(model)
        self.input_size = int(input_size)
        if not 64 <= self.input_size <= 2048:
            raise ValueError(f"input_size must be 64..2048 pixels, not {input_size!r}")
        self.min_keypoint_score = float(min_keypoint_score)
        self.min_keypoints = max(1, int(min_keypoints))
        self.max_width_factor = float(max_width_factor)
        self.prototxt_path, self.caffemodel_path = resolve_model_files(
            requested, prototxt, caffemodel, models_folder, download_prototxt)
        self.spec = self._check_definition(requested)
        self.format = self.spec.format
        self.device = resolve_device(device)
        self._net, self.engine = load_net(self.prototxt_path, self.caffemodel_path, engine,
                                          self.device)
        self.label = f"OpenPose {self.spec.name} (OpenCV DNN)"
        self.options = {"model": self.spec.key, "input_size": self.input_size,
                        "prototxt": str(self.prototxt_path),
                        "caffemodel": str(self.caffemodel_path), "device": str(device),
                        "engine": str(engine), "min_keypoint_score": self.min_keypoint_score,
                        "min_keypoints": self.min_keypoints}

    def _check_definition(self, requested: str) -> OpenPoseModel:
        """The model of the prototxt (by its output channels; the files decide when they do
        not match ``model``)."""
        text = self.prototxt_path.read_text(encoding="utf-8", errors="replace")
        outs = net_output_channels(text)
        chans = {c for c in outs.values() if c}
        by_channels = {m.n_channels: m for m in MODELS.values()}
        found = [by_channels[c] for c in chans if c in by_channels]
        if not found:
            raise ValueError(
                f"{self.prototxt_path.name} is not an OpenPose BODY_25 or COCO body model "
                f"(network outputs {outs}; expected {MODELS['body25'].n_channels} or "
                f"{MODELS['coco18'].n_channels} channels)")
        spec = found[0]
        if spec.key != requested:
            log.warning("OpenPose: the model files are %s, not %s; using %s", spec.name,
                        MODELS[requested].name, spec.name)
        return spec

    def detect(self, image_bgr: np.ndarray, t: float, cam_name: str) -> list[Person2D]:
        if self._net is None:
            raise RuntimeError("OpenPose detector is closed")
        img = _as_bgr(image_bgr)
        if img is None:
            return []
        blob, scale, content = preprocess(img, self.input_size, self.max_width_factor)
        self._net.setInput(blob)
        out = np.asarray(self._net.forward())
        if out.ndim != 4 or out.shape[1] < self.spec.n_parts:
            raise RuntimeError(f"OpenPose: unexpected network output shape {out.shape}")
        kp, sc = decode_heatmaps(out[0], self.spec.n_parts, blob.shape[2:], scale, content,
                                 self.min_keypoint_score)
        found = sc > 0
        if int(found.sum()) < self.min_keypoints:
            return []
        return [Person2D(kp, sc, bbox=keypoint_bbox(kp, sc, self.min_keypoint_score),
                         score=float(sc[found].mean()))]

    def close(self) -> None:
        self._net = None

    def info(self) -> dict:
        d = super().info()
        d.update({"openpose_model": self.spec.name, "engine_used": self.engine,
                  "device_used": self.device, "license": LICENSE_NOTE,
                  "prototxt": str(self.prototxt_path), "caffemodel": str(self.caffemodel_path),
                  "decoding": "single person (heatmap maxima; no PAF grouping)"})
        return d


def create(model: str = "body25", input_size: int = 368, prototxt: str = "",
           caffemodel: str = "", model_type: str | None = None, **kwargs) -> OpenPoseDnnDetector:
    """Factory of the ``openpose_dnn`` backend. ``model``: "body25" or "coco18" (``model_type``
    "BODY_25"/"COCO" is an alias); ``input_size``: network input height (368; 256 is faster,
    480/656 more precise); ``prototxt`` / ``caffemodel``: the OpenPose model files (see the
    module docstring). Extra options: device (auto/cpu/cuda), engine (auto/caffe/onnx),
    min_keypoint_score (0.1), min_keypoints (3), max_width_factor (3.0), models_folder,
    download_prototxt (True)."""
    if model_type:
        model = model_type
    return OpenPoseDnnDetector(model=model, input_size=input_size, prototxt=prototxt,
                               caffemodel=caffemodel, **kwargs)
