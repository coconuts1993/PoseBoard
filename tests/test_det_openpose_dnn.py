"""The ``openpose_dnn`` backend (OpenPose BODY_25 / COCO Caffe models through OpenCV DNN).

Always running (only OpenCV + numpy):

* the prototxt / caffemodel readers and the pure-Python Caffe-to-ONNX converter, checked
  against an independent numpy implementation of the Caffe layers, on a small network with
  every supported layer and (when the OpenPose prototxt files can be fetched from GitHub) on
  the real BODY_25 and COCO architectures with random weights;
* the single-person heatmap decoding, preprocessing and coordinate rescaling, with the OpenCV
  network replaced by a fake that returns synthetic heatmaps.

The real-model part downloads the OpenPose weights (~105 MB BODY_25 and ~209 MB COCO; the CMU
server is offline, so mirrors are tried, and with PyTorch installed also PyTorch copies of the
same weights hosted on GitHub, converted back to .caffemodel) and runs them on a real image; it
is skipped with the reason when the files cannot be obtained. Set POSEBOARD_OPENPOSE_BODY25 /
POSEBOARD_OPENPOSE_COCO to local .caffemodel files to use them instead, or
POSEBOARD_TEST_SKIP_LARGE_DOWNLOADS=1 to skip these downloads.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from poseboard.calibration import approximate_calibration
from poseboard.pose import mediapipe_backend
from poseboard.pose.base import MODE_2D_ONLY
from poseboard.pose.detectors import backend_available, create_detector
from poseboard.pose.detectors import openpose_dnn_det as od
from poseboard.pose.formats import BODY25, COCO18
from poseboard.pose.multiview import MultiViewEstimator
from tests.conftest import cache_dir, download_cached

ROOT = Path(__file__).resolve().parents[1]

# OpenPose part order, copied from src/openpose/pose/poseParameters.cpp (lines 7-55,
# POSE_BODY_25_BODY_PARTS / POSE_COCO_BODY_PARTS, background excluded). R/L = the person's side.
OPENPOSE_BODY_25 = ("Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
                    "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
                    "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe",
                    "RSmallToe", "RHeel")
OPENPOSE_COCO = ("Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
                 "LWrist", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye",
                 "REar", "LEar")


def _snake(openpose_name: str) -> str:
    n = openpose_name
    side = {"R": "right_", "L": "left_"}.get(n[0]) if n[:1] in "RL" and n[1:2].isupper() else None
    if side:
        n = n[1:]
    words = "".join("_" + c.lower() if c.isupper() else c for c in n).lstrip("_")
    return (side or "") + words


def test_openpose_part_order_is_the_registered_format():
    assert tuple(_snake(n) for n in OPENPOSE_BODY_25) == BODY25.names
    assert tuple(_snake(n) for n in OPENPOSE_COCO) == COCO18.names
    assert od.MODELS["body25"].format is BODY25 and od.MODELS["coco18"].format is COCO18
    assert od.MODELS["body25"].n_channels == 78 and od.MODELS["coco18"].n_channels == 57


def test_module_import_loads_no_model_library():
    code = ("import sys, poseboard.pose.detectors.openpose_dnn_det; "
            "print(','.join(m for m in ('torch', 'onnxruntime', 'ultralytics', 'mmpose') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_model_names():
    for alias in ("body25", "BODY_25", "body-25", " Body_25 "):
        assert od.model_key(alias) == "body25"
    for alias in ("coco", "COCO", "coco18", "COCO_18"):
        assert od.model_key(alias) == "coco18"
    with pytest.raises(ValueError, match="body25"):
        od.model_key("mpi")


# ------------------------------------------------------------------ Caffe files (test encoder)
def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def _ld(num: int, payload: bytes) -> bytes:
    return _varint((num << 3) | 2) + _varint(len(payload)) + payload


def _blob_proto(a: np.ndarray, legacy: bool = False) -> bytes:
    a = np.asarray(a, "<f4")
    if legacy:  # BlobProto num=1, channels=2, height=3, width=4
        dims = (1,) * (4 - a.ndim) + a.shape
        head = b"".join(_varint(i << 3) + _varint(d) for i, d in zip((1, 2, 3, 4), dims))
    else:  # BlobProto shape=7 {dim=1 packed}
        head = _ld(7, _ld(1, b"".join(_varint(d) for d in a.shape)))
    return head + _ld(5, a.tobytes())


def caffemodel_bytes(weights: dict[str, list[np.ndarray]], v1: bool = False) -> bytes:
    """A binary NetParameter with ``layer`` (field 100) or legacy V1 ``layers`` (field 2)."""
    out = [_ld(1, b"test net")]
    for name, blobs in weights.items():
        if v1:  # V1LayerParameter name=4, type=5 (enum, CONVOLUTION=4), blobs=6
            body = _ld(4, name.encode()) + _varint(5 << 3) + _varint(4) + b"".join(
                _ld(6, _blob_proto(b, legacy=True)) for b in blobs)
            out.append(_ld(2, body))
        else:  # LayerParameter name=1, type=2, blobs=7
            body = _ld(1, name.encode()) + _ld(2, b"Convolution") + b"".join(
                _ld(7, _blob_proto(b)) for b in blobs)
            out.append(_ld(100, body))
    return b"".join(out)


# ------------------------------------------------------------------ numpy reference of Caffe
def _conv(x, w, b, pad, stride, dil, group):
    ph, pw = pad
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    o, ci, kh, kw = w.shape
    win = np.lib.stride_tricks.sliding_window_view(
        xp, ((kh - 1) * dil[0] + 1, (kw - 1) * dil[1] + 1), axis=(1, 2))
    win = win[:, ::stride[0], ::stride[1], ::dil[0], ::dil[1]]
    og = o // group
    y = np.concatenate([np.einsum("chwij,ocij->ohw", win[g * ci:(g + 1) * ci],
                                  w[g * og:(g + 1) * og], optimize=True)
                        for g in range(group)])
    return y + (b[:, None, None] if b is not None else 0.0)


def _pool(x, k, s, mode):
    """Caffe pooling without padding: output size rounded up, windows clipped to the input
    (AVE divides by the clipped window size)."""
    c, h, w = x.shape
    oh, ow = -(-(h - k[0]) // s[0]) + 1, -(-(w - k[1]) // s[1]) + 1
    y = np.empty((c, oh, ow))
    for i in range(oh):
        for j in range(ow):
            win = x[:, i * s[0]:i * s[0] + k[0], j * s[1]:j * s[1] + k[1]]
            y[:, i, j] = win.max(axis=(1, 2)) if mode == "MAX" else win.mean(axis=(1, 2))
    return y


def _pair(p, name, default):
    if f"{name}_h" in p or f"{name}_w" in p:
        return (int(p.get(f"{name}_h", [default])[0]), int(p.get(f"{name}_w", [default])[0]))
    v = [int(x) for x in p.get("kernel_size" if name == "kernel" else name, [])] or [default]
    return (v[0], v[0]) if len(v) == 1 else (v[0], v[1])


def caffe_reference(prototxt: str, weights: dict, x: np.ndarray, output: str) -> np.ndarray:
    """Forward pass of a Caffe deploy network in float64 numpy (independent of the module's
    converter; only ``od.parse_prototxt`` is shared)."""
    net = od.parse_prototxt(prototxt)
    names = net.get("input") or [lay["top"][0] for lay in net["layer"]
                                 if lay["type"][0] == "Input"]
    blobs = {names[0]: x[0].astype(np.float64)}
    for lay in net["layer"]:
        typ, name = lay["type"][0], lay["name"][0]
        bot, top = lay.get("bottom", []), lay.get("top", [])
        wb = [a.astype(np.float64) for a in weights.get(name, [])]
        if typ == "Input":
            continue
        v = blobs[bot[0]] if bot else None
        if typ == "Convolution":
            p = lay["convolution_param"][0]
            dil = [int(d) for d in p.get("dilation", ["1"])]
            bias = wb[1] if len(wb) > 1 else None
            ys = [_conv(v, wb[0], bias, _pair(p, "pad", 0), _pair(p, "stride", 1),
                        (dil[0], dil[-1]), int(p.get("group", ["1"])[0]))]
        elif typ == "ReLU":
            slope = float(lay.get("relu_param", [{}])[0].get("negative_slope", ["0"])[0])
            ys = [np.where(v > 0, v, v * slope)]
        elif typ == "PReLU":
            s = wb[0].reshape(-1)
            s = s[0] if s.size == 1 else s[:, None, None]
            ys = [np.where(v > 0, v, v * s)]
        elif typ == "Pooling":
            p = lay["pooling_param"][0]
            ys = [_pool(v, _pair(p, "kernel", 1), _pair(p, "stride", 1), p["pool"][0])]
        elif typ == "Concat":
            ys = [np.concatenate([blobs[b] for b in bot], axis=0)]
        elif typ == "Eltwise":
            ys = [sum(blobs[b] for b in bot)]
        elif typ == "Sigmoid":
            ys = [1.0 / (1.0 + np.exp(-v))]
        elif typ in ("Split", "Dropout"):
            ys = [v] * len(top)
        else:
            raise AssertionError(typ)
        for t, y in zip(top, ys):
            blobs[t] = y
    return blobs[output][None]


TINY_NET = """
name: "tiny" # a comment
layer { name: "data" type: "Input" top: "data"
        input_param { shape { dim: 1 dim: 3 dim: 17 dim: 23 } } }
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param { lr_mult: 1.0 decay_mult: 1 }
  convolution_param { num_output: 6 pad: 1 kernel_size: 3
                      weight_filler { type: "gaussian" std: 0.01 } }
}
layer { name: "relu1" type: "ReLU" bottom: "conv1" top: "conv1" }
layer { name: "pool1" type: "Pooling" bottom: "conv1" top: "pool1"
        pooling_param { pool: MAX kernel_size: 2 stride: 2 } }
layer { name: "conv2" type: "Convolution" bottom: "pool1" top: "conv2"
        convolution_param { num_output: 4 pad: 2 kernel_size: 3 dilation: 2 group: 2 } }
layer { name: "prelu2" type: "PReLU" bottom: "conv2" top: "conv2" }
layer { name: "conv3" type: "Convolution" bottom: "pool1" top: "conv3"
        convolution_param { num_output: 4 kernel_h: 1 kernel_w: 3 pad_h: 0 pad_w: 1
                            bias_term: false } }
layer { name: "leaky3" type: "ReLU" bottom: "conv3" top: "leaky3"
        relu_param { negative_slope: 0.1 } }
layer { name: "split" type: "Split" bottom: "leaky3" top: "a" top: "b" }
layer { name: "sum" type: "Eltwise" bottom: "a" bottom: "b" top: "sum"
        eltwise_param { operation: SUM } }
layer { name: "drop" type: "Dropout" bottom: "sum" top: "sum"
        dropout_param { dropout_ratio: 0.5 } }
layer { name: "prelu_s" type: "PReLU" bottom: "sum" top: "sum_p"
        prelu_param { channel_shared: true } }
layer { name: "concat" type: "Concat" bottom: "conv2" bottom: "sum_p" top: "cat"
        concat_param { axis: 1 } }
layer { name: "conv4" type: "Convolution" bottom: "cat" top: "conv4"
        convolution_param { num_output: 5 kernel_size: 3 stride: 2 pad: 1 } }
layer { name: "sig" type: "Sigmoid" bottom: "conv4" top: "out" }
"""


def _tiny_weights(rng):
    n = rng.normal
    return {
        "conv1": [n(0, 0.5, (6, 3, 3, 3)), n(0, 0.1, 6)],
        "conv2": [n(0, 0.3, (4, 3, 3, 3)), n(0, 0.1, 4)],
        "prelu2": [rng.uniform(0, 0.5, 4)],
        "conv3": [n(0, 0.3, (4, 6, 1, 3))],
        "prelu_s": [np.array([0.25])],
        "conv4": [n(0, 0.3, (5, 8, 3, 3)), n(0, 0.1, 5)],
    }


def test_prototxt_parser():
    msg = od.parse_prototxt(TINY_NET + '\nextra: "a \\" quote" list: [1, 2 ,3] m: { x: 1 } '
                                      "n < y: 'q' >")
    assert msg["name"] == ["tiny"] and len(msg["layer"]) == 15
    conv1 = msg["layer"][1]
    assert conv1["bottom"] == ["data"] and conv1["top"] == ["conv1"]
    p = conv1["convolution_param"][0]
    assert p["num_output"] == ["6"] and p["weight_filler"][0]["std"] == ["0.01"]
    assert msg["layer"][0]["input_param"][0]["shape"][0]["dim"] == ["1", "3", "17", "23"]
    assert msg["layer"][8]["top"] == ["a", "b"]
    assert msg["extra"] == ['a \\" quote'] and msg["list"] == ["1", "2", "3"]
    assert msg["m"] == [{"x": ["1"]}] and msg["n"] == [{"y": ["q"]}]
    for bad in ("layer {", "layer }", "a: {", ": 3"):
        with pytest.raises(ValueError, match="prototxt"):
            od.parse_prototxt(bad)


@pytest.mark.parametrize("v1", [False, True], ids=["layer", "v1-layers"])
def test_caffemodel_reader(v1):
    rng = np.random.default_rng(1)
    w = _tiny_weights(rng)
    parsed = od.parse_caffemodel(caffemodel_bytes(w, v1=v1))
    assert set(parsed) == set(w)
    for name, blobs in w.items():
        assert len(parsed[name]) == len(blobs)
        for got, ref in zip(parsed[name], blobs):
            expect = np.asarray(ref, np.float32)
            if v1:  # legacy num/channels/height/width blobs are always 4-D
                expect = expect.reshape((1,) * (4 - expect.ndim) + expect.shape)
            assert got.dtype == np.float32 and got.shape == expect.shape
            np.testing.assert_array_equal(got, expect)
    with pytest.raises(ValueError, match="caffemodel"):
        od.parse_caffemodel(b"<html>not found</html>")
    with pytest.raises(ValueError, match="caffemodel"):
        od.parse_caffemodel(b"")


def _run_onnx_cv2(onnx: bytes, x: np.ndarray) -> np.ndarray:
    net = cv2.dnn.readNetFromONNX(np.frombuffer(onnx, np.uint8))
    net.setInput(x)
    return np.asarray(net.forward())


def test_converter_matches_numpy_reference():
    rng = np.random.default_rng(2)
    w = _tiny_weights(rng)
    onnx = od.caffe_to_onnx(TINY_NET, od.parse_caffemodel(caffemodel_bytes(w)))
    assert od.net_output_channels(TINY_NET) == {"out": 5}
    net = cv2.dnn.readNetFromONNX(np.frombuffer(onnx, np.uint8))
    for hw in ((17, 23), (20, 16), (9, 31)):  # odd sizes: Caffe's rounded-up pooling
        x = rng.uniform(-1, 1, (1, 3) + hw).astype(np.float32)
        net.setInput(x)
        got = np.asarray(net.forward())
        ref = caffe_reference(TINY_NET, w, x, "out")
        assert got.shape == ref.shape, hw
        np.testing.assert_allclose(got, ref, atol=2e-5)


def test_converter_output_runs_in_onnxruntime():
    ort = pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(3)
    w = _tiny_weights(rng)
    onnx = od.caffe_to_onnx(TINY_NET, od.parse_caffemodel(caffemodel_bytes(w)))
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["data"]
    assert [o.name for o in sess.get_outputs()] == ["out"]
    x = rng.uniform(-1, 1, (1, 3, 15, 22)).astype(np.float32)
    np.testing.assert_allclose(sess.run(None, {"data": x})[0],
                               caffe_reference(TINY_NET, w, x, "out"), atol=2e-5)


def test_average_pooling_and_legacy_inputs():
    proto = """input: "img"
input_dim: 1
input_dim: 2
input_dim: 8
input_dim: 12
layer { name: "p" type: "Pooling" bottom: "img" top: "p"
        pooling_param { pool: AVE kernel_size: 2 stride: 2 } }
layer { name: "c" type: "Convolution" bottom: "p" top: "c"
        convolution_param { num_output: 3 kernel_size: 1 } }
"""
    rng = np.random.default_rng(4)
    w = {"c": [rng.normal(0, 1, (3, 2, 1, 1)), rng.normal(0, 1, 3)]}
    onnx = od.caffe_to_onnx(proto, od.parse_caffemodel(caffemodel_bytes(w)))
    x = rng.uniform(-1, 1, (1, 2, 8, 12)).astype(np.float32)
    np.testing.assert_allclose(_run_onnx_cv2(onnx, x), caffe_reference(proto, w, x, "c"),
                               atol=2e-5)


def test_caffe_importer_agrees_with_the_converter(tmp_path):
    """OpenCV 4.x: readNetFromCaffe (engine "caffe") and the ONNX conversion give the same."""
    if not hasattr(cv2.dnn, "readNetFromCaffe"):
        pytest.skip(f"OpenCV {cv2.__version__} has no Caffe importer (removed in OpenCV 5)")
    rng = np.random.default_rng(5)
    w = _tiny_weights(rng)
    (tmp_path / "net.prototxt").write_text(TINY_NET)
    (tmp_path / "net.caffemodel").write_bytes(caffemodel_bytes(w))
    x = rng.uniform(-1, 1, (1, 3, 17, 23)).astype(np.float32)
    outs = []
    for engine in ("caffe", "onnx"):
        net, used = od.load_net(tmp_path / "net.prototxt", tmp_path / "net.caffemodel", engine)
        assert used == engine
        net.setInput(x)
        outs.append(np.asarray(net.forward()))
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-4)
    np.testing.assert_allclose(outs[1], caffe_reference(TINY_NET, w, x, "out"), atol=2e-5)


def test_converter_errors():
    rng = np.random.default_rng(6)
    w = _tiny_weights(rng)
    bad_layer = TINY_NET.replace('type: "Sigmoid"', 'type: "Deconvolution"')
    with pytest.raises(ValueError, match="Deconvolution.*not supported"):
        od.caffe_to_onnx(bad_layer, w)
    wrong = dict(w, conv4=[np.zeros((7, 8, 3, 3)), np.zeros(7)])
    with pytest.raises(ValueError, match="different models"):
        od.caffe_to_onnx(TINY_NET, wrong)
    missing = {k: v for k, v in w.items() if k != "conv2"}
    with pytest.raises(ValueError, match="no weights for layer 'conv2'"):
        od.caffe_to_onnx(TINY_NET, missing)
    with pytest.raises(ValueError, match="no input"):
        od.caffe_to_onnx('layer { name: "c" type: "ReLU" bottom: "x" top: "y" }', {})
    if not hasattr(cv2.dnn, "readNetFromCaffe"):  # OpenCV 5
        with pytest.raises(RuntimeError, match="no Caffe importer"):
            od.load_net("a.prototxt", "a.caffemodel", engine="caffe")
    with pytest.raises(ValueError, match="engine"):
        od.load_net("a.prototxt", "a.caffemodel", engine="tensorrt")


# ------------------------------------------------------------------ real OpenPose architectures
@pytest.fixture(scope="module", params=["body25", "coco18"])
def openpose_prototxt(request):
    spec = od.MODELS[request.param]
    path = download_cached(spec.prototxt_url, f"openpose_{spec.folder}_{spec.prototxt}")
    return spec, path


def _random_openpose_weights(text: str, rng) -> dict:
    net = od.parse_prototxt(text)
    ch = {net["input"][0]: 3}
    w = {}
    for lay in net["layer"]:
        typ, name, bot, top = lay["type"][0], lay["name"][0], lay["bottom"], lay["top"]
        if typ == "Convolution":
            p = lay["convolution_param"][0]
            o, k = int(p["num_output"][0]), int(p["kernel_size"][0])
            w[name] = [rng.normal(0, np.sqrt(2.0 / (ch[bot[0]] * k * k)), (o, ch[bot[0]], k, k)),
                       rng.normal(0, 0.1, o)]
            ch[top[0]] = o
        elif typ == "PReLU":
            w[name] = [rng.uniform(0, 0.3, ch[bot[0]])]
            ch[top[0]] = ch[bot[0]]
        elif typ == "Concat":
            ch[top[0]] = sum(ch[b] for b in bot)
        else:
            ch[top[0]] = ch[bot[0]]
    return w


def test_real_openpose_architectures_convert_exactly(openpose_prototxt):
    """The downloaded prototxt is the pinned file, and the whole OpenPose network (random
    weights) runs in OpenCV exactly like the numpy reference."""
    spec, path = openpose_prototxt
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec.prototxt_sha256
    text = path.read_text()
    assert od.net_output_channels(text) == {"net_output": spec.n_channels}
    rng = np.random.default_rng(7)
    w = _random_openpose_weights(text, rng)
    data = caffemodel_bytes(w)
    assert abs(len(data) / 1e6 - spec.caffemodel_mb) < 2  # same parameter count as the real file
    onnx = od.caffe_to_onnx(text, od.parse_caffemodel(data))
    net = cv2.dnn.readNetFromONNX(np.frombuffer(onnx, np.uint8))
    for hw in ((40, 56), (33, 47)):
        x = rng.uniform(-0.5, 0.5, (1, 3) + hw).astype(np.float32)
        net.setInput(x)
        got = np.asarray(net.forward())
        ref = caffe_reference(text, w, x, "net_output")
        assert got.shape == ref.shape == (1, spec.n_channels, -(-hw[0] // 8), -(-hw[1] // 8))
        np.testing.assert_allclose(got, ref, atol=1e-4 * max(1.0, np.abs(ref).max()))


# ------------------------------------------------------------------ decoding with a fake net
FAKE_PROTOTXT = """input: "image"
input_dim: 1
input_dim: 3
input_dim: 16
input_dim: 16
layer {{ name: "out" type: "Convolution" bottom: "image" top: "net_output"
        convolution_param {{ num_output: {c} kernel_size: 1 }} }}
"""


class FakeNet:
    """Stands in for cv2.dnn.Net: forward() = maps(blob) (blob = the preprocessed input)."""

    def __init__(self, maps):
        self.maps, self.inputs = maps, []

    def setInput(self, blob):
        self.inputs.append(np.array(blob))

    def forward(self):
        return self.maps(self.inputs[-1])


def gaussian_maps(blob, n_channels, points, sigma=1.2, extra=()):
    """Net output (1, C, H/8, W/8) with a Gaussian of peak ``v`` at net-input pixel (u, v)
    for each ``points[k] = (u, v, peak)``; ``extra`` = (channel, cell_x, cell_y, value)."""
    _, _, h, w = blob.shape
    oh, ow = h // 8, w // 8
    yy, xx = np.mgrid[0:oh, 0:ow]
    out = np.zeros((1, n_channels, oh, ow), np.float32)
    for k, (u, v, peak) in points.items():
        cx, cy = u / 8.0 - 0.5, v / 8.0 - 0.5  # cell i is centered on input pixel 8 i + 4
        out[0, k] = peak * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    for ch, cx, cy, val in extra:
        out[0, ch, cy, cx] = val
    return out


@pytest.fixture
def fake_openpose(monkeypatch, tmp_path):
    """Model files in tmp_path and a fake network; returns a factory make(maps, model=...)."""
    nets = []

    def fake_load(prototxt, caffemodel, engine="auto", device="cpu"):
        net = FakeNet(lambda blob: fake_load.maps(blob))
        nets.append(net)
        return net, "fake"

    fake_load.maps = lambda blob: np.zeros((1, 78, blob.shape[2] // 8, blob.shape[3] // 8))
    monkeypatch.setattr(od, "load_net", fake_load)
    monkeypatch.setattr(mediapipe_backend, "MODEL_DIR", tmp_path / "models")
    files = {}
    for key in ("body25", "coco18"):
        spec = od.MODELS[key]
        d = tmp_path / spec.folder
        d.mkdir()
        (d / spec.prototxt).write_text(FAKE_PROTOTXT.format(c=spec.n_channels))
        (d / spec.caffemodel).write_bytes(b"\0")
        files[key] = (d / spec.prototxt, d / spec.caffemodel)

    def make(maps, model="body25", **kw):
        fake_load.maps = maps
        proto, weights = files[model]
        return create_detector("openpose_dnn", model=model, prototxt=str(proto),
                               caffemodel=str(weights), **kw)

    make.nets, make.files = nets, files
    return make


def _expected_scale(h, w, input_size=368):
    s = input_size / h
    return round(w * s) / w, round(h * s) / h


def test_decoding_rescales_to_original_pixels(fake_openpose):
    h, w = 480, 640
    sx, sy = _expected_scale(h, w)
    rng = np.random.default_rng(8)
    truth = np.column_stack([rng.uniform(20, w - 20, 25), rng.uniform(20, h - 20, 25)])
    peaks = rng.uniform(0.3, 0.95, 25)
    pts = {k: (truth[k, 0] * sx, truth[k, 1] * sy, peaks[k]) for k in range(25)}
    # decoys: background channel (25) and PAF channels must be ignored
    det = fake_openpose(lambda b: gaussian_maps(b, 78, pts, extra=[(25, 3, 3, 5.0),
                                                                    (40, 10, 10, 9.0)]))
    assert det.format is BODY25 and det.key == "openpose_dnn" and not det.provides_3d
    img = np.full((h, w, 3), 128, np.uint8)
    (p,) = det.detect(img, 0.0, "cam0")
    np.testing.assert_allclose(p.keypoints, truth, atol=0.05)  # log-parabola: exact peaks
    np.testing.assert_allclose(p.scores, peaks, atol=1e-6)
    assert p.score == pytest.approx(peaks.mean())
    np.testing.assert_allclose(p.bbox, [truth[:, 0].min(), truth[:, 1].min(),
                                        truth[:, 0].max(), truth[:, 1].max()], atol=0.05)
    blob = fake_openpose.nets[-1].inputs[-1]
    assert blob.shape == (1, 3, 368, 496) and blob.dtype == np.float32
    # OpenPose normalization value / 256 - 0.5, black padding on the right
    assert blob[0, :, :, :491].mean() == pytest.approx(128 / 256 - 0.5, abs=1e-6)
    assert np.all(blob[0, :, :, 491:] == -0.5)


def test_keypoints_follow_their_channels(fake_openpose):
    """Channel k of the network is FORMATS['body25'].names[k]: a person facing the camera has
    the left shoulder (channel 5) on the image right."""
    h, w = 368, 368
    names = {"left_shoulder": (250, 100), "right_shoulder": (120, 100), "left_hip": (230, 200),
             "right_hip": (140, 200), "left_ankle": (240, 340), "right_ankle": (130, 340)}
    pts = {BODY25.index(n): (x, y, 0.9) for n, (x, y) in names.items()}
    det = fake_openpose(lambda b: gaussian_maps(b, 78, pts))
    (p,) = det.detect(np.zeros((h, w, 3), np.uint8), 0.0, "cam0")
    for n, xy in names.items():
        np.testing.assert_allclose(p.keypoints[BODY25.index(n)], xy, atol=0.05)
    assert OPENPOSE_BODY_25[5] == "LShoulder" and BODY25.names[5] == "left_shoulder"
    missing = [i for i in range(25) if i not in pts]
    assert np.all(np.isnan(p.keypoints[missing])) and np.all(p.scores[missing] == 0)


def test_coco_model_and_score_handling(fake_openpose):
    h, w = 300, 200
    sx, sy = _expected_scale(h, w)
    pts = {0: (50 * sx, 40 * sy, 1.4),  # > 1: clipped
           5: (120 * sx, 90 * sy, 0.5), 2: (60 * sx, 90 * sy, 0.5),
           11: (110 * sx, 200 * sy, 0.08)}  # below min_keypoint_score 0.1: missing
    det = fake_openpose(lambda b: gaussian_maps(b, 57, pts), model="coco18")
    assert det.format is COCO18 and det.options["model"] == "coco18"
    (p,) = det.detect(np.zeros((h, w, 3), np.uint8), 0.0, "cam0")
    assert p.keypoints.shape == (18, 2)
    assert p.scores[0] == 1.0 and p.scores[5] == pytest.approx(0.5)
    np.testing.assert_allclose(p.keypoints[5], (120, 90), atol=0.05)
    assert np.isnan(p.keypoints[11]).all() and p.scores[11] == 0
    assert np.all((p.scores >= 0) & (p.scores <= 1))
    # min_keypoints: two found keypoints are no person
    two = {5: pts[5], 2: pts[2]}
    det2 = fake_openpose(lambda b: gaussian_maps(b, 57, two), model="coco18")
    assert det2.detect(np.zeros((h, w, 3), np.uint8), 0.0, "cam0") == []
    det3 = fake_openpose(lambda b: gaussian_maps(b, 57, two), model="coco18", min_keypoints=2)
    assert len(det3.detect(np.zeros((h, w, 3), np.uint8), 0.0, "cam0")) == 1


def test_padding_area_is_ignored(fake_openpose):
    h, w = 368, 401  # net input 416 wide: heatmap column 51 is padding only
    pts = {k: (100.0 + 10 * k, 50.0 + 8 * k, 0.6) for k in range(25)}
    det = fake_openpose(lambda b: gaussian_maps(b, 78, pts, extra=[(3, 51, 20, 0.99)]))
    (p,) = det.detect(np.zeros((h, w, 3), np.uint8), 0.0, "cam0")
    assert fake_openpose.nets[-1].inputs[-1].shape == (1, 3, 368, 416)
    np.testing.assert_allclose(p.keypoints[3], (130.0, 74.0), atol=0.05)
    assert p.scores[3] == pytest.approx(0.6)


def test_input_size_and_wide_images(fake_openpose):
    pts = {k: (40.0 + 5 * k, 30.0 + 2 * k, 0.7) for k in range(25)}
    det = fake_openpose(lambda b: gaussian_maps(b, 78, pts), input_size=256)
    det.detect(np.zeros((512, 512, 3), np.uint8), 0.0, "cam0")
    assert fake_openpose.nets[-1].inputs[-1].shape == (1, 3, 256, 256)
    det.detect(np.zeros((100, 1000, 3), np.uint8), 0.0, "cam0")  # width limited to 3 x 256
    assert fake_openpose.nets[-1].inputs[-1].shape == (1, 3, 80, 768)
    with pytest.raises(ValueError, match="input_size"):
        fake_openpose(lambda b: None, input_size=10)


def test_empty_images_gray_images_and_close(fake_openpose):
    pts = {k: (40.0 + 5 * k, 30.0 + 2 * k, 0.7) for k in range(25)}
    det = fake_openpose(lambda b: gaussian_maps(b, 78, pts))
    assert det.detect(np.zeros((0, 0, 3), np.uint8), 0.0, "cam0") == []
    assert det.detect(None, 0.0, "cam0") == []
    assert len(det.detect(np.zeros((120, 160), np.uint8), 0.0, "cam0")) == 1
    assert len(det.detect(np.zeros((120, 160, 4), np.uint8), 0.0, "cam0")) == 1
    empty = fake_openpose(lambda b: gaussian_maps(b, 78, {}))
    assert empty.detect(np.zeros((120, 160, 3), np.uint8), 0.0, "cam0") == []
    det.close()
    with pytest.raises(RuntimeError, match="closed"):
        det.detect(np.zeros((8, 8, 3), np.uint8), 0.0, "cam0")


def test_model_files_decide_the_format(fake_openpose, caplog):
    body = fake_openpose.files["body25"]
    det = create_detector("openpose_dnn", model="coco18", prototxt=str(body[0]),
                          caffemodel=str(body[1]))
    assert det.format is BODY25 and det.spec.name == "BODY_25"
    assert "BODY_25" in caplog.text
    det2 = create_detector("openpose_dnn", model_type="COCO",
                           prototxt=str(fake_openpose.files["coco18"][0]),
                           caffemodel=str(fake_openpose.files["coco18"][1]))
    assert det2.format is COCO18
    info = det2.info()
    assert info["backend"] == "openpose_dnn" and info["keypoint_format"] == "coco18"
    assert info["pose2sim_model"] == "COCO" and "non-commercial" in info["license"]
    assert info["engine_used"] == "fake" and "no PAF" in info["decoding"]


def test_prototxt_that_is_not_openpose(fake_openpose, tmp_path):
    proto = tmp_path / "other.prototxt"
    proto.write_text(FAKE_PROTOTXT.format(c=21))
    with pytest.raises(ValueError, match="not an OpenPose"):
        create_detector("openpose_dnn", prototxt=str(proto),
                        caffemodel=str(fake_openpose.files["body25"][1]))


def test_model_file_lookup(fake_openpose, tmp_path, monkeypatch):
    # nothing given, nothing in the models folder: clear instructions
    monkeypatch.setattr(mediapipe_backend, "MODEL_DIR", tmp_path / "empty")
    with pytest.raises(FileNotFoundError) as e:
        od.resolve_model_files("body25")
    msg = str(e.value)
    assert "pose_iter_584000.caffemodel" in msg and "vcl.snu.ac.kr" in msg
    assert "non-commercial" in msg and str(tmp_path / "empty" / "openpose" / "body_25") in msg
    with pytest.raises(FileNotFoundError, match="dl.opencv.org"):
        od.resolve_model_files("coco18")
    with pytest.raises(FileNotFoundError, match="not found"):
        od.resolve_model_files("body25", caffemodel=tmp_path / "nope.caffemodel")
    # the standard files in the models folder are found
    folder = tmp_path / "empty" / "openpose" / "body_25"
    folder.mkdir(parents=True)
    shutil.copy(fake_openpose.files["body25"][1], folder / "pose_iter_584000.caffemodel")
    shutil.copy(fake_openpose.files["body25"][0], folder / "pose_deploy.prototxt")
    assert od.resolve_model_files("body25") == (folder / "pose_deploy.prototxt",
                                                folder / "pose_iter_584000.caffemodel")
    # a caffemodel elsewhere: the prototxt next to it wins
    p, _ = od.resolve_model_files("body25", caffemodel=fake_openpose.files["body25"][1])
    assert p == fake_openpose.files["body25"][0]


def test_prototxt_download_is_verified(tmp_path, monkeypatch):
    spec = od.MODELS["body25"]
    assert od.OPENPOSE_COMMIT in spec.prototxt_url
    assert spec.prototxt_url.startswith("https://raw.githubusercontent.com/")
    weights = tmp_path / "w" / spec.caffemodel
    weights.parent.mkdir()
    weights.write_bytes(b"\0")
    calls = []

    class Resp:
        def __init__(self, data):
            self.data = data

        def read(self, n=-1):
            out, self.data = (self.data, b"") if n < 0 else (self.data[:n], self.data[n:])
            return out

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(url, timeout=None):
        calls.append(url)
        return Resp(b"layer { tampered }")

    monkeypatch.setattr(od.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch") as e:
        od.resolve_model_files("body25", caffemodel=weights, folder=tmp_path / "models")
    assert calls == [spec.prototxt_url] and spec.prototxt_url in str(e.value)
    assert not list((tmp_path / "models").glob("*"))  # no partial file left
    with pytest.raises(FileNotFoundError, match="network definition"):
        od.resolve_model_files("body25", caffemodel=weights, folder=tmp_path / "models",
                               download_prototxt=False)


def test_device_resolution(monkeypatch):
    monkeypatch.setattr(od, "_cuda_devices", lambda: 0)
    assert od.resolve_device("auto") == "cpu" and od.resolve_device(None) == "cpu"
    assert od.resolve_device("CPU") == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        od.resolve_device("cuda")
    with pytest.raises(ValueError, match="device"):
        od.resolve_device("tpu")
    monkeypatch.setattr(od, "_cuda_devices", lambda: 1)
    assert od.resolve_device("auto") == "cuda" and od.resolve_device("cuda:0") == "cuda"


def test_registry_and_pipeline(fake_openpose):
    ok, why = backend_available("openpose_dnn")
    assert ok, why
    pts = {k: (40.0 + 9 * k, 30.0 + 6 * k, 0.8) for k in range(25)}
    det = fake_openpose(lambda b: gaussian_maps(b, 78, pts))
    assert det.options["input_size"] == 368 and det.options["model"] == "body25"
    est = MultiViewEstimator(det)
    cams = {"cam0": approximate_calibration("cam0", 368, 368)}
    pose = est.process({"cam0": (1.0, np.zeros((368, 368, 3), np.uint8))}, cams)
    assert pose.mode == MODE_2D_ONLY and pose.format_key == "body25"
    np.testing.assert_allclose(pose.per_camera_2d["cam0"].keypoints,
                               [(u, v) for u, v, _ in pts.values()], atol=0.05)
    est.close()


# ------------------------------------------------------------------ real OpenPose weights
BODY25_URLS = (
    # community mirror (Hugging Face), then the official mirror of OpenPose's getModels.sh
    ("https://huggingface.co/camenduru/openpose/resolve/f4a22b0e6fa2a4a2b1e2d50bd589e8bb11ebea7c/"
     "pose_iter_584000.caffemodel"),
    "https://huggingface.co/camenduru/openpose/resolve/main/pose_iter_584000.caffemodel",
    "http://vcl.snu.ac.kr/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel",
)
COCO_URLS = (
    "https://dl.opencv.org/models/openpose_pose_coco.caffemodel",
    "http://vcl.snu.ac.kr/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
)
# PyTorch copies of the same CMU weights on GitHub (tensor names = Caffe layer names + .weight /
# .bias; the COCO file is pytorch-openpose's body_pose_model.pth). Used, when PyTorch is
# installed, if no Caffe mirror is reachable: converted back to a .caffemodel for the test.
PYTORCH_WEIGHTS = {
    "body25": (("https://github.com/Qalxry/SIT-VTON/releases/download/models/"
                "pose_iter_584000.caffemodel.pt"),
               "36b517810e0da6703cc84898aa41c8bbd55e1db46d8f9873d1f230e2fe8657b4"),
    "coco18": (("https://github.com/yunfan1202/intellegent_design/releases/download/checkpoints/"
                "body_pose_model.pth"),
               "25a948c16078b0f08e236bda51a385d855ef4c153598947c28c0d47ed94bb746"),
}


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo, usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _valid_weights(path: Path, key: str) -> str | None:
    """None if ``path`` is the original CMU file (MD5 from OpenPose's CMakeLists.txt), else
    the reason."""
    digest, expected = _digest(path, "md5"), od.MODELS[key].caffemodel_md5
    return None if digest == expected else f"MD5 {digest} != {expected}"


def _download(url: str, target: Path, check) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        bad = check(tmp)
        if bad:
            raise RuntimeError(bad)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()


def _from_pytorch(key: str, folder: Path) -> Path:
    """The CMU weights rebuilt as a .caffemodel from their PyTorch copy (needs torch)."""
    spec = od.MODELS[key]
    target = folder / (Path(spec.caffemodel).stem + "_from_pytorch.caffemodel")
    if target.is_file():
        return target
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch is not installed (needed to convert the PyTorch copy)") from e
    url, sha = PYTORCH_WEIGHTS[key]
    src = folder / Path(url).name
    if not (src.is_file() and _digest(src, "sha256") == sha):
        _download(url, src, lambda p: None if _digest(p, "sha256") == sha else "SHA-256 mismatch")
    state = torch.load(src, map_location="cpu", weights_only=True)
    layers: dict[str, dict] = {}
    for name, tensor in state.items():
        layer, kind = name.rsplit(".", 1)
        layers.setdefault(layer, {})[kind] = tensor.float().numpy()
    weights = {n: [d["weight"]] + ([d["bias"]] if "bias" in d else []) for n, d in layers.items()}
    tmp = target.with_suffix(".part")
    tmp.write_bytes(caffemodel_bytes(weights))
    tmp.replace(target)
    return target


def _fetch_weights(key: str) -> Path:
    spec = od.MODELS[key]
    env = os.environ.get("POSEBOARD_OPENPOSE_BODY25" if key == "body25"
                         else "POSEBOARD_OPENPOSE_COCO")
    if env:
        if not Path(env).is_file():
            pytest.skip(f"{env} does not exist")
        return Path(env)
    folder = cache_dir() / "openpose" / spec.folder
    target = folder / spec.caffemodel
    if target.is_file() and _valid_weights(target, key) is None:
        return target
    converted = folder / (Path(spec.caffemodel).stem + "_from_pytorch.caffemodel")
    if converted.is_file():
        return converted
    if os.environ.get("POSEBOARD_TEST_SKIP_LARGE_DOWNLOADS"):
        pytest.skip("POSEBOARD_TEST_SKIP_LARGE_DOWNLOADS is set")
    folder.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in (BODY25_URLS if key == "body25" else COCO_URLS):
        try:
            _download(url, target, lambda p: _valid_weights(p, key))
            return target
        except Exception as e:  # noqa: BLE001  (blocked host, 404, proxy, bad file)
            errors.append(f"{url}: {type(e).__name__}: {e}")
    try:
        return _from_pytorch(key, folder)
    except Exception as e:  # noqa: BLE001
        errors.append(f"PyTorch copy {PYTORCH_WEIGHTS[key][0]}: {type(e).__name__}: {e}")
    pytest.skip(f"OpenPose {spec.name} weights not available: " + "; ".join(errors))


@pytest.fixture(scope="module", params=["body25", "coco18"])
def openpose_real(request):
    key = request.param
    weights = _fetch_weights(key)
    spec = od.MODELS[key]
    try:
        proto, _ = od.resolve_model_files(key, caffemodel=weights,
                                          folder=cache_dir() / "openpose" / spec.folder)
    except RuntimeError as e:
        pytest.skip(f"OpenPose prototxt not available: {e}")
    det = create_detector("openpose_dnn", model=key, prototxt=str(proto),
                          caffemodel=str(weights), device="cpu")
    yield det
    det.close()


@pytest.fixture(scope="module")
def mediapipe_points(person_image_path):
    """MediaPipe's keypoints (pixels) on the test image, as the reference."""
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


def test_real_openpose_on_person_image(openpose_real, person_image):
    h, w = person_image.shape[:2]
    fmt = openpose_real.format
    people = openpose_real.detect(person_image, 0.0, "cam0")
    assert len(people) == 1
    p = people[0]
    assert p.keypoints.shape == (len(fmt), 2) and np.all((p.scores >= 0) & (p.scores <= 1))
    kp = {n: p.keypoints[fmt.index(n)] for n in fmt.names}
    for side in ("left", "right"):
        assert all(p.scores[fmt.index(f"{side}_{j}")] > 0.3 for j in LIMBS)
        ys = [kp[f"{side}_{j}"][1] for j in LIMBS]
        assert ys == sorted(ys), (side, ys)  # shoulder above hip above knee above ankle
    # the subject faces the camera: their left side is on the image right
    assert kp["left_shoulder"][0] > kp["right_shoulder"][0]
    assert kp["left_hip"][0] > kp["right_hip"][0]
    body = np.array([kp[f"{s}_{j}"] for s in ("left", "right") for j in LIMBS])
    assert np.all((body >= 0) & (body < [w, h]))


def test_real_openpose_agrees_with_mediapipe(openpose_real, person_image, mediapipe_points):
    mp = mediapipe_points
    assert mp["left_shoulder"][0] > mp["right_shoulder"][0]  # same left/right convention
    (p,) = openpose_real.detect(person_image, 0.0, "cam0")
    fmt = openpose_real.format
    tol = 0.08 * np.hypot(*person_image.shape[:2])
    for side in ("left", "right"):
        for j in LIMBS:
            name = f"{side}_{j}"
            d = np.linalg.norm(p.keypoints[fmt.index(name)] - mp[name])
            assert d < tol, f"{name}: {d:.1f} px from MediaPipe (limit {tol:.1f})"


def test_real_openpose_returns_original_image_pixels(openpose_real, person_image):
    """Padding (other aspect ratio) and resizing are undone: same body, same pixels."""
    (base,) = openpose_real.detect(person_image, 0.0, "cam0")
    fmt = openpose_real.format
    idx = [fmt.index(f"{s}_{j}") for s in ("left", "right") for j in LIMBS]
    tol = 0.03 * np.hypot(*person_image.shape[:2])
    padded = cv2.copyMakeBorder(person_image, 90, 150, 240, 0, cv2.BORDER_CONSTANT,
                                value=(0, 0, 0))
    (pp,) = openpose_real.detect(padded, 0.0, "cam0")
    err = np.linalg.norm(pp.keypoints[idx] - (240.0, 90.0) - base.keypoints[idx], axis=1)
    assert np.all(err < tol), err
    half = cv2.resize(person_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    (ph,) = openpose_real.detect(half, 0.0, "cam0")
    err = np.linalg.norm(ph.keypoints[idx] * 2.0 - base.keypoints[idx], axis=1)
    assert np.all(err < tol), err
