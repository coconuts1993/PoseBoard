"""Registry of the 2D pose backends.

Every backend is registered here up front; its module is imported only by ``create_detector``,
so starting PoseBoard never imports PyTorch, ONNX Runtime, ... . ``backend_available`` checks
the required packages with ``importlib.util.find_spec`` (no import) and whether the backend
module is present.

Use::

    from poseboard.pose.detectors import create_detector
    from poseboard.pose.multiview import MultiViewEstimator

    est = MultiViewEstimator(create_detector("rtmpose_halpe26", device="cpu"))
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

from poseboard.pose.detectors.base import BackendSpec, Detector2D, Person2D, keypoint_bbox

__all__ = ["BACKENDS", "BackendSpec", "BackendUnavailable", "Detector2D", "Person2D",
           "backend_available", "create_detector", "keypoint_bbox", "list_backends"]

_DEVICE = ("auto", "cpu", "cuda")
_RTMLIB = ("rtmlib", "onnxruntime")
_RTMLIB_INSTALL = "pip install rtmlib onnxruntime   (NVIDIA GPU: onnxruntime-gpu)"
_RTMLIB_MODES = ("balanced", "performance", "lightweight")
_RTMLIB_NOTE = ("rtmlib downloads the ONNX models on first use (download.openmmlab.com / "
                "huggingface.co) into its cache folder (~/.cache/rtmlib/hub/checkpoints); "
                "without access, select model files downloaded elsewhere (empty = the default "
                "model of the mode).")


def _rtm(key, label, factory, fmt, notes, modes=_RTMLIB_MODES, provides_3d=False,
         one_stage=False) -> BackendSpec:
    files = ("pose_model",) if one_stage else ("pose_model", "det_model")
    return BackendSpec(
        key=key, label=label, module="poseboard.pose.detectors.rtmlib_det", factory=factory,
        requires=_RTMLIB, install=_RTMLIB_INSTALL,
        options={"mode": modes, "device": _DEVICE, **{f: () for f in files}},
        defaults={"mode": modes[0], "device": "auto", **{f: "" for f in files}},
        keypoint_format=fmt, license="Apache-2.0", notes=notes + " " + _RTMLIB_NOTE,
        provides_3d=provides_3d, needs_files=files)


BACKENDS: dict[str, BackendSpec] = {s.key: s for s in (
    BackendSpec(
        key="mediapipe", label="MediaPipe Pose Landmarker",
        module="poseboard.pose.detectors.mediapipe_det", factory="create",
        requires=("mediapipe",), install="pip install mediapipe",
        options={"model": ("full", "lite", "heavy"), "num_poses": (1, 2, 3, 4)},
        defaults={"model": "full", "num_poses": 1},
        keypoint_format="mediapipe33", license="Apache-2.0",
        notes="Built in. Also gives a metric 3D skeleton, so one camera is enough for 3D "
              "(depth approximate). Max. persons (num_poses) > 1 lets PoseBoard pick the person "
              "on the board.",
        provides_3d=True),
    _rtm("rtmpose", "RTMPose body (COCO-17, rtmlib)", "create_rtmpose", "coco17",
         "Top-down: YOLOX person detector + RTMPose. Fast and accurate; no feet."),
    _rtm("rtmpose_halpe26", "RTMPose body + feet (Halpe-26, rtmlib)", "create_rtmpose_halpe26",
         "halpe26", "Recommended: heels and toes improve the center of mass. YOLOX + RTMPose."),
    _rtm("rtmw_wholebody", "RTMW / DWPose whole body (133, rtmlib)", "create_rtmw_wholebody",
         "wholebody133", "Body, feet, face and hands (COCO-WholeBody). Slower."),
    _rtm("rtmo", "RTMO one-stage (COCO-17, rtmlib)", "create_rtmo", "coco17",
         "One-stage multi-person model: speed does not depend on the number of people.",
         one_stage=True),
    _rtm("vitpose_onnx", "ViTPose (COCO-17, rtmlib ONNX)", "create_vitpose_onnx", "coco17",
         "YOLOX person detector + ViTPose (ONNX)."),
    _rtm("rtmpose3d", "RTMPose3D whole body 3D (rtmlib)", "create_rtmpose3d", "wholebody133",
         "YOLOX + RTMW3D: 2D keypoints plus a 3D skeleton, so one camera gives 3D (depth "
         "approximate).", modes=("balanced",), provides_3d=True),
    BackendSpec(
        key="yolo_pose", label="YOLO11 / YOLOv8 / YOLO26 pose (Ultralytics)",
        module="poseboard.pose.detectors.ultralytics_det", factory="create",
        requires=("ultralytics",), install="pip install ultralytics",
        options={"model": ("n", "s", "m", "l", "x"), "version": ("11", "8", "26"),
                 "device": _DEVICE},
        defaults={"model": "n", "version": "11", "device": "auto"},
        keypoint_format="coco17", license="AGPL-3.0",
        notes="One-stage multi-person detector. Weights download from GitHub on first use. "
              "AGPL-3.0: check the license before commercial use."),
    BackendSpec(
        key="keypoint_rcnn", label="Keypoint R-CNN (torchvision)",
        module="poseboard.pose.detectors.torchvision_det", factory="create",
        requires=("torch", "torchvision"), install="pip install torch torchvision",
        options={"device": _DEVICE, "weights_path": ()},
        defaults={"device": "auto", "weights_path": ""},
        keypoint_format="coco17", license="BSD-3-Clause",
        notes="Two-stage multi-person model. The weights (~226 MB) download from "
              "download.pytorch.org into the PyTorch hub cache on first use; without internet, "
              "put keypointrcnn_resnet50_fpn_coco-fc266e95.pth into the models folder or "
              "select it as weights_path. Slow without a GPU.",
        needs_files=("weights_path",)),
    BackendSpec(
        key="vitpose_hf", label="ViTPose (Hugging Face transformers)",
        module="poseboard.pose.detectors.vitpose_hf_det", factory="create",
        requires=("transformers", "torch"), install="pip install transformers torch torchvision",
        options={"model": ("usyd-community/vitpose-base-simple", "usyd-community/vitpose-plus-small",
                           "usyd-community/vitpose-plus-base", "usyd-community/vitpose-plus-large",
                           "usyd-community/vitpose-plus-huge"),
                 "person_detector": ("rtdetr", "yolo", "none"),
                 "detector_model": (), "device": _DEVICE},
        defaults={"model": "usyd-community/vitpose-base-simple", "person_detector": "rtdetr",
                  "detector_model": "", "device": "auto"},
        keypoint_format="coco17", license="Apache-2.0",
        notes="Top-down: RT-DETR person detector (or Ultralytics YOLO, AGPL-3.0; or 'none' = "
              "the whole image is one person) + ViTPose. Models download from huggingface.co "
              "on first use; without access, type a local model folder as Model / Detector "
              "model. Slow without a GPU.",
        editable=("model",)),
    BackendSpec(
        key="openpose_dnn", label="OpenPose (OpenCV DNN, Caffe model files)",
        module="poseboard.pose.detectors.openpose_dnn_det", factory="create",
        requires=("cv2",), install="Included (OpenCV). Download the OpenPose model files "
                                   "(pose_deploy.prototxt + pose_iter_*.caffemodel) yourself.",
        options={"model": ("body25", "coco18"), "input_size": (368, 256, 480, 656),
                 "device": _DEVICE, "prototxt": (), "caffemodel": ()},
        defaults={"model": "body25", "input_size": 368, "device": "auto", "prototxt": "",
                  "caffemodel": ""},
        keypoint_format="body25",
        license="Academic / non-commercial use only (OpenPose license)",
        notes="Single person (heatmap maxima, no multi-person grouping). Slow on CPU. The "
              "prototxt is downloaded from GitHub if left empty; the .caffemodel (100-200 MB) is "
              "never downloaded: select it (see the README).",
        needs_files=("prototxt", "caffemodel")),
    BackendSpec(
        key="mmpose", label="MMPose (MMPoseInferencer, e.g. HRNet)",
        module="poseboard.pose.detectors.mmpose_det", factory="create",
        requires=("mmpose", "mmengine", "mmcv"),
        install="pip install -U openmim && mim install mmengine \"mmcv>=2.0.1\" mmdet mmpose",
        options={"model": ("td-hm_hrnet-w32_8xb64-210e_coco-256x192",
                           "td-hm_hrnet-w48_8xb32-210e_coco-256x192", "human"),
                 "device": _DEVICE},
        defaults={"model": "td-hm_hrnet-w32_8xb64-210e_coco-256x192", "device": "auto"},
        keypoint_format="coco17", license="Apache-2.0",
        notes="Any MMPose 2D model (config name or alias; type it as Model); the keypoint "
              "layout depends on the model (COCO-17 by default). Weights download from "
              "download.openmmlab.com.",
        editable=("model",)),
    BackendSpec(
        key="movenet", label="MoveNet single pose (ONNX Runtime)",
        module="poseboard.pose.detectors.movenet_det", factory="create",
        requires=("onnxruntime",), install="pip install onnxruntime",
        options={"model": ("lightning", "thunder"), "device": _DEVICE, "model_path": ()},
        defaults={"model": "lightning", "device": "auto", "model_path": ""},
        keypoint_format="coco17", license="Apache-2.0",
        notes="Very fast, single person. The model downloads from GitHub on first use; select "
              "a MoveNet .onnx file (model_path) if it cannot be downloaded.",
        needs_files=("model_path",)),
)}


class BackendUnavailable(RuntimeError):
    """The backend is unknown, its packages are missing, or its module is not included."""


def list_backends() -> list[BackendSpec]:
    return list(BACKENDS.values())


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def backend_available(key: str) -> tuple[bool, str]:
    """(available, reason). The reason names the missing packages and how to install them."""
    spec = BACKENDS.get(key)
    if spec is None:
        return False, f"Unknown pose backend {key!r}; known: {', '.join(BACKENDS)}"
    missing = [m for m in spec.requires if not _has_module(m)]
    if missing:
        msg = f"{spec.label} needs {', '.join(missing)}: {spec.install}"
        if getattr(sys, "frozen", False):
            msg += (" (the packaged PoseBoard.exe cannot load extra packages; use the source "
                    "install)")
        return False, msg
    if not _has_module(spec.module):
        return False, (f"{spec.label}: backend not implemented/installed in this version of "
                       f"PoseBoard (module {spec.module} not found)")
    return True, "available"


def create_detector(key: str, **options) -> Detector2D:
    """Create the detector of backend ``key``. ``options`` override ``BackendSpec.defaults``
    (None values are ignored). Raises ``BackendUnavailable`` with a clear message when the
    backend cannot be used."""
    ok, why = backend_available(key)
    if not ok:
        raise BackendUnavailable(why)
    spec = BACKENDS[key]
    opts = dict(spec.defaults)
    opts.update({k: v for k, v in options.items() if v is not None})
    mod = importlib.import_module(spec.module)
    factory = getattr(mod, spec.factory, None)
    if factory is None:
        raise BackendUnavailable(f"{spec.label}: {spec.module} has no {spec.factory}() "
                                 "(backend not implemented)")
    det = factory(**opts)
    if not isinstance(det, Detector2D):
        raise TypeError(f"{spec.module}.{spec.factory}() must return a Detector2D, "
                        f"not {type(det).__name__}")
    if not getattr(det, "options", None):
        det.options = opts
    return det
