"""2D person/keypoint detector interface shared by all pose backends.

A backend module (``poseboard/pose/detectors/<name>_det.py``) provides a factory
``factory(**options) -> Detector2D``. ``Detector2D.detect`` returns every person found in one
image; ``poseboard.pose.multiview.MultiViewEstimator`` then picks the subject in each camera,
triangulates (>= 2 calibrated cameras) or lifts a single view, and records the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from poseboard.pose.formats import COCO17, KeypointFormat


@dataclass
class Person2D:
    keypoints: np.ndarray  # (K, 2) pixels in the ORIGINAL image (undo any resize/crop); NaN if missing
    scores: np.ndarray  # (K,) confidence 0..1
    bbox: np.ndarray | None = None  # (4,) x1, y1, x2, y2 pixels (None: derived from the keypoints)
    score: float = 1.0  # person (detection) confidence 0..1
    # Optional (K, 3) metric, body-centred 3D skeleton (meters, any rotation/origin), e.g.
    # MediaPipe world landmarks or RTMPose3D; used to place the person with one camera (PnP).
    keypoints_3d: np.ndarray | None = None

    def __post_init__(self):
        self.keypoints = np.asarray(self.keypoints, np.float64).reshape(-1, 2)
        self.scores = np.asarray(self.scores, np.float64).reshape(-1)
        if self.scores.shape != (len(self.keypoints),):
            raise ValueError(f"Person2D: {len(self.keypoints)} keypoints but "
                             f"{self.scores.shape[0]} scores")
        if self.bbox is not None:
            self.bbox = np.asarray(self.bbox, np.float64).reshape(4)
        if self.keypoints_3d is not None:
            self.keypoints_3d = np.asarray(self.keypoints_3d, np.float64).reshape(-1, 3)
            if len(self.keypoints_3d) != len(self.keypoints):
                raise ValueError("Person2D.keypoints_3d must have one row per keypoint")
        self.score = float(self.score)

    def box(self, min_score: float = 0.0) -> np.ndarray | None:
        """``bbox`` if given, else the bounding box of the keypoints with a score >= min_score
        (None if there are none)."""
        if self.bbox is not None and np.all(np.isfinite(self.bbox)):
            return self.bbox
        return keypoint_bbox(self.keypoints, self.scores, min_score)


def keypoint_bbox(keypoints: np.ndarray, scores: np.ndarray | None = None,
                  min_score: float = 0.0) -> np.ndarray | None:
    """(x1, y1, x2, y2) of the finite keypoints with score >= min_score; None if there are none."""
    kp = np.asarray(keypoints, np.float64).reshape(-1, 2)
    ok = np.all(np.isfinite(kp), axis=1)
    if scores is not None:
        ok &= np.asarray(scores, np.float64).reshape(-1) >= min_score
    if not ok.any():
        return None
    p = kp[ok]
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])


class Detector2D:
    """Base class of the 2D pose backends.

    Subclasses set ``key`` (the ``BACKENDS`` key), ``label``, ``format`` (the keypoint layout
    of ``Person2D.keypoints``; authoritative, e.g. an MMPose model may differ from the
    registry default) and ``provides_3d`` (True when ``Person2D.keypoints_3d`` is filled), and
    implement ``detect``. ``detect`` is called from one worker thread for all cameras in turn;
    ``cam_name`` lets a backend keep per-camera state (e.g. a video-mode tracker, which needs
    increasing timestamps per camera). Load models in ``__init__`` (or lazily), never at import
    time of the module, and free them in ``close``.
    """

    key: str = "base"
    label: str = "Base detector"
    format: KeypointFormat = COCO17
    provides_3d: bool = False

    def __init__(self):
        self.options: dict = {}  # the options the detector was created with (for session.json)

    def detect(self, image_bgr: np.ndarray, t: float, cam_name: str) -> list[Person2D]:
        """All persons in ``image_bgr`` (BGR, uint8); ``t`` = capture time (perf_counter, s)."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    def info(self) -> dict:
        """Description for session.json."""
        return {"backend": self.key, "label": self.label, "keypoint_format": self.format.key,
                "pose2sim_model": self.format.pose2sim_model, "provides_3d": self.provides_3d,
                "options": dict(getattr(self, "options", {}) or {})}


@dataclass(frozen=True)
class BackendSpec:
    """Registry entry of a pose backend (no heavy import happens until ``create_detector``)."""

    key: str
    label: str
    module: str  # dotted module path, imported lazily
    factory: str  # name of ``factory(**options) -> Detector2D`` in ``module``
    requires: tuple[str, ...]  # importable module names checked with importlib.util.find_spec
    install: str  # pip hint shown when something is missing
    options: dict[str, tuple] = field(default_factory=dict)  # option -> allowed choices
    defaults: dict = field(default_factory=dict)  # default option values
    keypoint_format: str = "coco17"  # FORMATS key of the default model's output
    license: str = ""
    notes: str = ""
    # Option names whose value is a user-supplied file path (e.g. OpenPose prototxt/caffemodel)
    needs_files: tuple[str, ...] = ()
    provides_3d: bool = False  # the detector fills Person2D.keypoints_3d (single-view 3D)
    # Options with listed choices whose value may also be typed in (e.g. a model id or a local
    # model folder): an editable list in the GUI
    editable: tuple[str, ...] = ()
