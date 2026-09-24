"""Test doubles for the pose pipeline: a scripted ``Detector2D`` and synthetic persons.

``FakeDetector`` returns whatever its ``script`` says; ``standing_person`` builds a standing
skeleton (world coordinates, Z up, meters) in any keypoint format; ``Scene`` projects a subject
(and optional bystanders) through given cameras, with optional pixel noise, gross outliers and
cameras that miss the subject.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from poseboard.calibration import CameraCalibration
from poseboard.geometry import project_to_image
from poseboard.pose.detectors.base import Detector2D, Person2D, keypoint_bbox
from poseboard.pose.formats import KeypointFormat, get_format

# Standing person facing -Y (towards cameras placed at negative Y), feet at Z = 0, centred at
# the origin; "left" is +X. Values in meters.
_BASE = {
    "nose": (0.0, -0.10, 1.65), "head": (0.0, 0.0, 1.72), "neck": (0.0, 0.0, 1.50),
    "left_eye": (0.035, -0.08, 1.68), "right_eye": (-0.035, -0.08, 1.68),
    "left_eye_inner": (0.02, -0.085, 1.68), "right_eye_inner": (-0.02, -0.085, 1.68),
    "left_eye_outer": (0.05, -0.075, 1.68), "right_eye_outer": (-0.05, -0.075, 1.68),
    "left_ear": (0.08, 0.0, 1.65), "right_ear": (-0.08, 0.0, 1.65),
    "mouth_left": (0.025, -0.09, 1.60), "mouth_right": (-0.025, -0.09, 1.60),
    "left_shoulder": (0.18, 0.0, 1.45), "right_shoulder": (-0.18, 0.0, 1.45),
    "left_elbow": (0.20, 0.0, 1.15), "right_elbow": (-0.20, 0.0, 1.15),
    "left_wrist": (0.20, -0.02, 0.90), "right_wrist": (-0.20, -0.02, 0.90),
    "left_pinky": (0.21, -0.03, 0.82), "right_pinky": (-0.21, -0.03, 0.82),
    "left_index": (0.20, -0.05, 0.82), "right_index": (-0.20, -0.05, 0.82),
    "left_thumb": (0.19, -0.06, 0.86), "right_thumb": (-0.19, -0.06, 0.86),
    "hip": (0.0, 0.0, 0.95), "mid_hip": (0.0, 0.0, 0.95),
    "left_hip": (0.10, 0.0, 0.95), "right_hip": (-0.10, 0.0, 0.95),
    "left_knee": (0.10, -0.02, 0.50), "right_knee": (-0.10, -0.02, 0.50),
    "left_ankle": (0.10, 0.0, 0.08), "right_ankle": (-0.10, 0.0, 0.08),
    "left_heel": (0.10, 0.05, 0.03), "right_heel": (-0.10, 0.05, 0.03),
    "left_big_toe": (0.08, -0.15, 0.02), "right_big_toe": (-0.08, -0.15, 0.02),
    "left_small_toe": (0.13, -0.13, 0.02), "right_small_toe": (-0.13, -0.13, 0.02),
    "left_foot_index": (0.10, -0.15, 0.02), "right_foot_index": (-0.10, -0.15, 0.02),
}


def _generated(name: str) -> tuple[float, float, float] | None:
    """Positions of the face and hand points of COCO-WholeBody."""
    if name.startswith("face_"):
        i = int(name[5:])
        a = 2 * np.pi * i / 68
        return (0.07 * np.cos(a), -0.09, 1.64 + 0.08 * np.sin(a))
    for side, sx in (("left_hand_", 1.0), ("right_hand_", -1.0)):
        if name.startswith(side):
            i = int(name[len(side):])
            finger, joint = (i - 1) // 4, (i - 1) % 4
            x = sx * (0.20 + (0.0 if i == 0 else 0.008 * (finger - 2)))
            z = 0.90 - (0.0 if i == 0 else 0.03 + 0.02 * joint)
            return (x, -0.02, z)
    return None


def standing_person(fmt: str | KeypointFormat, offset=(0.0, 0.0, 0.0), yaw_deg: float = 0.0,
                    scale: float = 1.0) -> np.ndarray:
    """(K, 3) world coordinates (Z up) of a standing person in format ``fmt``, rotated by
    ``yaw_deg`` about Z, scaled, then moved by ``offset`` (feet at Z = offset[2])."""
    fmt = get_format(fmt)
    pts = []
    for n in fmt.names:
        p = _BASE.get(n) or _generated(n)
        if p is None:
            raise KeyError(f"no synthetic position for keypoint {n!r}")
        pts.append(p)
    pts = np.asarray(pts, np.float64) * scale
    a = np.deg2rad(yaw_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    return pts @ Rz.T + np.asarray(offset, np.float64)


def body_centred(pts3d: np.ndarray, fmt: str | KeypointFormat) -> np.ndarray:
    """The skeleton relative to the midpoint of its hips (like MediaPipe world landmarks)."""
    fmt = get_format(fmt)
    hips = [fmt.find("left_hip"), fmt.find("right_hip")]
    c = pts3d[[i for i in hips if i is not None]].mean(axis=0)
    return pts3d - c


def project_person(cam: CameraCalibration, pts3d: np.ndarray, fmt: str | KeypointFormat, *,
                   score: float = 0.9, noise_px: float = 0.0, rng: np.random.Generator | None = None,
                   with_3d: bool = False, person_score: float = 0.95) -> Person2D:
    """``Person2D`` of a 3D skeleton seen by ``cam`` (world frame; the camera frame when the
    camera has no extrinsics), with optional Gaussian pixel noise and the body-centred 3D
    skeleton as ``keypoints_3d`` (``with_3d``)."""
    kp = project_to_image(cam, pts3d)
    if noise_px:
        rng = rng or np.random.default_rng(0)
        kp = kp + rng.normal(0.0, noise_px, kp.shape)
    sc = np.full(len(kp), float(score))
    return Person2D(kp, sc, bbox=keypoint_bbox(kp), score=person_score,
                    keypoints_3d=body_centred(pts3d, fmt) if with_3d else None)


class FakeDetector(Detector2D):
    """A ``Detector2D`` that returns scripted persons.

    ``script``: a callable ``(cam_name, t, image) -> list[Person2D]`` (e.g. a ``Scene``), or
    a dict ``{camera name: list[Person2D]}``. Every call is logged in ``calls``."""

    key = "fake"
    label = "Fake detector"

    def __init__(self, fmt: str | KeypointFormat = "coco17",
                 script: Callable | dict | None = None, provides_3d: bool = False):
        super().__init__()
        self.format = get_format(fmt)
        self.provides_3d = provides_3d
        self.script = script
        self.calls: list[tuple[str, float]] = []
        self.closed = False

    def detect(self, image_bgr, t, cam_name):
        self.calls.append((cam_name, t))
        s = self.script
        if s is None:
            return []
        if callable(s):
            return list(s(cam_name, t, image_bgr) or [])
        return list(s.get(cam_name, []))

    def close(self):
        self.closed = True


def create_fake(fmt: str = "coco17", provides_3d: bool = False, **options) -> FakeDetector:
    """Factory used to test the backend registry."""
    det = FakeDetector(fmt, provides_3d=provides_3d)
    det.options = dict(fmt=fmt, provides_3d=provides_3d, **options)
    return det


class Scene:
    """Script for ``FakeDetector``: the subject (and bystanders) projected through ``cams``.

    ``subject``: (K, 3) world points, or a callable ``t -> (K, 3)`` for a moving subject.
    ``bystanders``: more (K, 3) skeletons detected in every camera. ``noise_px``: Gaussian
    pixel noise. ``outliers``: {camera: (K, 2) pixel offset added to the subject there} (gross
    errors). ``missing``: cameras that do not see the subject. ``with_3d``: fill
    ``keypoints_3d`` (for a detector with ``provides_3d``)."""

    def __init__(self, cams: dict[str, CameraCalibration], subject, fmt: str | KeypointFormat, *,
                 bystanders=(), noise_px: float = 0.0, outliers: dict | None = None,
                 missing=(), with_3d: bool = False, score: float = 0.9, seed: int = 0):
        self.cams = cams
        self.subject = subject
        self.fmt = get_format(fmt)
        self.bystanders = list(bystanders)
        self.noise_px = noise_px
        self.outliers = dict(outliers or {})
        self.missing = set(missing)
        self.with_3d = with_3d
        self.score = score
        self.rng = np.random.default_rng(seed)

    def subject_at(self, t: float) -> np.ndarray:
        return self.subject(t) if callable(self.subject) else self.subject

    def __call__(self, cam_name: str, t: float, image) -> list[Person2D]:
        cam = self.cams[cam_name]
        people = []
        if cam_name not in self.missing:
            p = project_person(cam, self.subject_at(t), self.fmt, score=self.score,
                               noise_px=self.noise_px, rng=self.rng, with_3d=self.with_3d)
            if cam_name in self.outliers:
                p.keypoints = p.keypoints + np.asarray(self.outliers[cam_name], np.float64)
                p.bbox = keypoint_bbox(p.keypoints)
            people.append(p)
        for b in self.bystanders:
            people.append(project_person(cam, b, self.fmt, score=self.score, with_3d=self.with_3d))
        return people
