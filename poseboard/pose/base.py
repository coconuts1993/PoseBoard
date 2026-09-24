"""Pose estimation interface.

Any 3D pose source (MediaPipe, your existing PoseAssess, Pose2Sim output, ...) can be
plugged into PoseBoard by implementing ``PoseEstimator.process`` and returning a
``Pose3D`` in world coordinates.
World frame = the frame defined by the checkerboard (the same camera extrinsics used
to locate the balance board).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from poseboard.calibration import CameraCalibration


# Pose3D.mode values
MODE_TRIANGULATED = "triangulated"  # 2D keypoints of >= 2 calibrated cameras triangulated
MODE_SINGLE_VIEW_3D = "single_view_3d"  # a body-centred 3D skeleton placed with PnP (1 camera)
MODE_2D_ONLY = "2d_only"  # no 3D possible: keypoints are NaN, per_camera_2d holds the 2D poses
POSE_MODES = (MODE_TRIANGULATED, MODE_SINGLE_VIEW_3D, MODE_2D_ONLY)


@dataclass
class Pose2D:
    keypoints: np.ndarray  # (K, 2) pixels in the original image; NaN if missing
    scores: np.ndarray  # (K,) 0..1
    t: float | None = None  # capture time (perf_counter) of the frame this pose comes from
    # Recorded video frame number (the "frame" column of camN_timestamps.csv) while recording;
    # None when the frame was not written to a video.
    frame_index: int | None = None


@dataclass
class Pose3D:
    t: float  # perf_counter timestamp (mean over frames when using multiple cameras)
    names: list[str]
    keypoints: np.ndarray  # (K, 3) world coordinates, meters; NaN if missing
    scores: np.ndarray  # (K,)
    # 2D subject of every camera in which one was found (also when no 3D was possible)
    per_camera_2d: dict[str, Pose2D] = field(default_factory=dict)
    mode: str = MODE_TRIANGULATED  # see POSE_MODES
    views_used: list[str] = field(default_factory=list)  # cameras that contributed to the 3D
    reproj_error_px: float | None = None  # mean reprojection error of the 3D keypoints (pixels)
    notes: list[str] = field(default_factory=list)  # e.g. "cam1 not used for 3D: no extrinsics"
    format_key: str | None = None  # key in poseboard.pose.formats.FORMATS (None: unknown layout)
    # Every camera whose frame was processed for this pose (with or without a subject):
    # {camera: (capture time, recorded video frame number or None)}. Set by the estimator
    # (frame number None) and completed by ``attach_frame_info``; the recorder writes an empty
    # 2D entry for processed cameras without a subject.
    camera_frames: dict[str, tuple[float, int | None]] = field(default_factory=dict)

    def get(self, name: str) -> np.ndarray | None:
        try:
            p = self.keypoints[self.names.index(name)]
        except ValueError:
            return None
        return None if np.any(np.isnan(p)) else p


def check_pose3d(pose: Pose3D) -> Pose3D:
    """Validate a pose returned by an estimator (e.g. a plugin); raises ValueError with a clear
    message instead of failing later while writing pose3d.csv."""
    if not isinstance(pose, Pose3D):
        raise ValueError(f"process() must return a Pose3D or None, not {type(pose).__name__}")
    try:
        t = float(pose.t)
    except (TypeError, ValueError):
        raise ValueError(f"Pose3D.t must be the frame time (a number), not {pose.t!r}") from None
    if not np.isfinite(t):
        raise ValueError("Pose3D.t is not finite")
    k = len(pose.names)
    kp = np.asarray(pose.keypoints, np.float64)
    if kp.shape != (k, 3):
        raise ValueError(f"Pose3D.keypoints must have shape ({k}, 3) for {k} names, got {kp.shape}")
    sc = np.asarray(pose.scores, np.float64).reshape(-1)
    if sc.shape != (k,):
        raise ValueError(f"Pose3D.scores must have {k} values, got {sc.shape[0]}")
    pose.t, pose.keypoints, pose.scores = t, kp, sc
    per2d = pose.per_camera_2d if pose.per_camera_2d is not None else {}
    if not isinstance(per2d, Mapping):
        raise ValueError("Pose3D.per_camera_2d must be a dict {camera name: Pose2D}")
    for cam, p in per2d.items():
        if not isinstance(p, Pose2D):
            raise ValueError(f"Pose3D.per_camera_2d[{cam!r}] must be a Pose2D, not {type(p).__name__}")
        k2 = np.asarray(p.keypoints, np.float64)
        s2 = (np.ones(len(k2)) if p.scores is None
              else np.asarray(p.scores, np.float64).reshape(-1))
        if k2.ndim != 2 or k2.shape[1] != 2 or s2.shape != (k2.shape[0],):
            raise ValueError(f"Pose3D.per_camera_2d[{cam!r}]: keypoints must be (K, 2) and scores "
                             f"(K,), got {k2.shape} and {s2.shape}")
        p.keypoints, p.scores = k2, s2
    pose.per_camera_2d = dict(per2d)
    pose.camera_frames = dict(pose.camera_frames or {})
    pose.views_used = list(pose.views_used or [])
    pose.notes = list(pose.notes or [])
    if not isinstance(pose.mode, str) or not pose.mode:
        raise ValueError(f"Pose3D.mode must be a text such as {', '.join(POSE_MODES)}, "
                         f"not {pose.mode!r}")
    return pose


def attach_frame_info(pose: Pose3D, frames: Mapping[str, Any]) -> Pose3D:
    """Record which camera frames a pose was computed from.

    ``frames``: {camera name: frame} for every frame passed to ``process()``, where a frame has
    ``t`` (capture time) and ``rec_index`` (recorded video frame number or None), e.g.
    ``poseboard.camera.Frame``. Fills ``pose.camera_frames`` and ``t`` / ``frame_index`` of
    the matching ``per_camera_2d`` entries, so the recorder can reference the exact video
    frames (pose2d_<cam>.csv, OpenPose JSON)."""
    for name, f in frames.items():
        t = float(f.t)
        idx = getattr(f, "rec_index", None)
        pose.camera_frames[name] = (t, None if idx is None else int(idx))
        p2 = pose.per_camera_2d.get(name)
        if p2 is not None:
            p2.t = t
            p2.frame_index = None if idx is None else int(idx)
    return pose


class PoseEstimator:
    """Base class for pose estimators."""

    name = "base"
    keypoint_names: list[str] = []
    skeleton: list[tuple[str, str]] = []
    # When no camera has extrinsics, the world frame is the camera frame of this camera (the one
    # the balance board was registered with); PoseBoard sets it. Single-view results must then
    # come from this camera only, so pose and board share one frame.
    world_camera: str | None = None

    def process(self, frames: dict[str, tuple[float, np.ndarray]],
                cams: dict[str, CameraCalibration]) -> Pose3D | None:
        """frames: {camera name: (timestamp, BGR image)}; cams: {camera name: calibration}.

        If any camera in ``cams`` has extrinsics, return world (checkerboard) coordinates computed
        only from cameras with extrinsics; otherwise use only ``world_camera`` (the first camera
        in ``cams`` when it is None). When ``world_camera`` is set but not in ``frames`` / ``cams``
        (it is not running), return no 3D (None, or a pose with NaN keypoints): lifting the pose
        in another camera's frame would mix it with the board's frame."""
        raise NotImplementedError

    def close(self) -> None:
        pass
