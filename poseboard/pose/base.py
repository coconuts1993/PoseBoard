"""Pose estimation interface.

Any 3D pose source (MediaPipe, your existing PoseAssess, Pose2Sim output, ...) can be
plugged into PoseBoard by implementing ``PoseEstimator.process`` and returning a
``Pose3D`` in world coordinates.
World frame = the frame defined by the checkerboard (the same camera extrinsics used
to locate the balance board).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from poseboard.calibration import CameraCalibration


@dataclass
class Pose2D:
    keypoints: np.ndarray  # (K, 2) pixels
    scores: np.ndarray  # (K,)


@dataclass
class Pose3D:
    t: float  # perf_counter timestamp (mean over frames when using multiple cameras)
    names: list[str]
    keypoints: np.ndarray  # (K, 3) world coordinates, meters; NaN if missing
    scores: np.ndarray  # (K,)
    per_camera_2d: dict[str, Pose2D] = field(default_factory=dict)

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
        only from cameras with extrinsics; otherwise use ``world_camera`` (default: the first
        camera in ``cams``)."""
        raise NotImplementedError

    def close(self) -> None:
        pass
