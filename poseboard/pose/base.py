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


class PoseEstimator:
    """Base class for pose estimators."""

    name = "base"
    keypoint_names: list[str] = []
    skeleton: list[tuple[str, str]] = []

    def process(self, frames: dict[str, tuple[float, np.ndarray]],
                cams: dict[str, CameraCalibration]) -> Pose3D | None:
        """frames: {camera name: (timestamp, BGR image)}; cams: {camera name: calibration}."""
        raise NotImplementedError

    def close(self) -> None:
        pass
