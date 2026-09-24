"""姿态估计接口。

任何 3D 姿态来源（MediaPipe、你已有的 PoseAssess、Pose2Sim 输出……）只要实现
``PoseEstimator.process`` 并返回世界坐标系下的 ``Pose3D`` 即可接入 PoseBoard。
世界坐标系 = 棋盘格定义的坐标系（与平衡板定位使用同一套相机外参）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from poseboard.calibration import CameraCalibration


@dataclass
class Pose2D:
    keypoints: np.ndarray  # (K, 2) 像素
    scores: np.ndarray  # (K,)


@dataclass
class Pose3D:
    t: float  # perf_counter 时间戳（多相机时取各帧平均）
    names: list[str]
    keypoints: np.ndarray  # (K, 3) 世界坐标，米；缺失为 NaN
    scores: np.ndarray  # (K,)
    per_camera_2d: dict[str, Pose2D] = field(default_factory=dict)

    def get(self, name: str) -> np.ndarray | None:
        try:
            p = self.keypoints[self.names.index(name)]
        except ValueError:
            return None
        return None if np.any(np.isnan(p)) else p


class PoseEstimator:
    """姿态估计器基类。"""

    name = "base"
    keypoint_names: list[str] = []
    skeleton: list[tuple[str, str]] = []

    def process(self, frames: dict[str, tuple[float, np.ndarray]],
                cams: dict[str, CameraCalibration]) -> Pose3D | None:
        """frames: {相机名: (时间戳, BGR 图像)}；cams: {相机名: 标定}。"""
        raise NotImplementedError

    def close(self) -> None:
        pass
