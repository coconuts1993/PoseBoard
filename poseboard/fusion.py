"""把 Wii 压力中心与 3D 姿态放到同一坐标系中。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poseboard.geometry import BoardPose
from poseboard.pose.base import Pose3D
from poseboard.pose.com import center_of_mass
from poseboard.wii.device import ForceSample
from poseboard.wii.protocol import KG_TO_N


@dataclass
class FusedState:
    t: float
    total_kg: float
    force_n: float
    cop_board: np.ndarray  # (2,) 板坐标
    cop_world: np.ndarray  # (3,) 世界坐标
    force_world: np.ndarray  # (3,) 地面反力向量（N，沿板法向）
    com_world: np.ndarray | None = None
    com_board: np.ndarray | None = None  # (3,) 重心在板坐标系中的位置（z 为离板面高度）

    @property
    def com_minus_cop(self) -> np.ndarray | None:
        """COM 在板面上的投影减 COP（板坐标，米）。"""
        if self.com_board is None:
            return None
        return self.com_board[:2] - self.cop_board


def cop_to_world(board: BoardPose, cop_board: tuple[float, float]) -> np.ndarray:
    return board.board_to_world.apply(np.array([cop_board[0], cop_board[1], 0.0]))


def fuse(board: BoardPose | None, force: ForceSample | None, pose: Pose3D | None) -> FusedState | None:
    if board is None or force is None:
        return None
    cop_b = np.array(force.cop_board, np.float64)
    cop_w = cop_to_world(board, force.cop_board) if np.all(np.isfinite(cop_b)) else np.full(3, np.nan)
    f_n = force.total_kg * KG_TO_N
    state = FusedState(force.t, force.total_kg, f_n, cop_b, cop_w, board.up_world * f_n)
    if pose is not None:
        com = center_of_mass(pose.keypoints, pose.names)
        if com is not None:
            state.com_world = com
            state.com_board = board.world_to_board.apply(com)
    return state
