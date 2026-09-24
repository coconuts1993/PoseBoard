"""Bring the Wii center of pressure and the 3D pose into a common coordinate frame."""

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
    cop_board: np.ndarray  # (2,) board coordinates
    cop_world: np.ndarray  # (3,) world coordinates
    force_world: np.ndarray  # (3,) ground reaction force vector (N, along the board normal)
    com_world: np.ndarray | None = None
    com_board: np.ndarray | None = None  # (3,) COM in the board frame (z = height above the board surface)

    @property
    def com_minus_cop(self) -> np.ndarray | None:
        """COM projected onto the board plane minus COP (board coordinates, meters)."""
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
