"""合成测试场景：地面上的棋盘格 + 一块平衡板，用于无硬件时的端到端测试/演示。"""

from __future__ import annotations

import cv2
import numpy as np

from poseboard.calibration import CheckerboardSpec
from poseboard.geometry import BoardGeometry, RigidTransform
from tests.test_core import make_cam

SPEC = CheckerboardSpec(9, 6, 60)
BOARD_T = RigidTransform(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]),
                         np.array([0.95, 0.2, 0.053]))


def _poly(cam, pts_w):
    """亚像素精度的多边形顶点（配合 shift=4 使用）。"""
    return np.round(cam.project(pts_w) * 16).astype(np.int32)


def scene_camera():
    return make_cam("cam0", [0.6, -0.9, 1.3], [0.6, 0.25, 0.0], size=(1280, 720), f=900.0)


def render_scene(cam=None, geo: BoardGeometry | None = None) -> np.ndarray:
    cam = cam or scene_camera()
    geo = geo or BoardGeometry()
    w, h = cam.image_size
    img = np.full((h, w, 3), 110, np.uint8)
    # 棋盘格（第一个内角点为世界原点）
    sq_m = SPEC.square_mm / 1000
    for r in range(-1, SPEC.rows):
        for c in range(-1, SPEC.cols):
            if (r + c) % 2:
                continue
            quad = np.array([[c, r, 0], [c + 1, r, 0], [c + 1, r + 1, 0], [c, r + 1, 0]], float) * sq_m
            cv2.fillConvexPoly(img, _poly(cam, quad), (20, 20, 20), cv2.LINE_AA, 4)
    # 棋盘格白色边框
    border = np.array([[-2, -2, 0], [SPEC.cols + 1, -2, 0], [SPEC.cols + 1, SPEC.rows + 1, 0],
                       [-2, SPEC.rows + 1, 0]], float) * sq_m
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, _poly(cam, border), 255, cv2.LINE_8, 4)
    white = np.full_like(img, 245)
    ink = img.astype(np.float32)
    # 棋盘格区域：黑块保持，其余（包括抗锯齿边缘）按黑度混合到白色
    alpha = np.clip((110 - ink[:, :, :1]) / 90.0, 0, 1)
    blended = (alpha * 20 + (1 - alpha) * white).astype(np.uint8)
    img[mask > 0] = blended[mask > 0]
    # 平衡板
    corners = _poly(cam, BOARD_T.apply(geo.landmarks()[:4]))
    cv2.fillConvexPoly(img, corners, (235, 235, 240), cv2.LINE_AA, 4)
    cv2.polylines(img, [corners], True, (160, 160, 160), 2, cv2.LINE_AA, 4)
    return img


def write_scene_video(path, n=20):
    img = render_scene()
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (img.shape[1], img.shape[0]))
    for _ in range(n):
        vw.write(img)
    vw.release()
    return img
