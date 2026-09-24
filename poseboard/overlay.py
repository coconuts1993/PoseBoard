"""在相机画面上绘制：点选的标注点、平衡板轮廓/传感器/坐标轴、COP 与力、骨架、COM。"""

from __future__ import annotations

import cv2
import numpy as np

from poseboard.calibration import CameraCalibration
from poseboard.geometry import LANDMARK_NAMES, BoardGeometry, BoardPose
from poseboard.pose.base import Pose3D

CLICK_COLORS = [(0, 0, 255), (0, 200, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255)]


def project_world(cam: CameraCalibration, pts_w: np.ndarray) -> np.ndarray:
    """投影世界坐标点；相机没有外参时世界坐标系即相机坐标系。"""
    pts_w = np.asarray(pts_w, np.float64).reshape(-1, 3)
    out = np.full((len(pts_w), 2), np.nan)
    ok = np.all(np.isfinite(pts_w), axis=1)
    if not ok.any():
        return out
    rvec = cam.rvec if cam.has_extrinsics else np.zeros(3)
    tvec = cam.tvec if cam.has_extrinsics else np.zeros(3)
    # 相机背后的点不投影
    pc = pts_w[ok] @ cv2.Rodrigues(np.asarray(rvec, float))[0].T + np.asarray(tvec, float).ravel()
    img, _ = cv2.projectPoints(pts_w[ok], np.asarray(rvec, float), np.asarray(tvec, float), cam.K, cam.dist)
    img = img.reshape(-1, 2)
    img[pc[:, 2] <= 1e-3] = np.nan
    out[ok] = img
    return out


def _p(pt) -> tuple[int, int] | None:
    if pt is None or not np.all(np.isfinite(pt)) or np.any(np.abs(pt) > 1e5):
        return None
    return int(round(pt[0])), int(round(pt[1]))


def _line(img, a, b, color, th=2):
    a, b = _p(a), _p(b)
    if a and b:
        cv2.line(img, a, b, color, th, cv2.LINE_AA)


def draw_clicks(img: np.ndarray, clicks: list, next_hint: str | None = None) -> None:
    for i, c in enumerate(clicks):
        p = _p(c)
        if p:
            col = CLICK_COLORS[i % len(CLICK_COLORS)]
            cv2.circle(img, p, 6, col, 2, cv2.LINE_AA)
            cv2.drawMarker(img, p, col, cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
            cv2.putText(img, f"{i + 1}:{LANDMARK_NAMES[i]}", (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    if len(clicks) >= 4:
        for i in range(4):
            _line(img, clicks[i], clicks[(i + 1) % 4], (200, 200, 200), 1)
    if next_hint:
        cv2.putText(img, next_hint, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


def draw_board(img: np.ndarray, cam: CameraCalibration, board: BoardPose, geo: BoardGeometry) -> None:
    T = board.board_to_world
    lm = project_world(cam, T.apply(geo.landmarks()))
    for i in range(4):
        _line(img, lm[i], lm[(i + 1) % 4], (255, 255, 0), 2)
    sensors = project_world(cam, T.apply(geo.sensors()))
    for name, s in zip(("TR", "BR", "TL", "BL"), sensors):
        p = _p(s)
        if p:
            cv2.circle(img, p, 5, (180, 180, 180), -1, cv2.LINE_AA)
            cv2.putText(img, name, (p[0] + 6, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (230, 230, 230), 1, cv2.LINE_AA)
    axes = project_world(cam, T.apply(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])))
    for k, col in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        _line(img, axes[0], axes[k], col, 2)


def draw_cop(img: np.ndarray, cam: CameraCalibration, board: BoardPose,
             cop_board: tuple[float, float], total_kg: float, m_per_kg: float = 0.005) -> None:
    if not np.all(np.isfinite(cop_board)):
        return
    T = board.board_to_world
    base = T.apply([cop_board[0], cop_board[1], 0.0])
    tip = base + board.up_world * total_kg * m_per_kg
    pts = project_world(cam, np.vstack([base, tip]))
    _line(img, pts[0], pts[1], (255, 80, 0), 3)
    p = _p(pts[0])
    if p:
        cv2.circle(img, p, 8, (255, 80, 0), -1, cv2.LINE_AA)
        cv2.putText(img, f"COP {total_kg:.1f}kg", (p[0] + 10, p[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 80, 0), 2, cv2.LINE_AA)


def draw_pose(img: np.ndarray, cam: CameraCalibration, cam_name: str, pose: Pose3D,
              skeleton: list[tuple[str, str]], board: BoardPose | None,
              com_world: np.ndarray | None) -> None:
    if cam_name in pose.per_camera_2d:
        pts2d = pose.per_camera_2d[cam_name].keypoints
    else:
        pts2d = project_world(cam, pose.keypoints)
    idx = {n: i for i, n in enumerate(pose.names)}
    for a, b in skeleton:
        if a in idx and b in idx:
            _line(img, pts2d[idx[a]], pts2d[idx[b]], (0, 230, 0), 2)
    for q in pts2d:
        p = _p(q)
        if p:
            cv2.circle(img, p, 3, (0, 255, 255), -1, cv2.LINE_AA)
    if com_world is not None:
        pts = [com_world]
        if board is not None:
            # 沿板法向投影到板面
            n = board.up_world
            d = np.dot(com_world - board.board_to_world.t, n)
            pts.append(com_world - d * n)
        pp = project_world(cam, np.vstack(pts))
        if len(pp) > 1:
            _line(img, pp[0], pp[1], (0, 140, 255), 1)
            q = _p(pp[1])
            if q:
                cv2.drawMarker(img, q, (0, 140, 255), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
        q = _p(pp[0])
        if q:
            cv2.circle(img, q, 9, (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "COM", (q[0] + 10, q[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 140, 255), 2, cv2.LINE_AA)
