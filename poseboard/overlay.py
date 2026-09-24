"""Drawing on camera frames: clicked landmarks, board outline/sensors/axes, COP and force,
skeleton (any keypoint format), COM."""

from __future__ import annotations

import cv2
import numpy as np

from poseboard.calibration import CameraCalibration
from poseboard.geometry import LANDMARK_NAMES, BoardGeometry, BoardPose
from poseboard.pose.base import Pose3D
from poseboard.pose.formats import FORMATS

CLICK_COLORS = [(0, 0, 255), (0, 200, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255)]


def project_world(cam: CameraCalibration, pts_w: np.ndarray) -> np.ndarray:
    """Project world points; if the camera has no extrinsics, the world frame is the camera frame."""
    pts_w = np.asarray(pts_w, np.float64).reshape(-1, 3)
    out = np.full((len(pts_w), 2), np.nan)
    ok = np.all(np.isfinite(pts_w), axis=1)
    if not ok.any():
        return out
    rvec = cam.rvec if cam.has_extrinsics else np.zeros(3)
    tvec = cam.tvec if cam.has_extrinsics else np.zeros(3)
    # Points behind the camera are not projected
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


SIDE_COLORS = {"left": (255, 160, 60), "right": (60, 160, 255), "center": (0, 230, 0)}  # BGR
POINT_COLOR = (0, 255, 255)


def _side(name: str) -> str:
    n = name.lower()
    if n.startswith(("left_", "l_")) or (name[:1] == "L" and name[1:2].isupper()):
        return "left"  # left_knee, l_knee, LKnee
    if n.startswith(("right_", "r_")) or (name[:1] == "R" and name[1:2].isupper()):
        return "right"
    return "center"


def _is_detail(name: str) -> bool:
    """Face and hand points of whole-body formats: drawn smaller."""
    return name.startswith("face_") or "_hand_" in name


def pose_skeleton(pose: Pose3D, skeleton: list[tuple[str, str]] | None = None
                  ) -> list[tuple[str, str]]:
    """The bone list to draw for ``pose``: ``skeleton`` if given, else the skeleton of the pose's
    keypoint format (``Pose3D.format_key``, or a format whose names match exactly), else []."""
    if skeleton:
        return list(skeleton)
    fmt = FORMATS.get(getattr(pose, "format_key", None) or "")
    if fmt is None:
        names = tuple(pose.names)
        fmt = next((f for f in FORMATS.values() if f.names == names), None)
    return list(fmt.skeleton) if fmt is not None else []


def draw_pose(img: np.ndarray, cam: CameraCalibration, cam_name: str, pose: Pose3D,
              skeleton: list[tuple[str, str]] | None = None, board: BoardPose | None = None,
              com_world: np.ndarray | None = None, min_score: float = 0.0) -> None:
    """Skeleton of ``pose`` in camera ``cam_name``: the 2D keypoints detected in that camera
    (``per_camera_2d``, drawn for every mode, also ``2d_only``) or else the 3D keypoints projected
    into it; bones from ``skeleton`` or the pose's keypoint format (left side blue, right side
    orange). 2D keypoints with a score below ``min_score`` are not drawn. Then the COM and its
    projection on the board."""
    p2 = pose.per_camera_2d.get(cam_name) if pose.per_camera_2d else None
    if p2 is not None:
        pts2d = np.asarray(p2.keypoints, np.float64).reshape(-1, 2)
        sc = (np.ones(len(pts2d)) if p2.scores is None
              else np.asarray(p2.scores, np.float64).reshape(-1))
        ok = np.all(np.isfinite(pts2d), axis=1) & (np.nan_to_num(sc, nan=0.0) >= min_score)
    else:
        pts2d = project_world(cam, pose.keypoints)
        ok = np.all(np.isfinite(pts2d), axis=1)
    names = list(pose.names)
    if len(names) != len(pts2d):  # a plugin's 2D layout differs from its 3D names
        names = [f"kp{i}" for i in range(len(pts2d))]
    idx = {n: i for i, n in enumerate(names)}
    for a, b in pose_skeleton(pose, skeleton):
        i, j = idx.get(a), idx.get(b)
        if i is None or j is None or not (ok[i] and ok[j]):
            continue
        sa, sb = _side(a), _side(b)
        color = SIDE_COLORS[sa if sa == sb else "center"]
        _line(img, pts2d[i], pts2d[j], color, 1 if _is_detail(a) or _is_detail(b) else 2)
    for i, q in enumerate(pts2d):
        p = _p(q) if ok[i] else None
        if p:
            cv2.circle(img, p, 1 if _is_detail(names[i]) else 3, POINT_COLOR, -1, cv2.LINE_AA)
    if com_world is not None:
        pts = [com_world]
        if board is not None:
            # Project onto the board surface along the board normal
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


def draw_caption(img: np.ndarray, text: str, line: int = 0) -> None:
    """A short text on a dark band at the bottom-left of the image (``line`` 0 = lowest)."""
    if not text:
        return
    scale = max(0.45, min(1.0, img.shape[1] / 1600))
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = img.shape[0] - 10 - line * (h + base + 8)
    x0, y0 = 6, y - h - 4
    cv2.rectangle(img, (x0, y0), (min(img.shape[1] - 1, x0 + w + 8), y + base + 2), (30, 30, 30),
                  -1)
    cv2.putText(img, text, (x0 + 4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1,
                cv2.LINE_AA)
