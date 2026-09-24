"""Pick the subject (the person standing on the Balance Board) among the persons detected in one
camera image.

Priority:

1. the person whose feet (midpoint of the confident ankle/heel keypoints) lie inside the
   board's projected outline, or nearest to it (within about one board diagonal);
2. otherwise the person whose box overlaps most with the subject's box in the previous frame
   of this camera (tracking continuity);
3. otherwise the person with the largest box area x detection score.
"""

from __future__ import annotations

import cv2
import numpy as np

from poseboard.calibration import CameraCalibration
from poseboard.geometry import BoardGeometry, BoardPose
from poseboard.pose.detectors.base import Person2D
from poseboard.pose.formats import KeypointFormat

FOOT_NAMES = ("left_ankle", "right_ankle", "left_heel", "right_heel")
FOOT_MIN_SCORE = 0.3
# A person "near" the board: feet at most this many board diagonals (in pixels) outside it
NEAR_BOARD_DIAGONALS = 1.0
MIN_IOU = 0.1  # below this the previous box does not identify anybody


def board_polygon_px(board: BoardPose | None, cam: CameraCalibration,
                     geometry: BoardGeometry | None = None, lift_m: float = 0.12) -> np.ndarray | None:
    """Outline (N, 2) in pixels of the region where the subject's feet appear in this camera:
    the convex hull of the board's 4 top-surface corners and the same corners raised by
    ``lift_m`` along the board normal (ankles are about 8 cm above the surface).

    None when the camera is in another frame than the board (the frame rule: a board registered
    on the checkerboard needs a camera with extrinsics; a board registered without extrinsics
    is in the camera frame of ``board.world_camera`` and only usable in that camera), or when
    a corner is behind the camera."""
    if board is None or cam is None:
        return None
    if board.world_camera is None:
        if not cam.has_extrinsics:
            return None
        rvec, tvec = np.asarray(cam.rvec, np.float64), np.asarray(cam.tvec, np.float64)
    else:
        if cam.has_extrinsics or cam.name != board.world_camera:
            return None
        rvec, tvec = np.zeros(3), np.zeros(3)
    geo = geometry or BoardGeometry()
    corners = geo.landmarks()[:4]
    lifted = corners + np.array([0.0, 0.0, lift_m])
    pts_w = board.board_to_world.apply(np.vstack([corners, lifted]))
    R = cv2.Rodrigues(rvec)[0]
    if np.any(pts_w @ R.T @ np.array([0, 0, 1.0]) + tvec.reshape(3)[2] <= 1e-3):
        return None  # behind the camera
    img, _ = cv2.projectPoints(pts_w, rvec, tvec, cam.K, cam.dist)
    img = img.reshape(-1, 2)
    if not np.all(np.isfinite(img)):
        return None
    return cv2.convexHull(img.astype(np.float32)).reshape(-1, 2).astype(np.float64)


def foot_point(person: Person2D, fmt: KeypointFormat, min_score: float = FOOT_MIN_SCORE
               ) -> np.ndarray | None:
    """Midpoint (2,) of the confident ankle/heel keypoints; None if there are none."""
    pts = []
    for n in FOOT_NAMES:
        i = fmt.find(n)
        if i is None or i >= len(person.keypoints):
            continue
        p = person.keypoints[i]
        if person.scores[i] >= min_score and np.all(np.isfinite(p)):
            pts.append(p)
    return np.mean(pts, axis=0) if pts else None


def iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(a) + _area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def _area(b: np.ndarray | None) -> float:
    if b is None:
        return 0.0
    return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def _size_score(p: Person2D) -> float:
    return _area(p.box(FOOT_MIN_SCORE)) * max(p.score, 1e-6)


def select_subject(people: list[Person2D], *, board_polygon_px: np.ndarray | None = None,
                   previous_bbox: np.ndarray | None = None,
                   fmt: KeypointFormat) -> Person2D | None:
    """The subject among ``people`` (see the module docstring); None if the list is empty."""
    people = [p for p in people if p is not None]
    if not people:
        return None
    if len(people) == 1:
        return people[0]
    prev = None if previous_bbox is None else np.asarray(previous_bbox, np.float64).reshape(4)

    if board_polygon_px is not None and len(board_polygon_px) >= 3:
        poly = np.asarray(board_polygon_px, np.float32).reshape(-1, 1, 2)
        diag = float(np.linalg.norm(np.ptp(poly.reshape(-1, 2), axis=0)))
        cands = []
        for p in people:
            f = foot_point(p, fmt)
            if f is None:
                continue
            d = cv2.pointPolygonTest(poly, (float(f[0]), float(f[1])), True)  # > 0 inside
            cands.append((p, d))
        inside = [p for p, d in cands if d >= 0]
        if inside:
            # several feet on the board (overlapping persons in the image): tracking, then size
            return max(inside, key=lambda p: (iou(p.box(), prev) >= MIN_IOU, _size_score(p)))
        near = [(p, d) for p, d in cands if -d <= NEAR_BOARD_DIAGONALS * diag]
        if near:
            return max(near, key=lambda pd: pd[1])[0]

    if prev is not None:
        best = max(people, key=lambda p: iou(p.box(), prev))
        if iou(best.box(), prev) >= MIN_IOU:
            return best
    return max(people, key=_size_score)
