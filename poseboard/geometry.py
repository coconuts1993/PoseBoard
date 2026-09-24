"""Wii Balance Board geometry model, locating it in the camera (clicking the 4 corners +
center), and triangulation utilities.

Balance Board frame (units: meters)
-----------------------------------
Top view of the board, standing on it facing "forward" (toward the TL/TR side):

        TL ─────────── TR          +Y (front, "top")
        │               │           ↑
        │       C       │           └──→ +X (right)
        │               │
        BL ─────────── BR

* Origin C is at the center of the top surface; Z points up.
* TL/TR/BR/BL name both the four sensors (as ordered in the Wii data) and the nearest
  outer corners of the board surface.
* The click order is fixed: TL, TR, BR, BL, C.

Default dimensions (official Wii Balance Board): outer surface about 511 x 316 mm,
sensor center spacing 433 mm (left-right) x 238 mm (front-back).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from poseboard.calibration import CameraCalibration

LANDMARK_NAMES = ("TL", "TR", "BR", "BL", "C")
SENSOR_NAMES = ("TR", "BR", "TL", "BL")  # matches the Wii data order


@dataclass
class BoardGeometry:
    length_mm: float = 511.0  # outer surface size along X (left-right)
    width_mm: float = 316.0  # outer surface size along Y (front-back)
    sensor_dx_mm: float = 433.0  # left-right sensor center spacing
    sensor_dy_mm: float = 238.0  # front-back sensor center spacing

    def landmarks(self) -> np.ndarray:
        """Landmarks TL, TR, BR, BL, C in the board frame (5,3), meters."""
        hx, hy = self.length_mm / 2000.0, self.width_mm / 2000.0
        return np.array([[-hx, hy, 0], [hx, hy, 0], [hx, -hy, 0], [-hx, -hy, 0], [0, 0, 0]],
                        dtype=np.float64)

    def sensors(self) -> np.ndarray:
        """Positions of the four sensors (TR, BR, TL, BL) (4,3), meters."""
        hx, hy = self.sensor_dx_mm / 2000.0, self.sensor_dy_mm / 2000.0
        return np.array([[hx, hy, 0], [hx, -hy, 0], [-hx, hy, 0], [-hx, -hy, 0]], dtype=np.float64)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RigidTransform:
    """X_dst = R @ X_src + t"""

    R: np.ndarray
    t: np.ndarray

    def apply(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, np.float64)
        return pts @ self.R.T + self.t

    def inverse(self) -> "RigidTransform":
        return RigidTransform(self.R.T, -self.R.T @ self.t)

    def to_dict(self) -> dict:
        return {"R": self.R.tolist(), "t": self.t.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "RigidTransform":
        return cls(np.asarray(d["R"], np.float64), np.asarray(d["t"], np.float64))


@dataclass
class BoardPose:
    board_to_world: RigidTransform
    method: str  # "pnp" / "triangulation"
    reproj_error_px: dict[str, float]

    @property
    def world_to_board(self) -> RigidTransform:
        return self.board_to_world.inverse()

    @property
    def up_world(self) -> np.ndarray:
        """Board surface normal (board +Z) expressed in world coordinates."""
        return self.board_to_world.R[:, 2]

    def to_dict(self) -> dict:
        return {"board_to_world": self.board_to_world.to_dict(), "method": self.method,
                "reproj_error_px": self.reproj_error_px}

    @classmethod
    def from_dict(cls, d: dict) -> "BoardPose":
        return cls(RigidTransform.from_dict(d["board_to_world"]), d["method"],
                   d.get("reproj_error_px", {}))


def rotate_board_frame(pose: BoardPose, degrees: int) -> BoardPose:
    """Rotate the board frame about the board normal (use when the clicks were oriented
    wrongly, e.g. 180°)."""
    a = np.deg2rad(degrees)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    T = pose.board_to_world
    return BoardPose(RigidTransform(T.R @ Rz, T.t.copy()), pose.method, pose.reproj_error_px)


# ------------------------------------------------------------------ utilities
def kabsch(src: np.ndarray, dst: np.ndarray) -> RigidTransform:
    """Find the rigid transform such that dst ≈ R @ src + t (least squares)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return RigidTransform(R, cd - R @ cs)


def triangulate_point(cams: list[CameraCalibration], pts_px: list[np.ndarray],
                      weights: list[float] | None = None) -> np.ndarray:
    """Multi-view DLT triangulation of a single point. pts_px holds the pixel coordinates
    in each camera."""
    if len(cams) < 2:
        raise ValueError("Triangulation requires at least 2 cameras")
    weights = weights or [1.0] * len(cams)
    rows = []
    for cam, p, w in zip(cams, pts_px, weights):
        x, y = cam.undistort_normalized(np.asarray(p).reshape(1, 2))[0]
        P = cam.projection_matrix(normalized=True)
        rows.append(w * (x * P[2] - P[0]))
        rows.append(w * (y * P[2] - P[1]))
    A = np.asarray(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def reprojection_error(cam: CameraCalibration, pts_w: np.ndarray, pts_px: np.ndarray) -> float:
    return float(np.linalg.norm(cam.project(pts_w) - np.asarray(pts_px).reshape(-1, 2), axis=1).mean())


# ---------------------------------------------------------- board registration
def register_board(geometry: BoardGeometry, cams: list[CameraCalibration],
                   clicks: list[np.ndarray | None]) -> BoardPose:
    """Compute the board pose in the world frame from the 5 landmarks (TL, TR, BR, BL, C)
    clicked in each camera image.

    * If ≥2 cameras with extrinsics have all 5 clicks: triangulate the 5 points, then align
      them to the board model with Kabsch.
    * Otherwise: solve PnP from a single camera's clicks (planar IPPE + iterative refinement),
      then transform to the world frame using the camera extrinsics.
      If that camera has no extrinsics, the world frame is that camera's frame.
    """
    model = geometry.landmarks()
    usable = [(c, np.asarray(k, np.float64).reshape(5, 2)) for c, k in zip(cams, clicks)
              if k is not None and len(k) == 5]
    if not usable:
        raise ValueError("No camera has all 5 landmarks clicked")

    with_ext = [(c, k) for c, k in usable if c.has_extrinsics]
    if len(with_ext) >= 2:
        cs = [c for c, _ in with_ext]
        pts_w = np.array([triangulate_point(cs, [k[i] for _, k in with_ext]) for i in range(5)])
        T = kabsch(model, pts_w)
        method = "triangulation"
    else:
        cam, k = with_ext[0] if with_ext else usable[0]
        T_bc = solve_board_pnp(cam, model, k)
        if cam.has_extrinsics:
            T_cw = RigidTransform(cam.R, cam.t).inverse()  # camera -> world
            T = RigidTransform(T_cw.R @ T_bc.R, T_cw.R @ T_bc.t + T_cw.t)
        else:
            T = T_bc
        method = "pnp"

    errors = {}
    for c, k in usable:
        if c.has_extrinsics or method == "pnp":
            pts_w = T.apply(model)
            if c.has_extrinsics:
                errors[c.name] = reprojection_error(c, pts_w, k)
            else:  # world frame = this camera frame
                img, _ = cv2.projectPoints(pts_w, np.zeros(3), np.zeros(3), c.K, c.dist)
                errors[c.name] = float(np.linalg.norm(img.reshape(-1, 2) - k, axis=1).mean())
    return BoardPose(T, method, errors)


def solve_board_pnp(cam: CameraCalibration, model: np.ndarray, img_px: np.ndarray) -> RigidTransform:
    """Board frame -> camera frame."""
    obj = np.asarray(model, np.float64).reshape(-1, 1, 3)
    img = np.asarray(img_px, np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("Balance board PnP failed; check the click order")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, cam.K, cam.dist, rvec, tvec)
    return RigidTransform(cv2.Rodrigues(rvec)[0], tvec.ravel())
