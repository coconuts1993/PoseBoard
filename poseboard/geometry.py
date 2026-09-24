"""Wii Balance Board geometry model, locating it in the camera (clicking the 4 corners +
center), and triangulation utilities.

Balance Board frame (units: meters)
-----------------------------------
Top view of the board, standing on it facing "forward" (toward the TL/TR edge):

        TL ─────────── TR          +Y (front, "top")
        │               │           ↑
        │       C       │           └──→ +X (right)
        │               │
        BL ─────────── BR
          [power button]

* Origin C is at the center of the top surface; Z points up.
* TL/TR/BR/BL name both the four sensors (as ordered in the Wii data) and the nearest
  outer corners of the board surface.
* TL/TR is the long edge OPPOSITE the power button (blue LED); BL/BR is the power-button
  edge. The subject stands facing away from the power button (the Wii Fit placement, with
  the power button facing away from the TV), so +Y is the subject's front and +X their right.
* The click order is fixed: TL, TR, BR, BL, C.

Default dimensions (official Wii Balance Board): outer surface about 511 x 316 mm,
sensor center spacing 433 mm (left-right) x 238 mm (front-back), top surface about 53 mm
above the floor.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from poseboard.calibration import CameraCalibration

LANDMARK_NAMES = ("TL", "TR", "BR", "BL", "C")
SENSOR_NAMES = ("TR", "BR", "TL", "BL")  # matches the Wii data order

# The board is assumed to lie flat on the checkerboard floor when its unconstrained fit is
# tilted at most this much; beyond it the 6-DoF fit is kept and a warning is given.
FLOOR_MAX_TILT_DEG = 10.0
# Tilt of the unconstrained fit above which a note is shown (click noise or an uneven floor).
TILT_NOTE_DEG = 1.0

MIRRORED_CLICKS_MSG = (
    "The click order is mirrored (left/right or front/back swapped): the board would face "
    "downwards. Redo the clicks in the order TL, TR, BR, BL, C, where TL/TR is the long edge "
    "OPPOSITE the power button and left/right are the subject's left/right when standing on "
    "the board facing that edge (not left/right in the image).")


@dataclass
class BoardGeometry:
    length_mm: float = 511.0  # outer surface size along X (left-right)
    width_mm: float = 316.0  # outer surface size along Y (front-back)
    sensor_dx_mm: float = 433.0  # left-right sensor center spacing
    sensor_dy_mm: float = 238.0  # front-back sensor center spacing
    # Height of the top (standing) surface above the checkerboard surface. Used to place the
    # board on the floor with a single camera (subtract the checkerboard's own thickness if
    # it lies on a mat or foam board).
    height_mm: float = 53.0

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
    # Camera whose frame is the world frame when no camera had extrinsics (None: the world
    # frame is the floor checkerboard). Pose estimation must use the same camera.
    world_camera: str | None = None
    # True when the board was fitted flat on the checkerboard floor (x, y, yaw only).
    floor_constrained: bool = False
    # Tilt (degrees) of the unconstrained 6-DoF fit relative to world +Z (None without
    # extrinsics). With floor_constrained this tilt was discarded.
    tilt_deg: float | None = None
    warnings: list[str] = field(default_factory=list)  # problems the user should look at
    notes: list[str] = field(default_factory=list)  # information (cameras not used, tilt removed)

    @property
    def world_to_board(self) -> RigidTransform:
        return self.board_to_world.inverse()

    @property
    def up_world(self) -> np.ndarray:
        """Board surface normal (board +Z) expressed in world coordinates."""
        return self.board_to_world.R[:, 2]

    def to_dict(self) -> dict:
        return {"board_to_world": self.board_to_world.to_dict(), "method": self.method,
                "reproj_error_px": self.reproj_error_px, "world_camera": self.world_camera,
                "floor_constrained": self.floor_constrained, "tilt_deg": self.tilt_deg,
                "warnings": list(self.warnings), "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, d: dict) -> "BoardPose":
        return cls(RigidTransform.from_dict(d["board_to_world"]), d["method"],
                   d.get("reproj_error_px", {}), d.get("world_camera"),
                   bool(d.get("floor_constrained", False)), d.get("tilt_deg"),
                   list(d.get("warnings", [])), list(d.get("notes", [])))


def rotate_board_frame(pose: BoardPose, degrees: int) -> BoardPose:
    """Rotate the board frame by 180° about the board normal (use when the clicks were made
    with front and back swapped). Only multiples of 180° are allowed: the TL-TR sensor axis is
    always the board's long axis, so a 90° turn would give an impossible sensor layout."""
    if degrees % 180:
        raise ValueError("Only 180° rotations are possible; for other mix-ups redo the clicks")
    a = np.deg2rad(degrees)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    T = pose.board_to_world
    return dataclasses.replace(pose, board_to_world=RigidTransform(T.R @ Rz, T.t.copy()),
                               warnings=list(pose.warnings), notes=list(pose.notes))


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


def procrustes_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Planar rotation R (2x2, det +1) and translation t with dst ≈ R @ src + t (least squares)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    a = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return R, cd - R @ cs


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


def backproject_to_plane(cam: CameraCalibration, pts_px: np.ndarray, z: float) -> np.ndarray:
    """Intersect the viewing rays of pixels with the world plane Z = z (camera needs extrinsics)."""
    n = cam.undistort_normalized(np.asarray(pts_px, np.float64).reshape(-1, 2))
    d_w = np.column_stack([n, np.ones(len(n))]) @ cam.R  # R^T @ ray (camera -> world)
    c = cam.center_world
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (z - c[2]) / d_w[:, 2]
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("The clicked points do not lie on the floor seen by this camera; check "
                         "the camera extrinsics (the checkerboard must lie flat on the floor)")
    return c + s[:, None] * d_w


def project_to_image(cam: CameraCalibration, pts_w: np.ndarray) -> np.ndarray:
    """Project world points; without extrinsics the world frame is the camera frame."""
    if cam.has_extrinsics:
        return cam.project(pts_w)
    img, _ = cv2.projectPoints(np.asarray(pts_w, np.float64).reshape(-1, 3), np.zeros(3),
                               np.zeros(3), cam.K, cam.dist)
    return img.reshape(-1, 2)


def reprojection_error(cam: CameraCalibration, pts_w: np.ndarray, pts_px: np.ndarray) -> float:
    return float(np.linalg.norm(project_to_image(cam, pts_w) - np.asarray(pts_px).reshape(-1, 2),
                                axis=1).mean())


def _camera_center(cam: CameraCalibration) -> np.ndarray:
    return cam.center_world if cam.has_extrinsics else np.zeros(3)


def _faces_camera(T: RigidTransform, center: np.ndarray) -> bool:
    """True if the board's top surface (normal = board +Z) faces a camera at ``center``."""
    return float(T.R[:, 2] @ (np.asarray(center) - T.t)) > 0


# ---------------------------------------------------------- board registration
def register_board(geometry: BoardGeometry, cams: list[CameraCalibration],
                   clicks: list[np.ndarray | None], floor: bool = True) -> BoardPose:
    """Compute the board pose in the world frame from the 5 landmarks (TL, TR, BR, BL, C)
    clicked in each camera image.

    * If any clicked camera has extrinsics, the world frame is the floor checkerboard and only
      cameras with extrinsics are used.
      - ≥2 such cameras: triangulate the 5 points; otherwise single-camera PnP (IPPE, choosing
        the solution whose normal is closest to world +Z).
      Cameras without extrinsics are listed in ``notes``.
      - With ``floor`` (default) the board is then fitted flat on the floor: x, y and yaw only
        (single camera: the clicks are intersected with the plane Z = board height; several
        cameras: planar fit to the triangulated points). This removes the 0.5-2° tilt that a
        full 6-DoF fit gets from click noise, which would otherwise bias the COM in board
        coordinates by centimeters. The discarded tilt is reported in ``tilt_deg``; above
        ``FLOOR_MAX_TILT_DEG`` the 6-DoF fit is kept and a warning is given instead.
    * If no clicked camera has extrinsics: PnP with the first camera; the world frame is that
      camera's frame (``world_camera``). Of the two planar PnP solutions the one whose normal
      points up in the image is used (upright camera).

    Raises ValueError when the clicks are mirrored (the board would face away from the camera).
    """
    model = geometry.landmarks()
    usable = [(c, np.asarray(k, np.float64).reshape(5, 2)) for c, k in zip(cams, clicks)
              if k is not None and len(k) == 5]
    if not usable:
        raise ValueError("No camera has all 5 landmarks clicked")

    warnings: list[str] = []
    notes: list[str] = []
    with_ext = [(c, k) for c, k in usable if c.has_extrinsics]
    world_camera = None
    if with_ext:
        ignored = [c.name for c, _ in usable if not c.has_extrinsics]
        if ignored:
            notes.append(f"Not used: {', '.join(ignored)} (no extrinsics; the world frame is "
                            "the floor checkerboard). Set their extrinsics in tab 2 to use them.")
        used = with_ext
    else:
        used = usable[:1]
        world_camera = used[0][0].name
        others = [c.name for c, _ in usable[1:]]
        notes.append(f"Camera extrinsics not set: the world frame is the camera frame of "
                        f"{world_camera}" + (f" (clicks in {', '.join(others)} not used)" if others
                                             else "") + ".")

    pts_w = None
    if len(used) >= 2:
        cs = [c for c, _ in used]
        pts_w = np.array([triangulate_point(cs, [k[i] for _, k in used]) for i in range(5)])
        T6 = kabsch(model, pts_w)
        method = "triangulation"
        if not all(_faces_camera(T6, _camera_center(c)) for c in cs):
            raise ValueError(MIRRORED_CLICKS_MSG)
    else:
        cam, k = used[0]
        T_bc, ambiguous = solve_board_pnp(cam, model, k, return_ambiguity=True)
        if cam.has_extrinsics:
            T_cw = RigidTransform(cam.R, cam.t).inverse()  # camera -> world
            T6 = RigidTransform(T_cw.R @ T_bc.R, T_cw.R @ T_bc.t + T_cw.t)
        else:
            T6 = T_bc
        method = "pnp"
        if ambiguous and not (floor and cam.has_extrinsics):
            warnings.append("Two board orientations fit these clicks almost equally well (single "
                            "camera): check the board normal. Setting the camera extrinsics (tab 2) "
                            "or clicking in a second camera removes this ambiguity.")

    T, constrained, tilt = T6, False, None
    if with_ext:
        tilt = float(np.degrees(np.arccos(np.clip(T6.R[2, 2], -1.0, 1.0))))
        if floor and tilt <= FLOOR_MAX_TILT_DEG:
            if pts_w is not None:
                xy, z = pts_w[:, :2], float(pts_w[:, 2].mean())
            else:
                z = geometry.height_mm / 1000.0
                xy = backproject_to_plane(used[0][0], used[0][1], z)[:, :2]
            R2, t2 = procrustes_2d(model[:, :2], xy)
            R = np.eye(3)
            R[:2, :2] = R2
            T, constrained = RigidTransform(R, np.array([t2[0], t2[1], z])), True
            if tilt > TILT_NOTE_DEG:
                notes.append(f"The unconstrained fit was tilted {tilt:.1f}° from the floor; the "
                                "board is assumed to lie flat on the checkerboard floor and this "
                                "tilt was removed. Large values mean imprecise clicks or a board "
                                "that is not on the same floor as the checkerboard.")
        elif tilt > FLOOR_MAX_TILT_DEG:
            warnings.append(f"The board is tilted {tilt:.0f}° relative to the checkerboard floor. "
                            "Check the click order (TL-TR must be a LONG edge), the board "
                            "dimensions, and that the checkerboard lay flat on the same floor as "
                            "the board, then redo the clicks. The pose was not fitted to the floor.")

    errors = {c.name: reprojection_error(c, T.apply(model), k) for c, k in used}
    return BoardPose(T, method, errors, world_camera, constrained, tilt, warnings, notes)


def solve_board_pnp(cam: CameraCalibration, model: np.ndarray, img_px: np.ndarray,
                    return_ambiguity: bool = False):
    """Board frame -> camera frame from clicked points (planar IPPE + Levenberg-Marquardt).

    A planar target has two IPPE solutions and, with noisy clicks, the wrong one can have the
    lower reprojection error. With extrinsics the solution whose normal is closest to world +Z
    is used, otherwise the one whose normal points up in the image (upright camera). Raises
    ValueError when the chosen board faces away from the camera (mirrored clicks). With
    ``return_ambiguity`` returns ``(transform, ambiguous)``, where ``ambiguous`` means both
    solutions fit within a factor of 2.
    """
    obj = np.asarray(model, np.float64).reshape(-1, 1, 3)
    img = np.asarray(img_px, np.float64).reshape(-1, 1, 2)
    sols: list[tuple[np.ndarray, np.ndarray, float]] = []
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, cam.K, cam.dist,
                                                    flags=cv2.SOLVEPNP_IPPE)
        errs = np.ravel(errs) if errs is not None else np.full(n, np.nan)
        sols = [(np.asarray(r, np.float64).reshape(3, 1), np.asarray(t, np.float64).reshape(3, 1),
                 float(e)) for r, t, e in zip(rvecs[:n], tvecs[:n], errs[:n])]
    except cv2.error:
        pass
    if not sols:
        ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("Balance board PnP failed; check the click order")
        sols = [(rvec, tvec, float("nan"))]

    def normal(s):
        return cv2.Rodrigues(s[0])[0][:, 2]

    if cam.has_extrinsics:
        rvec, tvec, _ = max(sols, key=lambda s: float((cam.R.T @ normal(s))[2]))
    else:  # camera y points down in the image: most negative y = normal pointing up
        rvec, tvec, _ = min(sols, key=lambda s: float(normal(s)[1]))
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, cam.K, cam.dist, rvec.copy(), tvec.copy())
    T = RigidTransform(cv2.Rodrigues(rvec)[0], tvec.ravel())
    if not _faces_camera(T, np.zeros(3)):
        raise ValueError(MIRRORED_CLICKS_MSG)
    if not return_ambiguity:
        return T
    e = sorted(s[2] for s in sols if np.isfinite(s[2]))
    ambiguous = len(e) == 2 and e[1] <= 2 * max(e[0], 0.5)
    return T, ambiguous
