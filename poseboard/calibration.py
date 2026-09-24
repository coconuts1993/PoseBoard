"""Camera calibration: checkerboard intrinsics, checkerboard-defined world frame
(extrinsics), and reading/writing calibration files.

Conventions
-----------
* World coordinates are in meters.
* Extrinsics are world -> camera: X_cam = R @ X_world + t  (rvec is a Rodrigues vector).
* Checkerboard world frame: the origin is the inner corner at one corner of the board
  (the one whose diagonally outward square is black), X runs along the board's "cols"
  direction, Y along its "rows" direction, and Z = X × Y points toward the camera
  (Z is up when the board lies flat on the floor). See ``canonical_corners``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    name: str
    image_size: tuple[int, int]  # (width, height)
    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray | None = None  # world -> camera
    tvec: np.ndarray | None = None
    intrinsic_rms: float | None = None
    extrinsic_rms: float | None = None

    # ------------------------------------------------------------------ props
    @property
    def has_extrinsics(self) -> bool:
        return self.rvec is not None and self.tvec is not None

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(np.asarray(self.rvec, dtype=np.float64))[0]

    @property
    def t(self) -> np.ndarray:
        return np.asarray(self.tvec, dtype=np.float64).reshape(3)

    @property
    def center_world(self) -> np.ndarray:
        """Camera optical center in world coordinates."""
        return -self.R.T @ self.t

    def projection_matrix(self, normalized: bool = False) -> np.ndarray:
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return Rt if normalized else self.K @ Rt

    # ------------------------------------------------------------ transforms
    def world_to_camera(self, pts_w: np.ndarray) -> np.ndarray:
        pts_w = np.asarray(pts_w, dtype=np.float64).reshape(-1, 3)
        return pts_w @ self.R.T + self.t

    def project(self, pts_w: np.ndarray) -> np.ndarray:
        """Project world points to image pixel coordinates (N,2)."""
        pts_w = np.asarray(pts_w, dtype=np.float64).reshape(-1, 3)
        if len(pts_w) == 0:
            return np.zeros((0, 2))
        img, _ = cv2.projectPoints(pts_w, np.asarray(self.rvec, np.float64),
                                   np.asarray(self.tvec, np.float64), self.K, self.dist)
        return img.reshape(-1, 2)

    def undistort_normalized(self, pts_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)

    # ---------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "image_size": list(self.image_size),
            "K": np.asarray(self.K).tolist(),
            "dist": np.asarray(self.dist).ravel().tolist(),
            "rvec": None if self.rvec is None else np.asarray(self.rvec).ravel().tolist(),
            "tvec": None if self.tvec is None else np.asarray(self.tvec).ravel().tolist(),
            "intrinsic_rms": self.intrinsic_rms,
            "extrinsic_rms": self.extrinsic_rms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraCalibration":
        return cls(
            name=d["name"],
            image_size=tuple(d["image_size"]),
            K=np.asarray(d["K"], dtype=np.float64),
            dist=np.asarray(d["dist"], dtype=np.float64),
            rvec=None if d.get("rvec") is None else np.asarray(d["rvec"], dtype=np.float64),
            tvec=None if d.get("tvec") is None else np.asarray(d["tvec"], dtype=np.float64),
            intrinsic_rms=d.get("intrinsic_rms"),
            extrinsic_rms=d.get("extrinsic_rms"),
        )


def approximate_calibration(name: str, width: int, height: int,
                            hfov_deg: float = 60.0) -> CameraCalibration:
    """Approximate intrinsics for an uncalibrated camera (pinhole, no distortion).
    For quick trials only; accuracy is limited."""
    f = (width / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], np.float64)
    return CameraCalibration(name, (width, height), K, np.zeros(5))


# ---------------------------------------------------------------- checkerboard
@dataclass
class CheckerboardSpec:
    cols: int = 9  # number of inner corners (horizontal)
    rows: int = 6  # number of inner corners (vertical)
    square_mm: float = 25.0

    @property
    def pattern_size(self) -> tuple[int, int]:
        return (self.cols, self.rows)

    def object_points(self) -> np.ndarray:
        """3D coordinates (meters) of the inner corners in the checkerboard frame."""
        grid = np.zeros((self.rows * self.cols, 3), np.float64)
        grid[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        return grid * (self.square_mm / 1000.0)


def find_checkerboard(image: np.ndarray, spec: CheckerboardSpec,
                      fast: bool = False) -> np.ndarray | None:
    """Detect checkerboard inner corners (N,2). fast=True is for live preview
    (returns quickly when no board is found)."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, spec.pattern_size, flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2)


@dataclass
class IntrinsicCollector:
    """Accumulates checkerboard corners from multiple images for intrinsic calibration."""

    spec: CheckerboardSpec
    image_size: tuple[int, int] | None = None
    corners: list[np.ndarray] = field(default_factory=list)

    def add(self, image: np.ndarray) -> np.ndarray | None:
        c = find_checkerboard(image, self.spec)
        if c is not None:
            self.corners.append(c)
            self.image_size = (image.shape[1], image.shape[0])
        return c

    def __len__(self) -> int:
        return len(self.corners)

    def calibrate(self, name: str) -> CameraCalibration:
        if len(self.corners) < 3:
            raise ValueError("Need at least 3 images with a detected checkerboard (15+ recommended)")
        obj = self.spec.object_points().astype(np.float32)
        objpoints = [obj] * len(self.corners)
        imgpoints = [c.astype(np.float32).reshape(-1, 1, 2) for c in self.corners]
        rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, self.image_size, None, None)
        return CameraCalibration(name=name, image_size=self.image_size, K=K,
                                 dist=dist.ravel(), intrinsic_rms=float(rms))


def canonical_corners(image: np.ndarray, corners: np.ndarray, spec: CheckerboardSpec,
                      cam: "CameraCalibration") -> np.ndarray:
    """Normalize the checkerboard corner order so all cameras share the same world frame.

    OpenCV may start the corner list at any of the four corners. Of the 4 possible
    orderings, this picks the one where:
    (1) the camera is on the +Z side of the board (Z points toward the camera, i.e.
        Z is up when the board lies flat on the floor);
    (2) the square diagonally outward from the origin is black.
    Requires a board whose inner-corner "cols + rows" is odd (e.g. 9x6); otherwise
    the 180° symmetry cannot be resolved.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    grid = np.asarray(corners, np.float64).reshape(spec.rows, spec.cols, 2)
    obj = spec.object_points()
    sq = spec.square_mm / 1000.0
    probes = np.array([[-0.5 * sq, -0.5 * sq, 0], [0.5 * sq, -0.5 * sq, 0]])
    best, best_score = None, -np.inf
    for cand in (grid, grid[::-1], grid[:, ::-1], grid[::-1, ::-1]):
        pts = np.ascontiguousarray(cand.reshape(-1, 2))
        ok, rvec, tvec = cv2.solvePnP(obj, pts, cam.K, cam.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        R = cv2.Rodrigues(rvec)[0]
        if (-R.T @ tvec.ravel())[2] <= 0:  # camera on the -Z side (mirrored ordering)
            continue
        img_pts, _ = cv2.projectPoints(probes, rvec, tvec, cam.K, cam.dist)
        vals = []
        for q in img_pts.reshape(-1, 2):
            if not (2 <= q[0] < gray.shape[1] - 2 and 2 <= q[1] < gray.shape[0] - 2):
                vals.append(np.nan)
                continue
            vals.append(float(cv2.getRectSubPix(gray, (5, 5), (float(q[0]), float(q[1]))).mean()))
        score = vals[1] - vals[0]  # darker outer square at origin and whiter neighbor is better
        if not np.isfinite(score):
            score = -1e6
        if score > best_score:
            best, best_score = pts, score
    return np.asarray(corners).reshape(-1, 2) if best is None else best


def extrinsics_from_checkerboard(cam: CameraCalibration, image: np.ndarray,
                                 spec: CheckerboardSpec) -> float:
    """Detect the checkerboard and use it as the world frame; writes cam.rvec/cam.tvec.
    Returns the reprojection error (pixels).

    Corner order is normalized by ``canonical_corners``: Z points toward the camera
    (up when the board lies flat on the floor), and the origin is the inner corner whose
    diagonally outward square is black. Multiple cameras should all see the board in the
    same placement.
    """
    corners = find_checkerboard(image, spec)
    if corners is None:
        raise RuntimeError("Checkerboard not detected")
    corners = canonical_corners(image, corners, spec, cam)
    return extrinsics_from_points(cam, spec.object_points(), corners)


def extrinsics_from_points(cam: CameraCalibration, obj_w: np.ndarray, img_px: np.ndarray) -> float:
    obj_w = np.asarray(obj_w, np.float64).reshape(-1, 3)
    img_px = np.asarray(img_px, np.float64).reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj_w, img_px, cam.K, cam.dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    cam.rvec, cam.tvec = rvec.ravel(), tvec.ravel()
    err = np.linalg.norm(cam.project(obj_w) - img_px, axis=1).mean()
    cam.extrinsic_rms = float(err)
    return float(err)


# ------------------------------------------------------------------- file I/O
def save_calibrations(path: str | Path, cams: list[CameraCalibration]) -> None:
    Path(path).write_text(json.dumps({"cameras": [c.to_dict() for c in cams]}, indent=2),
                          encoding="utf-8")


def load_calibrations(path: str | Path) -> list[CameraCalibration]:
    path = Path(path)
    if path.suffix.lower() == ".toml":
        return load_pose2sim_toml(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [CameraCalibration.from_dict(d) for d in data["cameras"]]


def load_pose2sim_toml(path: str | Path) -> list[CameraCalibration]:
    """Read a Pose2Sim-format Calib.toml (matrix / distortions / rotation / translation).

    Note: Pose2Sim translations are in meters and rotations are Rodrigues vectors
    (world -> camera), consistent with this project.
    """
    try:
        import tomllib  # py>=3.11
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    cams = []
    for key, d in data.items():
        if not isinstance(d, dict) or "matrix" not in d:
            continue
        dist = np.zeros(5)
        src = np.asarray(d.get("distortions", [0, 0, 0, 0]), np.float64)
        dist[: len(src)] = src
        cams.append(CameraCalibration(
            name=d.get("name", key),
            image_size=tuple(int(v) for v in d.get("size", [0, 0])),
            K=np.asarray(d["matrix"], np.float64),
            dist=dist,
            rvec=np.asarray(d["rotation"], np.float64) if "rotation" in d else None,
            tvec=np.asarray(d["translation"], np.float64) if "translation" in d else None,
        ))
    return cams


def save_pose2sim_toml(path: str | Path, cams: list[CameraCalibration]) -> None:
    def arr(a):
        return "[ " + ", ".join(repr(float(x)) for x in np.ravel(a)) + ",]"

    lines = []
    for i, c in enumerate(cams, 1):
        lines += [
            f"[cam_{i:02d}]",
            f'name = "{c.name}"',
            f"size = [ {float(c.image_size[0])}, {float(c.image_size[1])},]",
            "matrix = [ " + ", ".join(arr(r) for r in np.asarray(c.K)) + ",]",
            f"distortions = {arr(np.ravel(c.dist)[:4])}",
            f"rotation = {arr(c.rvec if c.rvec is not None else np.zeros(3))}",
            f"translation = {arr(c.tvec if c.tvec is not None else np.zeros(3))}",
            "fisheye = false",
            "",
        ]
    lines += ["[metadata]", "adjusted = false", "error = 0.0", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
