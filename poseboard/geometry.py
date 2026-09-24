"""Wii Balance Board 的几何模型、在相机中的定位（点选四角+中心）以及三角化工具。

Balance Board 坐标系（单位：米）
--------------------------------
俯视平衡板，站在板上面朝 "前方"（TL/TR 一侧）：

        TL ─────────── TR          +Y (前, "top")
        │               │           ↑
        │       C       │           └──→ +X (右)
        │               │
        BL ─────────── BR

* 原点 C 在板上表面中心，Z 轴朝上。
* TL/TR/BR/BL 同时指四个传感器（Wii 数据里的顺序）和与之最近的板面外角。
* 标注顺序固定为：TL, TR, BR, BL, C。

默认尺寸（官方 Wii Balance Board）：板面外形约 511 x 316 mm，
传感器中心间距 433 mm（左右） x 238 mm（前后）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from poseboard.calibration import CameraCalibration

LANDMARK_NAMES = ("TL", "TR", "BR", "BL", "C")
SENSOR_NAMES = ("TR", "BR", "TL", "BL")  # 与 Wii 数据顺序一致


@dataclass
class BoardGeometry:
    length_mm: float = 511.0  # 板面外形 X 方向（左右）
    width_mm: float = 316.0  # 板面外形 Y 方向（前后）
    sensor_dx_mm: float = 433.0  # 左右传感器中心间距
    sensor_dy_mm: float = 238.0  # 前后传感器中心间距

    def landmarks(self) -> np.ndarray:
        """标注点 TL, TR, BR, BL, C 在板坐标系中的坐标 (5,3)，米。"""
        hx, hy = self.length_mm / 2000.0, self.width_mm / 2000.0
        return np.array([[-hx, hy, 0], [hx, hy, 0], [hx, -hy, 0], [-hx, -hy, 0], [0, 0, 0]],
                        dtype=np.float64)

    def sensors(self) -> np.ndarray:
        """四个传感器位置 (TR, BR, TL, BL) (4,3)，米。"""
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
        """板面法向（板坐标 +Z）在世界坐标中的方向。"""
        return self.board_to_world.R[:, 2]

    def to_dict(self) -> dict:
        return {"board_to_world": self.board_to_world.to_dict(), "method": self.method,
                "reproj_error_px": self.reproj_error_px}

    @classmethod
    def from_dict(cls, d: dict) -> "BoardPose":
        return cls(RigidTransform.from_dict(d["board_to_world"]), d["method"],
                   d.get("reproj_error_px", {}))


def rotate_board_frame(pose: BoardPose, degrees: int) -> BoardPose:
    """绕板法向旋转板坐标系（标注方向搞反时用，例如 180°）。"""
    a = np.deg2rad(degrees)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    T = pose.board_to_world
    return BoardPose(RigidTransform(T.R @ Rz, T.t.copy()), pose.method, pose.reproj_error_px)


# ------------------------------------------------------------------ utilities
def kabsch(src: np.ndarray, dst: np.ndarray) -> RigidTransform:
    """求刚体变换使 dst ≈ R @ src + t（最小二乘）。"""
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
    """多视角 DLT 三角化单个点。pts_px 为每台相机中的像素坐标。"""
    if len(cams) < 2:
        raise ValueError("三角化至少需要 2 台相机")
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
    """根据在各相机图像中点选的 5 个标注点（TL, TR, BR, BL, C）求平衡板在世界坐标系中的位姿。

    * 若有 ≥2 台已标定外参的相机都完成了标注：先三角化 5 个点，再用 Kabsch 与板模型对齐。
    * 否则：用唯一一台相机的点做 PnP（平面 IPPE + 迭代细化），再经相机外参变换到世界坐标系。
      若该相机没有外参，则世界坐标系即为该相机坐标系。
    """
    model = geometry.landmarks()
    usable = [(c, np.asarray(k, np.float64).reshape(5, 2)) for c, k in zip(cams, clicks)
              if k is not None and len(k) == 5]
    if not usable:
        raise ValueError("没有完成 5 点标注的相机")

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
            else:  # 世界坐标系 = 该相机坐标系
                img, _ = cv2.projectPoints(pts_w, np.zeros(3), np.zeros(3), c.K, c.dist)
                errors[c.name] = float(np.linalg.norm(img.reshape(-1, 2) - k, axis=1).mean())
    return BoardPose(T, method, errors)


def solve_board_pnp(cam: CameraCalibration, model: np.ndarray, img_px: np.ndarray) -> RigidTransform:
    """板坐标 -> 相机坐标。"""
    obj = np.asarray(model, np.float64).reshape(-1, 1, 3)
    img = np.asarray(img_px, np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("平衡板 PnP 求解失败，请检查点选顺序")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, cam.K, cam.dist, rvec, tvec)
    return RigidTransform(cv2.Rodrigues(rvec)[0], tvec.ravel())
