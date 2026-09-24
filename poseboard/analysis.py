"""录制后处理：时间对齐、融合表、静态平衡指标，以及导入外部 3D 姿态（如 PoseAssess 输出）。

命令行::

    python -m poseboard.analysis recordings/20260924_153000
    python -m poseboard.analysis recordings/20260924_153000 --external pose.trc --offset 0.0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from poseboard.geometry import BoardPose
from poseboard.pose.com import center_of_mass
from poseboard.pose.external import read_keypoint_csv, read_trc

FORCE_COLS = ["TR_kg", "BR_kg", "TL_kg", "BL_kg", "total_kg", "cop_x_board", "cop_y_board",
              "cop_x_world", "cop_y_world", "cop_z_world"]


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    out = {}
    for i, h in enumerate(head):
        out[h] = np.array([float(r[i]) if i < len(r) and r[i] != "" else np.nan for r in body])
    return out


def interp(t_src: np.ndarray, v: np.ndarray, t_dst: np.ndarray, max_gap: float = 0.1) -> np.ndarray:
    """线性插值；目标时间离最近源样本超过 max_gap 秒或源值为 NaN 时返回 NaN。"""
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.full(len(t_dst), np.nan)
    ts, vs = t_src[ok], v[ok]
    out = np.interp(t_dst, ts, vs, left=np.nan, right=np.nan)
    j = np.clip(np.searchsorted(ts, t_dst), 1, len(ts) - 1)
    gap = np.minimum(np.abs(t_dst - ts[j - 1]), np.abs(ts[j] - t_dst))
    out[gap > max_gap] = np.nan
    return out


# --------------------------------------------------------------------- metrics
def sway_metrics(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """常用静态平衡 COP 指标（输入单位米，输出 mm / mm²）。"""
    ok = np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok] * 1000, y[ok] * 1000
    if len(t) < 10:
        return {}
    dur = float(t[-1] - t[0])
    path = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    xc, yc = x - x.mean(), y - y.mean()
    cov = np.cov(np.vstack([xc, yc]))
    eig = np.linalg.eigvalsh(cov)
    # 95% 置信椭圆面积：pi * chi2(0.95, 2) * sqrt(λ1 λ2)
    area95 = float(np.pi * 5.991 * np.sqrt(max(eig[0], 0) * max(eig[1], 0)))
    return {
        "duration_s": dur,
        "samples": int(len(t)),
        "mean_x_mm": float(x.mean()), "mean_y_mm": float(y.mean()),
        "range_ml_mm": float(np.ptp(x)), "range_ap_mm": float(np.ptp(y)),
        "rms_ml_mm": float(np.sqrt(np.mean(xc ** 2))), "rms_ap_mm": float(np.sqrt(np.mean(yc ** 2))),
        "path_length_mm": path,
        "mean_velocity_mm_s": path / dur if dur > 0 else float("nan"),
        "ellipse95_area_mm2": area95,
    }


# ----------------------------------------------------------------------- fuse
def _write_fused(folder: Path, t: np.ndarray, t0: float, force: dict[str, np.ndarray],
                 com: np.ndarray, com_b: np.ndarray, name: str = "fused.csv") -> Path:
    path = folder / name
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "t_rel", *FORCE_COLS, "com_x", "com_y", "com_z",
                    "com_x_board", "com_y_board", "com_z_board",
                    "com_minus_cop_x", "com_minus_cop_y"])
        for i in range(len(t)):
            fv = [force[c][i] for c in FORCE_COLS]
            dx = com_b[i, 0] - force["cop_x_board"][i]
            dy = com_b[i, 1] - force["cop_y_board"][i]
            vals = [t[i], t[i] - t0, *fv, *com[i], *com_b[i], dx, dy]
            w.writerow(["" if not np.isfinite(v) else f"{v:.6f}" for v in vals])
    return path


def analyze_session(folder: str | Path) -> dict:
    folder = Path(folder)
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    t0 = meta.get("t0", 0.0)
    summary: dict = {"session": folder.name}

    wii = read_csv_columns(folder / "wii.csv") if (folder / "wii.csv").exists() else None
    if wii is not None and len(wii["t"]):
        summary["mean_total_kg"] = float(np.nanmean(wii["total_kg"]))
        summary["wii_rate_hz"] = float((len(wii["t"]) - 1) / max(np.ptp(wii["t"]), 1e-9))
        summary["cop"] = sway_metrics(wii["t"], wii["cop_x_board"], wii["cop_y_board"])

    pose_path = folder / "pose3d.csv"
    if wii is not None and pose_path.exists():
        pose = read_csv_columns(pose_path)
        t = pose["t"]
        if len(t):
            force = {c: interp(wii["t"], wii[c], t) for c in FORCE_COLS}
            com = np.column_stack([pose["com_x"], pose["com_y"], pose["com_z"]])
            com_b = np.column_stack([pose["com_x_board"], pose["com_y_board"], pose["com_z_board"]])
            _write_fused(folder, t, t0, force, com, com_b)
            summary["pose_rate_hz"] = float((len(t) - 1) / max(np.ptp(t), 1e-9))
            summary["com"] = sway_metrics(t, com_b[:, 0], com_b[:, 1])
            d = np.hypot(com_b[:, 0] - force["cop_x_board"], com_b[:, 1] - force["cop_y_board"])
            if np.isfinite(d).any():
                summary["com_cop_distance_rms_mm"] = float(np.sqrt(np.nanmean(d ** 2)) * 1000)

    (folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    return summary


# Pose2Sim 写 TRC 时把 Z-up 转成了 Y-up：(X', Y', Z') = (Y, Z, X)。其逆变换：
POSE2SIM_YUP_TO_ZUP = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)


def fuse_external(folder: str | Path, pose_file: str | Path, offset_s: float = 0.0,
                  world_transform: np.ndarray | None = None) -> Path:
    """把外部 3D 姿态（TRC 或 CSV）与本次录制的 Wii 数据对齐并融合。

    外部数据的时间 0 对应录制开始 (t0) + offset_s（外部文件的时间列需从录制开始计时）。外部姿态须在同一棋盘格世界坐标系中；
    若不是，可传入 4x4 ``world_transform``（外部坐标 -> PoseBoard 世界坐标）。
    """
    folder = Path(folder)
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    t0 = meta["t0"]
    board = BoardPose.from_dict(meta["board_pose"]) if meta.get("board_pose") else None
    pose_file = Path(pose_file)
    if pose_file.suffix.lower() == ".trc":
        t_ext, names, kps, _ = read_trc(pose_file)
    else:
        t_ext, names, kps = read_keypoint_csv(pose_file)
    if world_transform is not None:
        T = np.asarray(world_transform, np.float64)
        kps = kps @ T[:3, :3].T + T[:3, 3]
    t = t0 + offset_s + t_ext
    com = np.array([c if (c := center_of_mass(k, names)) is not None else np.full(3, np.nan)
                    for k in kps])
    com_b = board.world_to_board.apply(com) if board is not None else np.full_like(com, np.nan)
    wii = read_csv_columns(folder / "wii.csv")
    force = {c: interp(wii["t"], wii[c], t) for c in FORCE_COLS}
    return _write_fused(folder, t, t0, force, com, com_b, name=f"fused_{pose_file.stem}.csv")


def export_trc(folder: str | Path) -> Path:
    """把 pose3d.csv 导出为 TRC（毫米），方便在 OpenSim / Pose2Sim 中使用。"""
    folder = Path(folder)
    pose = read_csv_columns(folder / "pose3d.csv")
    names = [k[:-2] for k in pose if k.endswith("_x") and not k.startswith("com")]
    t = pose["t_rel"]
    rate = (len(t) - 1) / max(np.ptp(t), 1e-9) if len(t) > 1 else 30.0
    path = folder / "pose3d.trc"
    with open(path, "w", newline="") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{path.name}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{rate:.2f}\t{rate:.2f}\t{len(t)}\t{len(names)}\tmm\t{rate:.2f}\t1\t{len(t)}\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(names) + "\n")
        f.write("\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, len(names) + 1)) + "\n")
        for i in range(len(t)):
            vals = []
            for n in names:
                for a in "xyz":
                    v = pose[f"{n}_{a}"][i] * 1000
                    vals.append("" if not np.isfinite(v) else f"{v:.3f}")
            f.write(f"{i + 1}\t{t[i]:.6f}\t" + "\t".join(vals) + "\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="PoseBoard 录制后处理")
    ap.add_argument("folder")
    ap.add_argument("--external", help="外部 3D 姿态文件（.trc 或 .csv）")
    ap.add_argument("--offset", type=float, default=0.0, help="外部数据相对录制开始的时间偏移（秒）")
    ap.add_argument("--pose2sim-yup", action="store_true",
                    help="外部 TRC 是 Pose2Sim 导出的 Y-up 坐标，转换回棋盘格 Z-up 坐标")
    ap.add_argument("--trc", action="store_true", help="导出 pose3d.trc")
    a = ap.parse_args(argv)
    print(json.dumps(analyze_session(a.folder), indent=2, ensure_ascii=False))
    if a.external:
        T = POSE2SIM_YUP_TO_ZUP if a.pose2sim_yup else None
        print("写入", fuse_external(a.folder, a.external, a.offset, T))
    if a.trc:
        print("写入", export_trc(a.folder))


if __name__ == "__main__":
    main()
