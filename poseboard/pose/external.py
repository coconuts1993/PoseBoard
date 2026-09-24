"""接入外部 3D 姿态：

1. 插件：一个 Python 文件里定义 ``create_estimator(**kwargs) -> PoseEstimator``，
   在界面里选择该文件即可在实时采集中使用（例如把你的 PoseAssess 包装成插件）。
   参见 ``plugins/poseassess_plugin_template.py``。
2. 离线导入：读取 TRC（Pose2Sim / OpenSim 格式）或 CSV 的 3D 关键点文件，
   再与 PoseBoard 录制的 Wii 数据按时间对齐（见 ``poseboard.analysis``）。
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np

from poseboard.pose.base import PoseEstimator


def load_plugin(path: str | Path, **kwargs) -> PoseEstimator:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"poseboard_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载插件 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "create_estimator"):
        raise ImportError("插件必须定义 create_estimator(**kwargs) -> PoseEstimator")
    est = mod.create_estimator(**kwargs)
    if not isinstance(est, PoseEstimator):
        raise TypeError("create_estimator 必须返回 PoseEstimator 实例")
    return est


def read_trc(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    """读取 TRC 文件，返回 (时间 (N,), 关键点名, 坐标 (N,K,3) 米, 头信息)。"""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    header_keys = lines[1].split("\t")
    header_vals = lines[2].split("\t")
    header = dict(zip(header_keys, header_vals))
    names = [n for n in lines[3].split("\t")[2:] if n.strip()]
    units = header.get("Units", "m").strip().lower()
    scale = {"mm": 1e-3, "cm": 1e-2}.get(units, 1.0)
    times, data = [], []
    for line in lines[5:]:
        parts = line.strip().split("\t")
        if len(parts) < 2 + 3 * len(names):
            parts = line.split()
        if len(parts) < 2:
            continue
        times.append(float(parts[1]))
        vals = [float(v) if v not in ("", "nan", "NaN") else np.nan for v in parts[2:2 + 3 * len(names)]]
        vals += [np.nan] * (3 * len(names) - len(vals))
        data.append(vals)
    arr = np.asarray(data, np.float64).reshape(len(data), len(names), 3) * scale
    return np.asarray(times), names, arr, header


def read_keypoint_csv(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """读取 CSV：第一列时间（列名 t/time），之后为 <name>_x, <name>_y, <name>_z 列（米）。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    tcol = next(i for i, h in enumerate(head) if h.strip().lower() in ("t", "time", "timestamp"))
    names = []
    for h in head:
        if h.endswith("_x") and h[:-2] + "_y" in head and h[:-2] + "_z" in head:
            names.append(h[:-2])
    cols = [[head.index(f"{n}_{a}") for a in "xyz"] for n in names]
    t = np.array([float(r[tcol]) for r in rows[1:]])
    arr = np.array([[[float(r[c]) if r[c] else np.nan for c in cc] for cc in cols] for r in rows[1:]])
    return t, names, arr
