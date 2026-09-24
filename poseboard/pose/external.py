"""Using external 3D pose sources:

1. Plugin: a Python file that defines ``create_estimator(**kwargs) -> PoseEstimator``.
   Select the file in the GUI to use it for live acquisition (e.g. wrap your PoseAssess
   as a plugin). See ``plugins/poseassess_plugin_template.py``.
2. Offline import: read a 3D keypoint file in TRC (Pose2Sim / OpenSim format) or CSV,
   then align it in time with the Wii data recorded by PoseBoard (see ``poseboard.analysis``).
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
        raise ImportError(f"Cannot load plugin {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "create_estimator"):
        raise ImportError("Plugin must define create_estimator(**kwargs) -> PoseEstimator")
    est = mod.create_estimator(**kwargs)
    if not isinstance(est, PoseEstimator):
        raise TypeError("create_estimator must return a PoseEstimator instance")
    return est


def read_trc(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    """Read a TRC file; returns (times (N,), keypoint names, coordinates (N,K,3) in meters, header)."""
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
    """Read a CSV: first column is time (header t/time), then <name>_x, <name>_y, <name>_z (meters)."""
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
