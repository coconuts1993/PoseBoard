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
import re
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


def _num(v: str) -> float:
    v = v.strip()
    if v == "" or v.lower() == "nan":
        return np.nan
    return float(v)


def read_trc(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    """Read a TRC file; returns (times (N,), keypoint names, coordinates (N,K,3) in meters, header).

    Data rows are split on tabs without stripping them, so empty fields of missing markers
    (NaN, as written by pandas / Pose2Sim or ``export_trc``) keep every marker in its column,
    also when the last markers of a row are missing. Only rows without any tab are split on
    whitespace.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    header_keys = lines[1].split("\t")
    header_vals = lines[2].split("\t")
    header = dict(zip(header_keys, header_vals))
    names = [n.strip() for n in lines[3].split("\t")[2:] if n.strip()]
    units = header.get("Units", "m").strip().lower()
    scale = {"mm": 1e-3, "cm": 1e-2}.get(units, 1.0)
    n_vals = 3 * len(names)
    times, data = [], []
    for line in lines[5:]:
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2 or parts[1].strip() == "":
            continue
        times.append(float(parts[1]))
        vals = [_num(v) for v in parts[2:2 + n_vals]]
        vals += [np.nan] * (n_vals - len(vals))
        data.append(vals)
    arr = np.asarray(data, np.float64).reshape(len(data), len(names), 3) * scale
    return np.asarray(times), names, arr, header


# Accepted names of the time column of a keypoint CSV (compared case-insensitively, ignoring
# spaces and a unit in brackets, e.g. "Time (s)"). "t_unix" is used when none of these exists.
TIME_COLUMNS = ("t", "time", "timestamp", "time_s", "t_s", "t_rel")


def _norm_header(h: str) -> str:
    h = re.sub(r"[\(\[][^\)\]]*[\)\]]", "", h.strip().lower())  # drop "(s)" / "[s]"
    return re.sub(r"\s+", "", h).strip("_")


def csv_time_column(head: list[str]) -> str | None:
    """Name of the time column in a keypoint CSV header: one of ``TIME_COLUMNS``, otherwise
    ``t_unix``; None if there is none."""
    norm = [_norm_header(h) for h in head]
    for cand in TIME_COLUMNS + ("t_unix",):
        if cand in norm:
            return head[norm.index(cand)]
    return None


def read_keypoint_csv(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Read a CSV: a time column (``t``, ``time``, ``timestamp``, ``time_s``, ``Time (s)`` ...;
    ``t_unix`` if there is no other) and <name>_x, <name>_y, <name>_z columns (meters)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path}: empty file")
    head = [h.strip() for h in rows[0]]
    tname = csv_time_column(head)
    if tname is None:
        raise ValueError(f"{path}: no time column; name it one of "
                         f"{', '.join(TIME_COLUMNS + ('t_unix',))} (seconds)")
    tcol = head.index(tname)
    names = []
    for h in head:
        if h.endswith("_x") and h[:-2] + "_y" in head and h[:-2] + "_z" in head:
            names.append(h[:-2])
    cols = [[head.index(f"{n}_{a}") for a in "xyz"] for n in names]
    body = [r for r in rows[1:] if any(v.strip() for v in r)]
    t = np.array([_num(r[tcol]) for r in body])
    arr = np.array([[[_num(r[c]) if c < len(r) else np.nan for c in cc] for cc in cols]
                    for r in body]).reshape(len(body), len(names), 3)
    return t, names, arr
