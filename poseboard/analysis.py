"""Post-processing of recordings: time alignment, fused table, static balance metrics,
force events for synchronization (step-on/off, jumps, stomps) and import of external 3D pose
(e.g. PoseAssess output).

Command line::

    python -m poseboard.analysis recordings/20260924_153000
    python -m poseboard.analysis recordings/20260924_153000 --events
    python -m poseboard.analysis recordings/20260924_153000 --external pose.trc --offset 0.0
    python -m poseboard.analysis recordings/20260924_153000 --external other.csv --time-base unix
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from poseboard.geometry import BoardPose
from poseboard.pose.com import center_of_mass
from poseboard.pose.external import csv_time_column, read_keypoint_csv, read_trc

FORCE_COLS = ["TR_kg", "BR_kg", "TL_kg", "BL_kg", "total_kg", "cop_x_board", "cop_y_board",
              "cop_x_world", "cop_y_world", "cop_z_world"]
EVENT_COLS = ["type", "t", "t_rel", "t_unix", "flight_s", "peak_kg"]


def _float(v: str) -> float:
    try:
        return float(v) if v != "" else np.nan
    except ValueError:  # a text column, e.g. "mode" in pose3d.csv
        return np.nan


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    """All columns of a CSV as float arrays (empty or non-numeric values are NaN)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    out = {}
    for i, h in enumerate(head):
        out[h] = np.array([_float(r[i]) if i < len(r) else np.nan for r in body])
    return out


def read_csv_column(path: Path, name: str) -> np.ndarray | None:
    """One numeric column of a CSV file by header name (None if the column does not exist)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    head = [h.strip() for h in rows[0]] if rows else []
    if name not in head:
        return None
    i = head.index(name)
    return np.array([float(r[i]) if i < len(r) and r[i].strip() != "" else np.nan for r in rows[1:]])


def read_events(folder: str | Path, name: str = "events.csv") -> list[dict]:
    """Event markers written during recording (``SessionRecorder.add_event``): t, t_rel, t_unix, label."""
    path = Path(folder) / name
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:  # with or without a BOM
        rows = list(csv.DictReader(f))

    def num(v: str | None) -> float:
        return float(v) if v not in (None, "") else float("nan")

    return [{"t": num(r.get("t")), "t_rel": num(r.get("t_rel")), "t_unix": num(r.get("t_unix")),
             "label": r.get("label", "")} for r in rows]


def interp(t_src: np.ndarray, v: np.ndarray, t_dst: np.ndarray, max_gap: float = 0.1) -> np.ndarray:
    """Linear interpolation; returns NaN where the target time is more than max_gap seconds
    from the nearest source sample, or where the source value is NaN."""
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
    """Common static balance COP metrics (input in meters, output in mm / mm²)."""
    ok = np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok] * 1000, y[ok] * 1000
    if len(t) < 10:
        return {}
    dur = float(t[-1] - t[0])
    path = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    xc, yc = x - x.mean(), y - y.mean()
    cov = np.cov(np.vstack([xc, yc]))
    eig = np.linalg.eigvalsh(cov)
    # 95% confidence ellipse area: pi * chi2(0.95, 2) * sqrt(λ1 λ2)
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
                 com: np.ndarray, com_b: np.ndarray, name: str = "fused.csv",
                 clock_offset_unix: float | None = None) -> Path:
    path = folder / name
    off = np.nan if clock_offset_unix is None else clock_offset_unix
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "t_rel", "t_unix", *FORCE_COLS, "com_x", "com_y", "com_z",
                    "com_x_board", "com_y_board", "com_z_board",
                    "com_minus_cop_x", "com_minus_cop_y"])
        for i in range(len(t)):
            fv = [force[c][i] for c in FORCE_COLS]
            dx = com_b[i, 0] - force["cop_x_board"][i]
            dy = com_b[i, 1] - force["cop_y_board"][i]
            vals = [t[i], t[i] - t0, t[i] + off, *fv, *com[i], *com_b[i], dx, dy]
            w.writerow(["" if not np.isfinite(v) else f"{v:.6f}" for v in vals])
    return path


# ---------------------------------------------------------------- force events
def _load_force(data, t0: float | None, clock_offset_unix: float | None):
    """(t, total_kg, t0, clock_offset_unix) from a session folder, a wii.csv path, a dict of
    columns or a (t, total_kg) pair."""
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.is_dir():
            path = path / "wii.csv"
        cols = read_csv_columns(path)
        meta_path = path.parent / "session.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            t0 = meta.get("t0") if t0 is None else t0
            if clock_offset_unix is None:
                clock_offset_unix = meta.get("clock_offset_unix")
    elif isinstance(data, dict):
        cols = data
    else:
        t, total = data
        cols = {"t": t, "total_kg": total}
    t = np.asarray(cols["t"], np.float64)
    total = np.asarray(cols["total_kg"], np.float64)
    if t0 is None and "t_rel" in cols and np.isfinite(cols["t_rel"]).any():
        t0 = float(np.nanmedian(t - np.asarray(cols["t_rel"], np.float64)))
    if clock_offset_unix is None and "t_unix" in cols and np.isfinite(cols["t_unix"]).any():
        clock_offset_unix = float(np.nanmedian(np.asarray(cols["t_unix"], np.float64) - t))
    if t0 is None:
        t0 = float(t[0]) if len(t) else 0.0
    return t, total, t0, clock_offset_unix


def detect_force_events(data, *, t0: float | None = None, clock_offset_unix: float | None = None,
                        body_kg: float | None = None, min_load_kg: float = 10.0,
                        on_fraction: float = 0.5, hysteresis: float = 0.1, flight_kg: float = 5.0,
                        min_flight_s: float = 0.05, max_flight_s: float = 1.0,
                        peak_factor: float = 1.5, jump_guard_s: float = 0.5,
                        max_gap_s: float = 0.1) -> list[dict]:
    """Find events in the total force that are easy to see in other systems too, for time alignment.

    ``data`` is a session folder, a wii.csv path, a dict with ``t`` and ``total_kg`` (optionally
    ``t_rel``/``t_unix``) or a ``(t, total_kg)`` pair. Event types:

    * ``step_on`` / ``step_off``: total crosses ``on_fraction`` x body weight (hysteresis
      +-``hysteresis`` x body weight); time of that crossing, interpolated.
    * ``takeoff`` / ``landing``: an unloaded phase (total < ``flight_kg``) of at most
      ``max_flight_s`` between two standing phases, i.e. a jump; times where the total crosses
      ``flight_kg``. Both carry ``flight_s``; ``landing`` also carries the impact ``peak_kg``
      when it exceeds ``peak_factor`` x body weight.
    * ``stomp``: a sharp peak above ``peak_factor`` x body weight that is not a jump push-off
      or landing; time of the maximum (parabolic interpolation), with ``peak_kg``.

    Body weight is the median load while somebody is on the board (> ``min_load_kg``) unless
    given. Crossings next to a gap of more than ``max_gap_s`` in the data are skipped. Returns
    a list of dicts sorted by time: ``type``, ``t`` (perf_counter), ``t_rel``, ``t_unix``
    (None if unknown) and, where applicable, ``flight_s`` / ``peak_kg``.
    """
    t, f, t0, off = _load_force(data, t0, clock_offset_unix)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if len(t) < 3:
        return []
    if body_kg is None:
        loaded = f[f > min_load_kg]
        if not len(loaded):
            return []
        body_kg = float(np.median(loaded))
    mid = on_fraction * body_kg
    hi, lo = mid + hysteresis * body_kg, mid - hysteresis * body_kg
    n = len(t)

    def crossing(i: int, level: float, rising: bool) -> float | None:
        """Time where f crosses ``level`` last before sample i (inclusive), interpolated."""
        j = i - 1
        while j >= 0 and ((f[j] >= level) if rising else (f[j] <= level)):
            j -= 1
        if j < 0 or t[j + 1] - t[j] > max_gap_s:
            return None
        return float(t[j] + (level - f[j]) / (f[j + 1] - f[j]) * (t[j + 1] - t[j]))

    # Hysteresis state machine: indices where the subject gets on / off the board
    on = f[0] >= mid
    flips: list[tuple[int, bool]] = []
    for i in range(1, n):
        if not on and f[i] >= hi:
            on = True
            flips.append((i, True))
        elif on and f[i] <= lo:
            on = False
            flips.append((i, False))

    events: list[dict] = []

    def add(kind: str, tt: float | None, **extra) -> dict | None:
        if tt is None:
            return None
        ev = {"type": kind, "t": tt, "t_rel": tt - t0, "t_unix": None if off is None else tt + off}
        ev.update(extra)
        events.append(ev)
        return ev

    landings: list[dict] = []
    takeoffs: list[float] = []
    for k, (i, rising) in enumerate(flips):
        if rising:
            if k == 0:  # started empty, then stepped on
                add("step_on", crossing(i, mid, True))
            continue
        # i: the subject left the board (or unloaded it); look at what happens next
        nxt = flips[k + 1][0] if k + 1 < len(flips) else None
        if nxt is None:
            add("step_off", crossing(i, mid, False))
            continue
        t_off, t_on = crossing(i, mid, False), crossing(nxt, mid, True)
        dur = (t_on - t_off) if t_off is not None and t_on is not None else t[nxt] - t[i]
        seg = slice(i, nxt)
        no_gap = np.all(np.diff(t[i - 1:nxt + 1]) <= max_gap_s)
        if dur <= max_flight_s and no_gap and f[seg].min() < flight_kg:
            idx = np.flatnonzero(f[seg] < flight_kg) + i
            t_up = crossing(int(idx[0]), flight_kg, False)
            q = int(idx[-1]) + 1
            t_down = crossing(q, flight_kg, True) if q < n else None
            if t_up is not None and t_down is not None and t_down - t_up >= min_flight_s:
                add("takeoff", t_up, flight_s=t_down - t_up)
                takeoffs.append(t_up)
                landings.append(add("landing", t_down, flight_s=t_down - t_up))
        elif dur > max_flight_s:
            add("step_off", t_off)
            add("step_on", t_on)
        # else: a short partial unloading (e.g. countermovement) - not an event

    # Sharp peaks above peak_factor x body weight
    above = np.concatenate([[False], f > peak_factor * body_kg, [False]])
    edges = np.diff(above.astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for a, b in zip(starts, ends):
        p = a + int(np.argmax(f[a:b]))
        tp = float(t[p])
        if 0 < p < n - 1 and t[p + 1] - t[p - 1] <= 2 * max_gap_s:
            den = f[p - 1] - 2 * f[p] + f[p + 1]
            if den < 0:
                tp += float(np.clip(0.5 * (f[p - 1] - f[p + 1]) / den, -0.5, 0.5)) * (t[p + 1] - t[p - 1]) / 2
        peak = float(f[p])
        land = next((e for e in landings if e["t"] - max_gap_s <= tp <= e["t"] + jump_guard_s), None)
        if land is not None:
            land["peak_kg"] = max(peak, land.get("peak_kg", 0.0))
        elif not any(to - jump_guard_s <= tp <= to for to in takeoffs):
            add("stomp", tp, peak_kg=peak)
    events.sort(key=lambda e: e["t"])
    return events


def write_events_csv(path: str | Path, events: list[dict]) -> Path:
    """Write detected force events (columns: type, t, t_rel, t_unix, flight_s, peak_kg)."""
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EVENT_COLS)
        for e in events:
            w.writerow([e["type"]] + ["" if e.get(c) is None or not np.isfinite(e[c]) else f"{e[c]:.6f}"
                                      for c in EVENT_COLS[1:]])
    return path


def estimate_time_offset(t_ref, t_other, tolerance: float = 0.05) -> tuple[float, int]:
    """Offset ``d`` such that ``t_other + d`` matches ``t_ref``, from two lists of times of the
    same physical events seen by two systems (e.g. jump landings in wii.csv and in another
    app's data). Tries every pairwise difference, keeps the one that matches the most events
    within ``tolerance`` (then the smallest residual) and refines it with the median of the
    matched differences. Returns (offset, number of matched events)."""
    a = np.sort(np.asarray(t_ref, np.float64).ravel())
    b = np.sort(np.asarray(t_other, np.float64).ravel())
    if not len(a) or not len(b):
        raise ValueError("Both event lists must contain at least one time")
    best = (-1, np.inf, 0.0)
    for d in np.unique((a[:, None] - b[None, :]).ravel()):
        r = np.abs(a[:, None] - (b[None, :] + d)).min(axis=1)
        m = r <= tolerance
        cand = (int(m.sum()), float(r[m].mean()), float(d))
        if cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
            best = cand
    d = best[2]
    j = np.abs(a[:, None] - (b[None, :] + d)).argmin(axis=1)
    diff = a - b[j]
    m = np.abs(diff - d) <= tolerance
    return float(np.median(diff[m])), int(m.sum())


def _fmt_time(t_unix: float | None) -> str:
    if t_unix is None or not np.isfinite(t_unix):
        return ""
    return datetime.fromtimestamp(t_unix).astimezone().isoformat(timespec="milliseconds")


def analyze_session(folder: str | Path) -> dict:
    folder = Path(folder)
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    t0 = meta.get("t0", 0.0)
    off = meta.get("clock_offset_unix")
    summary: dict = {"session": folder.name, "t0": t0, "t0_unix": meta.get("t0_unix"),
                     "start_time_iso": meta.get("start_time_iso")}

    wii = read_csv_columns(folder / "wii.csv") if (folder / "wii.csv").exists() else None
    if wii is not None and not len(wii.get("t", [])):
        wii = None
    pose_path = folder / "pose3d.csv"
    pose = read_csv_columns(pose_path) if pose_path.exists() else None
    if pose is not None and not len(pose.get("t", [])):
        pose = None
    summary["has_wii"], summary["has_pose"] = wii is not None, pose is not None
    summary["events_marked"] = len(read_events(folder))

    if wii is not None:
        summary["mean_total_kg"] = float(np.nanmean(wii["total_kg"]))
        summary["wii_rate_hz"] = float((len(wii["t"]) - 1) / max(np.ptp(wii["t"]), 1e-9))
        summary["cop"] = sway_metrics(wii["t"], wii["cop_x_board"], wii["cop_y_board"])
        summary["force_events_detected"] = len(
            detect_force_events(wii, t0=t0, clock_offset_unix=off))

    if pose is not None:
        t = pose["t"]
        com = np.column_stack([pose["com_x"], pose["com_y"], pose["com_z"]])
        com_b = np.column_stack([pose["com_x_board"], pose["com_y_board"], pose["com_z_board"]])
        summary["pose_rate_hz"] = float((len(t) - 1) / max(np.ptp(t), 1e-9))
        summary["com"] = sway_metrics(t, com_b[:, 0], com_b[:, 1])
        if wii is not None:
            force = {c: interp(wii["t"], wii[c], t) for c in FORCE_COLS}
            _write_fused(folder, t, t0, force, com, com_b, clock_offset_unix=off)
            d = np.hypot(com_b[:, 0] - force["cop_x_board"], com_b[:, 1] - force["cop_y_board"])
            if np.isfinite(d).any():
                summary["com_cop_distance_rms_mm"] = float(np.sqrt(np.nanmean(d ** 2)) * 1000)

    (folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    return summary


# When writing TRC, Pose2Sim converts Z-up to Y-up: (X', Y', Z') = (Y, Z, X). Its inverse:
POSE2SIM_YUP_TO_ZUP = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)


def fuse_external(folder: str | Path, pose_file: str | Path, offset_s: float = 0.0,
                  world_transform: np.ndarray | None = None, time_base: str = "rel") -> Path:
    """Align an external 3D pose file (TRC or CSV) with this recording's Wii data and fuse them.

    ``time_base="rel"``: the external time column counts from the recording start, i.e. time 0
    of the external data corresponds to t0 + offset_s (a CSV whose only time column is
    ``t_unix`` is refused). ``time_base="unix"``: the external times are wall-clock Unix seconds
    (for a CSV, a column named ``t_unix`` is used when present, otherwise the time column) and
    are mapped with the session's ``clock_offset_unix``; ``offset_s`` is added on top (e.g. a
    known clock difference between two computers). The external pose must be in the same
    checkerboard world frame; if it is not, pass a 4x4 ``world_transform`` (external
    coordinates -> PoseBoard world coordinates). Without wii.csv the force columns are empty.
    """
    if time_base not in ("rel", "unix"):
        raise ValueError(f"time_base must be 'rel' or 'unix', not {time_base!r}")
    folder = Path(folder)
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    t0 = meta["t0"]
    off = meta.get("clock_offset_unix")
    board = BoardPose.from_dict(meta["board_pose"]) if meta.get("board_pose") else None
    pose_file = Path(pose_file)
    is_trc = pose_file.suffix.lower() == ".trc"
    if is_trc:
        t_ext, names, kps, _ = read_trc(pose_file)
    else:
        t_ext, names, kps = read_keypoint_csv(pose_file)
        with open(pose_file, newline="", encoding="utf-8-sig") as f:
            tname = csv_time_column(next(csv.reader(f), []))
        if time_base == "rel" and tname is not None and tname.strip().lower() == "t_unix":
            raise ValueError(f"{pose_file.name} has only a t_unix time column (Unix seconds): "
                             "use time_base='unix' (--time-base unix), or add a t/time column "
                             "with seconds since the recording start")
    if world_transform is not None:
        T = np.asarray(world_transform, np.float64)
        kps = kps @ T[:3, :3].T + T[:3, 3]
    if time_base == "unix":
        if off is None:
            raise ValueError("This session has no clock_offset_unix (recorded by an older version); "
                             "use time_base='rel'")
        if not is_trc:
            tu = read_csv_column(pose_file, "t_unix")
            if tu is not None:
                t_ext = tu
        t = t_ext - off + offset_s
    else:
        t = t0 + offset_s + t_ext
    com = np.array([c if (c := center_of_mass(k, names)) is not None else np.full(3, np.nan)
                    for k in kps]).reshape(-1, 3)
    com_b = board.world_to_board.apply(com) if board is not None else np.full_like(com, np.nan)
    wii_path = folder / "wii.csv"
    if wii_path.exists():
        wii = read_csv_columns(wii_path)
        force = {c: interp(wii["t"], wii[c], t) for c in FORCE_COLS}
    else:
        force = {c: np.full(len(t), np.nan) for c in FORCE_COLS}
    return _write_fused(folder, t, t0, force, com, com_b, name=f"fused_{pose_file.stem}.csv",
                        clock_offset_unix=off)


def export_trc(folder: str | Path) -> Path:
    """Export pose3d.csv to TRC (millimeters) for use in OpenSim / Pose2Sim."""
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
    ap = argparse.ArgumentParser(description="PoseBoard recording post-processing")
    ap.add_argument("folder")
    ap.add_argument("--external", help="external 3D pose file (.trc or .csv)")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="time offset (s) added to the external times (with --time-base rel: "
                         "relative to the recording start)")
    ap.add_argument("--time-base", choices=["rel", "unix"], default="rel",
                    help="time base of the external file: 'rel' = seconds since the recording start "
                         "(CSV column t/time/timestamp/time_s), 'unix' = wall-clock Unix seconds (a "
                         "CSV column 't_unix' is used if present, otherwise the time column)")
    ap.add_argument("--pose2sim-yup", action="store_true",
                    help="external TRC uses Pose2Sim's Y-up coordinates; convert back to checkerboard Z-up")
    ap.add_argument("--events", action="store_true",
                    help="detect step-on/off, jumps and stomps in wii.csv, print them and "
                         "write events_detected.csv")
    ap.add_argument("--trc", action="store_true", help="export pose3d.trc")
    a = ap.parse_args(argv)
    print(json.dumps(analyze_session(a.folder), indent=2, ensure_ascii=False))
    if a.events:
        folder = Path(a.folder)
        if not (folder / "wii.csv").exists():
            print("No wii.csv in this session: no force events")
        else:
            events = detect_force_events(folder)
            print(f"{len(events)} force event(s):")
            for e in events:
                extra = "".join(f"  {k}={e[k]:.3f}" for k in ("flight_s", "peak_kg") if k in e)
                print(f"  {e['type']:<9} t_rel={e['t_rel']:9.3f} s  t_unix={e['t_unix'] or float('nan'):.3f}"
                      f"  {_fmt_time(e['t_unix'])}{extra}")
            print("Wrote", write_events_csv(folder / "events_detected.csv", events))
    if a.external:
        T = POSE2SIM_YUP_TO_ZUP if a.pose2sim_yup else None
        print("Wrote", fuse_external(a.folder, a.external, a.offset, T, time_base=a.time_base))
    if a.trc:
        print("Wrote", export_trc(a.folder))


if __name__ == "__main__":
    main()
