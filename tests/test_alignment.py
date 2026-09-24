"""Alignment support: t_unix columns, event markers, optional streams, force event detection
and wall-clock fusion of external data."""

import csv
import json
import time
from datetime import datetime

import numpy as np
import pytest

from poseboard.analysis import (analyze_session, detect_force_events, estimate_time_offset,
                                fuse_external, main, read_csv_columns, read_events)
from poseboard.camera import CameraStream
from poseboard.geometry import BoardGeometry
from poseboard.pose.base import Pose3D
from poseboard.pose.mediapipe_backend import MP_NAMES
from poseboard.session import WII_HEADER, SessionRecorder, capture_clock_offset
from poseboard.wii.device import SimulatedBoard
from tests.test_core import board_transform, standing_skeleton
from tests.test_session import make_video

W = 70.0
# Piecewise-linear total force: step on (50 % at 2.0 s), countermovement jump (total crosses
# 5 kg at 5.60 s going up and at 5.90 s coming down: 0.30 s flight, 3 BW landing peak), stomp
# (1.8 BW peak at 8.0 s), step off (50 % at 11.0 s).
KNOTS = [(0.0, 0), (1.9, 0), (2.1, W), (5.0, W), (5.15, 0.35 * W), (5.3, W), (5.45, 2.0 * W),
         (5.55, 145), (5.60, 5), (5.61, 0), (5.89, 0), (5.90, 5), (5.93, 3 * W), (6.0, 50),
         (6.3, W), (7.95, W), (8.0, 1.8 * W), (8.05, W), (10.9, W), (11.1, 0), (12.0, 0)]
TRUTH = [("step_on", 2.0), ("takeoff", 5.60), ("landing", 5.90), ("stomp", 8.0), ("step_off", 11.0)]


def synthetic_force(rate=100.0, phase=0.0037, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 12, 1 / rate) + phase
    kx, ky = np.array(KNOTS).T
    f = np.interp(t, kx, ky)
    f += np.where(f > 10, 0.01 * W * np.sin(2 * np.pi * 0.7 * t), 0) + rng.normal(0, noise, len(t))
    return t, np.maximum(f, -1.0)


# ------------------------------------------------------------------ detection
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_detect_force_events(seed):
    t, f = synthetic_force(seed=seed)
    t_perf = t + 1000.0  # perf_counter-like times
    off = 1.7e9
    events = detect_force_events((t_perf, f), t0=1000.0, clock_offset_unix=off)
    assert [e["type"] for e in events] == [k for k, _ in TRUTH]
    for e, (_, tt) in zip(events, TRUTH):
        assert abs(e["t_rel"] - tt) < 0.02, e
        assert abs(e["t"] - (1000.0 + tt)) < 0.02
        assert e["t_unix"] == pytest.approx(e["t"] + off, abs=1e-6)
    take, land, stomp = events[1], events[2], events[3]
    assert land["flight_s"] == pytest.approx(0.30, abs=0.02)
    assert take["flight_s"] == land["flight_s"]
    assert land["peak_kg"] > 1.5 * W  # the landing impact is attached, not reported as a stomp
    assert 1.6 * W < stomp["peak_kg"] < 1.85 * W  # sampled at 100 Hz: a little below the true 1.8 BW


def test_detect_force_events_dict_and_no_subject():
    t, f = synthetic_force()
    cols = {"t": t + 5.0, "t_rel": t, "t_unix": t + 5.0 + 1.6e9, "total_kg": f}
    events = detect_force_events(cols)
    assert len(events) == 5
    assert events[0]["t_rel"] == pytest.approx(2.0, abs=0.02)  # t0 derived from t_rel
    assert events[0]["t_unix"] == pytest.approx(1.6e9 + 5.0 + events[0]["t_rel"], abs=1e-3)
    # Nobody on the board: no events; t_unix unknown without an offset
    assert detect_force_events((t, np.random.default_rng(0).normal(0, 0.3, len(t)))) == []
    assert detect_force_events((t, f))[0]["t_unix"] is None


def test_estimate_time_offset():
    ref = np.array([2.0, 5.9, 8.0, 11.0, 14.2])
    rng = np.random.default_rng(1)
    other = ref - 123.456 + rng.normal(0, 0.004, len(ref))
    other = np.append(np.delete(other, 3), 40.0)  # one event missed, one spurious
    d, n = estimate_time_offset(ref, other)
    assert d == pytest.approx(123.456, abs=0.01) and n == 4


# ------------------------------------------------------------------ recording
def test_clock_offset_capture():
    t, u = capture_clock_offset()
    assert abs((time.time() - time.perf_counter()) - (u - t)) < 0.05


def test_wii_only_session_with_events(tmp_path):
    sim = SimulatedBoard()
    sim.start()
    rec = SessionRecorder(tmp_path)
    try:
        folder = rec.start(cams=[], calibrations={}, force=sim, board=None, geometry=BoardGeometry(),
                           pose_backend=None, subject="wii")
        with pytest.raises(RuntimeError):
            rec.start(cams=[], calibrations={}, force=sim, board=None, geometry=BoardGeometry(),
                      pose_backend=None)
        time.sleep(0.3)
        ev = rec.add_event("sync jump")
        t_mark = time.perf_counter()
        rec.add_event("  marker\nwith, comma ", t=t_mark)
        time.sleep(0.2)
        assert rec.stop() == folder
    finally:
        sim.stop()
    # stop() is idempotent; late pose/events are ignored
    assert rec.stop() is None
    assert rec.add_event("late") is None
    rec.add_pose(Pose3D(time.perf_counter(), list(MP_NAMES), standing_skeleton(), np.ones(len(MP_NAMES))))
    assert not (folder / "pose3d.csv").exists()

    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["has_wii"] and not meta["has_pose"] and not meta["has_video"]
    assert meta["camera_names"] == [] and meta["samples"]["events"] == 2
    off = meta["clock_offset_unix"]
    assert off == pytest.approx(meta["t0_unix"] - meta["t0"], abs=1e-6)
    assert datetime.fromisoformat(meta["start_time_iso"]).tzinfo is not None
    assert meta["t_stop_unix"] - meta["t0_unix"] == pytest.approx(meta["duration_s"], abs=1e-6)

    with open(folder / "wii.csv", newline="") as f:
        assert next(csv.reader(f)) == WII_HEADER
    wii = read_csv_columns(folder / "wii.csv")
    assert len(wii["t"]) > 20
    assert np.all(np.diff(wii["t_unix"]) > 0)
    np.testing.assert_allclose(wii["t_unix"] - wii["t"], off, atol=2e-6)
    np.testing.assert_allclose(wii["t_rel"], wii["t"] - meta["t0"], atol=2e-6)
    assert abs(wii["t_unix"][0] - time.time()) < 60

    events = read_events(folder)
    assert [e["label"] for e in events] == ["sync jump", "marker with, comma"]
    assert events[0]["t_unix"] == pytest.approx(ev["t_unix"], abs=2e-6)
    assert events[1]["t"] == pytest.approx(t_mark, abs=2e-6)
    assert events[1]["t_unix"] - events[1]["t"] == pytest.approx(off, abs=2e-6)
    assert 0.2 < events[1]["t_rel"] < 1.0

    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["t0_unix"] == meta["t0_unix"] and summary["events_marked"] == 2
    assert summary["has_wii"] and not summary["has_pose"]
    assert summary["force_events_detected"] == 0  # the simulator stands still the whole time
    assert 60 < summary["mean_total_kg"] < 80


def test_pose_only_session_then_board_attached(tmp_path):
    video = tmp_path / "in.mp4"
    make_video(video)
    cam = CameraStream("cam0", str(video))
    cam.start()
    sim = SimulatedBoard()
    try:
        for _ in range(100):
            if cam.latest() is not None:
                break
            time.sleep(0.01)
        rec = SessionRecorder(tmp_path / "rec")
        folder = rec.start(cams=[cam], calibrations={}, force=None, board=None,
                           geometry=BoardGeometry(), pose_backend="test")
        kp = board_transform().apply(standing_skeleton())
        for _ in range(12):
            rec.add_pose(Pose3D(time.perf_counter(), list(MP_NAMES), kp, np.ones(len(kp))))
            time.sleep(0.02)
        assert not (folder / "wii.csv").exists()
        # A board that appears during the recording is picked up
        sim.start()
        rec.attach_force(sim)
        time.sleep(0.3)
        rec.stop()
    finally:
        sim.stop()
        cam.stop()
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    off = meta["clock_offset_unix"]
    assert meta["has_pose"] and meta["has_video"] and meta["camera_names"] == ["cam0"]
    assert meta["has_wii"] and meta["force_source"]["type"] == "SimulatedBoard"
    pose = read_csv_columns(folder / "pose3d.csv")
    assert len(pose["t"]) == 12
    np.testing.assert_allclose(pose["t_unix"] - pose["t"], off, atol=2e-6)
    ts = read_csv_columns(folder / "cam0_timestamps.csv")
    assert len(ts["t"]) > 3
    np.testing.assert_allclose(ts["t_unix"] - ts["t"], off, atol=2e-6)
    wii = read_csv_columns(folder / "wii.csv")
    assert len(wii["t"]) > 10 and wii["t_rel"].min() > 0.2
    fused = read_csv_columns(folder / "fused.csv")
    np.testing.assert_allclose(fused["t_unix"] - fused["t"], off, atol=2e-6)


def test_pose_only_analysis(tmp_path):
    rec = SessionRecorder(tmp_path)
    folder = rec.start(cams=[], calibrations={}, force=None, board=None, geometry=BoardGeometry(),
                       pose_backend="plugin")
    kp = standing_skeleton()
    for i in range(20):
        rec.add_pose(Pose3D(rec.t0 + 0.05 * i, list(MP_NAMES), kp, np.ones(len(kp))))
    rec.stop()
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["has_pose"] and not summary["has_wii"]
    assert summary["pose_rate_hz"] == pytest.approx(20, rel=0.01)
    assert "force_events_detected" not in summary and not (folder / "fused.csv").exists()


def test_camera_default_offset(tmp_path):
    video = tmp_path / "in.mp4"
    make_video(video)
    cam = CameraStream("c", str(video))
    cam.start()
    try:
        for _ in range(100):
            if cam.latest() is not None:
                break
            time.sleep(0.01)
        cam.start_recording(tmp_path / "c.mp4")  # backward compatible call
        time.sleep(0.2)
        cam.stop_recording()
    finally:
        cam.stop()
    ts = read_csv_columns(tmp_path / "c_timestamps.csv")
    assert len(ts["t"]) > 2
    assert np.all(np.abs(ts["t_unix"] - ts["t"] - (time.time() - time.perf_counter())) < 0.1)


# ------------------------------------------------------------ offline analysis
def write_session(folder, t0=500.0, off=1.75e9, with_offset=True):
    """A hand-made session folder with the synthetic force profile in wii.csv."""
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"t0": t0, "board_pose": None}
    if with_offset:
        meta.update(t0_unix=t0 + off, clock_offset_unix=off)
    (folder / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    t, f = synthetic_force()
    t = t + t0
    with open(folder / "wii.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(WII_HEADER)
        for ti, fi in zip(t, f):
            w.writerow([f"{ti:.6f}", f"{ti - t0:.6f}", f"{ti + off:.6f}" if with_offset else "",
                        *[f"{fi / 4:.6f}"] * 4, f"{fi:.6f}", "", "", "", "", ""])
    return folder


def write_external_csv(path, t_col, kp, t_unix=None):
    names = list(MP_NAMES)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["time"] + (["t_unix"] if t_unix is not None else [])
        w.writerow(head + [f"{n}_{a}" for n in names for a in "xyz"])
        for i, ti in enumerate(t_col):
            row = [f"{ti:.6f}"] + ([f"{t_unix[i]:.6f}"] if t_unix is not None else [])
            w.writerow(row + [f"{v:.6f}" for v in kp.ravel()])
    return path


def test_fuse_external_unix_time_base(tmp_path):
    t0, off = 500.0, 1.75e9
    folder = write_session(tmp_path / "s", t0, off)
    kp = standing_skeleton()
    t_rel = np.arange(1.0, 10.0, 0.5)
    # The other app's own time column is meaningless here (frame numbers); t_unix is preferred
    ext = write_external_csv(tmp_path / "ext.csv", np.arange(len(t_rel)), kp, t_unix=t0 + off + t_rel)
    fused = read_csv_columns(fuse_external(folder, ext, time_base="unix"))
    np.testing.assert_allclose(fused["t_rel"], t_rel, atol=1e-5)
    np.testing.assert_allclose(fused["t_unix"], t0 + off + t_rel, atol=1e-5)
    t, f = synthetic_force()
    np.testing.assert_allclose(fused["total_kg"], np.interp(t_rel, t, f), atol=1.0)
    # Unix seconds in the plain time column + extra offset_s
    ext2 = write_external_csv(tmp_path / "ext2.csv", t0 + off + t_rel - 0.25, kp)
    fused2 = read_csv_columns(fuse_external(folder, ext2, offset_s=0.25, time_base="unix"))
    np.testing.assert_allclose(fused2["t_rel"], t_rel, atol=1e-5)
    # 'rel' is unchanged: the time column counts from the recording start
    ext3 = write_external_csv(tmp_path / "ext3.csv", t_rel, kp, t_unix=np.zeros_like(t_rel))
    np.testing.assert_allclose(read_csv_columns(fuse_external(folder, ext3))["t_rel"], t_rel, atol=1e-5)
    with pytest.raises(ValueError):
        fuse_external(folder, ext, time_base="gps")
    old = write_session(tmp_path / "old", with_offset=False)
    with pytest.raises(ValueError):
        fuse_external(old, ext, time_base="unix")


def test_cli_events_and_time_base(tmp_path, capsys):
    t0, off = 500.0, 1.75e9
    folder = write_session(tmp_path / "s", t0, off)
    kp = standing_skeleton()
    ext = write_external_csv(tmp_path / "ext.csv", [0, 1], kp, t_unix=[t0 + off + 3.0, t0 + off + 4.0])
    main([str(folder), "--events", "--external", str(ext), "--time-base", "unix"])
    out = capsys.readouterr().out
    assert "5 force event(s)" in out and "landing" in out
    rows = list(csv.DictReader(open(folder / "events_detected.csv", newline="")))
    assert [r["type"] for r in rows] == [k for k, _ in TRUTH]
    for r, (_, tt) in zip(rows, TRUTH):
        assert abs(float(r["t_rel"]) - tt) < 0.02
        assert float(r["t_unix"]) == pytest.approx(float(r["t"]) + off, abs=2e-6)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["force_events_detected"] == 5 and summary["t0_unix"] == t0 + off
    fused = read_csv_columns(folder / "fused_ext.csv")
    np.testing.assert_allclose(fused["t_rel"], [3.0, 4.0], atol=1e-5)
    assert analyze_session(folder)["has_wii"]
