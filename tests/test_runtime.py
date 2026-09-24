"""Runtime robustness: camera thread and video writer, session folders and files, the pose
worker thread, and the Wii-only command-line recorder."""

import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from poseboard import camera as camera_mod
from poseboard.camera import CameraStream, Frame, configure_capture, open_video_writer
from poseboard.geometry import BoardGeometry
from poseboard.pose.base import Pose3D, PoseEstimator
from poseboard.pose.mediapipe_backend import MP_NAMES
from poseboard.session import SessionRecorder
from poseboard.wii import device as device_mod
from poseboard.wii.device import HIDAPI_MISSING, HidapiUnavailable, SimulatedBoard
from poseboard.wii.record import main as record_main
from tests.test_core import standing_skeleton

ROOT = Path(__file__).resolve().parents[1]


def wait_until(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ------------------------------------------------------------------ camera
class FakeCap:
    """Stand-in for cv2.VideoCapture."""

    def __init__(self, sizes=((240, 320),), block_after=None, block_s=0.0):
        self.calls, self.sizes, self.n = [], list(sizes), 0
        self.block_after, self.block_s = block_after, block_s
        self.reading = False
        self.released = False
        self.released_during_read = False
        self.fourcc = 0

    def set(self, prop, value):
        self.calls.append((prop, value))
        if prop == cv2.CAP_PROP_FOURCC:
            self.fourcc = int(value)
        return True

    def get(self, prop):
        return self.fourcc if prop == cv2.CAP_PROP_FOURCC else 30.0

    def isOpened(self):
        return True

    def read(self):
        self.reading = True
        try:
            if self.block_after is not None and self.n >= self.block_after:
                time.sleep(self.block_s)  # e.g. a stalled network stream
            else:
                time.sleep(0.005)
            h, w = self.sizes[min(self.n // 10, len(self.sizes) - 1)]
            self.n += 1
            return True, np.full((h, w, 3), self.n % 255, np.uint8)
        finally:
            self.reading = False

    def release(self):
        self.released_during_read |= self.reading
        self.released = True


def test_directshow_mjpg_is_requested_last():
    cap = FakeCap()
    configure_capture(cap, 1280, 720, 30, mjpg=True)
    props = [p for p, _ in cap.calls]
    assert props == [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                     cv2.CAP_PROP_FOURCC]
    assert cap.calls[-1][1] == cv2.VideoWriter_fourcc(*"MJPG")


def test_camera_stop_never_releases_during_a_blocking_read(monkeypatch):
    cap = FakeCap(block_after=3, block_s=0.6)
    monkeypatch.setattr(camera_mod, "open_capture", lambda *a, **k: cap)
    s = CameraStream("net", "rtsp://example/stream")
    s.stop_timeout_s = 0.1
    s.start()
    assert wait_until(lambda: cap.n >= 3 and cap.reading)
    t = time.perf_counter()
    s.stop()
    assert time.perf_counter() - t < 0.5 and not cap.released  # left to the capture thread
    assert wait_until(lambda: cap.released, 2.0)
    assert not cap.released_during_read


def test_video_writer_failure_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="Cannot write video"):
        open_video_writer(tmp_path / "missing_dir" / "cam0.mkv", 30, (320, 240))


def test_frames_of_another_size_are_not_timestamped(tmp_path, monkeypatch):
    cap = FakeCap(sizes=((240, 320), (120, 160), (240, 320)))
    monkeypatch.setattr(camera_mod, "open_capture", lambda *a, **k: cap)
    s = CameraStream("c", "0")
    s.start()
    try:
        assert wait_until(lambda: s.latest() is not None)
        path = s.start_recording(tmp_path / "c.mkv")
        assert wait_until(lambda: cap.n > 35)
        s.stop_recording()
    finally:
        s.stop()
    assert s.dropped_frames > 0
    rows = list(csv.DictReader(open(tmp_path / "c_timestamps.csv", newline="")))
    v = cv2.VideoCapture(str(path))
    n = 0
    while v.read()[0]:
        n += 1
    v.release()
    assert n == len(rows) > 5 and set(rows[0]) == {"frame", "t", "t_rel", "t_unix"}


def test_recording_write_error_does_not_kill_the_camera(tmp_path, monkeypatch):
    cap = FakeCap()
    monkeypatch.setattr(camera_mod, "open_capture", lambda *a, **k: cap)
    s = CameraStream("c", "0")
    s.start()
    try:
        assert wait_until(lambda: s.latest() is not None)
        s.start_recording(tmp_path / "c.mkv")

        def boom(*a):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(s, "_write_frame", boom)
        assert wait_until(lambda: s.error is not None and "recording stopped" in s.error)
        n = cap.n
        assert wait_until(lambda: cap.n > n + 5) and s.running  # still capturing
        assert s._writer is None
    finally:
        s.stop()


CHILD = r'''
import os, sys, time
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from poseboard.camera import CameraStream
from poseboard.geometry import BoardGeometry
from poseboard.session import SessionRecorder
from poseboard.wii.device import SimulatedBoard
import cv2
import numpy as np

out = Path(sys.argv[2])
# Noisy frames like a real camera sensor (flat synthetic frames compress to almost nothing and
# would just sit in the writer's buffer)
vw = cv2.VideoWriter(str(out / "in.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
rng = np.random.default_rng(0)
for _ in range(30):
    vw.write(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
vw.release()
cam = CameraStream("cam0", str(out / "in.mp4"))
cam.start()
while cam.latest() is None:
    time.sleep(0.01)
sim = SimulatedBoard()
sim.start()
rec = SessionRecorder(out / "rec")
rec.start(cams=[cam], calibrations={}, force=sim, board=None, geometry=BoardGeometry(),
          pose_backend=None)
time.sleep(2.5)
os._exit(1)  # crash / killed / power loss: nothing is closed
'''


def test_recording_survives_an_abrupt_exit(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    r = subprocess.run([sys.executable, str(script), str(ROOT), str(tmp_path)], timeout=60,
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stderr
    (folder,) = list((tmp_path / "rec").iterdir())
    v = cv2.VideoCapture(str(folder / "cam0.mkv"))
    n = 0
    while v.read()[0]:
        n += 1
    v.release()
    assert n > 10  # an MP4 would be unreadable without its index
    ts = list(csv.reader(open(folder / "cam0_timestamps.csv", newline="")))
    wii = list(csv.reader(open(folder / "wii.csv", newline="")))
    assert len(ts) > 10 and len(wii) > 50  # flushed about once per second


# ------------------------------------------------------------------ session
def test_recordings_in_the_same_second_get_separate_folders(tmp_path):
    rec = SessionRecorder(tmp_path)
    kw = dict(cams=[], calibrations={}, force=None, board=None, geometry=BoardGeometry(),
              pose_backend="x", subject="S")
    f1 = rec.start(**kw)
    rec.add_event("sync")
    rec.add_pose(Pose3D(time.perf_counter(), list(MP_NAMES), standing_skeleton(), np.ones(33)))
    rec.stop()
    f2 = rec.start(**kw)
    rec.stop()
    assert f1 != f2
    if f1.name[:15] == f2.name[:15]:  # same second
        assert f2.name == f1.name + "_2"
    assert not (f2 / "events.csv").exists() and not (f2 / "pose3d.csv").exists()
    summary = json.loads((f2 / "summary.json").read_text(encoding="utf-8"))
    assert summary["events_marked"] == 0 and not summary["has_pose"]


def test_pose_names_changing_mid_recording_are_matched_by_name(tmp_path):
    rec = SessionRecorder(tmp_path)
    folder = rec.start(cams=[], calibrations={}, force=None, board=None, geometry=BoardGeometry(),
                       pose_backend="a")
    rec.add_pose(Pose3D(rec.t0 + 0.1, ["a", "b", "c"], np.arange(9.0).reshape(3, 3), np.ones(3)))
    rec.add_pose(Pose3D(rec.t0 + 0.2, ["c", "x", "a"], np.array([[7, 8, 9], [0, 0, 0], [1, 2, 3.0]]),
                        np.ones(3)))
    rec.stop()
    rows = list(csv.DictReader(open(folder / "pose3d.csv", newline="")))
    assert len(rows) == 2 and "x_x" not in rows[0]
    assert rows[1]["a_x"] == "1.000000" and rows[1]["c_z"] == "9.000000" and rows[1]["b_x"] == ""
    labels = [r["label"] for r in csv.DictReader(open(folder / "events.csv", newline="",
                                                      encoding="utf-8-sig"))]
    assert labels == ["pose_keypoints_changed"]


def test_background_post_processing_and_force_history(tmp_path):
    sim = SimulatedBoard()
    sim.start()
    rec = SessionRecorder(tmp_path)
    try:
        folder = rec.start(cams=[], calibrations={}, force=sim, board=None,
                           geometry=BoardGeometry(), pose_backend=None)
        time.sleep(0.3)
        rec.add_event("yeux fermés")  # non-ASCII label
        t = time.perf_counter()
        assert rec.stop(background=True) == folder
        assert time.perf_counter() - t < 1.0
        assert rec.wait_post_processing(30) and not rec.post_processing
    finally:
        sim.stop()
    assert (folder / "summary.json").exists()
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    (h,) = meta["force_source_history"]
    assert h["reason"] == "start" and h["device"] == "simulator" and h["min_total_kg"] == 5.0
    assert meta["force_source"]["min_total_kg"] == 5.0
    raw = (folder / "events.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM: Excel detects UTF-8


# ------------------------------------------------------------------ pose worker
class FakeRecorder:
    def __init__(self):
        self.recording = True
        self.poses = []
        self.fail: Exception | None = None

    def add_pose(self, pose):
        if self.fail is not None:
            raise self.fail
        self.poses.append(pose)


class FakeEstimator(PoseEstimator):
    def __init__(self, delay=0.0, t_value=None):
        self.delay, self.t_value = delay, t_value
        self.seen: list[set] = []
        self.busy = False
        self.closed = False
        self.closed_while_busy = False

    def process(self, frames, cams):
        self.busy = True
        try:
            self.seen.append(set(frames))
            time.sleep(self.delay)
            t = self.t_value if self.t_value is not None else max(t for t, _ in frames.values())
            return Pose3D(t, ["a"], np.zeros((1, 3)), np.ones(1))
        finally:
            self.busy = False

    def close(self):
        self.closed_while_busy |= self.busy
        self.closed = True


def make_worker(est, frames_fn, rec=None):
    from poseboard.gui.app import PoseWorker

    return PoseWorker(est, frames_fn, dict, rec or FakeRecorder(), max_skew_s=0.25)


def live_frames(frozen_t):
    n = [0]

    def get():
        n[0] += 1
        now = time.perf_counter()
        img = np.zeros((2, 2, 3), np.uint8)
        return {"live": Frame(n[0], now, img), "frozen": Frame(0, frozen_t, img)}
    return get


def test_pose_worker_leaves_out_a_frozen_camera():
    est = FakeEstimator()
    w = make_worker(est, live_frames(time.perf_counter() - 2.0))
    w.start()
    try:
        assert wait_until(lambda: len(est.seen) > 5)
        assert all(s == {"live"} for s in est.seen) and w.stale_cameras == ["frozen"]
        assert abs(w.latest.t - time.perf_counter()) < 0.2  # no lag from the frozen view
    finally:
        w.stop()
    assert est.closed and not est.closed_while_busy


def test_pose_worker_survives_errors_and_clears_them():
    rec = FakeRecorder()
    rec.fail = OSError(28, "No space left on device")
    est = FakeEstimator()
    w = make_worker(est, live_frames(time.perf_counter()), rec)
    w.start()
    try:
        assert wait_until(lambda: w.error is not None and "space" in w.error)
        assert w.alive
        rec.fail = None
        assert wait_until(lambda: w.error is None and rec.poses)
        est.t_value = float("nan")  # invalid plugin output
        assert wait_until(lambda: w.error is not None and "t" in w.error) and w.alive
    finally:
        w.stop()


def test_pose_worker_stop_never_closes_during_process():
    rec = FakeRecorder()
    est = FakeEstimator(delay=0.5)
    w = make_worker(est, live_frames(time.perf_counter()), rec)
    w.start()
    assert wait_until(lambda: est.busy)
    n = len(rec.poses)
    w.stop(timeout=0.05)
    assert not est.closed  # still processing: closed later by the thread itself
    assert wait_until(lambda: est.closed, 2.0)
    assert not est.closed_while_busy and len(rec.poses) == n  # the late result was dropped


# ------------------------------------------------------------------ Wii-only CLI
def only_session(root):
    folders = [p for p in Path(root).iterdir() if p.is_dir()]
    assert len(folders) == 1
    return folders[0]


def test_cli_ignores_enter_before_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("\nlabel typed too early\n"))
    assert record_main(["--simulate", "--seconds", "0.8", "--tare", "0.3", "--out", str(tmp_path)]) == 0
    folder = only_session(tmp_path)
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["duration_s"] > 0.7 and meta["samples"]["wii"] > 30
    assert meta["samples"]["events"] == 0
    assert meta["force_source"]["min_total_kg"] == 5.0


def test_cli_exit_codes(tmp_path, monkeypatch, capsys):
    # Taring fails: defined exit code, nothing recorded
    def no_data(self, seconds=1.0):
        raise RuntimeError("No data available for taring")

    with monkeypatch.context() as m:
        m.setattr(SimulatedBoard, "do_tare", no_data)
        assert record_main(["--simulate", "--tare", "0.1", "--out", str(tmp_path / "a")]) == 3
    assert not (tmp_path / "a").exists()
    assert "Taring failed" in capsys.readouterr().out

    # hidapi unusable: fails at once instead of waiting forever
    def missing():
        raise HidapiUnavailable(HIDAPI_MISSING)

    with monkeypatch.context() as m:
        m.setattr(device_mod.BalanceBoardHID, "list_devices", staticmethod(missing))
        t = time.monotonic()
        assert record_main(["--out", str(tmp_path / "b")]) == 2
        assert time.monotonic() - t < 5.0
    assert "hidapi is not installed" in capsys.readouterr().out

    # A board that stops sending before the recording: no samples -> exit code 4
    def one_sample(self):
        self._set_state("connected")
        self._emit_kg(time.perf_counter(), np.array([20.0, 20, 20, 20]))
        while not self._stop.is_set():
            time.sleep(0.01)

    with monkeypatch.context() as m:
        m.setattr(SimulatedBoard, "_run", one_sample)
        assert record_main(["--simulate", "--seconds", "0.3", "--min-kg", "2",
                            "--out", str(tmp_path / "c")]) == 4
    meta = json.loads((only_session(tmp_path / "c") / "session.json").read_text(encoding="utf-8"))
    assert meta["force_source"]["min_total_kg"] == 2.0


def test_cli_device_by_number(tmp_path, monkeypatch):
    from tests.test_wii_auto import FakeBus

    bus = FakeBus(monkeypatch)
    bus.present = True
    assert record_main(["--device", "3", "--out", str(tmp_path / "x")]) == 2
    assert record_main(["--device", "0", "--seconds", "0.3", "--out", str(tmp_path / "y")]) == 0
    assert len(bus.opened) == 1


def test_cli_close_handler_restores_signals(tmp_path):
    import signal

    before = signal.getsignal(signal.SIGTERM)
    assert record_main(["--simulate", "--seconds", "0.2", "--out", str(tmp_path)]) == 0
    assert signal.getsignal(signal.SIGTERM) is before
