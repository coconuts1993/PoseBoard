"""Offscreen GUI: the Wii Balance Board is optional (cameras only, Wii only, or both), plug-and-play
auto-connect, and event markers."""

import csv
import json
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from poseboard.wii.device import BalanceBoardHID, SimulatedBoard, WiiAutoConnect  # noqa: E402
from tests.synthetic import write_scene_video  # noqa: E402
from tests.test_wii_auto import FakeBus  # noqa: E402

# A pose plugin that "detects" a standing person (Halpe names, meters, Z up) at the frame time
PLUGIN = '''
import numpy as np
from poseboard.pose.base import Pose3D, PoseEstimator

KP = {"Nose": (0.0, 0.08, 1.62), "LEar": (-0.07, 0.0, 1.62), "REar": (0.07, 0.0, 1.62),
      "LShoulder": (-0.18, 0.0, 1.42), "RShoulder": (0.18, 0.0, 1.42),
      "LElbow": (-0.22, 0.0, 1.12), "RElbow": (0.22, 0.0, 1.12),
      "LWrist": (-0.24, 0.02, 0.86), "RWrist": (0.24, 0.02, 0.86),
      "LHip": (-0.1, 0.0, 0.95), "RHip": (0.1, 0.0, 0.95),
      "LKnee": (-0.1, 0.02, 0.52), "RKnee": (0.1, 0.02, 0.52),
      "LAnkle": (-0.1, 0.0, 0.1), "RAnkle": (0.1, 0.0, 0.1)}


class Standing(PoseEstimator):
    name = "standing"
    keypoint_names = list(KP)

    def process(self, frames, cams):
        t = float(np.mean([t for t, _ in frames.values()]))
        kp = np.array(list(KP.values())) + np.array([0.95, 0.2, 0.05])
        return Pose3D(t, list(KP), kp, np.ones(len(KP)))


def create_estimator(**kwargs):
    return Standing()
'''


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def no_boards(monkeypatch):
    """No Balance Board is paired/switched on (independent of the machine running the tests)."""
    monkeypatch.setattr(BalanceBoardHID, "list_devices", staticmethod(lambda: []))


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.01)


def pump_until(app, cond, timeout=5.0) -> bool:
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def make_window(**kw):
    from poseboard.gui.app import MainWindow

    w = MainWindow(**kw)
    w.warnings = []
    w._warn = w.warnings.append  # never open a modal dialog in a test
    w.resize(1400, 850)
    w.show()
    return w


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def only_session(root):
    folders = [p for p in root.iterdir() if p.is_dir()]
    assert len(folders) == 1
    return folders[0]


def test_record_cameras_only_without_wii(app, tmp_path, no_boards):
    video = tmp_path / "scene.mp4"
    write_scene_video(video)
    plugin = tmp_path / "standing_plugin.py"
    plugin.write_text(PLUGIN, encoding="utf-8")

    w = make_window()  # auto-connect on (default), but no board is around
    try:
        assert w.chk_auto.isChecked() and isinstance(w.force, WiiAutoConnect)
        w.cam_res.setCurrentText("Default")
        w.add_camera(str(video))
        w.pose_backend.setCurrentText("Plugin (.py)")
        w.plugin_path.setText(str(plugin))
        w.b_pose.setChecked(True)
        assert w.pose_worker is not None
        assert pump_until(app, lambda: w.pose_worker.latest is not None)
        pump(app, 0.1)
        text = w.streams_label.text()
        assert "cam0" in text and "pose" in text and "Wii not connected" in text
        assert "waiting" in w.wii_label.text().lower()

        w.out_dir.setText(str(tmp_path / "rec"))
        w.b_rec.setChecked(True)
        assert w.recorder.recording, w.warnings
        pump(app, 0.8)
        w.mark_event()  # no label -> mark_1
        pump(app, 0.3)
        w.b_rec.setChecked(False)
        assert w.recorder.wait_post_processing(30)
        assert not w.warnings
    finally:
        w.close()

    folder = only_session(tmp_path / "rec")
    assert (folder / "cam0.mkv").stat().st_size > 0
    ts = read_rows(folder / "cam0_timestamps.csv")
    assert ts and set(ts[0]) == {"frame", "t", "t_rel", "t_unix"}
    pose = read_rows(folder / "pose3d.csv")
    assert len(pose) > 3 and pose[0]["com_z"] != ""
    assert not (folder / "wii.csv").exists()
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["has_wii"] is False and meta["has_pose"] is True and meta["has_video"] is True
    assert meta["force_source"] is None
    events = read_rows(folder / "events.csv")
    assert [e["label"] for e in events] == ["mark_1"]
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["has_wii"] is False and summary["has_pose"] is True


def test_record_wii_only_with_simulator(app, tmp_path, no_boards):
    w = make_window()
    try:
        w.connect_sim()
        assert not w.chk_auto.isChecked(), "a manual source switches plug and play off"
        assert isinstance(w.force, SimulatedBoard)
        assert pump_until(app, lambda: w.force.connected and w.force.latest() is not None)
        pump(app, 0.1)
        assert "Wii" in w.streams_label.text() and "No cameras" in w.streams_label.text()
        assert "Simulator: connected" in w.wii_label.text()
        assert not w.act_mark.isEnabled() and not w.b_mark.isEnabled()

        w.out_dir.setText(str(tmp_path / "rec"))
        w.subject.setText("S01")
        w.b_rec.setChecked(True)
        assert w.recorder.recording, w.warnings
        assert w.act_mark.isEnabled() and w.b_mark.isEnabled()
        pump(app, 0.5)
        w.event_label.setText("sync jump")
        w.act_mark.trigger()  # the keyboard shortcut (F9 / M)
        pump(app, 0.5)
        assert "Events: 1" in w.events_info.text()
        w.b_rec.setChecked(False)
        assert w.recorder.wait_post_processing(30)
        assert not w.act_mark.isEnabled()
        assert not w.warnings
    finally:
        w.close()

    folder = only_session(tmp_path / "rec")
    assert folder.name.endswith("_S01")
    wii = read_rows(folder / "wii.csv")
    assert len(wii) > 50 and "t_unix" in wii[0]
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["has_wii"] is True and meta["has_video"] is False and meta["camera_names"] == []
    assert meta["force_source"]["type"] == "SimulatedBoard"
    (ev,) = read_rows(folder / "events.csv")
    assert ev["label"] == "sync jump"
    t_rel, t_unix = float(ev["t_rel"]), float(ev["t_unix"])
    assert 0.4 < t_rel < 3.0
    assert t_unix == pytest.approx(meta["t0_unix"] + t_rel, abs=1e-3)
    assert not list(folder.glob("*.mp4")) and not list(folder.glob("*.mkv"))


def test_auto_connect_toggle_searching_does_not_block(app, tmp_path, no_boards):
    w = make_window(auto_connect_wii=False)
    try:
        assert w.force is None and not w.chk_auto.isChecked()
        assert w.wii_label.text() == "Not connected (optional)"
        t = time.perf_counter()
        w.chk_auto.setChecked(True)
        assert time.perf_counter() - t < 0.5
        src = w.force
        assert isinstance(src, WiiAutoConnect)
        assert pump_until(app, lambda: src.status == "searching")
        pump(app, 0.1)
        assert "waiting for a paired board" in w.wii_label.text()
        assert "Waiting for the Wii Balance Board" in w.streams_label.text()

        # Nothing to record yet (no camera, board not connected): refused with a message
        w.out_dir.setText(str(tmp_path / "rec"))
        w.b_rec.setChecked(True)
        assert not w.recorder.recording and not w.b_rec.isChecked()
        assert w.warnings and "Nothing to record" in w.warnings[-1]
        assert not (tmp_path / "rec").exists()

        # Off: the plug-and-play reader stops (quickly)
        t = time.perf_counter()
        w.chk_auto.setChecked(False)
        assert time.perf_counter() - t < 1.0
        assert w.force is None and not src.running

        # Manual choices never leave two sources running
        w.chk_auto.setChecked(True)
        auto = w.force
        w.connect_sim()
        assert not w.chk_auto.isChecked() and isinstance(w.force, SimulatedBoard)
        assert not auto.running
        w.chk_auto.setChecked(True)  # replaces the simulator
        assert isinstance(w.force, WiiAutoConnect)
        pump(app, 0.2)
        w.disconnect_wii()
        assert w.force is None and not w.chk_auto.isChecked()
        pump(app, 0.1)
        assert w.wii_label.text() == "Not connected (optional)"
    finally:
        w.close()


def test_board_appears_and_drops_during_recording(app, tmp_path, monkeypatch):
    bus = FakeBus(monkeypatch)  # fake hidapi: no board until bus.present = True
    video = tmp_path / "scene.mp4"
    write_scene_video(video)
    w = make_window(auto_connect_wii=False)
    try:
        w.wii_poll_interval_s = 0.05
        w.chk_auto.setChecked(True)
        w.cam_res.setCurrentText("Default")
        w.add_camera(str(video))
        pump(app, 0.3)
        w.out_dir.setText(str(tmp_path / "rec"))
        w.b_rec.setChecked(True)  # camera only: the board is not there yet
        assert w.recorder.recording, w.warnings
        pump(app, 0.2)
        assert w.recorder.counts["wii"] == 0

        bus.present = True  # board switched on: connected and added to the running recording
        assert pump_until(app, lambda: w.recorder.counts["wii"] > 20)
        assert "Auto-connect: connected" in w.wii_label.text()
        assert "Wii" in w.streams_label.text()

        bus.opened[-1].broken = True  # Bluetooth drop -> reconnects by itself
        assert pump_until(app, lambda: w.force.disconnects == 1 and w.force.connections == 2)
        n = w.recorder.counts["wii"]
        assert pump_until(app, lambda: w.recorder.counts["wii"] > n + 20)
        w.b_rec.setChecked(False)
        assert w.recorder.wait_post_processing(30)
        assert not w.warnings
    finally:
        w.close()

    folder = only_session(tmp_path / "rec")
    labels = [e["label"] for e in read_rows(folder / "events.csv")]
    assert labels == ["wii_connected", "wii_disconnected", "wii_connected"]
    assert len(read_rows(folder / "wii.csv")) > 40
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["has_wii"] is True and meta["force_source"]["type"] == "WiiAutoConnect"
    assert (folder / "cam0.mkv").exists()


def test_manual_connect_keeps_plug_and_play_for_selected_board(app, monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    w = make_window(auto_connect_wii=False)
    try:
        w.scan_wii()
        assert w.wii_devices.count() == 1
        w.connect_wii()  # Auto-connect off: a one-shot connection
        assert type(w.force) is BalanceBoardHID
        assert pump_until(app, lambda: w.force.connected)
        w.chk_auto.setChecked(True)  # replaces the one-shot connection: any board, reconnects
        assert type(w.force) is WiiAutoConnect and w.force.target_path is None
        assert pump_until(app, lambda: w.force.connected)
        w.connect_wii()  # Auto-connect on: plug and play, limited to the selected board
        assert isinstance(w.force, WiiAutoConnect) and w.force.target_path == bus.list_devices()[0]["path"]
        assert pump_until(app, lambda: w.force.connected)
        assert len(bus.opened) == 3 and all(d.closed for d in bus.opened[:2])
    finally:
        w.close()


def test_slot_exceptions_show_a_message(app, tmp_path, no_boards, monkeypatch):
    w = make_window(auto_connect_wii=False)
    try:
        w.connect_sim()
        assert pump_until(app, lambda: w.force.connected)
        w.out_dir.setText(str(tmp_path / "rec"))
        w.b_rec.setChecked(True)
        assert w.recorder.recording

        def boom(*a, **k):
            raise OSError("disk full")

        with monkeypatch.context() as m:
            m.setattr(w.recorder, "add_event", boom)
            w.mark_event()  # must not raise
        assert w.warnings and "disk full" in w.warnings[-1]
        w.b_rec.setChecked(False)
        assert w.recorder.wait_post_processing(30)
        w.mark_event()  # not recording: ignored
        assert len(w.warnings) == 1
    finally:
        w.close()
