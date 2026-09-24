"""Offscreen GUI: calibrations kept across resolution changes, project resolution, corner check,
tare handling, camera naming/removal, board recomputation, lost cameras, checkerboard parity."""

import csv
import json
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from poseboard.calibration import CameraCalibration  # noqa: E402
from poseboard.geometry import BoardGeometry, BoardPose, RigidTransform  # noqa: E402
from poseboard.wii.device import ForceSource  # noqa: E402
from tests.synthetic import BOARD_T, SPEC, scene_camera, write_scene_video  # noqa: E402
from tests.test_gui_optional import make_window, only_session, pump, pump_until  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def no_boards(monkeypatch):
    from poseboard.wii.device import BalanceBoardHID

    monkeypatch.setattr(BalanceBoardHID, "list_devices", staticmethod(lambda: []))


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "scene.mp4"
    write_scene_video(p)  # 1280x720
    return str(p)


def events(folder):
    with open(folder / "events.csv", newline="", encoding="utf-8-sig") as f:
        return [r["label"] for r in csv.DictReader(f)]


class ManualBoard(ForceSource):
    """Force source whose four sensor loads (TR, BR, TL, BL) the test sets."""

    name = "manual"

    def __init__(self):
        super().__init__()
        self.kg = np.zeros(4)

    @property
    def device_key(self):
        return "manual"

    def _run(self):
        self._set_state("connected")
        while not self._stop.is_set():
            self._emit_kg(time.perf_counter(), self.kg.copy())
            time.sleep(0.01)


def test_calibration_is_kept_when_the_resolution_differs(app, tmp_path, no_boards, video):
    w = make_window(auto_connect_wii=False)
    try:
        w.cam_res.setCurrentText("Default")
        w.add_camera(video)
        truth = scene_camera()
        calib = CameraCalibration("cam0", (1920, 1080), truth.K * 1.5, np.zeros(5), truth.rvec,
                                  truth.tvec, intrinsic_rms=0.3)
        calib.K[2, 2] = 1.0
        w.calibs["cam0"] = calib
        assert pump_until(app, lambda: "cam0" in w._calib_mismatch)
        pump(app, 0.2)
        assert w.calibs["cam0"] is calib and calib.has_extrinsics  # never overwritten
        live = w._live_calibs["cam0"]  # approximate intrinsics, but the calibrated camera pose
        assert live.image_size == (1280, 720) and live is not calib and live.intrinsic_rms is None
        np.testing.assert_allclose(live.center_world, calib.center_world)
        assert w.warn_label.isVisible() and "1920x1080" in w.warn_label.text()

        proj = tmp_path / "p.json"
        from unittest import mock
        with mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(proj), "")):
            w.save_project()
        data = json.loads(proj.read_text(encoding="utf-8"))
        assert data["cameras"][0]["width"] is None  # "Default" was requested
        assert data["calibrations"][0]["image_size"] == [1920, 1080]
    finally:
        w.close()

    # Opening a project uses the saved resolution, not whatever the Resolution box shows
    data["cameras"][0].update(width=1280, height=720)
    data["calibrations"][0]["image_size"] = [1280, 720]
    proj.write_text(json.dumps(data), encoding="utf-8")
    w = make_window(auto_connect_wii=False)
    try:
        w.cam_res.setCurrentText("640x480")
        w.load_project(str(proj))
        assert w.streams["cam0"].req[:2] == (1280, 720)
        pump(app, 0.3)
        assert not w._calib_mismatch and w._live_calibs["cam0"].has_extrinsics
        assert not w.warn_label.isVisible()
        # An older project without the resolution: the calibration's size is used
        del data["cameras"][0]["width"], data["cameras"][0]["height"]
        proj.write_text(json.dumps(data), encoding="utf-8")
        w2 = make_window(auto_connect_wii=False)
        w2.load_project(str(proj))
        assert w2.streams["cam0"].req[:2] == (1280, 720)
        w2.close()
        w.load_project(str(tmp_path / "missing.json"))  # shown as a message, no exception
        assert w.warnings and "missing.json" in w.warnings[-1]
    finally:
        w.close()


def test_corner_check(app, no_boards):
    w = make_window(auto_connect_wii=False)
    try:
        board = ManualBoard()
        w._set_force(board)
        assert pump_until(app, lambda: board.latest() is not None)
        pump(app, 0.3)

        def press(sensor_index, expect):
            board.kg[:] = 0
            pump(app, 0.4)
            w.start_corner_check()
            board.kg[sensor_index] = 15.0
            assert pump_until(app, lambda: expect in w.corner_label.text(), 3.0), w.corner_label.text()

        press(2, "front-left")  # TL, no board registered: just names the corner
        w.board = BoardPose(RigidTransform(np.eye(3), np.zeros(3)), "pnp", {})
        press(2, "OK")
        assert w.board_rotation == 0
        press(1, "rotated 180")  # BR responded: front/back swapped -> fixed
        assert w.board_rotation == 180 and w.board.board_to_world.R[0, 0] == pytest.approx(-1)
        press(2, "OK")  # the corner now labelled TL is the TL sensor
        press(0, "Redo the clicks")  # TR: mirrored / turned 90°
    finally:
        w.close()


def test_tare_is_kept_on_reconnect_and_locked_while_recording(app, tmp_path, no_boards):
    w = make_window(auto_connect_wii=False)
    try:
        w.connect_sim()
        assert pump_until(app, lambda: len(w.force.recent(1.0)) > 50)
        w.tare()
        tare = w.force.tare.copy()
        assert tare.sum() > 50
        w.connect_sim()  # a new source object for the same (simulated) board
        np.testing.assert_allclose(w.force.tare, tare)
        assert pump_until(app, lambda: w.force.latest() is not None)
        w.out_dir.setText(str(tmp_path / "rec"))
        w.b_rec.setChecked(True)
        assert w.recorder.recording
        assert not w.b_tare.isEnabled() and not w.pose_backend.isEnabled()
        w.tare()  # refused while recording
        assert "not possible during a recording" in w.warnings[-1]
        np.testing.assert_allclose(w.force.tare, tare)
        pump(app, 0.3)
        w.b_rec.setChecked(False)
        assert w.recorder.wait_post_processing(30)
        assert w.b_tare.isEnabled() and w.pose_backend.isEnabled()
    finally:
        w.close()
    meta = json.loads((only_session(tmp_path / "rec") / "session.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(meta["force_source_history"][0]["tare_kg"], tare)


def test_camera_names_and_removal(app, tmp_path, no_boards, video):
    w = make_window(auto_connect_wii=False)
    try:
        w.cam_res.setCurrentText("Default")
        w.add_camera(video)
        assert list(w.streams) == ["cam0"]
        w.cam_list.setCurrentRow(0)
        w.remove_camera()
        assert not w.streams
        w.add_camera(video)  # the same source gets its old name (and calibration) back
        assert list(w.streams) == ["cam0"]
        w.cam_source.setText(video)
        w.cam_name.setText("left cam")
        w.add_camera()  # the "Add Camera" button
        assert "left cam" in w.streams
        s = w.streams["left cam"]
        w.cam_list.setCurrentRow(1)
        w.remove_camera()  # a name with a space is removed properly
        assert "left cam" not in w.streams and not s.running
        # Clicks of a camera that is not running are not used
        w.clicks = {"ghost": [np.array([10.0 * i, 5.0]) for i in range(5)]}
        w.calibs["ghost"] = scene_camera()
        w.compute_board()
        assert w.board is None and "running camera" in w.warnings[-1]
    finally:
        w.close()


def test_board_is_recomputed_after_new_extrinsics(app, no_boards, video):
    w = make_window(auto_connect_wii=False)
    try:
        w.cam_res.setCurrentText("Default")
        w.add_camera(video)
        pump(app, 0.3)
        truth = scene_camera()
        w.calibs["cam0"] = CameraCalibration("cam0", truth.image_size, truth.K.copy(), truth.dist.copy())
        w.b_click.setChecked(True)
        for p in truth.project(BOARD_T.apply(BoardGeometry().landmarks())):
            w._on_video_click(float(p[0]), float(p[1]), 1)
        w.compute_board()
        assert w.board.world_camera == "cam0"  # no extrinsics yet: camera frame
        w.rotate_board(180)
        w.cb_cols.setValue(SPEC.cols)
        w.cb_rows.setValue(SPEC.rows)
        w.cb_square.setValue(SPEC.square_mm)
        w.compute_extrinsics_all()
        b = w.board
        assert b.world_camera is None and w.board_stale is None and b.floor_constrained
        np.testing.assert_allclose(b.board_to_world.t, BOARD_T.t, atol=0.02)
        assert w.board_rotation == 180  # kept
        np.testing.assert_allclose(b.board_to_world.R[:, 0], -BOARD_T.R[:, 0], atol=0.02)
        assert not w.warnings
    finally:
        w.close()


def test_lost_camera_is_shown_and_logged(app, tmp_path, no_boards, video, monkeypatch):
    import poseboard.gui.app as gui_app

    monkeypatch.setattr(gui_app, "CAMERA_LOST_S", 0.3)
    w = make_window(auto_connect_wii=False)
    try:
        w.cam_res.setCurrentText("Default")
        w.add_camera(video)
        pump(app, 0.3)
        w.out_dir.setText(str(tmp_path / "rec"))
        w.b_rec.setChecked(True)
        assert w.recorder.recording
        pump(app, 0.3)
        w.streams["cam0"].stop()  # unplugged: the stream stays in the list without new frames
        assert pump_until(app, lambda: "cam0" in w._cam_lost)
        pump(app, 0.1)
        assert "no new frames" in w.warn_label.text()
        w.b_rec.setChecked(False)
        assert w.recorder.wait_post_processing(30)
    finally:
        w.close()
    assert events(only_session(tmp_path / "rec")) == ["camera_lost cam0"]


def test_symmetric_checkerboard_refused_for_two_cameras(app, no_boards, video):
    w = make_window(auto_connect_wii=False)
    try:
        w.cam_res.setCurrentText("Default")
        w.add_camera(video)
        w.add_camera(video, "cam1")
        pump(app, 0.3)
        w.cb_cols.setValue(8)
        w.cb_rows.setValue(6)
        w.compute_extrinsics_all()
        assert w.warnings and "symmetric" in w.warnings[-1]
        assert not any(c.has_extrinsics for c in w.calibs.values())
    finally:
        w.close()
