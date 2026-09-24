"""Offscreen end-to-end GUI flow: camera (video file) → checkerboard extrinsics →
click the balance board → simulated Wii → record (with an event marker).

Set POSEBOARD_SCREENSHOT=<file.png> to save screenshots: the Record tab while recording, and
the Devices tab (<file>_devices.png) at the end."""

import json
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from poseboard.geometry import BoardGeometry  # noqa: E402
from poseboard.wii.device import BalanceBoardHID  # noqa: E402
from tests.synthetic import BOARD_T, SPEC, scene_camera, write_scene_video  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.01)


def test_full_flow(app, tmp_path, monkeypatch):
    from poseboard.gui.app import MainWindow

    # Auto-connect starts with the window; make sure no real board on this machine interferes
    monkeypatch.setattr(BalanceBoardHID, "list_devices", staticmethod(lambda: []))
    shot = os.environ.get("POSEBOARD_SCREENSHOT")
    video = tmp_path / "scene.mp4"
    write_scene_video(video)
    w = MainWindow()
    assert w.chk_auto.isChecked()
    w.resize(1400, 850)
    w.show()
    w.cam_res.setCurrentText("Default")
    w.add_camera(str(video))
    pump(app, 0.3)
    assert "cam0" in w.streams

    # Use ground-truth intrinsics (as if already calibrated), then solve extrinsics from the checkerboard
    truth = scene_camera()
    from poseboard.calibration import CameraCalibration
    w.calibs["cam0"] = CameraCalibration("cam0", truth.image_size, truth.K.copy(), truth.dist.copy())
    w.cb_cols.setValue(SPEC.cols)
    w.cb_rows.setValue(SPEC.rows)
    w.cb_square.setValue(SPEC.square_mm)
    w.compute_extrinsics_all()
    cam = w.calibs["cam0"]
    assert cam.has_extrinsics and cam.extrinsic_rms < 1.0
    np.testing.assert_allclose(cam.center_world, truth.center_world, atol=0.02)

    # Click the 5 balance board points
    w.tabs.setCurrentIndex(2)
    w.b_click.setChecked(True)
    for p in truth.project(BOARD_T.apply(BoardGeometry().landmarks())):
        w._on_video_click(float(p[0]), float(p[1]), 1)
    w._on_video_click(0, 0, 1)  # the 6th click is ignored
    assert len(w.clicks["cam0"]) == 5
    w.compute_board()
    assert w.board is not None
    np.testing.assert_allclose(w.board.board_to_world.t, BOARD_T.t, atol=0.02)
    assert w.board.up_world[2] > 0.99

    # Simulated Wii + recording
    w.connect_sim()
    assert not w.chk_auto.isChecked()
    w.out_dir.setText(str(tmp_path / "rec"))
    w.subject.setText("test")
    pump(app, 0.3)
    w.tabs.setCurrentIndex(3)
    w.b_rec.setChecked(True)
    pump(app, 0.5)
    w.event_label.setText("sync")
    w.mark_event()
    pump(app, 0.5)
    w.grab().save(shot or str(tmp_path / "screenshot.png"))
    w.b_rec.setChecked(False)
    # Post-processing runs in the background: the window stays responsive meanwhile
    assert "Post-processing" in w.summary.toPlainText()
    assert w.recorder.wait_post_processing(30)
    pump(app, 0.1)
    assert '"cop"' in w.summary.toPlainText()
    folders = list((tmp_path / "rec").iterdir())
    assert len(folders) == 1
    meta = json.loads((folders[0] / "session.json").read_text(encoding="utf-8"))
    assert meta["board_pose"]["method"] == "pnp"
    assert meta["samples"]["wii"] > 50
    assert meta["has_wii"] and meta["has_video"] and meta["t0_unix"] > 1e9
    summary = json.loads((folders[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["cop"]["samples"] > 50
    assert summary["events_marked"] == 1

    # Save/load config
    proj = tmp_path / "proj.json"
    from unittest import mock
    with mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(proj), "")):
        w.save_project()
    w.board = None
    w.load_project(str(proj))
    np.testing.assert_allclose(w.board.board_to_world.t, BOARD_T.t, atol=0.02)
    if shot:
        w.tabs.setCurrentIndex(0)
        pump(app, 0.2)
        stem, ext = os.path.splitext(shot)
        w.grab().save(f"{stem}_devices{ext or '.png'}")
    w.close()
