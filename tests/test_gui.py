"""Offscreen end-to-end GUI flow: camera (video file) → checkerboard extrinsics →
click the balance board → simulated Wii → record."""

import json
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from poseboard.geometry import BoardGeometry  # noqa: E402
from tests.synthetic import BOARD_T, SPEC, scene_camera, write_scene_video  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.01)


def test_full_flow(app, tmp_path):
    from poseboard.gui.app import MainWindow

    video = tmp_path / "scene.mp4"
    write_scene_video(video)
    w = MainWindow()
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
    w.out_dir.setText(str(tmp_path / "rec"))
    w.subject.setText("test")
    pump(app, 0.3)
    w.b_rec.setChecked(True)
    pump(app, 1.0)
    w.grab().save(str(tmp_path / "screenshot.png"))
    w.b_rec.setChecked(False)
    folders = list((tmp_path / "rec").iterdir())
    assert len(folders) == 1
    meta = json.loads((folders[0] / "session.json").read_text(encoding="utf-8"))
    assert meta["board_pose"]["method"] == "pnp"
    assert meta["samples"]["wii"] > 50
    summary = json.loads((folders[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["cop"]["samples"] > 50

    # Save/load config
    proj = tmp_path / "proj.json"
    from unittest import mock
    with mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(proj), "")):
        w.save_project()
    w.board = None
    w.load_project(str(proj))
    np.testing.assert_allclose(w.board.board_to_world.t, BOARD_T.t, atol=0.02)
    if os.environ.get("POSEBOARD_SCREENSHOT"):
        w.grab().save(os.environ["POSEBOARD_SCREENSHOT"])
    w.close()
