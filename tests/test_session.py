import json
import time

import cv2
import numpy as np

from poseboard.analysis import analyze_session, export_trc, fuse_external, read_csv_columns
from poseboard.camera import CameraStream
from poseboard.geometry import BoardGeometry, register_board
from poseboard.pose.base import Pose3D
from poseboard.pose.mediapipe_backend import MP_NAMES
from poseboard.session import SessionRecorder
from poseboard.wii.device import SimulatedBoard
from tests.test_core import board_transform, make_cam, standing_skeleton


def make_video(path, n=30, size=(320, 240)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    for i in range(n):
        img = np.zeros((size[1], size[0], 3), np.uint8)
        cv2.circle(img, (10 + 5 * i, 120), 10, (255, 255, 255), -1)
        w.write(img)
    w.release()


def test_record_and_analyze(tmp_path):
    video = tmp_path / "in.mp4"
    make_video(video)
    cam_stream = CameraStream("cam0", str(video))
    cam_stream.start()
    sim = SimulatedBoard()
    sim.start()
    geo, T = BoardGeometry(), board_transform()
    cam = make_cam("cam0", [0.8, -1.5, 1.6], [0.8, 0.5, 0.0])
    board = register_board(geo, [cam], [cam.project(T.apply(geo.landmarks()))])

    for _ in range(100):
        if cam_stream.latest() is not None:
            break
        time.sleep(0.01)

    rec = SessionRecorder(tmp_path / "rec")
    folder = rec.start(cams=[cam_stream], calibrations={"cam0": cam}, force=sim, board=board,
                       geometry=geo, pose_backend="test", subject="s1")
    kp = T.apply(standing_skeleton())
    for _ in range(15):
        rec.add_pose(Pose3D(time.perf_counter(), list(MP_NAMES), kp, np.ones(len(kp))))
        time.sleep(0.033)
    rec.stop()
    sim.stop()
    cam_stream.stop()

    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["samples"]["wii"] > 20 and meta["samples"]["pose"] == 15
    video = folder / meta["streams"][0]["video"]
    assert video.name == "cam0.mkv" and video.stat().st_size > 0
    cap = cv2.VideoCapture(str(video))
    n_frames = 0
    while cap.read()[0]:
        n_frames += 1
    cap.release()
    ts = read_csv_columns(folder / "cam0_timestamps.csv")
    assert len(ts["t"]) > 5
    assert n_frames == len(ts["t"])  # one timestamp row per stored frame
    np.testing.assert_allclose(ts["t_rel"], ts["t"] - meta["t0"], atol=2e-6)

    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert 60 < summary["mean_total_kg"] < 80
    assert summary["cop"]["path_length_mm"] > 0
    fused = read_csv_columns(folder / "fused.csv")
    assert np.isfinite(fused["total_kg"]).sum() >= 13
    # COP world coordinates should lie on the board surface (z ~ board height)
    assert np.nanmax(np.abs(fused["cop_z_world"] - T.t[2])) < 0.01

    trc = export_trc(folder)
    out = fuse_external(folder, trc)
    ext = read_csv_columns(out)
    np.testing.assert_allclose(np.nanmean(ext["com_x_board"]), np.nanmean(fused["com_x_board"]), atol=1e-3)
    assert analyze_session(folder)["pose_rate_hz"] > 10
