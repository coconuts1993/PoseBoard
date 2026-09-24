"""Board registration (IPPE branch, floor fit, mirrored clicks, mixed extrinsics), TRC/CSV
readers, COM keypoint aliases, calibration I/O and the pose backend's world frame."""

import csv
import logging

import cv2
import numpy as np
import pytest

from poseboard.analysis import export_trc, fuse_external, read_csv_columns
from poseboard.calibration import (CameraCalibration, CheckerboardSpec, IntrinsicCollector,
                                   check_world_checkerboard, find_checkerboard,
                                   load_calibrations, save_pose2sim_toml)
from poseboard.geometry import (BoardGeometry, BoardPose, RigidTransform, register_board,
                                rotate_board_frame, solve_board_pnp)
from poseboard.pose import mediapipe_backend as mpb
from poseboard.pose.base import Pose2D, Pose3D, check_pose3d
from poseboard.pose.com import center_of_mass
from poseboard.pose.external import read_keypoint_csv, read_trc
from poseboard.pose.mediapipe_backend import MP_NAMES, MediaPipePose
from tests.test_alignment import write_session
from tests.test_core import make_cam, standing_skeleton

GEO = BoardGeometry()
MODEL = GEO.landmarks()
MIRROR = [1, 0, 3, 2, 4]  # TR, TL, BL, BR, C: left and right swapped


def yaw_transform(deg=20.0, xy=(0.3, 0.2)):
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return RigidTransform(R, np.array([xy[0], xy[1], GEO.height_mm / 1000]))


def no_ext(cam):
    return CameraCalibration(cam.name, cam.image_size, cam.K, cam.dist)


# ------------------------------------------------------------------ board registration
def test_pnp_never_returns_the_flipped_ippe_solution():
    """Low, oblique camera + noisy clicks: plain IPPE picked a board tilted ~150° in ~14% of
    the cases; the branch closest to world +Z (or pointing up in the image) is always right."""
    T = yaw_transform()
    cam = make_cam("c", (2.5, -2.5, 1.0), (0.3, 0.2, 0))
    rng = np.random.default_rng(0)
    cos10 = np.cos(np.deg2rad(10))
    n_amb = 0
    for _ in range(300):
        k = cam.project(T.apply(MODEL)) + rng.normal(0, 2.0, (5, 2))
        free = register_board(GEO, [cam], [k], floor=False)
        assert free.up_world[2] > cos10
        assert free.tilt_deg < 10
        T_bc, amb = solve_board_pnp(no_ext(cam), MODEL, k, return_ambiguity=True)
        n_amb += amb
        assert T_bc.R[:, 2] @ (cam.R @ [0, 0, 1.0]) > cos10  # upright-camera prior, no extrinsics
    assert n_amb > 0  # close IPPE errors happen and are reported


def test_floor_fit_removes_tilt_bias_single_camera():
    T = yaw_transform()
    com = np.array([0.3, 0.2, 1.0])
    cam = make_cam("c", (1.5, -2.0, 1.6), (0.3, 0.2, 0))
    rng = np.random.default_rng(1)
    err_floor, err_free = [], []
    truth_b = T.inverse().apply(com)
    for _ in range(200):
        k = cam.project(T.apply(MODEL)) + rng.normal(0, 1.0, (5, 2))
        b = register_board(GEO, [cam], [k])
        assert b.floor_constrained and b.method == "pnp" and b.world_camera is None
        np.testing.assert_allclose(b.up_world, [0, 0, 1], atol=1e-12)
        assert b.board_to_world.t[2] == pytest.approx(GEO.height_mm / 1000)
        assert b.tilt_deg is not None and b.reproj_error_px["c"] < 5
        err_floor.append(np.hypot(*(b.world_to_board.apply(com) - truth_b)[:2]))
        free = register_board(GEO, [cam], [k], floor=False)
        err_free.append(np.hypot(*(free.world_to_board.apply(com) - truth_b)[:2]))
    assert np.median(err_floor) < 0.006  # a few mm instead of 1-1.5 cm
    assert np.median(err_floor) < 0.5 * np.median(err_free)


def test_floor_fit_two_cameras_and_large_tilt_warning():
    T = yaw_transform(-35)
    cams = [make_cam("a", (-0.5, -1.5, 1.8), (0.3, 0.2, 0)), make_cam("b", (2.2, -1.2, 1.7), (0.3, 0.2, 0))]
    rng = np.random.default_rng(2)
    clicks = [c.project(T.apply(MODEL)) + rng.normal(0, 1.0, (5, 2)) for c in cams]
    b = register_board(GEO, cams, clicks)
    assert b.method == "triangulation" and b.floor_constrained
    np.testing.assert_allclose(b.up_world, [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(b.board_to_world.t, T.t, atol=0.01)
    assert not b.warnings
    # A board that is really tilted (not on the checkerboard floor): kept 6-DoF, with a warning
    a = np.deg2rad(20)
    tilt = RigidTransform(T.R @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]), T.t)
    b = register_board(GEO, cams, [c.project(tilt.apply(MODEL)) for c in cams])
    assert not b.floor_constrained and b.tilt_deg == pytest.approx(20, abs=0.5)
    assert b.warnings and "tilted" in b.warnings[0]


@pytest.mark.parametrize("mirror", [MIRROR, [3, 2, 1, 0, 4]])  # left/right and front/back
def test_mirrored_clicks_are_rejected(mirror):
    T = yaw_transform()
    cam = make_cam("c", (1.5, -2.0, 1.6), (0.3, 0.2, 0))
    k = cam.project(T.apply(MODEL))[mirror]
    with pytest.raises(ValueError, match="mirrored"):
        register_board(GEO, [cam], [k])
    with pytest.raises(ValueError, match="mirrored"):
        register_board(GEO, [no_ext(cam)], [k])
    cams = [cam, make_cam("d", (-0.8, -1.4, 1.7), (0.3, 0.2, 0))]
    with pytest.raises(ValueError, match="mirrored"):
        register_board(GEO, cams, [c.project(T.apply(MODEL))[mirror] for c in cams])
    # The correct order still works everywhere
    assert register_board(GEO, [no_ext(cam)], [cam.project(T.apply(MODEL))]).method == "pnp"


def test_cameras_without_extrinsics_are_not_mixed_in():
    T = yaw_transform()
    cam0 = make_cam("cam0", (1.5, -2.0, 1.6), (0.3, 0.2, 0))
    cam1 = make_cam("cam1", (-1.0, -1.5, 1.5), (0.3, 0.2, 0))
    k0, k1 = cam0.project(T.apply(MODEL)), cam1.project(T.apply(MODEL))
    b = register_board(GEO, [cam0, no_ext(cam1)], [k0, k1])
    assert set(b.reproj_error_px) == {"cam0"} and b.reproj_error_px["cam0"] < 0.5
    assert any("cam1" in n for n in b.notes) and b.world_camera is None
    np.testing.assert_allclose(b.board_to_world.t, T.t, atol=1e-6)
    # No extrinsics at all: the first camera defines the world, errors only for it
    b = register_board(GEO, [no_ext(cam1), no_ext(cam0)], [k1, k0])
    assert b.world_camera == "cam1" and set(b.reproj_error_px) == {"cam1"}
    assert b.reproj_error_px["cam1"] < 0.5 and not b.floor_constrained


def test_rotate_board_frame_only_180_and_pose_roundtrip():
    cam = make_cam("c", (1.5, -2.0, 1.6), (0.3, 0.2, 0))
    b = register_board(GEO, [cam], [cam.project(yaw_transform().apply(MODEL))])
    with pytest.raises(ValueError):
        rotate_board_frame(b, 90)
    r = rotate_board_frame(b, 180)
    assert r.floor_constrained and r.tilt_deg == b.tilt_deg
    back = BoardPose.from_dict(r.to_dict())
    np.testing.assert_allclose(back.board_to_world.R, r.board_to_world.R)
    assert back.floor_constrained and back.notes == r.notes
    old = {"board_to_world": b.board_to_world.to_dict(), "method": "pnp", "reproj_error_px": {}}
    assert BoardPose.from_dict(old).world_camera is None  # sessions/projects of older versions


# ------------------------------------------------------------------ TRC / CSV readers
def test_read_trc_keeps_columns_with_missing_markers(tmp_path):
    p = tmp_path / "x.trc"
    p.write_text("PathFileType\t4\t(X/Y/Z)\tx.trc\nDataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\n"
                 "30\t30\t2\t4\tm\nFrame#\tTime\tA\t\t\tB\t\t\tC\t\t\tD\n"
                 "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\tX3\tY3\tZ3\tX4\tY4\tZ4\n"
                 "1\t0.000\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\n"
                 "2\t0.033\t1\t2\t3\t\t\t\t7\t8\t9\t\t\t\n", encoding="utf-8")
    t, names, arr, _ = read_trc(p)
    assert names == ["A", "B", "C", "D"] and len(t) == 2
    np.testing.assert_allclose(arr[1, 0], [1, 2, 3])
    assert np.isnan(arr[1, 1]).all() and np.isnan(arr[1, 3]).all()
    np.testing.assert_allclose(arr[1, 2], [7, 8, 9])  # not shifted into B's slot


def test_export_trc_roundtrip_with_occluded_points(tmp_path):
    names = ["a", "b", "c"]
    rows = [[0.0, 0.0], [0.1, 0.1]]
    with open(tmp_path / "pose3d.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "t_rel", "t_unix"] + [f"{n}_{a}" for n in names for a in "xyz"])
        w.writerow(rows[0] + [1] + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        w.writerow(rows[1] + [2] + [0.1, 0.2, 0.3, "", "", "", "", "", ""])  # b and c missing
    t, got, arr, _ = read_trc(export_trc(tmp_path))
    assert got == names
    np.testing.assert_allclose(arr[0, 2], [0.7, 0.8, 0.9])
    np.testing.assert_allclose(arr[1, 0], [0.1, 0.2, 0.3])
    assert np.isnan(arr[1, 1:]).all()


def write_csv(path, head, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    return path


def test_keypoint_csv_time_columns(tmp_path):
    kp = standing_skeleton()
    xyz = [f"{n}_{a}" for n in MP_NAMES for a in "xyz"]
    vals = [f"{v:.4f}" for v in kp.ravel()]
    p = write_csv(tmp_path / "a.csv", ["Time (s)"] + xyz, [["0.5"] + vals, ["1.0"] + vals])
    t, names, arr = read_keypoint_csv(p)
    np.testing.assert_allclose(t, [0.5, 1.0])
    assert names == list(MP_NAMES)
    np.testing.assert_allclose(arr[0], kp, atol=1e-4)
    with pytest.raises(ValueError, match="time column"):
        read_keypoint_csv(write_csv(tmp_path / "b.csv", ["frame"] + xyz, [["1"] + vals]))

    t0, off = 500.0, 1.75e9
    folder = write_session(tmp_path / "s", t0, off)
    unix_only = write_csv(tmp_path / "u.csv", ["t_unix"] + xyz,
                          [[f"{t0 + off + 3.0:.6f}"] + vals, [f"{t0 + off + 4.0:.6f}"] + vals])
    fused = read_csv_columns(fuse_external(folder, unix_only, time_base="unix"))
    np.testing.assert_allclose(fused["t_rel"], [3.0, 4.0], atol=1e-5)
    with pytest.raises(ValueError, match="t_unix"):
        fuse_external(folder, unix_only)  # time_base="rel" would misread Unix seconds


# ------------------------------------------------------------------ center of mass
H36M = {"Hip": (0, 0, .95), "RHip": (.1, 0, .95), "RKnee": (.1, .05, .5), "RFoot": (.1, 0, .08),
        "LHip": (-.1, 0, .95), "LKnee": (-.1, .05, .5), "LFoot": (-.1, 0, .08), "Spine": (0, 0, 1.2),
        "Thorax": (0, 0, 1.4), "Neck": (0, 0, 1.5), "Head": (0, 0, 1.65), "LShoulder": (-.18, 0, 1.45),
        "LElbow": (-.2, 0, 1.15), "LWrist": (-.2, 0, .9), "RShoulder": (.18, 0, 1.45),
        "RElbow": (.2, 0, 1.15), "RWrist": (.2, 0, .9)}


def test_com_h36m_and_bvh_foot_is_the_ankle():
    names, kp = list(H36M), np.array(list(H36M.values()))
    com, segs = center_of_mass(kp, names, return_segments=True)
    ankle = center_of_mass(kp, [n.replace("Foot", "Ankle") for n in names])
    np.testing.assert_allclose(com, ankle)
    assert {"l_shank", "r_shank"} <= set(segs)
    bvh = ["LeftShoulder", "RightShoulder", "LeftUpLeg", "LeftHip", "RightHip", "LeftKnee",
           "RightKnee", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase", "Head"]
    pts = {"LeftShoulder": (-.18, 0, 1.45), "RightShoulder": (.18, 0, 1.45), "LeftUpLeg": (-.1, 0, .95),
           "LeftHip": (-.1, 0, .95), "RightHip": (.1, 0, .95), "LeftKnee": (-.1, 0, .5),
           "RightKnee": (.1, 0, .5), "LeftFoot": (-.1, 0, .08), "RightFoot": (.1, 0, .08),
           "LeftToeBase": (-.1, .15, .02), "RightToeBase": (.1, .15, .02), "Head": (0, 0, 1.65)}
    _, segs = center_of_mass(np.array([pts[n] for n in bvh]), bvh, return_segments=True)
    assert {"l_shank", "r_shank", "l_foot", "r_foot"} <= set(segs)


def test_com_minimum_keypoints_and_log(caplog):
    names = ["LShoulder", "RShoulder", "LHip", "RHip", "LKnee", "RKnee", "LEar", "REar"]
    kp = np.array([[-.18, 0, 1.45], [.18, 0, 1.45], [-.1, 0, .95], [.1, 0, .95],
                   [-.1, 0, .5], [.1, 0, .5], [-.07, 0, 1.62], [.07, 0, 1.62]])
    with caplog.at_level(logging.WARNING, logger="poseboard.pose.com"):
        assert center_of_mass(kp[:4], names[:4]) is None  # shoulders + hips: only ~50 %
        assert center_of_mass(kp[[0, 1, 2, 3, 6, 7]], names[:4] + names[6:]) is None  # + head
        assert center_of_mass(kp[:6], names[:6]) is not None  # + both knees
        assert center_of_mass(kp[[0, 1, 2, 3, 4, 6, 7]], names[:5] + names[6:]) is not None
    assert "No COM" in caplog.text and "60%" in caplog.text


# ------------------------------------------------------------------ calibration
def test_pose2sim_export_skips_uncalibrated_and_zero_pose_import(tmp_path):
    a = make_cam("a", [0, -1, 1])
    a.dist = np.array([0.1, -0.05, 0, 0, 0.4])
    b = CameraCalibration("b", (1280, 720), a.K.copy(), np.zeros(5))
    warnings = save_pose2sim_toml(tmp_path / "Calib.toml", [a, b])
    assert any("b" in w and "extrinsics" in w for w in warnings) and any("k3" in w for w in warnings)
    back = load_calibrations(tmp_path / "Calib.toml")
    assert [c.name for c in back] == ["a"] and back[0].has_extrinsics
    with pytest.raises(ValueError):
        save_pose2sim_toml(tmp_path / "none.toml", [b])
    (tmp_path / "zero.toml").write_text(
        '[cam_01]\nname = "z"\nsize = [ 640.0, 480.0,]\nmatrix = [ [ 500.0, 0.0, 320.0,], '
        '[ 0.0, 500.0, 240.0,], [ 0.0, 0.0, 1.0,],]\ndistortions = [ 0.0, 0.0, 0.0, 0.0,]\n'
        'rotation = [ 0.0, 0.0, 0.0,]\ntranslation = [ 0.0, 0.0, 0.0,]\nfisheye = false\n',
        encoding="utf-8")
    assert not load_calibrations(tmp_path / "zero.toml")[0].has_extrinsics


def test_intrinsic_calibration_fixes_k3():
    spec = CheckerboardSpec(9, 6, 30)
    truth = CameraCalibration("t", (1280, 720), np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]),
                              np.array([0.12, -0.2, 0, 0, 0.0]))
    col = IntrinsicCollector(spec, image_size=(1280, 720))
    obj = spec.object_points()
    rng = np.random.default_rng(0)
    for _ in range(8):
        rvec = rng.normal(0, 0.3, 3)
        tvec = np.array([rng.uniform(-0.15, 0), rng.uniform(-0.1, 0), rng.uniform(0.5, 0.8)])
        img, _ = cv2.projectPoints(obj, rvec, tvec, truth.K, truth.dist)
        col.corners.append(img.reshape(-1, 2))
    cal = col.calibrate("t")
    assert cal.dist[4] == 0.0 and cal.intrinsic_rms < 0.1


def test_world_checkerboard_must_be_asymmetric():
    assert check_world_checkerboard(CheckerboardSpec(9, 6), 3) is None
    assert "symmetric" in check_world_checkerboard(CheckerboardSpec(8, 6), 1)
    with pytest.raises(ValueError, match="symmetric"):
        check_world_checkerboard(CheckerboardSpec(7, 7), 2)


def test_preview_checkerboard_detection_downscaled():
    from tests.synthetic import SPEC, render_scene
    img = render_scene()
    full = find_checkerboard(img, SPEC)
    small = find_checkerboard(img, SPEC, fast=True, max_width=640)
    assert full is not None and small is not None
    assert np.abs(small - full).max() < 2.0  # scaled back to full-resolution pixels


# ------------------------------------------------------------------ pose backend
class FakeMediaPipe(MediaPipePose):
    """MediaPipePose without a model: detect() returns a projected skeleton for chosen cameras."""

    def __init__(self, truth, seen):
        self.min_score, self.truth, self.seen = 0.5, truth, seen

    def detect(self, cam_name, t, image):
        if cam_name not in self.seen:
            return None
        cam = self.cams[cam_name]
        body = self.truth - self.truth[MP_NAMES.index("left_hip")]
        return Pose2D(cam.project(self.truth), np.ones(len(MP_NAMES))), body


def test_mediapipe_never_mixes_camera_frames():
    truth = standing_skeleton((0.3, 0.2, 0.05))
    c0 = make_cam("cam0", (0.3, -2.5, 1.3), (0.3, 0.2, 0.9))
    c1 = make_cam("cam1", (2.5, -1.0, 1.3), (0.3, 0.2, 0.9))
    frames = {"cam0": (1.0, np.zeros((4, 4, 3), np.uint8)), "cam1": (1.2, np.zeros((4, 4, 3), np.uint8))}
    # cam0 has extrinsics but misses the person; cam1 (no extrinsics) sees it: no pose, since
    # cam1's camera frame is not the checkerboard world frame
    est = FakeMediaPipe(truth, seen={"cam1"})
    est.cams = {"cam0": c0, "cam1": c1}
    assert est.process(frames, {"cam0": c0, "cam1": no_ext(c1)}) is None
    est.seen = {"cam0", "cam1"}
    pose = est.process(frames, {"cam0": c0, "cam1": no_ext(c1)})
    np.testing.assert_allclose(pose.keypoints, truth, atol=1e-3)
    assert pose.t == 1.0  # the time of the frame actually used
    # Without extrinsics only the world camera (the board's camera) is used
    cams = {"cam0": no_ext(c0), "cam1": no_ext(c1)}
    est.world_camera = "cam1"
    est.seen = {"cam0"}
    assert est.process(frames, cams) is None
    est.seen = {"cam0", "cam1"}
    pose = est.process(frames, cams)
    np.testing.assert_allclose(pose.keypoints, c1.world_to_camera(truth), atol=1e-3)
    assert pose.t == 1.2


def test_model_download_error_names_url_and_path(tmp_path, monkeypatch):
    monkeypatch.setattr(mpb, "MODEL_DIR", tmp_path / "models")

    def fail(url, timeout=None):
        assert timeout is not None  # never blocks forever
        raise OSError("blocked")

    monkeypatch.setattr(mpb.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError) as e:
        mpb.ensure_model("lite")
    msg = str(e.value)
    assert mpb.MODEL_URLS["lite"] in msg and str(tmp_path / "models" / "pose_landmarker_lite.task") in msg
    assert not list((tmp_path / "models").glob("*.part"))


def test_check_pose3d_rejects_bad_plugin_output():
    names = ["a", "b"]
    assert check_pose3d(Pose3D(1, names, np.zeros((2, 3)), [1, 1])).t == 1.0
    with pytest.raises(ValueError, match="t"):
        check_pose3d(Pose3D(None, names, np.zeros((2, 3)), np.ones(2)))
    with pytest.raises(ValueError, match="shape"):
        check_pose3d(Pose3D(1.0, names, np.zeros((2, 2)), np.ones(2)))
    with pytest.raises(ValueError, match="scores"):
        check_pose3d(Pose3D(1.0, names, np.zeros((2, 3)), np.ones(3)))
