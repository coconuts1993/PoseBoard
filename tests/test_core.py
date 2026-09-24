import time

import cv2
import numpy as np
import pytest

from poseboard.calibration import (CameraCalibration, CheckerboardSpec, IntrinsicCollector,
                                   extrinsics_from_checkerboard, load_calibrations,
                                   save_calibrations, save_pose2sim_toml)
from poseboard.fusion import fuse
from poseboard.geometry import (BoardGeometry, RigidTransform, kabsch, register_board,
                                rotate_board_frame)
from poseboard.pose.base import Pose2D, Pose3D
from poseboard.pose.com import center_of_mass
from poseboard.pose.mediapipe_backend import MP_NAMES, single_view_lift
from poseboard.pose.triangulation import triangulate_keypoints
from poseboard.wii import protocol as P
from poseboard.wii.device import ForceSample, SimulatedBoard


def make_cam(name, center, target=(0, 0, 0), size=(1280, 720), f=1000.0):
    """Build a camera looking at target (world Z up)."""
    center, target = np.asarray(center, float), np.asarray(target, float)
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ center
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
    return CameraCalibration(name, size, K, np.zeros(5), cv2.Rodrigues(R)[0].ravel(), t)


def board_transform():
    a = np.deg2rad(20)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return RigidTransform(R, np.array([0.8, 0.5, 0.053]))


# ------------------------------------------------------------------ Wii
def test_calibration_parse_and_kg():
    kg0, kg17, kg34 = [1000, 1100, 1200, 1300], [2700, 2800, 2900, 3000], [4400, 4500, 4600, 4700]
    data = bytes(4) + b"".join(v.to_bytes(2, "big") for v in kg0 + kg17 + kg34) + bytes(4)
    cal = P.Calibration.from_bytes(data)
    np.testing.assert_allclose(cal.to_kg(np.array(kg0)), 0)
    np.testing.assert_allclose(cal.to_kg(np.array(kg17)), 17)
    np.testing.assert_allclose(cal.to_kg(np.array(kg34)), 34)
    np.testing.assert_allclose(cal.to_kg(np.array([1850, 1950, 2050, 2150])), 8.5)
    np.testing.assert_allclose(cal.to_kg(np.array([3550, 3650, 3750, 3850])), 25.5)


def test_parse_sensor_report():
    vals = [0x1234, 0x0102, 0xABCD, 0x0FF0]
    rep = bytes([0x32, 0, 0]) + b"".join(v.to_bytes(2, "big") for v in vals) + bytes(11)
    np.testing.assert_array_equal(P.parse_sensor_raw(rep), vals)
    assert P.parse_sensor_raw(bytes([0x20] + [0] * 21)) is None


def test_output_reports():
    r = P.write_memory_report(0xA400F0, b"\x55")
    assert len(r) == P.OUTPUT_REPORT_LEN
    assert list(r[:7]) == [0x16, 0x04, 0xA4, 0x00, 0xF0, 0x01, 0x55]
    assert list(P.read_memory_report(0xA40020, 32)[:7]) == [0x17, 0x04, 0xA4, 0x00, 0x20, 0x00, 0x20]
    assert list(P.set_mode_report()[:3]) == [0x12, 0x04, 0x32]
    off, data, err = P.parse_read_data(bytes([0x21, 0, 0, 0xF0, 0x00, 0x20]) + bytes(range(16)))
    assert (off, len(data), err) == (0x20, 16, 0)


def test_cop():
    dx, dy = 0.433, 0.238
    assert P.center_of_pressure([10, 10, 10, 10], dx, dy) == (0, 0)
    x, y = P.center_of_pressure([20, 0, 0, 0], dx, dy)  # all load on TR
    assert x == pytest.approx(dx / 2) and y == pytest.approx(dy / 2)
    x, y = P.center_of_pressure([0, 0, 0, 20], dx, dy)  # BL
    assert x == pytest.approx(-dx / 2) and y == pytest.approx(-dy / 2)
    assert np.isnan(P.center_of_pressure([0.1, 0, 0, 0], dx, dy)[0])


def test_simulator_cop_matches_model():
    sim = SimulatedBoard()
    kg = sim.kg_at(1.234)
    x, y = P.center_of_pressure(kg, sim.sensor_dx_m, sim.sensor_dy_m)
    import math
    assert x == pytest.approx(0.06 * math.sin(2 * math.pi * 0.23 * 1.234), abs=1e-9)
    assert y == pytest.approx(0.04 * math.sin(2 * math.pi * 0.31 * 1.234 + 0.7), abs=1e-9)


def test_simulator_thread_and_tare():
    sim = SimulatedBoard(mass_kg=0.0)
    got = []
    sim.add_listener(got.append)
    sim.start()
    time.sleep(0.3)
    sim.do_tare(0.2)
    sim.stop()
    assert len(got) > 10
    assert np.all(np.abs(sim.tare) < 0.2)


# -------------------------------------------------------------- geometry
def test_register_board_single_camera_pnp():
    geo, T = BoardGeometry(), board_transform()
    cam = make_cam("c0", [0.8, -1.5, 1.6], [0.8, 0.5, 0.0])
    clicks = cam.project(T.apply(geo.landmarks())) + np.random.default_rng(0).normal(0, 0.3, (5, 2))
    pose = register_board(geo, [cam], [clicks])
    assert pose.method == "pnp"
    np.testing.assert_allclose(pose.board_to_world.t, T.t, atol=0.01)
    np.testing.assert_allclose(pose.board_to_world.R, T.R, atol=0.02)
    assert pose.reproj_error_px["c0"] < 1.0


def test_register_board_triangulation():
    geo, T = BoardGeometry(), board_transform()
    cams = [make_cam("a", [-0.5, -1.5, 1.8], [0.8, 0.5, 0]), make_cam("b", [2.2, -1.2, 1.7], [0.8, 0.5, 0])]
    clicks = [c.project(T.apply(geo.landmarks())) for c in cams]
    pose = register_board(geo, cams, clicks)
    assert pose.method == "triangulation"
    np.testing.assert_allclose(pose.board_to_world.t, T.t, atol=1e-6)
    np.testing.assert_allclose(pose.board_to_world.R, T.R, atol=1e-6)


def test_rotate_board_frame_180():
    pose = register_board(BoardGeometry(), [make_cam("c", [0, -1.5, 1.5])],
                          [make_cam("c", [0, -1.5, 1.5]).project(BoardGeometry().landmarks())])
    p2 = rotate_board_frame(pose, 180)
    np.testing.assert_allclose(p2.board_to_world.apply([0.1, 0.05, 0]),
                               pose.board_to_world.apply([-0.1, -0.05, 0]), atol=1e-9)


def test_kabsch():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(6, 3))
    T = board_transform()
    est = kabsch(src, T.apply(src))
    np.testing.assert_allclose(est.R, T.R, atol=1e-9)


def test_checkerboard_extrinsics_z_up():
    spec = CheckerboardSpec(9, 6, 40)
    cam = make_cam("c", [0.5, -1.2, 1.4], [0.16, 0.1, 0])
    # Render a synthetic checkerboard image
    sq = 40
    img_board = np.full(((spec.rows + 3) * sq, (spec.cols + 3) * sq), 255, np.uint8)
    for r in range(spec.rows + 1):
        for c in range(spec.cols + 1):
            if (r + c) % 2 == 0:
                y0, x0 = (r + 1) * sq, (c + 1) * sq
                img_board[y0:y0 + sq, x0:x0 + sq] = 0
    # Checkerboard pixels -> world (meters); the first inner corner is at pixel (2sq, 2sq)
    s = spec.square_mm / 1000 / sq
    src = np.array([[0, 0], [img_board.shape[1], 0], [img_board.shape[1], img_board.shape[0]],
                    [0, img_board.shape[0]]], float)
    world = np.column_stack([(src - 2 * sq) * s, np.zeros(4)])
    H = cv2.getPerspectiveTransform(src.astype(np.float32), cam.project(world).astype(np.float32))
    img = cv2.warpPerspective(img_board, H, cam.image_size, borderValue=255)
    cam2 = CameraCalibration("c", cam.image_size, cam.K, cam.dist)
    err = extrinsics_from_checkerboard(cam2, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), spec)
    assert err < 1.0
    np.testing.assert_allclose(cam2.center_world, cam.center_world, atol=0.01)


def test_checkerboard_world_frame_consistent_across_views():
    """Cameras at different positions (including opposite sides) must yield the same world frame."""
    from tests.synthetic import SPEC, render_scene
    for center in ([0.6, -0.9, 1.3], [1.6, 1.2, 1.3], [-0.8, 0.9, 1.4], [0.3, 1.5, 1.2]):
        cam = make_cam("x", center, [0.3, 0.15, 0])
        est = CameraCalibration("x", cam.image_size, cam.K, cam.dist)
        extrinsics_from_checkerboard(est, render_scene(cam), SPEC)
        np.testing.assert_allclose(est.center_world, center, atol=0.01)


def test_intrinsic_collector_requires_images():
    with pytest.raises(ValueError):
        IntrinsicCollector(CheckerboardSpec()).calibrate("c")


def test_calibration_io_roundtrip(tmp_path):
    cams = [make_cam("a", [0, -1, 1]), make_cam("b", [1, -1, 1])]
    save_calibrations(tmp_path / "c.json", cams)
    back = load_calibrations(tmp_path / "c.json")
    np.testing.assert_allclose(back[1].K, cams[1].K)
    save_pose2sim_toml(tmp_path / "Calib.toml", cams)
    back = load_calibrations(tmp_path / "Calib.toml")
    np.testing.assert_allclose(back[0].tvec, cams[0].tvec)
    np.testing.assert_allclose(back[1].R, cams[1].R, atol=1e-9)


# -------------------------------------------------------------------- pose
def standing_skeleton(offset=(0, 0, 0)):
    """Simplified standing skeleton (world coordinates, Z up), MediaPipe naming."""
    p = {n: np.full(3, np.nan) for n in MP_NAMES}
    for side, sx in (("left", 0.1), ("right", -0.1)):
        p[f"{side}_shoulder"] = [sx * 1.8, 0, 1.45]
        p[f"{side}_elbow"] = [sx * 2.0, 0, 1.15]
        p[f"{side}_wrist"] = [sx * 2.0, 0, 0.9]
        p[f"{side}_hip"] = [sx, 0, 0.95]
        p[f"{side}_knee"] = [sx, 0.02, 0.5]
        p[f"{side}_ankle"] = [sx, 0, 0.08]
        p[f"{side}_heel"] = [sx, -0.05, 0.03]
        p[f"{side}_foot_index"] = [sx, 0.15, 0.02]
        p[f"{side}_ear"] = [sx * 0.8, 0, 1.65]
        p[f"{side}_eye"] = [sx * 0.3, 0.08, 1.68]
    p["nose"] = [0, 0.1, 1.65]
    return np.array([p[n] for n in MP_NAMES], float) + np.asarray(offset)


def test_com_standing():
    kp = standing_skeleton((0.3, 0.2, 0))
    com = center_of_mass(kp, MP_NAMES)
    assert com[0] == pytest.approx(0.3, abs=1e-6)  # left-right symmetric
    assert 0.85 < com[2] < 1.1  # about 55% of body height


def test_com_other_naming_and_missing():
    names = ["LShoulder", "RShoulder", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "Nose"]
    kp = np.array([[0.18, 0, 1.45], [-0.18, 0, 1.45], [0.1, 0, .95], [-.1, 0, .95],
                   [.1, 0, .5], [-.1, 0, .5], [.1, 0, .08], [-.1, 0, .08], [0, .1, 1.65]])
    com = center_of_mass(kp, names)
    assert com is not None and 0.8 < com[2] < 1.15
    assert center_of_mass(kp[4:], names[4:]) is None


def test_triangulate_keypoints_and_single_view():
    kp = standing_skeleton((0.8, 0.5, 0.05))
    cams = [make_cam("a", [0.8, -2.5, 1.3], [0.8, 0.5, 0.9]), make_cam("b", [3.0, -1.0, 1.3], [0.8, 0.5, 0.9])]
    p2 = [c.project(kp) for c in cams]
    sc = [np.ones(len(kp))] * 2
    X, conf = triangulate_keypoints(cams, p2, sc)
    np.testing.assert_allclose(X, kp, atol=1e-6)

    body = kp - kp[MP_NAMES.index("left_hip")]  # hip as origin; any rotation works too
    lifted = single_view_lift(cams[0], Pose2D(p2[0], np.ones(len(kp))), body)
    np.testing.assert_allclose(lifted, kp, atol=1e-4)


def test_fuse_state():
    geo, T = BoardGeometry(), board_transform()
    cam = make_cam("c0", [0.8, -1.5, 1.6], [0.8, 0.5, 0.0])
    board = register_board(geo, [cam], [cam.project(T.apply(geo.landmarks()))])
    force = ForceSample(1.0, np.array([20, 20, 20, 20.0]), 80.0, (0.02, -0.01))
    kp = T.apply(standing_skeleton())  # standing at the board center
    st = fuse(board, force, Pose3D(1.0, list(MP_NAMES), kp, np.ones(len(kp))))
    np.testing.assert_allclose(st.cop_world, T.apply([0.02, -0.01, 0]), atol=1e-4)
    assert st.force_n == pytest.approx(80 * 9.80665)
    assert abs(st.com_board[0]) < 1e-3
    np.testing.assert_allclose(st.com_minus_cop, st.com_board[:2] - [0.02, -0.01])
