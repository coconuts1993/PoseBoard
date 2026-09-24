"""The PoseAssess plugin template must load through PoseBoard's plugin loader and run."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from poseboard.analysis import POSE2SIM_YUP_TO_ZUP
from poseboard.calibration import CameraCalibration, approximate_calibration
from poseboard.pose.base import Pose3D, PoseEstimator
from poseboard.pose.com import KeypointIndex, center_of_mass
from poseboard.pose.external import load_plugin

TEMPLATE = Path(__file__).resolve().parents[1] / "plugins" / "poseassess_plugin_template.py"


def look_at_cam(name, center, target, size=(1280, 720), f=1000.0) -> CameraCalibration:
    """Pinhole camera at ``center`` looking at ``target`` (world Z up)."""
    center, target = np.asarray(center, float), np.asarray(target, float)
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
    return CameraCalibration(name, size, K, np.zeros(5), cv2.Rodrigues(R)[0].ravel(), -R @ center)


def standing_halpe(names) -> np.ndarray:
    """A rough upright person (meters, Z up) standing at (0.8, 0.5) on the floor."""
    base = {
        "Nose": (0.0, 0.08, 1.62), "LEye": (-0.03, 0.07, 1.65), "REye": (0.03, 0.07, 1.65),
        "LEar": (-0.07, 0.0, 1.62), "REar": (0.07, 0.0, 1.62),
        "LShoulder": (-0.18, 0.0, 1.42), "RShoulder": (0.18, 0.0, 1.42),
        "LElbow": (-0.22, 0.0, 1.12), "RElbow": (0.22, 0.0, 1.12),
        "LWrist": (-0.24, 0.02, 0.86), "RWrist": (0.24, 0.02, 0.86),
        "LHip": (-0.1, 0.0, 0.95), "RHip": (0.1, 0.0, 0.95),
        "LKnee": (-0.1, 0.02, 0.52), "RKnee": (0.1, 0.02, 0.52),
        "LAnkle": (-0.1, 0.0, 0.1), "RAnkle": (0.1, 0.0, 0.1),
        "Head": (0.0, 0.0, 1.75), "Neck": (0.0, 0.0, 1.5), "Hip": (0.0, 0.0, 0.95),
        "LBigToe": (-0.1, 0.18, 0.06), "RBigToe": (0.1, 0.18, 0.06),
        "LSmallToe": (-0.14, 0.15, 0.06), "RSmallToe": (0.14, 0.15, 0.06),
        "LHeel": (-0.1, -0.05, 0.06), "RHeel": (0.1, -0.05, 0.06),
    }
    return np.array([base[n] for n in names]) + np.array([0.8, 0.5, 0.0])


@pytest.fixture
def plugin() -> PoseEstimator:
    est = load_plugin(TEMPLATE)
    yield est
    est.close()


def test_template_loads_and_runs_without_detector(plugin):
    assert isinstance(plugin, PoseEstimator)
    frames = {"cam0": (time.perf_counter(), np.zeros((480, 640, 3), np.uint8))}
    cams = {"cam0": approximate_calibration("cam0", 640, 480)}
    out = plugin.process(frames, cams)
    assert out is None or isinstance(out, Pose3D)
    assert plugin.process({}, {}) is None


def test_template_keypoints_support_com(plugin):
    names = plugin.keypoint_names
    idx = KeypointIndex(names)
    for side in ("l", "r"):
        for part in ("shoulder", "hip", "knee", "ankle", "heel", "toe"):
            assert idx.find(side, part) is not None, (side, part)
    for a, b in plugin.skeleton:
        assert a in names and b in names
    com = center_of_mass(standing_halpe(names), names)
    assert com is not None and 0.8 < com[2] < 1.2

    coco = load_plugin(TEMPLATE, keypoint_set="coco17")
    assert len(coco.keypoint_names) == 17
    assert center_of_mass(standing_halpe(coco.keypoint_names), coco.keypoint_names) is not None


def test_template_triangulates_2d_detections(plugin):
    truth = standing_halpe(plugin.keypoint_names)
    cams = {"cam0": look_at_cam("cam0", [0.0, -2.0, 1.4], [0.8, 0.5, 0.9]),
            "cam1": look_at_cam("cam1", [2.4, -1.5, 1.4], [0.8, 0.5, 0.9])}
    frames = {n: (10.0 + i * 0.01, np.zeros((720, 1280, 3), np.uint8)) for i, n in enumerate(cams)}

    def fake_detect(cam_name, t, image, cam):
        return {"kp2d": cam.project(truth), "scores": np.full(len(truth), 0.9)}

    plugin.detect_per_camera = fake_detect
    pose = plugin.process(frames, cams)
    assert isinstance(pose, Pose3D)
    np.testing.assert_allclose(pose.keypoints, truth, atol=1e-6)
    assert pose.t == pytest.approx(10.005)
    assert set(pose.per_camera_2d) == {"cam0", "cam1"}

    # Single camera: "camera"-frame 3D is mapped to the world with the extrinsics.
    cam = cams["cam0"]
    one = {"cam0": frames["cam0"]}
    plugin.detect_per_camera = lambda n, t, img, c: {
        "kp2d": c.project(truth), "kp3d": c.world_to_camera(truth), "kp3d_frame": "camera"}
    pose = plugin.process(one, {"cam0": cam})
    np.testing.assert_allclose(pose.keypoints, truth, atol=1e-9)

    # Single camera: body-centred 3D is placed into the world by PnP.
    plugin.detect_per_camera = lambda n, t, img, c: {
        "kp2d": c.project(truth), "kp3d": truth - truth.mean(0), "kp3d_frame": "body"}
    pose = plugin.process(one, {"cam0": cam})
    np.testing.assert_allclose(pose.keypoints, truth, atol=1e-4)

    # Single camera without 3D: no pose, but no crash.
    plugin.detect_per_camera = lambda n, t, img, c: {"kp2d": c.project(truth)}
    assert plugin.process(one, {"cam0": cam}) is None


def test_template_multiview_3d_with_world_transform(plugin):
    truth = standing_halpe(plugin.keypoint_names)
    # Their frame is Y-up (Pose2Sim style): (X', Y', Z') = (Y, Z, X)
    theirs = truth[:, [1, 2, 0]]
    est = load_plugin(TEMPLATE, world_transform=POSE2SIM_YUP_TO_ZUP)
    est.detect_3d_multiview = lambda frames, cams: (theirs, np.ones(len(theirs)))
    frames = {"cam0": (5.0, np.zeros((10, 10, 3), np.uint8))}
    pose = est.process(frames, {})
    assert isinstance(pose, Pose3D)
    assert pose.t == 5.0
    np.testing.assert_allclose(pose.keypoints, truth, atol=1e-12)

    est.detect_3d_multiview = lambda frames, cams: (theirs[:5], np.ones(5))
    with pytest.raises(ValueError, match="KEYPOINT_SET"):
        est.process(frames, {})
