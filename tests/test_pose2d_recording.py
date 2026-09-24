"""Recording of the per-camera 2D keypoints: pose2d_<cam>.csv, OpenPose JSON (for Pose2Sim),
the frame numbers of the recorded videos, and the pose format in session.json."""

import csv
import json
import time

import cv2
import numpy as np
import pytest

from poseboard.calibration import approximate_calibration
from poseboard.camera import CameraStream
from poseboard.geometry import BoardGeometry
from poseboard.pose.base import MODE_2D_ONLY, Pose2D, Pose3D
from poseboard.pose.detectors.base import Person2D
from poseboard.pose.formats import FORMATS
from poseboard.pose.multiview import MultiViewEstimator
from poseboard.session import SessionRecorder, openpose_json
from tests.fakes import FakeDetector

NAMES = list(FORMATS["coco17"].names)
POSE2D_HEAD = ["t", "t_rel", "t_unix", "frame"] + [f"{n}_{a}" for n in NAMES
                                                  for a in ("x", "y", "score")]
PERSON_KEYS = ["person_id", "pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d",
               "hand_right_keypoints_2d", "pose_keypoints_3d", "face_keypoints_3d",
               "hand_left_keypoints_3d", "hand_right_keypoints_3d"]
SIZE = (320, 240)


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def check_openpose_schema(d, n_people):
    assert list(d) == ["version", "people"] and d["version"] == 1.3
    assert len(d["people"]) == n_people
    for p in d["people"]:
        assert list(p) == PERSON_KEYS and p["person_id"] == [-1]
        assert all(p[k] == [] for k in PERSON_KEYS[2:])
        assert all(isinstance(v, float) for v in p["pose_keypoints_2d"])


# ------------------------------------------------------------------ video with frame ids
def encode_id(i: int) -> np.ndarray:
    """A frame whose number (0..255) is written as 8 black/white blocks (survives MPEG-4)."""
    img = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
    for bit in range(8):
        x0 = 40 * bit
        img[40:120, x0 + 4:x0 + 36] = 255 if (i >> bit) & 1 else 0
    return img


def decode_id(img: np.ndarray) -> int:
    return sum(1 << bit for bit in range(8) if img[60:100, 40 * bit + 12:40 * bit + 28].mean() > 128)


def id_person(image, t=None, cam=None) -> list[Person2D]:
    """Fake detection: the keypoints encode the frame number read from the image."""
    i = decode_id(image)
    kp = np.column_stack([50.0 + 10 * np.arange(17) + i, 100.0 + np.arange(17)])
    return [Person2D(kp, np.full(17, 0.8))]


def test_openpose_json_function():
    p = Pose2D(np.array([[1.23456, 2.0], [np.nan, 5.0]]), np.array([0.5, 0.9]))
    d = openpose_json(p)
    check_openpose_schema(d, 1)
    assert d["people"][0]["pose_keypoints_2d"] == [1.235, 2.0, 0.5, 0.0, 0.0, 0.0]
    check_openpose_schema(openpose_json(None), 0)
    json.dumps(d, allow_nan=False)  # valid JSON: no NaN


def test_pose2d_csv_and_json_without_camera_streams(tmp_path):
    """Poses with explicit frame numbers (e.g. offline processing): columns, empty rows for
    cameras without a subject, JSON sets only when every camera has a frame number."""
    rec = SessionRecorder(tmp_path, save_openpose_json=True)
    folder = rec.start(cams=[], calibrations={}, force=None, board=None, geometry=BoardGeometry(),
                       pose_backend="Fake", pose_backend_key="fake", keypoint_format="coco17")
    t = rec.t0 + 0.5
    kp = np.column_stack([np.arange(17.0), np.arange(17.0) + 100])
    for i in range(3):
        per2d = {"camA": Pose2D(kp + i, np.full(17, 0.7), t=t + i, frame_index=10 + i)}
        frames = {"camA": (t + i, 10 + i), "camB": (t + i + 0.01, 20 + i if i != 2 else None)}
        rec.add_pose(Pose3D(t + i, NAMES, np.full((17, 3), np.nan), np.zeros(17), per2d,
                            mode=MODE_2D_ONLY, format_key="coco17", camera_frames=frames,
                            notes=["camB: no person detected"]))
    rec.stop(analyze=False)

    rows = read_rows(folder / "pose2d_camA.csv")
    assert rows[0] == POSE2D_HEAD and len(rows) == 4
    assert rows[1][3] == "10" and rows[3][3] == "12"
    assert rows[1][4:7] == ["0.000", "100.000", "0.7000"] and rows[2][4] == "1.000"
    assert float(rows[2][0]) == pytest.approx(t + 1) and float(rows[2][1]) == pytest.approx(
        t + 1 - rec.t0)
    rows_b = read_rows(folder / "pose2d_camB.csv")  # processed, but nobody found
    assert rows_b[0] == POSE2D_HEAD and [r[3] for r in rows_b[1:]] == ["20", "21", ""]
    assert all(v == "" for r in rows_b[1:] for v in r[4:])

    pose3d = read_rows(folder / "pose3d.csv")
    assert pose3d[0][-2:] == ["mode", "reproj_error_px"] and pose3d[1][-2:] == ["2d_only", ""]

    jdir = folder / "pose2d_json"
    a = sorted(p.name for p in (jdir / "camA").iterdir())
    b = sorted(p.name for p in (jdir / "camB").iterdir())
    # the third pose has no frame number for camB: its set is skipped in both folders
    assert a == ["camA_000000000010_keypoints.json", "camA_000000000011_keypoints.json"]
    assert b == ["camB_000000000020_keypoints.json", "camB_000000000021_keypoints.json"]
    da = json.loads((jdir / "camA" / a[1]).read_text(encoding="utf-8"))
    check_openpose_schema(da, 1)
    assert da["people"][0]["pose_keypoints_2d"][:6] == [1.0, 101.0, 0.7, 2.0, 102.0, 0.7]
    assert len(da["people"][0]["pose_keypoints_2d"]) == 17 * 3
    check_openpose_schema(json.loads((jdir / "camB" / b[0]).read_text(encoding="utf-8")), 0)

    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["pose_backend"] == "Fake" and meta["pose_backend_key"] == "fake"
    assert meta["keypoint_format"] == "coco17" and meta["pose2sim_model"] == "COCO_17"
    assert meta["pose2d"]["cameras"] == ["camA", "camB"] and meta["pose2d"]["openpose_json"]
    assert meta["pose2d"]["json_dir"] == "pose2d_json"
    assert meta["pose2d"]["json_sets_written"] == 2 and meta["pose2d"]["json_sets_skipped"] == 1


def test_format_from_first_pose_and_no_json_by_default(tmp_path):
    rec = SessionRecorder(tmp_path)
    folder = rec.start(cams=[], calibrations={}, force=None, board=None, geometry=BoardGeometry(),
                       pose_backend="x")
    names = list(FORMATS["halpe26"].names)
    per2d = {"c": Pose2D(np.zeros((26, 2)), np.ones(26), frame_index=3)}
    rec.add_pose(Pose3D(rec.t0 + 0.1, names, np.zeros((26, 3)), np.ones(26), per2d,
                        format_key="halpe26", reproj_error_px=1.25))
    rec.stop(analyze=False)
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["keypoint_format"] == "halpe26" and meta["pose2sim_model"] == "HALPE_26"
    assert meta["pose2d"]["openpose_json"] is False and not (folder / "pose2d_json").exists()
    rows = read_rows(folder / "pose2d_c.csv")
    assert rows[0][4:7] == ["nose_x", "nose_y", "nose_score"] and rows[1][3] == "3"
    assert read_rows(folder / "pose3d.csv")[1][-2:] == ["triangulated", "1.250"]


def test_pose2d_frame_numbers_match_the_recorded_video(tmp_path):
    """End to end with a camera stream and the GUI's pose worker: every pose2d row and JSON file
    names the video frame that was actually processed."""
    from poseboard.gui.app import PoseWorker

    src = tmp_path / "ids.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, SIZE)
    for i in range(40):
        vw.write(encode_id(i))
    vw.release()

    stream = CameraStream("cam0", str(src))
    stream.start()
    cams = {"cam0": approximate_calibration("cam0", *SIZE)}
    det = FakeDetector("coco17", lambda cam, t, img: id_person(img))
    est = MultiViewEstimator(det)  # one camera, 2D-only detector: "2d_only" poses
    rec = SessionRecorder(tmp_path / "rec", save_openpose_json=True)

    def get_frames():
        f = stream.latest()
        return {} if f is None else {"cam0": f}

    worker = PoseWorker(est, get_frames, lambda: cams, rec)
    try:
        deadline = time.perf_counter() + 5
        while stream.latest() is None and time.perf_counter() < deadline:
            time.sleep(0.01)
        worker.start()
        time.sleep(0.2)  # poses before the recording: not written
        folder = rec.start(cams=[stream], calibrations=cams, force=None, board=None,
                           geometry=BoardGeometry(), pose_backend="Fake",
                           pose_backend_key=est.name, pose_info=est.info())
        time.sleep(1.5)
        rec.stop(analyze=False)
    finally:
        worker.stop()
        stream.stop()
    assert worker.error is None

    cap = cv2.VideoCapture(str(folder / "cam0.mkv"))
    video_ids = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        video_ids.append(decode_id(img))
    ts = read_rows(folder / "cam0_timestamps.csv")[1:]
    assert len(video_ids) == len(ts) > 20

    rows = read_rows(folder / "pose2d_cam0.csv")
    assert rows[0] == POSE2D_HEAD
    rows = rows[1:]
    with_frame = [r for r in rows if r[3] != ""]
    assert len(with_frame) >= len(rows) - 1 and len(with_frame) > 10
    frames_seen = [int(r[3]) for r in with_frame]
    assert frames_seen == sorted(set(frames_seen))  # increasing, no duplicates
    for r in with_frame:
        k = int(r[3])
        processed_id = round(float(r[4]) - 50.0)  # nose_x encodes the id of the processed image
        assert processed_id == video_ids[k], (k, processed_id, video_ids[k])
        assert r[0] == ts[k][1]  # the same capture time as camN_timestamps.csv
        j = json.loads((folder / "pose2d_json" / "cam0" / f"cam0_{k:012d}_keypoints.json")
                       .read_text(encoding="utf-8"))
        check_openpose_schema(j, 1)
        assert j["people"][0]["pose_keypoints_2d"][0] == pytest.approx(float(r[4]))
    n_json = len(list((folder / "pose2d_json" / "cam0").glob("*.json")))
    assert n_json == len(with_frame)

    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["pose_backend_key"] == "fake" and meta["keypoint_format"] == "coco17"
    assert meta["pose2sim_model"] == "COCO_17" and meta["pose_info"]["min_score"] == 0.3
    assert meta["pose2d"]["json_sets_written"] == n_json
    pose3d = read_rows(folder / "pose3d.csv")
    assert len(pose3d) - 1 == len(rows) and {r[-2] for r in pose3d[1:]} == {"2d_only"}
