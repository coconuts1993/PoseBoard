"""Recording of the per-camera 2D keypoints: pose2d_<cam>.csv, OpenPose JSON (for Pose2Sim),
the frame numbers of the recorded videos, and the pose format in session.json."""

import csv
import json
import re
import threading
import time

import cv2
import numpy as np
import pytest

from poseboard.calibration import approximate_calibration, load_calibrations
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


def pose2sim_frames(jdir, cams):
    """The JSON files Pose2Sim 0.10.49 reads for each frame f (triangulation.py: the folders
    sorted by their last number, f in range(0, min(number of files)), and in every folder the
    file whose last number == f; a missing file counts as nobody). Returns
    [(f, {camera: file name or None})]."""
    def last_number(name):
        numbers = re.findall(r"\d+", name)
        return (False, int(numbers[-1])) if numbers else (True, name)

    files = {c: sorted((p.name for p in (jdir / c).glob("*.json")), key=last_number)
             for c in cams}
    n = min(len(f) for f in files.values())
    out = []
    for f in range(n):
        out.append((f, {c: next((j for j in files[c] if int(re.split(r"(\d+)", j)[-2]) == f),
                                None) for c in cams}))
    return out


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
    cameras without a subject; JSON sets numbered 0, 1, 2, ... in every camera folder, with an
    empty file for a camera whose frame is not in its video, and sets.csv."""
    rec = SessionRecorder(tmp_path, save_openpose_json=True)
    folder = rec.start(cams=[], calibrations={}, force=None, board=None, geometry=BoardGeometry(),
                       pose_backend="Fake", pose_backend_key="fake", keypoint_format="coco17")
    t = rec.t0 + 0.5
    kp = np.column_stack([np.arange(17.0), np.arange(17.0) + 100])
    # a pose from frames captured before the recording started: no JSON set at all
    early = rec.t0 - 0.2
    rec.add_pose(Pose3D(early, NAMES, np.full((17, 3), np.nan), np.zeros(17),
                        {"camA": Pose2D(kp, np.full(17, 0.7), t=early, frame_index=3)},
                        mode=MODE_2D_ONLY, format_key="coco17",
                        camera_frames={"camA": (early, 3), "camB": (early, 4)}))
    for i in range(3):
        per2d = {"camA": Pose2D(kp + i, np.full(17, 0.7), t=t + i, frame_index=10 + i)}
        frames = {"camA": (t + i, 10 + i), "camB": (t + i + 0.01, 20 + i if i != 2 else None)}
        rec.add_pose(Pose3D(t + i, NAMES, np.full((17, 3), np.nan), np.zeros(17), per2d,
                            mode=MODE_2D_ONLY, format_key="coco17", camera_frames=frames,
                            notes=["camB: no person detected"]))
    rec.stop(analyze=False)

    rows = read_rows(folder / "pose2d_camA.csv")
    assert rows[0] == POSE2D_HEAD and len(rows) == 5
    assert rows[1][3] == "" and rows[2][3] == "10" and rows[4][3] == "12"  # 3: before the start
    assert rows[2][4:7] == ["0.000", "100.000", "0.7000"] and rows[3][4] == "1.000"
    assert float(rows[3][0]) == pytest.approx(t + 1) and float(rows[3][1]) == pytest.approx(
        t + 1 - rec.t0)
    rows_b = read_rows(folder / "pose2d_camB.csv")  # processed, but nobody found
    assert rows_b[0] == POSE2D_HEAD and [r[3] for r in rows_b[1:]] == ["", "20", "21", ""]
    assert all(v == "" for r in rows_b[1:] for v in r[4:])

    pose3d = read_rows(folder / "pose3d.csv")
    assert pose3d[0][-2:] == ["mode", "reproj_error_px"] and pose3d[1][-2:] == ["2d_only", ""]

    jdir = folder / "pose2d_json"
    a = sorted(p.name for p in (jdir / "camA").iterdir())
    b = sorted(p.name for p in (jdir / "camB").iterdir())
    # one set per pose with a new video frame, numbered without gaps in both folders; the third
    # pose has no video frame for camB: an empty file keeps the numbers aligned
    assert a == [f"camA_{i:06d}_keypoints.json" for i in range(3)]
    assert b == [f"camB_{i:06d}_keypoints.json" for i in range(3)]
    da = json.loads((jdir / "camA" / a[1]).read_text(encoding="utf-8"))
    check_openpose_schema(da, 1)
    assert da["people"][0]["pose_keypoints_2d"][:6] == [1.0, 101.0, 0.7, 2.0, 102.0, 0.7]
    assert len(da["people"][0]["pose_keypoints_2d"]) == 17 * 3
    for name in b:
        check_openpose_schema(json.loads((jdir / "camB" / name).read_text(encoding="utf-8")), 0)
    assert all(all(fs.values()) for _, fs in pose2sim_frames(jdir, ["camA", "camB"]))
    sets = read_rows(jdir / "sets.csv")
    assert sets[0] == ["set", "t", "t_rel", "t_unix", "camA_frame", "camA_t", "camB_frame",
                       "camB_t"]
    assert [r[0] for r in sets[1:]] == ["0", "1", "2"]
    assert [(r[4], r[6]) for r in sets[1:]] == [("10", "20"), ("11", "21"), ("12", "")]
    assert float(sets[2][1]) == pytest.approx(t + 1) and float(sets[2][5]) == pytest.approx(t + 1)
    assert sets[3][7] == ""

    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["pose_backend"] == "Fake" and meta["pose_backend_key"] == "fake"
    assert meta["keypoint_format"] == "coco17" and meta["pose2sim_model"] == "COCO_17"
    assert meta["pose2d"]["cameras"] == ["camA", "camB"] and meta["pose2d"]["openpose_json"]
    assert meta["pose2d"]["json_dir"] == "pose2d_json"
    assert meta["pose2d"]["json_cameras"] == ["camA", "camB"]
    assert meta["pose2d"]["json_sets_csv"] == "pose2d_json/sets.csv"
    assert meta["pose2d"]["json_sets_written"] == 3 and meta["pose2d"]["json_sets_skipped"] == 1


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
    sets = read_rows(folder / "pose2d_json" / "sets.csv")[1:]
    set_of_frame = {int(r[4]): int(r[0]) for r in sets}
    assert sorted(set_of_frame.values()) == list(range(len(set_of_frame)))
    for r in with_frame:
        k = int(r[3])
        processed_id = round(float(r[4]) - 50.0)  # nose_x encodes the id of the processed image
        assert processed_id == video_ids[k], (k, processed_id, video_ids[k])
        assert r[0] == ts[k][1]  # the same capture time as camN_timestamps.csv
        n = set_of_frame[k]
        j = json.loads((folder / "pose2d_json" / "cam0" / f"cam0_{n:06d}_keypoints.json")
                       .read_text(encoding="utf-8"))
        check_openpose_schema(j, 1)
        assert j["people"][0]["pose_keypoints_2d"][0] == pytest.approx(float(r[4]))
    n_json = len(list((folder / "pose2d_json" / "cam0").glob("*.json")))
    assert n_json == len(with_frame) == len(set_of_frame)

    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["pose_backend_key"] == "fake" and meta["keypoint_format"] == "coco17"
    assert meta["pose2sim_model"] == "COCO_17" and meta["pose_info"]["min_score"] == 0.3
    assert meta["pose2d"]["json_sets_written"] == n_json
    pose3d = read_rows(folder / "pose3d.csv")
    assert len(pose3d) - 1 == len(rows) and {r[-2] for r in pose3d[1:]} == {"2d_only"}


def _video_ids(path):
    cap = cv2.VideoCapture(str(path))
    ids = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        ids.append(decode_id(img))
    cap.release()
    return ids


def test_json_sets_of_two_cameras_pair_like_pose2sim(tmp_path):
    """Two cameras and a pose rate below the frame rate (frames are skipped, and the cameras'
    frame counters differ): Pose2Sim finds a file for every frame number in both camera folders,
    and the two files of one number are the two images of the same pose (sets.csv)."""
    from poseboard.gui.app import PoseWorker

    offsets = {"cam0": 0, "cam1": 100}
    streams = {}
    for name, off in offsets.items():
        src = tmp_path / f"{name}.mp4"
        vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, SIZE)
        for i in range(60):
            vw.write(encode_id(off + i))
        vw.release()
        streams[name] = CameraStream(name, str(src))
    cams = {n: approximate_calibration(n, *SIZE) for n in streams}

    def slow(cam, t, img):
        time.sleep(0.035)  # two cameras: about 14 poses per second at 30 fps
        return id_person(img)

    est = MultiViewEstimator(FakeDetector("coco17", slow))
    rec = SessionRecorder(tmp_path / "rec", save_openpose_json=True)

    def get_frames():
        return {n: f for n, s in streams.items() if (f := s.latest()) is not None}

    worker = PoseWorker(est, get_frames, lambda: cams, rec)
    try:
        for s in streams.values():
            s.start()
        deadline = time.perf_counter() + 5
        while len(get_frames()) < 2 and time.perf_counter() < deadline:
            time.sleep(0.01)
        time.sleep(0.1)  # the second stream starts a little later: different frame counters
        worker.start()
        folder = rec.start(cams=list(streams.values()), calibrations=cams, force=None,
                           board=None, geometry=BoardGeometry(), pose_backend="Fake")
        time.sleep(2.0)
        rec.stop(analyze=False)
    finally:
        worker.stop()
        for s in streams.values():
            s.stop()
    assert worker.error is None

    jdir = folder / "pose2d_json"
    sets = read_rows(jdir / "sets.csv")
    head, sets = sets[0], sets[1:]
    n_video = {n: len(read_rows(folder / f"{n}_timestamps.csv")) - 1 for n in streams}
    assert 10 < len(sets) < 0.8 * min(n_video.values())  # frames were skipped
    assert [int(r[0]) for r in sets] == list(range(len(sets)))
    frames = pose2sim_frames(jdir, ["cam0", "cam1"])
    assert len(frames) == len(sets)
    assert all(fs["cam0"] and fs["cam1"] for _, fs in frames), frames  # nothing missing
    ids = {n: _video_ids(folder / f"{n}.mkv") for n in streams}
    for (f, fs), row in zip(frames, sets):
        for n in streams:
            j = json.loads((jdir / n / fs[n]).read_text(encoding="utf-8"))
            k = row[head.index(f"{n}_frame")]
            if k == "":  # no new frame of this camera for this pose: nobody
                check_openpose_schema(j, 0)
                continue
            check_openpose_schema(j, 1)
            # the JSON holds the keypoints of exactly that video frame
            processed = round(j["people"][0]["pose_keypoints_2d"][0] - 50.0)
            assert processed == ids[n][int(k)], (n, f, k)
        t0, t1 = row[head.index("cam0_t")], row[head.index("cam1_t")]
        if t0 and t1:
            assert abs(float(t0) - float(t1)) < 0.25  # the two images of one pose
    both = sum(1 for r in sets if r[head.index("cam0_frame")] and r[head.index("cam1_frame")])
    assert both >= 0.8 * len(sets)
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["pose2d"]["json_sets_written"] == len(sets)
    assert meta["pose2d"]["calib_toml"] is None  # no extrinsics: no Calib.toml
    assert not (jdir / "Calib.toml").exists()


def test_recording_writes_calib_toml_of_the_recorded_cameras(tmp_path):
    """pose2d_json/Calib.toml: only the recorded cameras with extrinsics, in the order in which
    Pose2Sim pairs Calib.toml sections with the <camera>_json folders."""
    from tests.test_core import make_cam

    class Stream:  # enough of a CameraStream for the recorder
        fps, source = 30.0, "x"

        def __init__(self, name):
            self.name = name

        def start_recording(self, path, clock_offset_unix=None, t0=None):
            return path

        def stop_recording(self):
            pass

    calibs = {n: make_cam(n, [0.5 * i, -2.0, 1.2]) for i, n in enumerate(
        ("side", "cam10", "front", "cam2", "unused"))}
    calibs["noext"] = approximate_calibration("noext", 640, 480)
    rec = SessionRecorder(tmp_path, save_openpose_json=True)
    folder = rec.start(cams=[Stream(n) for n in ("side", "cam10", "front", "cam2", "noext")],
                       calibrations=calibs, force=None, board=None, geometry=BoardGeometry(),
                       pose_backend="Fake")
    rec.stop(analyze=False)
    back = load_calibrations(folder / "pose2d_json" / "Calib.toml")
    assert [c.name for c in back] == ["cam2", "cam10", "front", "side"]
    np.testing.assert_allclose(back[1].tvec, calibs["cam10"].tvec)
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    p2 = meta["pose2d"]
    assert p2["calib_toml"] == "pose2d_json/Calib.toml"
    assert p2["calib_toml_cameras"] == ["cam2", "cam10", "front", "side"]
    assert "noext" in p2["calib_toml_note"]  # recorded, but without extrinsics
    assert p2["json_cameras"] == ["cam2", "cam10", "front", "noext", "side"]


def test_wii_samples_do_not_wait_for_the_pose_output(tmp_path):
    """The Wii reader thread (samples are stamped when read) never waits while a pose, its JSON
    files or an event are written: wii.csv has its own lock."""
    from poseboard.wii.device import ForceSample, SimulatedBoard

    board = SimulatedBoard()
    rec = SessionRecorder(tmp_path)
    folder = rec.start(cams=[], calibrations={}, force=board, board=None,
                       geometry=BoardGeometry(), pose_backend=None)
    try:
        s = ForceSample(time.perf_counter(), np.array([10.0, 10.0, 10.0, 10.0]), 40.0,
                        np.array([0.01, -0.02]))
        with rec._lock:  # e.g. add_pose writing many CSV rows and JSON files
            th = threading.Thread(target=rec._on_force, args=(s,))
            th.start()
            th.join(2.0)
            assert not th.is_alive(), "the Wii sample waited for the pose output"
    finally:
        rec.stop(analyze=False)
    rows = read_rows(folder / "wii.csv")
    assert len(rows) == 2 and rows[1][7] == "40.000000"
