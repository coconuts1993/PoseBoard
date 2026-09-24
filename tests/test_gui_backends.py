"""Offscreen GUI: the pose backend list of the Record tab (built from the registry, unavailable
backends greyed out), the generated option widgets, starting a backend in the pose thread
(loading status, errors), and an end-to-end recording with two synthetic calibrated cameras
(triangulated 3D, the person on the board picked, pose2d CSV and OpenPose JSON written).

Set POSEBOARD_SCREENSHOT_BACKENDS=<file.png> to save a screenshot of the Record tab during the
two-camera recording."""

import csv
import json
import os
import threading
import time

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from poseboard.calibration import CameraCalibration, load_calibrations  # noqa: E402
from poseboard.geometry import BoardGeometry  # noqa: E402
from poseboard.pose.base import (MODE_2D_ONLY, MODE_TRIANGULATED, Pose2D, Pose3D,  # noqa: E402
                                 PoseEstimator)
from poseboard.pose.detectors import (BACKENDS, BackendSpec, backend_available,  # noqa: E402
                                      list_backends)
from poseboard.pose.formats import FORMATS  # noqa: E402
from poseboard.pose.multiview import MultiViewEstimator  # noqa: E402
from poseboard.wii.device import BalanceBoardHID  # noqa: E402
from tests.fakes import FakeDetector, Scene, standing_person  # noqa: E402
from tests.synthetic import BOARD_T, render_scene, scene_camera  # noqa: E402
from tests.test_core import make_cam  # noqa: E402

# ------------------------------------------------------------------ fake backends (by module path)
SCENE: dict = {}  # "scene": the Scene the fake MediaPipe detector plays; "created": its options
GATE = threading.Event()


def create_scene_detector(**options):
    """Factory registered in place of MediaPipe: a FakeDetector playing ``SCENE["scene"]``."""
    SCENE.setdefault("created", []).append(dict(options))
    det = FakeDetector("coco17", SCENE["scene"])
    det.key, det.label = "mediapipe", "MediaPipe Pose Landmarker"
    det.options = dict(options)
    return det


def create_failing_detector(**options):
    """Waits until the test opens ``GATE``, then fails like a blocked model download."""
    GATE.wait(10)
    raise RuntimeError("cannot download the model https://example.invalid/model.onnx (HTTP 403)")


LOADS = {"active": 0, "peak": 0, "count": 0}  # create_gated_detector calls


def create_gated_detector(**options):
    """Waits until the test opens ``GATE`` (a slow model download), then returns a
    FakeDetector; counts overlapping calls in ``LOADS``."""
    LOADS["active"] += 1
    LOADS["count"] += 1
    LOADS["peak"] = max(LOADS["peak"], LOADS["active"])
    try:
        GATE.wait(10)
        time.sleep(0.05)
        return FakeDetector("coco17")
    finally:
        LOADS["active"] -= 1


def fake_spec(key, factory, label, **kw):
    base = dict(key=key, label=label, module=__name__, factory=factory, requires=("numpy",),
                install="pip install nothing", options={"model": ("a", "b")},
                defaults={"model": "a"}, keypoint_format="coco17", license="MIT",
                notes="Test backend.")
    base.update(kw)
    return BackendSpec(**base)


# ------------------------------------------------------------------ helpers
@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def no_boards(monkeypatch):
    monkeypatch.setattr(BalanceBoardHID, "list_devices", staticmethod(lambda: []))


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.01)


def pump_until(app, cond, timeout=10.0) -> bool:
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def make_window(**kw):
    from poseboard.gui.app import MainWindow

    kw.setdefault("auto_connect_wii", False)
    w = MainWindow(**kw)
    w.warnings = []
    w._warn = w.warnings.append  # never open a modal dialog in a test
    w.resize(1500, 900)
    w.show()
    w.tabs.setCurrentIndex(3)
    return w


def select(w, key):
    i = w.pose_backend.findData(key)
    assert i >= 0, key
    w.pose_backend.setCurrentIndex(i)
    return i


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def only_session(root):
    folders = [p for p in root.iterdir() if p.is_dir()]
    assert len(folders) == 1
    return folders[0]


# ------------------------------------------------------------------ backend list and options
def test_backend_list_shows_every_registry_backend(app, no_boards):
    from poseboard.gui.app import PLUGIN_KEY, PLUGIN_LABEL

    w = make_window()
    try:
        combo = w.pose_backend
        keys = [combo.itemData(i) for i in range(combo.count())]
        assert keys == [s.key for s in list_backends()] + [PLUGIN_KEY]
        assert combo.itemText(combo.count() - 1) == PLUGIN_LABEL
        model = combo.model()
        disabled = []
        for i, key in enumerate(keys[:-1]):
            ok, why = backend_available(key)
            item = model.item(i)
            assert item.isEnabled() == ok, key
            assert BACKENDS[key].license in item.toolTip()
            if not ok:
                disabled.append(key)
                assert why in item.toolTip() and "not" in combo.itemText(i)
        # Packages missing in this environment -> greyed out with the pip command
        for key, spec in BACKENDS.items():
            missing = [m for m in spec.requires if not _has(m)]
            if missing:
                assert key in disabled
                assert spec.install in model.item(keys.index(key)).toolTip()
        if backend_available("mediapipe")[0]:
            assert w.current_backend() == "mediapipe"  # the default
    finally:
        w.close()


def _has(name):
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def test_option_widgets_follow_the_registry(app, no_boards, tmp_path):
    w = make_window()
    try:
        select(w, "openpose_dnn")
        kinds = {k: v[0] for k, v in w._opt_widgets.items()}
        assert kinds == {"model": "combo", "input_size": "combo", "device": "combo",
                         "prototxt": "file", "caffemodel": "file"}
        assert w.backend_options() == {"model": "body25", "input_size": 368, "device": "auto",
                                       "prototxt": "", "caffemodel": ""}
        assert "non-commercial" in w.backend_info.text()
        assert not w.plugin_row.isVisible()
        w._opt_widgets["input_size"][1].setCurrentText("256")
        w._opt_widgets["caffemodel"][1].setText(str(tmp_path / "x.caffemodel"))
        assert w.backend_options()["input_size"] == 256  # typed like the registry choice

        # a model file that does not exist: the start fails with the backend's message
        w._opt_widgets["caffemodel"][1].setText(str(tmp_path / "missing.caffemodel"))
        w.b_pose.setChecked(True)
        assert pump_until(app, lambda: w.pose_worker is None)
        assert not w.b_pose.isChecked() and w.pose_backend.isEnabled()
        assert "caffemodel file not found" in w.warnings[-1]
        w._opt_widgets["caffemodel"][1].setText(str(tmp_path / "x.caffemodel"))

        # per-backend values survive switching backends; min score is per backend
        select(w, "movenet")
        assert w._opt_widgets["model_path"][0] == "file"
        assert w.min_score.value() == pytest.approx(0.3)
        select(w, "mediapipe")
        assert set(w._opt_widgets) == {"model", "num_poses"}
        assert w.min_score.value() == pytest.approx(0.5)
        w._opt_widgets["num_poses"][1].setCurrentText("2")
        w.min_score.setValue(0.6)
        select(w, "openpose_dnn")
        assert w.backend_options()["input_size"] == 256
        select(w, "mediapipe")
        assert w.backend_options() == {"model": "full", "num_poses": 2}
        assert w.min_score.value() == pytest.approx(0.6)

        select(w, "__plugin__")
        assert w._opt_widgets == {} and not w.plugin_row.isHidden()

        # the settings are saved with the project and restored
        select(w, "openpose_dnn")
        w.reproj_thr.setValue(22.0)
        w.chk_json.setChecked(True)
        saved = json.loads(json.dumps(w.pose_settings()))
    finally:
        w.close()
    w2 = make_window()
    try:
        w2.apply_pose_settings(saved)
        assert w2.current_backend() == "openpose_dnn"
        assert w2.backend_options()["input_size"] == 256
        assert w2.reproj_thr.value() == pytest.approx(22.0) and w2.chk_json.isChecked()
        select(w2, "mediapipe")
        assert w2.backend_options()["num_poses"] == 2
    finally:
        w2.close()


def test_unavailable_backend_is_refused_with_the_install_hint(app, no_boards, monkeypatch):
    monkeypatch.setitem(BACKENDS, "needs_pkg", fake_spec(
        "needs_pkg", "create_scene_detector", "Needs a package",
        requires=("surely_not_installed_pkg",), install="pip install surely-not-installed"))
    w = make_window()
    try:
        i = select(w, "needs_pkg")
        assert not w.pose_backend.model().item(i).isEnabled()
        assert "NOT AVAILABLE" in w.backend_info.text()
        w.b_pose.setChecked(True)
        assert w.pose_worker is None and not w.b_pose.isChecked()
        assert "pip install surely-not-installed" in w.warnings[-1]
    finally:
        w.close()


def test_backend_loads_in_the_pose_thread_and_reports_failures(app, no_boards, monkeypatch):
    monkeypatch.setitem(BACKENDS, "slow_fail", fake_spec("slow_fail", "create_failing_detector",
                                                         "Slow failing backend"))
    GATE.clear()
    w = make_window()
    try:
        select(w, "slow_fail")
        t = time.perf_counter()
        w.b_pose.setChecked(True)
        assert time.perf_counter() - t < 1.0  # the window is not blocked by the model loading
        pw = w.pose_worker
        assert pw is not None and pw.loading
        pump(app, 0.3)
        assert "Loading Slow failing backend" in w.pose_label.text()
        assert not w.pose_backend.isEnabled()  # fixed while pose estimation runs
        GATE.set()
        assert pump_until(app, lambda: w.pose_worker is None)
        assert not w.b_pose.isChecked() and w.pose_backend.isEnabled()
        assert "HTTP 403" in w.warnings[-1] and "Slow failing backend" in w.warnings[-1]
        assert "HTTP 403" in w.pose_label.text()
        assert not pw.alive

        # Stop while loading: returns at once, the late estimator is closed by the thread
        GATE.clear()
        w.b_pose.setChecked(True)
        pw = w.pose_worker
        t = time.perf_counter()
        w.b_pose.setChecked(False)
        assert time.perf_counter() - t < 0.5 and w.pose_worker is None
        GATE.set()
        assert pump_until(app, lambda: not pw.alive)
        pump(app, 0.1)
        assert len(w.warnings) == 1  # a cancelled start is not reported
    finally:
        GATE.set()
        w.close()


# ------------------------------------------------------------------ two cameras, end to end
def write_video(path, cam, n=30):
    img = render_scene(cam)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (img.shape[1], img.shape[0]))
    for _ in range(n):
        vw.write(img)
    vw.release()


def test_two_calibrated_cameras_triangulate_and_record_2d(app, tmp_path, no_boards, monkeypatch):
    """"MediaPipe" (a scripted FakeDetector registered under its key) with two calibrated
    synthetic cameras: triangulated 3D of the person on the board (a larger bystander stands
    in front of the board), then a recording with pose2d CSV files and OpenPose JSON."""
    cams = {"cam0": make_cam("cam0", [0.95, -2.6, 1.2], [0.95, 0.2, 0.85], f=900.0),
            "cam1": make_cam("cam1", [3.0, -1.6, 1.2], [0.95, 0.2, 0.85], f=900.0)}
    board_top = BOARD_T.t  # the board surface center (world), 5.3 cm above the floor
    subject = standing_person("coco17", offset=(board_top[0], board_top[1], board_top[2]))
    # closer to both cameras, so it looks larger than the subject: picked without the board
    bystander = standing_person("coco17", offset=(0.55, -0.8, 0.0), scale=1.05)
    SCENE.clear()
    SCENE["scene"] = Scene(cams, subject, "coco17", bystanders=[bystander], noise_px=0.3)
    real = BACKENDS["mediapipe"]
    monkeypatch.setitem(BACKENDS, "mediapipe", BackendSpec(
        key="mediapipe", label=real.label, module=__name__, factory="create_scene_detector",
        requires=("numpy",), install=real.install, options=dict(real.options),
        defaults=dict(real.defaults), keypoint_format="coco17", license=real.license,
        notes=real.notes, provides_3d=False))
    videos = {}
    for n, c in cams.items():
        videos[n] = tmp_path / f"{n}.mp4"
        write_video(videos[n], c)

    w = make_window()
    try:
        w.cam_res.setCurrentText("Default")
        for n in cams:
            w.add_camera(str(videos[n]), n)
        assert pump_until(app, lambda: len(w._latest_frames()) == 2)
        for n, c in cams.items():  # calibrated (intrinsics + extrinsics on the checkerboard)
            w.calibs[n] = CameraCalibration(n, c.image_size, c.K.copy(), c.dist.copy(),
                                            np.array(c.rvec, float), np.array(c.tvec, float))
        # register the board by clicking its 5 points in cam0
        w.cam_select.setCurrentText("cam0")
        w.tabs.setCurrentIndex(2)
        w.b_click.setChecked(True)
        for p in cams["cam0"].project(BOARD_T.apply(BoardGeometry().landmarks())):
            w._on_video_click(float(p[0]), float(p[1]), 1)
        w.compute_board()
        assert w.board is not None and not w.warnings
        w.tabs.setCurrentIndex(3)

        select(w, "mediapipe")
        assert w.pose_backend.currentText() == real.label
        w._opt_widgets["num_poses"][1].setCurrentText("2")
        w.chk_json.setChecked(True)
        assert w.chk_pick_board.isChecked() and w.chk_save2d.isChecked()
        w.b_pose.setChecked(True)
        assert pump_until(app, lambda: w.pose_worker is not None and w.pose_worker.latest
                          is not None and w.pose_worker.latest.mode == MODE_TRIANGULATED)
        pw = w.pose_worker
        assert isinstance(pw.estimator, MultiViewEstimator) and pw.key == "mediapipe"
        assert SCENE["created"] == [{"model": "full", "num_poses": 2}]
        assert pw.estimator.min_score == pytest.approx(0.5)
        pose = pw.latest
        assert pose.views_used == ["cam0", "cam1"] and pose.format_key == "coco17"
        assert pose.reproj_error_px < 2.0
        err = np.linalg.norm(pose.keypoints - subject, axis=1)
        assert np.nanmax(err) < 0.02, "the person standing on the board must be triangulated"
        pump(app, 0.2)
        assert "triangulated from 2 views" in w.pose_label.text()
        assert "reprojection error" in w.pose_label.text()
        assert w.min_score.isEnabled()  # adjustable while not recording

        # Record
        w.out_dir.setText(str(tmp_path / "rec"))
        w.subject.setText("two_cams")
        w.b_rec.setChecked(True)
        assert w.recorder.recording, w.warnings
        assert not w.chk_json.isEnabled() and not w.min_score.isEnabled()
        pump(app, 1.5)
        shot = os.environ.get("POSEBOARD_SCREENSHOT_BACKENDS")
        if shot:
            w.grab().save(shot)
        w.b_rec.setChecked(False)
        assert w.recorder.wait_post_processing(30)
        assert w.chk_json.isEnabled()
        assert not w.warnings, w.warnings
    finally:
        w.close()

    folder = only_session(tmp_path / "rec")
    names = list(FORMATS["coco17"].names)
    pose3d = read_rows(folder / "pose3d.csv")
    head = pose3d[0]
    assert head[-2:] == ["mode", "reproj_error_px"] and len(pose3d) > 5
    assert {r[-2] for r in pose3d[1:]} == {MODE_TRIANGULATED}
    com_z = [float(r[head.index("com_z")]) for r in pose3d[1:]]
    assert all(0.8 < z < 1.2 for z in com_z)  # COM of a 1.7 m person standing on the board
    frames = {}
    for n in cams:
        rows = read_rows(folder / f"pose2d_{n}.csv")
        assert rows[0][:4] == ["t", "t_rel", "t_unix", "frame"]
        assert rows[0][4:7] == [f"{names[0]}_x", f"{names[0]}_y", f"{names[0]}_score"]
        assert len(rows) - 1 == len(pose3d) - 1
        frames[n] = [int(r[3]) for r in rows[1:] if r[3] != ""]
        assert len(frames[n]) >= len(rows) - 2 and frames[n] == sorted(frames[n])
        # a pose waits for a new frame of both cameras: a frame is used twice only when a
        # camera stalled for more than 0.1 s
        assert len(set(frames[n])) >= 0.8 * len(frames[n]), frames[n]
        ts = read_rows(folder / f"{n}_timestamps.csv")
        assert max(frames[n]) < len(ts) - 1  # a frame of the recorded video
    jdir = folder / "pose2d_json"
    files = {n: sorted((jdir / n).glob("*.json")) for n in cams}
    assert len(files["cam0"]) == len(files["cam1"]) > 3
    sets = read_rows(jdir / "sets.csv")
    head, sets = sets[0], sets[1:]
    assert len(sets) == len(files["cam0"])
    for n in cams:  # numbered by set, 0, 1, 2, ... in both folders
        assert [p.name for p in files[n]] == [f"{n}_{i:06d}_keypoints.json"
                                              for i in range(len(sets))]
        assert {int(r[head.index(f"{n}_frame")]) for r in sets
                if r[head.index(f"{n}_frame")]} <= set(frames[n])
        # a person exactly in the sets with a frame of this camera (a set just after the start
        # may have none: a frame captured before the camera's video writer was open)
        col = head.index(f"{n}_frame")
        docs = [json.loads(p.read_text(encoding="utf-8")) for p in files[n]]
        assert all(d["version"] == 1.3 for d in docs)
        assert [len(d["people"]) for d in docs] == [1 if r[col] else 0 for r in sets]
        with_person = [d for d in docs if d["people"]]
        assert len(with_person) >= len(docs) - 2, [len(d["people"]) for d in docs]
        assert len(with_person[0]["people"][0]["pose_keypoints_2d"]) == 17 * 3
    back = load_calibrations(jdir / "Calib.toml")  # the recorded cameras, Pose2Sim's order
    assert [c.name for c in back] == ["cam0", "cam1"]
    np.testing.assert_allclose(back[1].tvec, cams["cam1"].tvec)
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["pose_backend"] == real.label and meta["pose_backend_key"] == "mediapipe"
    assert meta["keypoint_format"] == "coco17" and meta["pose2sim_model"] == "COCO_17"
    assert meta["pose_info"]["options"] == {"model": "full", "num_poses": 2}
    assert meta["pose_info"]["min_score"] == pytest.approx(0.5)
    assert meta["pose2d"]["openpose_json"] is True
    assert meta["pose2d"]["cameras"] == ["cam0", "cam1"]
    assert meta["pose2d"]["json_sets_written"] == len(files["cam0"])
    assert meta["pose2d"]["calib_toml"] == "pose2d_json/Calib.toml"


def test_single_camera_2d_backend_draws_2d_and_says_why(app, tmp_path, no_boards, monkeypatch):
    """One camera and a 2D-only backend: "2d_only" poses; the 2D skeleton is drawn and the
    status explains that a second calibrated camera is needed for 3D."""
    cam = scene_camera()
    subject = standing_person("coco17", offset=tuple(BOARD_T.t))
    SCENE.clear()
    SCENE["scene"] = Scene({"cam0": cam}, subject, "coco17")
    monkeypatch.setitem(BACKENDS, "fake2d", fake_spec("fake2d", "create_scene_detector",
                                                      "Fake 2D backend"))
    video = tmp_path / "cam0.mp4"
    write_video(video, cam, n=10)
    w = make_window()
    try:
        w.cam_res.setCurrentText("Default")
        w.add_camera(str(video), "cam0")
        assert pump_until(app, lambda: len(w._latest_frames()) == 1)
        w.calibs["cam0"] = CameraCalibration("cam0", cam.image_size, cam.K.copy(), cam.dist.copy(),
                                             np.array(cam.rvec, float), np.array(cam.tvec, float))
        select(w, "fake2d")
        w.b_pose.setChecked(True)
        assert pump_until(app, lambda: w.pose_worker.latest is not None)
        pose = w.pose_worker.latest
        assert pose.mode == MODE_2D_ONLY and "cam0" in pose.per_camera_2d
        pump(app, 0.2)
        text = w.pose_label.text()
        assert "2D only" in text and "gives no 3D skeleton" in text
        raw = w.streams["cam0"].latest().image
        shown = w.video._img
        assert shown is not None
        from poseboard.gui.widgets import bgr_to_qimage

        assert shown != bgr_to_qimage(raw)  # skeleton (and caption) drawn over the frame
    finally:
        w.close()


# ------------------------------------------------------------------ overlay and board view
@pytest.mark.parametrize("fmt_key", sorted(FORMATS))
def test_draw_pose_uses_the_format_skeleton(fmt_key):
    from poseboard.overlay import SIDE_COLORS, draw_pose, pose_skeleton

    fmt = FORMATS[fmt_key]
    cam = scene_camera()
    pts = standing_person(fmt, offset=tuple(BOARD_T.t))
    kp2 = cam.project(pts)
    K = len(fmt)
    pose = Pose3D(0.0, list(fmt.names), np.full((K, 3), np.nan), np.zeros(K),
                  {"cam0": Pose2D(kp2, np.full(K, 0.9))}, mode=MODE_2D_ONLY, format_key=fmt_key)
    assert pose_skeleton(pose) == list(fmt.skeleton)
    img = np.zeros((720, 1280, 3), np.uint8)
    draw_pose(img, cam, "cam0", pose, None, None, None, min_score=0.5)
    for side in ("left", "right"):
        color = np.array(SIDE_COLORS[side])
        assert np.all(img == color, axis=2).sum() > 50, side
    # low scores are not drawn; a camera without a 2D subject and no 3D: nothing drawn
    empty = np.zeros_like(img)
    pose.per_camera_2d["cam0"].scores[:] = 0.2
    draw_pose(empty, cam, "cam0", pose, None, None, None, min_score=0.5)
    draw_pose(empty, cam, "cam1", pose, None, None, None)
    assert not empty.any()
    # the skeleton is found from the names when the format key is unknown
    pose.format_key = None
    assert pose_skeleton(pose) == list(fmt.skeleton)


def test_draw_pose_projects_3d_without_2d():
    from poseboard.overlay import draw_pose

    cam = scene_camera()
    fmt = FORMATS["halpe26"]
    pts = standing_person(fmt, offset=tuple(BOARD_T.t))
    pose = Pose3D(0.0, list(fmt.names), pts, np.ones(len(fmt)), {}, format_key="halpe26")
    img = np.zeros((720, 1280, 3), np.uint8)
    draw_pose(img, cam, "cam0", pose, None, None, np.array([0.95, 0.2, 1.0]))
    assert img.any()


@pytest.mark.parametrize("size", [(300, 200), (640, 260), (1000, 260), (420, 400)])
def test_board_view_sensor_labels_inside_the_board(app, size):
    from PySide6.QtCore import QRectF

    from poseboard.gui.widgets import CopView

    v = CopView()
    v.resize(*size)
    board = v.board_rect().adjusted(2, 2, -2, -2)  # inside the 2 px outline
    layout = v.sensor_layout()
    assert set(layout) == {"TL", "TR", "BL", "BR"}
    r_dot = v.SENSOR_DOT_PX
    rects = []
    for name, (c, r) in layout.items():
        assert board.contains(r), (name, r, board)
        dot = QRectF(c.x() - r_dot, c.y() - r_dot, 2 * r_dot, 2 * r_dot)
        assert not r.intersects(dot), name
        assert abs(r.center().y() - c.y()) < 1  # beside its dot
        rects.append(r)
    for i, a in enumerate(rects):
        assert not any(a.intersects(b) for b in rects[i + 1:])
    img = v.grab()  # paints without errors
    assert not img.isNull()


# ------------------------------------------------------------------ pose worker and recorder
class _IndexEstimator(PoseEstimator):
    """Records the frame numbers it is given (encoded in the images)."""

    name = "index"

    def __init__(self):
        self.seen: list[dict] = []
        self.closed = False

    def process(self, frames, cams):
        self.seen.append({n: int(img[0, 0, 0]) for n, (_, img) in frames.items()})
        t = max(t for t, _ in frames.values())
        return Pose3D(t, ["a"], np.zeros((1, 3)), np.ones(1))

    def close(self):
        self.closed = True


class _Rec:
    recording = False


def _two_cameras(period_b=0.03):
    """Camera a: a new frame at every call; camera b: every ``period_b`` seconds."""
    from poseboard.camera import Frame

    t0 = time.perf_counter()
    n = [0]

    def get():
        n[0] += 1
        now = time.perf_counter()
        ib = int((now - t0) / period_b)
        return {"a": Frame(n[0], now, np.full((2, 2, 3), n[0] % 256, np.uint8)),
                "b": Frame(ib, t0 + ib * period_b, np.full((2, 2, 3), ib % 256, np.uint8))}
    return get


@pytest.mark.parametrize("sync_wait_s", [0.2, 0.0])
def test_pose_worker_waits_for_a_new_frame_of_every_camera(sync_wait_s):
    from poseboard.gui.app import PoseWorker

    est = _IndexEstimator()
    w = PoseWorker(est, _two_cameras(), dict, _Rec(), max_skew_s=0.25, sync_wait_s=sync_wait_s)
    w.start()
    try:
        end = time.perf_counter() + 5
        while len(est.seen) < 15 and time.perf_counter() < end:
            time.sleep(0.01)
    finally:
        w.stop()
    b = [s["b"] for s in est.seen]
    assert len(b) >= 15 and all(set(s) == {"a", "b"} for s in est.seen)
    if sync_wait_s:
        assert all(x != y for x, y in zip(b, b[1:])), b  # camera b's frames are never reused
    else:
        assert any(x == y for x, y in zip(b, b[1:]))  # without waiting they are
    assert est.closed


def test_pose_worker_reports_a_failing_factory():
    from poseboard.gui.app import PoseWorker

    def factory():
        raise RuntimeError("cannot download the model (HTTP 403)")

    w = PoseWorker(None, dict, dict, _Rec(), factory=factory, label="X")
    assert w.loading and w.estimator is None
    w.start()
    end = time.perf_counter() + 5
    while w.alive and time.perf_counter() < end:
        time.sleep(0.01)
    assert not w.alive and not w.loading and w.estimator is None
    assert "HTTP 403" in w.load_error
    with pytest.raises(ValueError):
        PoseWorker(None, dict, dict, _Rec())


class _RecStream:
    """Enough of a recorded CameraStream for the recorder."""
    fps, source = 30.0, "x"

    def __init__(self, name):
        self.name = name

    def start_recording(self, path, clock_offset_unix=None, t0=None):
        return path

    def stop_recording(self):
        pass


def test_openpose_json_stays_aligned_when_a_camera_has_no_new_frame(tmp_path, caplog):
    """Every recorded camera gets a file in every JSON set, so the folders stay aligned for
    Pose2Sim: a camera frame used for a second pose (no new frame in time), a camera left out
    as stale, and a camera whose frames are not in its video (removed and added again during
    the recording) get an empty file; the other cameras' JSON goes on."""
    from poseboard.session import SessionRecorder
    from tests.test_pose2d_recording import pose2sim_frames

    rec = SessionRecorder(tmp_path, save_openpose_json=True)
    folder = rec.start(cams=[_RecStream("a"), _RecStream("b")], calibrations={}, force=None,
                       board=None, geometry=BoardGeometry(), pose_backend="Fake",
                       keypoint_format="coco17")
    names = list(FORMATS["coco17"].names)
    t = time.perf_counter() + 0.1  # captured after the videos started
    kp = np.column_stack([np.arange(17.0), np.arange(17.0)])
    # b's frame 1 is used twice; b is stale (not processed) for the 4th pose; from the 5th pose
    # on b is a new stream that does not record (frame number None)
    for a, b in ((1, 1), (2, 1), (3, 2), (4, "stale"), (5, None), (6, None)):
        per2d = {"a": Pose2D(kp, np.ones(17), t=t, frame_index=a)}
        frames = {"a": (t, a)}
        if b != "stale":
            per2d["b"] = Pose2D(kp, np.ones(17), t=t, frame_index=b)
            frames["b"] = (t, b)
        with caplog.at_level("WARNING", logger="poseboard.session"):
            rec.add_pose(Pose3D(t, names, np.zeros((17, 3)), np.ones(17), per2d,
                                format_key="coco17", camera_frames=frames))
        t += 0.1
    rec.stop(analyze=False)
    jdir = folder / "pose2d_json"
    ja = sorted(p.name for p in (jdir / "a").iterdir())
    jb = sorted(p.name for p in (jdir / "b").iterdir())
    assert ja == [f"a_{i:06d}_keypoints.json" for i in range(6)]
    assert jb == [f"b_{i:06d}_keypoints.json" for i in range(6)]
    people_b = [len(json.loads((jdir / "b" / n).read_text(encoding="utf-8"))["people"])
                for n in jb]
    assert people_b == [1, 0, 1, 0, 0, 0]
    assert all(len(json.loads((jdir / "a" / n).read_text(encoding="utf-8"))["people"]) == 1
               for n in ja)
    assert all(all(fs.values()) for _, fs in pose2sim_frames(jdir, ["a", "b"]))
    sets = read_rows(jdir / "sets.csv")[1:]
    assert [(r[4], r[6]) for r in sets] == [("1", "1"), ("2", ""), ("3", "2"), ("4", ""),
                                            ("5", ""), ("6", "")]
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["pose2d"]["json_sets_written"] == 6 and meta["pose2d"]["json_sets_skipped"] == 0
    assert [r[3] for r in read_rows(folder / "pose2d_b.csv")[1:]] == ["1", "1", "2", "", ""]
    warned = [r for r in caplog.records if "not in its recorded video" in r.getMessage()]
    assert len(warned) == 1 and "b" in warned[0].getMessage()  # reported once


class _SlowRecStream(_RecStream):
    """A camera whose video writer takes a while to open."""

    def start_recording(self, path, clock_offset_unix=None, t0=None):
        time.sleep(0.05)
        return path


def test_openpose_json_frames_before_the_video_started_are_not_reported(tmp_path, caplog):
    """A frame captured after t0 but before its camera's video writer was open is not in the
    video: its JSON file is empty, without the "not in its recorded video" warning (that is only
    for frames captured while the video is being written)."""
    from poseboard.session import SessionRecorder

    rec = SessionRecorder(tmp_path, save_openpose_json=True)
    folder = rec.start(cams=[_RecStream("a"), _SlowRecStream("b")], calibrations={},
                       force=None, board=None, geometry=BoardGeometry(), pose_backend="Fake",
                       keypoint_format="coco17")
    names = list(FORMATS["coco17"].names)
    kp = np.column_stack([np.arange(17.0), np.arange(17.0)])
    t_early, t_late = rec.t0 + 0.01, time.perf_counter() + 0.1  # before / after b's video
    for i, (t, b) in enumerate(((t_early, None), (t_late, 0), (t_late + 0.1, None))):
        per2d = {"a": Pose2D(kp, np.ones(17), t=t, frame_index=i),
                 "b": Pose2D(kp, np.ones(17), t=t, frame_index=b)}
        with caplog.at_level("WARNING", logger="poseboard.session"):
            rec.add_pose(Pose3D(t, names, np.zeros((17, 3)), np.ones(17), per2d,
                                format_key="coco17", camera_frames={"a": (t, i), "b": (t, b)}))
        if i == 0:
            assert not [r for r in caplog.records if "recorded video" in r.getMessage()]
    rec.stop(analyze=False)
    jdir = folder / "pose2d_json"
    people_b = [len(json.loads(p.read_text(encoding="utf-8"))["people"])
                for p in sorted((jdir / "b").iterdir())]
    assert people_b == [0, 1, 0]
    warned = [r for r in caplog.records if "not in its recorded video" in r.getMessage()]
    assert len(warned) == 1 and "b" in warned[0].getMessage()  # only the frame after the start


def test_save_pose2d_off_writes_no_2d_files(tmp_path):
    from poseboard.session import SessionRecorder

    rec = SessionRecorder(tmp_path, save_openpose_json=True)
    folder = rec.start(cams=[], calibrations={}, force=None, board=None, geometry=BoardGeometry(),
                       pose_backend="Fake", save_pose2d=False)
    names = list(FORMATS["coco17"].names)
    per2d = {"a": Pose2D(np.zeros((17, 2)), np.ones(17), frame_index=0)}
    rec.add_pose(Pose3D(rec.t0 + 0.1, names, np.zeros((17, 3)), np.ones(17), per2d,
                        format_key="coco17", camera_frames={"a": (rec.t0 + 0.1, 0)}))
    rec.stop(analyze=False)
    assert (folder / "pose3d.csv").exists()
    assert not list(folder.glob("pose2d_*.csv")) and not (folder / "pose2d_json").exists()
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["pose2d"]["csv"] is None and meta["pose2d"]["openpose_json"] is False


def test_exe_selftest_backend_checks(monkeypatch):
    """The backend part of ``PoseBoard.exe --selftest`` (installer/entry_gui.py), run from
    source: the registry, and with rtmlib ONNX Runtime plus the rtmlib detector class."""
    import importlib
    from pathlib import Path

    if not backend_available("mediapipe")[0]:
        pytest.skip("MediaPipe is not installed")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "installer"))
    entry = importlib.import_module("entry_gui")
    rtm = backend_available("rtmpose")[0]
    monkeypatch.setattr(entry, "_bundled_backends",
                        lambda: ["mediapipe"] + (["rtmpose"] if rtm else []))
    if rtm:
        assert isinstance(entry._rtmlib_cached(), bool)
    monkeypatch.setattr(entry, "_rtmlib_cached", lambda mode="lightweight": False)
    lines = entry.selftest_backends()
    assert "  mediapipe: available" in lines
    assert len([ln for ln in lines if ln.startswith("  ")]) == len(BACKENDS)
    if rtm:
        assert any(ln.startswith("onnxruntime") and ln.split(": ", 1)[1].startswith("OK")
                   for ln in lines)
        assert any("detector class OK" in ln for ln in lines)
        assert any("not in the download cache" in ln for ln in lines)
    else:
        assert "rtmlib: not bundled" in lines


# ------------------------------------------------------------------ backend info, model fields
def test_backend_info_follows_the_selected_options(app, no_boards):
    """The line below the backend list shows the keypoint format of the selected model (OpenPose
    COCO-18 is not BODY_25), also after switching backends."""
    from poseboard.gui.app import backend_description

    assert "OpenPose COCO-18, 18 keypoints" in backend_description("openpose_dnn",
                                                                   {"model": "coco18"})
    assert "OpenPose BODY_25, 25 keypoints" in backend_description("openpose_dnn")
    w = make_window()
    try:
        select(w, "openpose_dnn")
        assert "OpenPose BODY_25, 25 keypoints" in w.backend_info.text()
        w._opt_widgets["model"][1].setCurrentText("coco18")
        assert "OpenPose COCO-18, 18 keypoints" in w.backend_info.text()
        assert "COCO-18" in w.backend_info.toolTip()
        select(w, "mediapipe")
        select(w, "openpose_dnn")
        assert "OpenPose COCO-18, 18 keypoints" in w.backend_info.text()
        w._opt_widgets["model"][1].setCurrentText("body25")
        assert "OpenPose BODY_25, 25 keypoints" in w.backend_info.text()
    finally:
        w.close()


def test_model_fields_take_local_files_and_typed_models(app, no_boards, tmp_path, monkeypatch):
    """rtmlib: pose / person detector model files (empty = the default of the mode); ViTPose
    (transformers) and MMPose: the model list also takes a typed id or a local folder. The file
    picker starts in PoseBoard's models folder."""
    from poseboard.pose import mediapipe_backend as mpb

    w = make_window()
    try:
        select(w, "rtmpose")
        assert {k: v[0] for k, v in w._opt_widgets.items()} == {
            "mode": "combo", "device": "combo", "pose_model": "file", "det_model": "file"}
        assert w.backend_options()["pose_model"] == "" and w.backend_options()["det_model"] == ""
        w._opt_widgets["det_model"][1].setText(str(tmp_path / "yolox_tiny.onnx"))
        assert w.backend_options()["det_model"] == str(tmp_path / "yolox_tiny.onnx")
        monkeypatch.setattr(mpb, "MODEL_DIR", tmp_path / "models")
        seen = []
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (seen.append(a), ("", ""))[1]))
        w._opt_widgets["pose_model"][1].setText("")
        w._pick_option_file("pose_model", w._opt_widgets["pose_model"][1])
        assert seen[-1][2] == str(tmp_path / "models") and "*.zip" in seen[-1][3]
        select(w, "rtmo")
        assert "pose_model" in w._opt_widgets and "det_model" not in w._opt_widgets

        select(w, "vitpose_hf")
        kind, combo, choices = w._opt_widgets["model"]
        assert kind == "edit_combo" and combo.isEditable()
        assert w._opt_widgets["person_detector"][0] == "combo"
        assert w._opt_widgets["detector_model"][0] == "text"
        combo.setEditText(str(tmp_path / "vitpose-base-simple"))
        assert w.backend_options()["model"] == str(tmp_path / "vitpose-base-simple")
        combo.setCurrentIndex(2)
        assert w.backend_options()["model"] == choices[2]
        combo.setEditText("my-org/my-vitpose")
        assert w.backend_options()["detector_model"] == ""
        select(w, "mmpose")
        assert w._opt_widgets["model"][0] == "edit_combo"
        select(w, "vitpose_hf")
        assert w.backend_options()["model"] == "my-org/my-vitpose"  # kept per backend
        saved = json.loads(json.dumps(w.pose_settings()))
    finally:
        w.close()
    w2 = make_window()
    try:
        w2.apply_pose_settings(saved)
        assert w2.current_backend() == "vitpose_hf"
        assert w2.backend_options()["model"] == "my-org/my-vitpose"
        select(w2, "rtmpose")
        assert w2.backend_options()["det_model"] == str(tmp_path / "yolox_tiny.onnx")
    finally:
        w2.close()


# ------------------------------------------------------------------ loading and stopping
def test_settings_changed_while_loading_are_used(app, no_boards, monkeypatch, tmp_path):
    """Min confidence, outlier threshold, smoothing and the board's world camera changed while
    the model loads are applied once it exists, and session.json records the values in use."""
    from poseboard.geometry import BoardPose

    monkeypatch.setitem(BACKENDS, "gated", fake_spec("gated", "create_gated_detector",
                                                     "Gated backend"))
    GATE.clear()
    w = make_window()
    try:
        w.out_dir.setText(str(tmp_path / "rec"))
        w.connect_sim()
        assert pump_until(app, lambda: w._wii_connected())
        select(w, "gated")
        w.b_pose.setChecked(True)
        pw = w.pose_worker
        assert pw.loading and w.min_score.isEnabled()  # adjustable while loading
        w.min_score.setValue(0.8)
        w.reproj_thr.setValue(40.0)
        w.chk_smooth.setChecked(True)
        w.board = BoardPose(BOARD_T, "pnp", {}, world_camera="cam7")
        w.b_rec.setChecked(True)  # a recording started while the model still loads
        assert w.recorder.recording and pw.loading, w.warnings
        GATE.set()
        assert pump_until(app, lambda: pw.ready_seen)
        est = pw.estimator
        assert est.min_score == pytest.approx(0.8) and est.reproj_threshold_px == 40.0
        assert est.smoothing == "one_euro" and est.world_camera == "cam7"
        w.b_rec.setChecked(False)
        assert w.recorder.wait_post_processing(30)
    finally:
        GATE.set()
        w.close()
    folders = sorted(p for p in (tmp_path / "rec").iterdir() if p.is_dir())
    meta = json.loads((folders[-1] / "session.json").read_text(encoding="utf-8"))
    info = meta["pose_info"]
    assert info["min_score"] == pytest.approx(0.8) and info["reproj_threshold_px"] == 40.0
    assert info["smoothing"] == "one_euro"


def test_plugin_exiting_while_loading_is_reported(app, no_boards, tmp_path):
    """A plugin that calls sys.exit() at import (e.g. "needs CUDA") is reported like any other
    failed start, instead of a status stuck at "Loading..."."""
    from poseboard.gui.app import PLUGIN_KEY

    plugin = tmp_path / "exit_plugin.py"
    plugin.write_text("import sys\nsys.exit('this plugin needs CUDA')\n", encoding="utf-8")
    w = make_window()
    try:
        select(w, PLUGIN_KEY)
        w.plugin_path.setText(str(plugin))
        w.b_pose.setChecked(True)
        pw = w.pose_worker
        assert pump_until(app, lambda: w.pose_worker is None)
        assert "this plugin needs CUDA" in w.warnings[-1] and not w.b_pose.isChecked()
        assert not pw.alive and not pw.loading and "sys.exit" in pw.load_error
    finally:
        w.close()


def test_pose_thread_ending_while_loading_is_reported(app, no_boards):
    from poseboard.gui.app import PoseWorker

    class Dying(PoseWorker):
        def _run(self):
            """Ends without reporting a result (as if the thread was killed)."""

    w = make_window()
    try:
        pw = Dying(None, dict, dict, w.recorder, factory=lambda: None, label="Dying backend")
        pw.start()
        pw._thread.join(2)
        assert pw.loading and pw.load_crashed
        w.pose_worker = pw
        w._poll_pose_worker()
        assert w.pose_worker is None
        assert "ended while loading" in w.warnings[-1] and "Dying backend" in w.warnings[-1]
    finally:
        w.close()


def test_stop_while_loading_never_runs_two_loads_at_once(app, no_boards, monkeypatch):
    """Stop during "Loading..." abandons the start (its download goes on in the background); a
    new start waits for it, so two threads never create a model (write its files) at once."""
    monkeypatch.setitem(BACKENDS, "gated", fake_spec("gated", "create_gated_detector",
                                                     "Gated backend"))
    GATE.clear()
    LOADS.update(active=0, peak=0, count=0)
    w = make_window()
    try:
        select(w, "gated")
        w.b_pose.setChecked(True)
        first = w.pose_worker
        assert pump_until(app, lambda: LOADS["count"] == 1)
        w.b_pose.setChecked(False)
        assert w.pose_worker is None and first in w._abandoned_workers and first.alive
        w.b_pose.setChecked(True)
        second = w.pose_worker
        assert pump_until(app, lambda: second.load_waiting)
        pump(app, 0.1)
        assert "waiting for the previous model load" in w.pose_label.text()
        GATE.set()
        assert pump_until(app, lambda: second.ready_seen)
        assert LOADS["count"] == 2 and LOADS["peak"] == 1
        assert pump_until(app, lambda: not first.alive)  # it closed its own estimator
        assert first.estimator is not None and first.estimator.detector.closed
        assert not second.estimator.detector.closed
    finally:
        GATE.set()
        w.close()


class _SlowEstimator(PoseEstimator):
    name = "slow"

    def __init__(self):
        self.calls = 0
        self.closed = False

    def process(self, frames, cams):
        self.calls += 1
        time.sleep(1.0)
        return None

    def close(self):
        self.closed = True


def test_stop_does_not_block_the_window_during_a_slow_frame(app, no_boards):
    """Stop while a slow backend processes a frame returns at once; the thread finishes the
    frame, drops it and closes the estimator itself. Also: the warning when the board's world
    camera (no extrinsics) is not running."""
    from poseboard.camera import Frame
    from poseboard.geometry import BoardPose
    from poseboard.gui.app import PoseWorker

    n = [0]

    def frames():
        n[0] += 1
        return {"cam0": Frame(n[0], time.perf_counter(), np.zeros((4, 4, 3), np.uint8))}

    est = _SlowEstimator()
    w = make_window()
    try:
        pw = PoseWorker(est, frames, dict, w.recorder)
        pw.start()
        deadline = time.perf_counter() + 5
        while est.calls == 0 and time.perf_counter() < deadline:
            time.sleep(0.01)
        w.pose_worker = pw
        w.board = BoardPose(BOARD_T, "pnp", {}, world_camera="cam1")
        w._update_warnings(time.perf_counter())
        assert "registered in cam1" in w.warn_label.text() and "not running" in w.warn_label.text()
        t = time.perf_counter()
        w.toggle_pose(False)
        assert time.perf_counter() - t < 0.3
        assert w.pose_worker is None and pw in w._abandoned_workers
        pw._thread.join(3)
        assert not pw.alive and est.closed and est.calls == 1
    finally:
        w.close()


def test_exe_selftest_accepts_keypoints_in_the_aspect_widened_rtmpose_crop(monkeypatch):
    """Regression (Windows CI run 36020082305): on the black 320x240 test image RTMPose put
    keypoints at y ~ 386, outside the box but inside the crop it actually saw (rtmlib pads the
    box 1.25x and widens it to the model's 192:256 aspect ratio, so y spans -147 .. 387). Such
    keypoints must pass; keypoints outside that crop must still fail."""
    import importlib
    from pathlib import Path

    from poseboard.pose.detectors import rtmlib_det as rd

    if not backend_available("rtmpose")[0]:
        pytest.skip("rtmlib / onnxruntime not installed")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "installer"))
    entry = importlib.import_module("entry_gui")
    lo, hi = entry._rtmpose_crop([0.0, 0.0, 320.0, 240.0], (192, 256))
    np.testing.assert_allclose(lo, [-40.0, -146.6667], atol=1e-3)
    np.testing.assert_allclose(hi, [360.0, 386.6667], atol=1e-3)
    img = np.zeros((240, 320, 3), np.uint8)

    def run(y_max):
        def pose_model(image, bboxes=()):
            kp = np.column_stack([np.linspace(80, 359, 17), np.linspace(-10, y_max, 17)])
            return kp[None], np.full((1, 17), 0.8)

        pose_model.model_input_size = (192, 256)
        real = rd.RTMLibDetector("rtmpose", pose_model, lambda image: np.zeros((0, 4)))
        return entry._selftest_real_rtmpose(real, img)

    assert "OK (1 person with 17 keypoints" in run(385.625)  # the values seen on Windows CI
    with pytest.raises(AssertionError, match="not mapped back"):
        run(420.0)


def test_exe_selftest_runs_the_real_pose_model_on_a_forced_box(monkeypatch):
    """``PoseBoard.exe --selftest`` with the real rtmlib models: besides YOLOX on a black image,
    RTMPose itself runs (on a forced whole-image box) and its output is mapped to a person.
    Here the two models are stand-ins with the same call signatures."""
    import importlib
    from pathlib import Path

    from poseboard.pose import detectors
    from poseboard.pose.detectors import rtmlib_det as rd

    if not (backend_available("mediapipe")[0] and backend_available("rtmpose")[0]):
        pytest.skip("MediaPipe or rtmlib / onnxruntime not installed")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "installer"))
    entry = importlib.import_module("entry_gui")
    monkeypatch.setattr(entry, "_bundled_backends", lambda: ["mediapipe", "rtmpose"])
    monkeypatch.setattr(entry, "_rtmlib_cached", lambda mode="lightweight": True)
    calls = []

    def pose_model(image, bboxes=()):
        calls.append([list(map(float, b)) for b in bboxes])
        x1, y1, x2, y2 = bboxes[0]
        kp = np.column_stack([np.linspace(x1 + 10, x2 - 10, 17), np.linspace(y1 + 10, y2 - 10, 17)])
        return np.repeat(kp[None], len(bboxes), 0), np.full((len(bboxes), 17), 0.8)

    def create(key, **kw):
        assert key == "rtmpose" and kw == {"mode": "lightweight", "device": "cpu"}
        return rd.RTMLibDetector("rtmpose", pose_model, lambda image: np.zeros((0, 4)))

    monkeypatch.setattr(detectors, "create_detector", create)
    lines = entry.selftest_backends()
    assert any("lightweight YOLOX + RTMPose): OK (1 person with 17 keypoints" in ln
               for ln in lines), lines
    assert [[0.0, 0.0, 320.0, 240.0]] in calls
