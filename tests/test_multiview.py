"""MultiViewEstimator with a scripted detector: robust triangulation, subject selection, the
single-view / 2D-only paths, the camera-frame rule and One-Euro smoothing."""

import numpy as np
import pytest

from poseboard.geometry import BoardGeometry, BoardPose
from poseboard.pose.base import MODE_2D_ONLY, MODE_SINGLE_VIEW_3D, MODE_TRIANGULATED
from poseboard.pose.detectors.base import Person2D
from poseboard.pose.filters import OneEuroFilter
from poseboard.pose.formats import FORMATS
from poseboard.pose.multiview import MultiViewEstimator
from poseboard.pose.subject import board_polygon_px, foot_point, select_subject
from poseboard.pose.triangulation import triangulate_keypoints, triangulate_robust
from tests.fakes import FakeDetector, Scene, project_person, standing_person
from tests.test_core import board_transform, make_cam
from tests.test_geometry_io import no_ext

TARGET = (0.8, 0.5, 0.9)
T_BOARD = board_transform()  # board centre at (0.8, 0.5, 0.053), rotated 20 deg
BOARD = BoardPose(T_BOARD, "triangulation", {})
TRUTH = standing_person("coco17", offset=(0.8, 0.5, 0.053), yaw_deg=20)
IMG = np.zeros((4, 4, 3), np.uint8)


def cameras(n=4):
    all_cams = [make_cam("cam0", (0.8, -2.5, 1.3), TARGET),
                make_cam("cam1", (3.4, -0.2, 1.4), TARGET),
                make_cam("cam2", (-1.8, -0.9, 1.2), TARGET),
                make_cam("cam3", (0.6, 3.6, 1.6), TARGET)]
    return {c.name: c for c in all_cams[:n]}


def frames_for(cams, t=1.0, dt=0.01):
    return {n: (t + i * dt, IMG) for i, n in enumerate(cams)}


def estimator(scene, fmt="coco17", provides_3d=False, **kw):
    return MultiViewEstimator(FakeDetector(fmt, scene, provides_3d=provides_3d), **kw)


# ---------------------------------------------------------------- triangulation
def test_triangulation_exact_and_times():
    cams = cameras(2)
    est = estimator(Scene(cams, TRUTH, "coco17"))
    pose = est.process(frames_for(cams), cams)
    assert pose.mode == MODE_TRIANGULATED and pose.format_key == "coco17"
    np.testing.assert_allclose(pose.keypoints, TRUTH, atol=1e-6)
    assert pose.views_used == ["cam0", "cam1"] and pose.reproj_error_px < 1e-3
    assert pose.t == pytest.approx(1.005)  # mean capture time of the views used
    assert set(pose.per_camera_2d) == {"cam0", "cam1"}
    assert pose.per_camera_2d["cam1"].t == pytest.approx(1.01)
    assert pose.camera_frames == {"cam0": (1.0, None), "cam1": (1.01, None)}
    assert pose.names == list(FORMATS["coco17"].names) and np.all(pose.scores == 0.9)


def test_triangulation_rejects_a_gross_outlier_view():
    rng = np.random.default_rng(1)
    for n_cams in (3, 4):
        cams = cameras(n_cams)
        bad = "cam2"
        offsets = rng.normal(0, 40, (17, 2)) + np.array([120.0, -60.0])
        scene = Scene(cams, TRUTH, "coco17", outliers={bad: offsets})
        pose = estimator(scene).process(frames_for(cams), cams)
        assert pose.mode == MODE_TRIANGULATED
        np.testing.assert_allclose(pose.keypoints, TRUTH, atol=1e-5)
        assert bad not in pose.views_used and len(pose.views_used) == n_cams - 1
        assert any(n.startswith(f"{bad}: 17 of 17 keypoints rejected") for n in pose.notes)
        assert pose.reproj_error_px < 1e-3
        # plain weighted DLT (no rejection) is pulled away by the outlier view
        plain, _ = triangulate_keypoints(list(cams.values()),
                                         [scene(n, 1.0, IMG)[0].keypoints for n in cams],
                                         [np.ones(17)] * n_cams)
        assert np.nanmax(np.linalg.norm(plain - TRUTH, axis=1)) > 0.05


def test_triangulation_rejects_single_wrong_keypoints_and_keeps_the_view():
    cams = cameras(3)
    fmt = FORMATS["coco17"]
    lw, rw = fmt.index("left_wrist"), fmt.index("right_wrist")
    swap = np.zeros((17, 2))
    p = project_person(cams["cam1"], TRUTH, "coco17")
    swap[lw] = p.keypoints[rw] - p.keypoints[lw]  # left/right wrists swapped in cam1
    swap[rw] = p.keypoints[lw] - p.keypoints[rw]
    pose = estimator(Scene(cams, TRUTH, "coco17", outliers={"cam1": swap})).process(
        frames_for(cams), cams)
    np.testing.assert_allclose(pose.keypoints, TRUTH, atol=1e-5)
    assert pose.views_used == ["cam0", "cam1", "cam2"]
    assert "cam1: 2 of 17 keypoints rejected as outliers (reprojection error > 15 px)" in pose.notes


def test_triangulation_with_noise_and_low_scores():
    cams = cameras(3)
    pose = estimator(Scene(cams, TRUTH, "coco17", noise_px=1.0)).process(frames_for(cams), cams)
    err = np.linalg.norm(pose.keypoints - TRUTH, axis=1)
    assert err.max() < 0.02 and 0.2 < pose.reproj_error_px < 2.0
    # keypoints below min_score are not used; with a single view left they are NaN
    tri = triangulate_robust(list(cams.values()), [project_person(c, TRUTH, "coco17").keypoints
                                                   for c in cams.values()],
                             [np.r_[0.1, np.ones(16)], np.r_[0.1, np.ones(16)], np.ones(17)],
                             min_score=0.3)
    assert np.all(np.isnan(tri.points[0])) and np.all(np.isfinite(tri.points[1:]))
    assert tri.scores[0] == 0 and not tri.used[:, 0].any()


def test_outlier_rejection_respects_min_views():
    cams = cameras(3)
    scene = Scene(cams, TRUTH, "coco17", outliers={"cam2": np.full((17, 2), 90.0)})
    pose = estimator(scene, min_views=3).process(frames_for(cams), cams)  # nothing may be dropped
    assert pose.views_used == ["cam0", "cam1", "cam2"] and pose.reproj_error_px > 15
    assert any("mean reprojection error" in n for n in pose.notes)


# ---------------------------------------------------------------- subject selection
BYSTANDER = standing_person("coco17", offset=(0.6, -1.0, 0.0), yaw_deg=-10)  # closer to cam0


def test_board_polygon_contains_the_feet_of_the_subject():
    cam = cameras(1)["cam0"]
    poly = board_polygon_px(BOARD, cam, BoardGeometry())
    assert poly is not None and poly.shape[1] == 2 and len(poly) >= 4
    import cv2
    fmt = FORMATS["coco17"]
    f_subj = foot_point(project_person(cam, TRUTH, fmt), fmt)
    f_by = foot_point(project_person(cam, BYSTANDER, fmt), fmt)
    c = poly.astype(np.float32).reshape(-1, 1, 2)
    assert cv2.pointPolygonTest(c, tuple(map(float, f_subj)), False) > 0
    assert cv2.pointPolygonTest(c, tuple(map(float, f_by)), False) < 0
    # frame rule: a checkerboard-frame board is not drawn into a camera without extrinsics
    assert board_polygon_px(BOARD, no_ext(cam)) is None
    in_cam = BoardPose(T_BOARD, "pnp", {}, world_camera="cam0")
    assert board_polygon_px(in_cam, cam) is None
    assert board_polygon_px(in_cam, no_ext(cam)) is not None


def test_subject_on_the_board_wins_over_a_bigger_bystander():
    cams = cameras(1)
    scene = Scene(cams, TRUTH, "coco17", bystanders=[BYSTANDER], with_3d=True)
    people = scene("cam0", 1.0, IMG)
    area = [(p.box()[2] - p.box()[0]) * (p.box()[3] - p.box()[1]) for p in people]
    assert area[1] > 1.5 * area[0]  # the bystander is much bigger in the image
    # without the board: the biggest person
    est = estimator(scene, provides_3d=True)
    pose = est.process(frames_for(cams), cams)
    np.testing.assert_allclose(pose.keypoints, BYSTANDER, atol=1e-3)
    # with the board: the person standing on it (single camera, 3D skeleton -> PnP)
    est = estimator(scene, provides_3d=True, board_provider=lambda: (BOARD, BoardGeometry()))
    pose = est.process(frames_for(cams), cams)
    assert pose.mode == MODE_SINGLE_VIEW_3D
    np.testing.assert_allclose(pose.keypoints, TRUTH, atol=1e-3)
    # the provider may also return the BoardPose alone; two cameras triangulate the subject
    cams2 = cameras(2)
    scene2 = Scene(cams2, TRUTH, "coco17", bystanders=[BYSTANDER])
    pose = estimator(scene2, board_provider=lambda: BOARD).process(frames_for(cams2), cams2)
    assert pose.mode == MODE_TRIANGULATED
    np.testing.assert_allclose(pose.keypoints, TRUTH, atol=1e-5)


def test_select_subject_tracking_and_size_fallback():
    fmt = FORMATS["coco17"]

    def person(x0, size, score=0.9):
        kp = np.column_stack([np.linspace(x0, x0 + size / 3, 17), np.linspace(100, 100 + size, 17)])
        return Person2D(kp, np.full(17, 0.9), score=score)

    small, big = person(100, 200), person(500, 400)
    assert select_subject([], fmt=fmt) is None
    assert select_subject([small], fmt=fmt) is small
    assert select_subject([small, big], fmt=fmt) is big  # area x score
    assert select_subject([small, big], previous_bbox=small.box() + 5, fmt=fmt) is small  # IoU
    far_prev = np.array([2000, 2000, 2100, 2300.0])
    assert select_subject([small, big], previous_bbox=far_prev, fmt=fmt) is big
    weak_big = person(500, 400, score=0.1)
    assert select_subject([small, weak_big], fmt=fmt) is small


def test_tracking_keeps_the_subject_between_frames():
    """Without a board, the person chosen first stays the subject when a bigger one appears."""
    cams = cameras(1)
    scene = Scene(cams, TRUTH, "coco17", with_3d=True)
    est = estimator(scene, provides_3d=True)
    est.process(frames_for(cams), cams)
    scene.bystanders = [BYSTANDER]
    pose = est.process(frames_for(cams, t=1.1), cams)
    np.testing.assert_allclose(pose.keypoints, TRUTH, atol=1e-3)


# ---------------------------------------------------------------- single view / 2D only
def test_single_view_3d():
    cams = cameras(1)
    est = estimator(Scene(cams, TRUTH, "coco17", with_3d=True), provides_3d=True)
    pose = est.process({"cam0": (2.5, IMG)}, cams)
    assert pose.mode == MODE_SINGLE_VIEW_3D and pose.views_used == ["cam0"] and pose.t == 2.5
    np.testing.assert_allclose(pose.keypoints, TRUTH, atol=1e-3)
    assert pose.reproj_error_px < 0.1 and np.all(pose.scores == 0.9)


def test_single_view_3d_for_other_formats():
    cams = cameras(1)
    for key in ("halpe26", "wholebody133", "mediapipe33"):
        truth = standing_person(key, offset=(0.8, 0.5, 0.053), yaw_deg=20)
        est = estimator(Scene(cams, truth, key, with_3d=True), fmt=key, provides_3d=True)
        pose = est.process(frames_for(cams), cams)
        assert pose.mode == MODE_SINGLE_VIEW_3D and pose.format_key == key
        np.testing.assert_allclose(pose.keypoints, truth, atol=2e-3)


def test_2d_only_with_one_camera_and_a_2d_backend():
    cams = cameras(1)
    est = estimator(Scene(cams, TRUTH, "coco17"))  # no 3D skeleton from the detector
    pose = est.process({"cam0": (3.0, IMG)}, cams)
    assert pose.mode == MODE_2D_ONLY and pose.t == 3.0 and pose.views_used == []
    assert np.all(np.isnan(pose.keypoints)) and np.all(pose.scores == 0)
    assert pose.reproj_error_px is None
    np.testing.assert_allclose(pose.per_camera_2d["cam0"].keypoints, cams["cam0"].project(TRUTH))
    assert any("gives no 3D skeleton" in n for n in pose.notes)
    # emit_2d_only=False: None instead (the MediaPipePose behaviour)
    est = estimator(Scene(cams, TRUTH, "coco17"), emit_2d_only=False)
    assert est.process({"cam0": (3.0, IMG)}, cams) is None


def test_nobody_detected_returns_none_and_notes_missing_cameras():
    cams = cameras(2)
    assert estimator(Scene(cams, TRUTH, "coco17", missing={"cam0", "cam1"})).process(
        frames_for(cams), cams) is None
    pose = estimator(Scene(cams, TRUTH, "coco17", missing={"cam1"})).process(frames_for(cams), cams)
    assert pose.mode == MODE_2D_ONLY and "cam1: no person detected" in pose.notes
    assert set(pose.per_camera_2d) == {"cam0"} and set(pose.camera_frames) == {"cam0", "cam1"}


def test_wrong_keypoint_count_is_reported():
    cams = cameras(1)
    bad = {"cam0": [Person2D(np.zeros((5, 2)), np.ones(5))]}
    with pytest.raises(ValueError, match="5 keypoints"):
        MultiViewEstimator(FakeDetector("coco17", bad)).process(frames_for(cams), cams)


# ---------------------------------------------------------------- camera-frame rule
def test_frames_are_never_mixed():
    c = cameras(3)
    # cam0 has extrinsics but misses the subject; cam1 (no extrinsics) sees it: 2D only, never a
    # skeleton lifted in cam1's camera frame
    cams = {"cam0": c["cam0"], "cam1": no_ext(c["cam1"])}
    scene = Scene(c, TRUTH, "coco17", missing={"cam0"}, with_3d=True)
    pose = estimator(scene, provides_3d=True).process(frames_for(cams), cams)
    assert pose.mode == MODE_2D_ONLY and np.all(np.isnan(pose.keypoints))
    assert set(pose.per_camera_2d) == {"cam1"}
    assert any(n.startswith("cam1 not used for 3D: no extrinsics") for n in pose.notes)

    # two calibrated cameras + one without extrinsics (whose detection is wrong on purpose)
    cams = {"cam0": c["cam0"], "cam1": c["cam1"], "cam2": no_ext(c["cam2"])}
    scene = Scene(c, TRUTH, "coco17", outliers={"cam2": np.full((17, 2), 200.0)})
    pose = estimator(scene).process(frames_for(cams), cams)
    assert pose.mode == MODE_TRIANGULATED and pose.views_used == ["cam0", "cam1"]
    np.testing.assert_allclose(pose.keypoints, TRUTH, atol=1e-5)
    assert set(pose.per_camera_2d) == {"cam0", "cam1", "cam2"}  # 2D kept for display
    assert pose.t == pytest.approx(1.005)  # time of the views used only

    # no extrinsics at all: the world frame is the camera frame of world_camera, only it is used
    cams = {"cam0": no_ext(c["cam0"]), "cam1": no_ext(c["cam1"])}
    truth_cam1 = c["cam1"].world_to_camera(TRUTH)
    scene = Scene(c, TRUTH, "coco17", with_3d=True)  # what the (real) cameras see
    est = estimator(scene, provides_3d=True)
    est.world_camera = "cam1"
    pose = est.process(frames_for(cams), cams)
    assert pose.mode == MODE_SINGLE_VIEW_3D and pose.views_used == ["cam1"]
    np.testing.assert_allclose(pose.keypoints, truth_cam1, atol=1e-3)
    assert pose.t == pytest.approx(1.01)
    assert any("cam0 not used for 3D: without extrinsics the world frame is the camera frame "
               "of cam1" in n for n in pose.notes)
    scene.missing = {"cam1"}
    pose = est.process(frames_for(cams), cams)
    assert pose.mode == MODE_2D_ONLY and set(pose.per_camera_2d) == {"cam0"}


# ---------------------------------------------------------------- smoothing
def test_one_euro_filter_reduces_jitter_follows_steps_and_resets():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    ts = np.arange(0, 4, 1 / 30)
    raw = 1.0 + rng.normal(0, 0.01, len(ts))
    out = np.array([f(x, t) for x, t in zip(raw, ts)])
    assert np.std(out[30:]) < 0.4 * np.std(raw[30:])
    # a step is followed (the cutoff rises with the speed)
    step = np.r_[np.zeros(30), np.full(60, 0.5)]
    f = OneEuroFilter(min_cutoff=1.0, beta=1.0)
    out = np.array([f(x, t) for x, t in zip(step, ts[:90])])
    assert abs(out[-1] - 0.5) < 0.01
    # vectors: a NaN element is reset and restarts from the raw value
    f = OneEuroFilter()
    f(np.array([0.0, 0.0]), 0.0)
    f(np.array([0.1, 0.1]), 0.033)
    y = f(np.array([np.nan, 0.2]), 0.066)
    assert np.isnan(y[0]) and 0.0 < y[1] < 0.2
    y = f(np.array([5.0, 0.2]), 0.1)
    assert y[0] == 5.0 and y[1] < 0.2
    # time going backwards (a new session): full reset
    assert f(np.array([1.0, 1.0]), 0.0).tolist() == [1.0, 1.0]
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0)


def test_estimator_smoothing_reduces_jitter_and_resets_on_dropouts():
    cams = cameras(2)
    rng = np.random.default_rng(3)
    n = 60
    dets = [{c: [project_person(cams[c], TRUTH, "coco17", noise_px=2.0, rng=rng)] for c in cams}
            for _ in range(n)]
    dets[40] = {c: [] for c in cams}  # nobody detected in frame 40
    lw = FORMATS["coco17"].index("left_wrist")
    dets[50]["cam0"][0].scores[lw] = 0.05  # the left wrist disappears in frame 50
    frame = [0]

    def script(cam, t, img):
        return dets[frame[0]][cam]

    raw_est = estimator(script)
    smooth_est = estimator(script, smoothing="one_euro")
    raw, smooth = [], []
    for i in range(n):
        frame[0] = i
        fr = frames_for(cams, t=i / 30)
        a, b = raw_est.process(fr, cams), smooth_est.process(fr, cams)
        if i == 40:
            assert a is None and b is None
            continue
        raw.append(a.keypoints)
        smooth.append(b.keypoints)
        if i == 41:  # first pose after the dropout: the filter restarted
            np.testing.assert_array_equal(a.keypoints, b.keypoints)
        if i == 50:  # the left wrist is lost ...
            assert np.all(np.isnan(b.keypoints[lw]))
        if i == 51:  # ... and restarts from its raw value, while the others stay smoothed
            np.testing.assert_array_equal(a.keypoints[lw], b.keypoints[lw])
            assert not np.allclose(a.keypoints[0], b.keypoints[0])
    raw, smooth = np.array(raw[5:39]), np.array(smooth[5:39])
    assert np.mean(np.std(smooth, axis=0)) < 0.6 * np.mean(np.std(raw, axis=0))
    with pytest.raises(ValueError):
        estimator(script, smoothing="kalman")


def test_estimator_info_and_close():
    det = FakeDetector("halpe26")
    est = MultiViewEstimator(det, smoothing="one_euro")
    info = est.info()
    assert info["keypoint_format"] == "halpe26" and info["pose2sim_model"] == "HALPE_26"
    assert info["backend"] == "fake" and info["smoothing"] == "one_euro"
    assert est.keypoint_names == list(FORMATS["halpe26"].names)
    assert est.skeleton == list(FORMATS["halpe26"].skeleton) and est.name == "fake"
    est.close()
    assert det.closed


# ---------------------------------------------------------------- spotter next to the board
def _spotter(angle_deg=200.0, dist=0.55):
    a = np.deg2rad(angle_deg)
    return standing_person("coco17", offset=(0.8 + dist * np.cos(a), 0.5 + dist * np.sin(a), 0.0))


def test_spotter_next_to_the_board_does_not_replace_the_tracked_subject():
    """The subject's ankles become unsure in one camera (feet cut off, board edge, low scores)
    while a spotter stands next to the board: the tracked subject is kept, instead of
    triangulating two different people (which gave a plausible reprojection error)."""
    cams = cameras(2)
    fmt = FORMATS["coco17"]
    feet = [fmt.index(n) for n in ("left_ankle", "right_ankle")]
    spot = _spotter()
    state = {"weak_feet": False, "hidden": False}

    def script(cam, t, img):
        people = []
        if not (state["hidden"] and cam == "cam0"):
            p = project_person(cams[cam], TRUTH, "coco17")
            if state["weak_feet"] and cam == "cam0":
                p.scores[feet] = 0.25  # below FOOT_MIN_SCORE: no foot point
            people.append(p)
        people.append(project_person(cams[cam], spot, "coco17"))
        return people

    board = lambda: (BOARD, BoardGeometry())  # noqa: E731
    est = estimator(script, board_provider=board)
    pose = est.process(frames_for(cams), cams)
    np.testing.assert_allclose(pose.keypoints, TRUTH, atol=1e-6)
    state["weak_feet"] = True
    pose = est.process(frames_for(cams, t=1.1), cams)
    assert pose.mode == MODE_TRIANGULATED and pose.views_used == ["cam0", "cam1"]
    np.testing.assert_allclose(pose.per_camera_2d["cam0"].keypoints,
                               cams["cam0"].project(TRUTH), atol=1e-6)
    np.testing.assert_allclose(pose.keypoints[:-2], TRUTH[:-2], atol=1e-6)

    # the subject is hidden in cam0 and only the spotter is detected there: not used
    est = estimator(script, board_provider=board)
    state.update(weak_feet=False)
    est.process(frames_for(cams), cams)
    state["hidden"] = True
    pose = est.process(frames_for(cams, t=1.1), cams)
    assert "cam0" not in pose.per_camera_2d and pose.mode == MODE_2D_ONLY
    assert any(n.startswith("cam0: nobody detected on the board") for n in pose.notes)
    np.testing.assert_allclose(pose.per_camera_2d["cam1"].keypoints, cams["cam1"].project(TRUTH))


def test_select_subject_alone_or_tracked():
    """A person detected alone is used only on the board, tracked, or when the feet are hidden
    and nobody was tracked; without a board outline anybody alone is the subject."""
    cam = cameras(1)["cam0"]
    fmt = FORMATS["coco17"]
    poly = board_polygon_px(BOARD, cam, BoardGeometry())
    subject = project_person(cam, TRUTH, fmt)
    spotter = project_person(cam, _spotter(), fmt)
    assert select_subject([subject], board_polygon_px=poly, fmt=fmt) is subject
    assert select_subject([spotter], board_polygon_px=poly, fmt=fmt) is None
    assert select_subject([spotter], fmt=fmt) is spotter  # no board: anybody alone
    # the spotter's box overlaps the subject's previous box a lot: tracked, if the feet allow
    assert select_subject([spotter], board_polygon_px=poly, previous_bbox=spotter.box(),
                          fmt=fmt) is spotter
    far = project_person(cam, standing_person("coco17", offset=(-0.6, 1.4, 0.0)), fmt)
    assert select_subject([far], board_polygon_px=poly, previous_bbox=far.box(), fmt=fmt) is None
    # feet hidden: accepted when nobody was tracked, or when it is the tracked person
    lo = [fmt.index(n) for n in ("left_ankle", "right_ankle")]
    hidden = project_person(cam, _spotter(), fmt)
    hidden.scores[lo] = 0.1
    assert select_subject([hidden], board_polygon_px=poly, fmt=fmt) is hidden
    assert select_subject([hidden], board_polygon_px=poly, previous_bbox=subject.box(),
                          fmt=fmt) is None
    subj_hidden = project_person(cam, TRUTH, fmt)
    subj_hidden.scores[lo] = 0.1
    assert select_subject([subj_hidden, spotter], board_polygon_px=poly,
                          previous_bbox=subject.box(), fmt=fmt) is subj_hidden
    # without tracking the spotter near the board is still preferred over a far bystander
    assert select_subject([far, spotter], board_polygon_px=poly, fmt=fmt) is spotter


# ---------------------------------------------------------------- world camera not running
def test_world_camera_not_running_gives_no_3d_in_another_frame():
    """Without extrinsics the board is in the world camera's frame: when that camera is not
    running, no other camera may lift the pose (it would be in another frame)."""
    c = cameras(2)
    cams = {"cam0": no_ext(c["cam0"])}  # only cam0 runs; the board was registered in cam1
    est = estimator(Scene(c, TRUTH, "coco17", with_3d=True), provides_3d=True)
    est.world_camera = "cam1"
    pose = est.process({"cam0": (1.0, IMG)}, cams)
    assert pose.mode == MODE_2D_ONLY and np.all(np.isnan(pose.keypoints))
    assert set(pose.per_camera_2d) == {"cam0"}
    assert any("world camera cam1" in n and "not running" in n for n in pose.notes)
    from poseboard.pose.multiview import world_view
    assert world_view(cams, "cam1") == "cam1" and world_view(cams) == "cam0"
    assert world_view(c, "cam1") is None  # extrinsics: the checkerboard frame


# ---------------------------------------------------------------- too few triangulated points
def test_nothing_triangulated_falls_back_to_single_view_or_2d_only():
    """The second camera's keypoints are all below min_score: not an all-NaN "triangulated"
    pose, but the single-view lift of the confident camera (or 2D only without a 3D skeleton)."""
    cams = cameras(2)
    scene = Scene(cams, TRUTH, "coco17", with_3d=True)

    def script(cam, t, img):
        (p,) = scene(cam, t, img)
        p.scores[:] = 0.9 if cam == "cam0" else 0.4
        return [p]

    pose = estimator(script, provides_3d=True, min_score=0.5).process(frames_for(cams), cams)
    assert pose.mode == MODE_SINGLE_VIEW_3D and pose.views_used == ["cam0"]
    np.testing.assert_allclose(pose.keypoints, TRUTH, atol=1e-3)
    assert any("single-view 3D from cam0" in n for n in pose.notes)
    pose = estimator(script, min_score=0.5).process(frames_for(cams), cams)
    assert pose.mode == MODE_2D_ONLY and np.all(np.isnan(pose.keypoints))
    assert any("no keypoint was seen confidently by two cameras" in n for n in pose.notes)
    assert estimator(script, min_score=0.5, emit_2d_only=False).process(
        frames_for(cams), cams) is None
    # enough triangulated keypoints: triangulation is kept
    pose = estimator(script, provides_3d=True, min_score=0.3).process(frames_for(cams), cams)
    assert pose.mode == MODE_TRIANGULATED


# ---------------------------------------------------------------- One-Euro defaults (meters)
def test_one_euro_defaults_follow_fast_movements_and_keep_sway():
    """Default parameters on coordinates in meters at 30 Hz: a 0.3 m rise in 0.2 s (1.5 m/s)
    lags by at most one frame, a 0.5-1 Hz sway of 2 cm keeps its amplitude, and the jitter of
    a still keypoint is reduced."""
    fs = 30.0
    t = np.arange(0, 6, 1 / fs)

    def run(x):
        f = OneEuroFilter()
        return np.array([f(v, tt) for v, tt in zip(x, t)])

    rise = np.clip((t - 2.0) / 0.2, 0, 1) * 0.3
    y = run(rise)
    lag = t[np.argmax(y >= 0.15)] - t[np.argmax(rise >= 0.15)]
    assert lag <= 1.01 / fs, lag
    assert y[np.argmin(abs(t - 2.2))] > 0.25  # the old beta=0.01 reached 0.14 m
    for freq, kept in ((0.5, 0.95), (1.0, 0.88)):
        s = 0.02 * np.sin(2 * np.pi * freq * t)
        ys = run(s)[len(t) // 2:]
        assert (ys.max() - ys.min()) / 2 > kept * 0.02, (freq, ys.max())
    noise = 0.005 * np.random.default_rng(0).standard_normal(len(t))
    assert np.std(run(1.0 + noise)[30:]) < 0.6 * np.std(noise[30:])
