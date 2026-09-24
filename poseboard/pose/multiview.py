"""Multi-camera 3D pose from any 2D pose backend (``Detector2D``).

``MultiViewEstimator`` runs the detector on every camera image, picks the subject in each
camera (``select_subject``: the person on the Balance Board, else tracking, else the largest
person) and then, following the frame rule (camera frames are never mixed):

* **triangulated**: >= 2 cameras with extrinsics see the subject: weighted DLT with iterative
  outlier rejection per keypoint (``triangulate_robust``). When fewer than
  ``MIN_TRIANGULATED_KEYPOINTS`` keypoints are seen confidently by two cameras and the detector
  gives a 3D skeleton, the single-view lift of the camera with the most confident keypoints is
  used instead (with a note); with no triangulated keypoint at all and no 3D skeleton the pose
  is ``2d_only``;
* **single_view_3d**: one usable camera and a detector that gives a metric body-centred 3D
  skeleton (``Person2D.keypoints_3d``, e.g. MediaPipe): the skeleton is placed with PnP
  (``single_view_lift``);
* **2d_only**: no 3D possible (e.g. one camera and a 2D-only backend): the 3D keypoints are NaN,
  but ``per_camera_2d`` holds the 2D subject of every camera, and ``notes`` say why.

If any camera has extrinsics the world frame is the floor checkerboard and only cameras with
extrinsics are used for 3D; otherwise the world frame is the camera frame of ``world_camera``
(the camera the board was registered with) and only that camera is used: when it is not
running the pose is ``2d_only``, never lifted in another camera's frame.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import cv2
import numpy as np

from poseboard.calibration import CameraCalibration
from poseboard.geometry import BoardGeometry, BoardPose, project_to_image
from poseboard.pose.base import (MODE_2D_ONLY, MODE_SINGLE_VIEW_3D, MODE_TRIANGULATED, Pose2D,
                                 Pose3D, PoseEstimator)
from poseboard.pose.detectors.base import Detector2D, Person2D
from poseboard.pose.filters import OneEuroFilter
from poseboard.pose.formats import KeypointFormat
from poseboard.pose.subject import board_polygon_px, select_subject
from poseboard.pose.triangulation import triangulate_robust

log = logging.getLogger(__name__)

SMOOTHING = (None, "one_euro")
# Fewer triangulated keypoints than this: use the single-view 3D skeleton instead, if any
MIN_TRIANGULATED_KEYPOINTS = 6


def world_view(cams: dict[str, CameraCalibration], world_camera: str | None = None) -> str | None:
    """None if the world frame is the checkerboard (some camera has extrinsics); otherwise the
    camera whose frame is the world frame: ``world_camera`` when set (even if it is not in
    ``cams``: the board and the COP are in its frame, so no other camera may be used), else the
    first camera (no board registered)."""
    if any(c.has_extrinsics for c in cams.values()):
        return None
    if world_camera is not None:
        return world_camera
    return next(iter(cams), None)


def single_view_lift(cam: CameraCalibration, pose2d: Pose2D | Person2D, body_pts: np.ndarray,
                     min_score: float = 0.5) -> np.ndarray | None:
    """Place a metric body-centred 3D skeleton in the world frame using PnP (single camera).

    ``pose2d``: anything with ``keypoints`` (K, 2) pixels and ``scores`` (K,) (a ``Pose2D`` or
    ``Person2D``); ``body_pts`` (K, 3) in meters, any origin and rotation. Needs >= 6 keypoints
    with a score >= ``min_score``. Returns (K, 3) world points (camera frame if the camera
    has no extrinsics), or None."""
    kp2 = np.asarray(pose2d.keypoints, np.float64)
    sc = np.asarray(pose2d.scores, np.float64)
    body_pts = np.asarray(body_pts, np.float64)
    mask = ((sc >= min_score) & np.all(np.isfinite(body_pts), axis=1)
            & np.all(np.isfinite(kp2), axis=1))
    if mask.sum() < 6:
        return None
    obj = body_pts[mask]
    img = kp2[mask]
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, cam.dist, rvec, tvec, True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2] <= 0:
        return None
    R_bc = cv2.Rodrigues(rvec)[0]
    pts_c = body_pts @ R_bc.T + tvec.ravel()
    if not cam.has_extrinsics:
        return pts_c
    return (pts_c - cam.t) @ cam.R  # camera -> world: R^T (X - t)


class _State:
    """Per-estimator tracking/smoothing state (created lazily, see ``MultiViewEstimator``)."""

    def __init__(self):
        self.prev_bbox: dict[str, np.ndarray] = {}
        self.filter: OneEuroFilter | None = None
        self.last_mode: str | None = None
        self.board_error_logged = False


class MultiViewEstimator(PoseEstimator):
    """3D pose from a ``Detector2D`` over one or more cameras (see the module docstring).

    ``min_score``: minimum keypoint confidence used for 3D; ``reproj_threshold_px`` /
    ``min_views``: outlier rejection of the triangulation; ``smoothing``: None or
    ``"one_euro"`` (``smoothing_params`` are passed to ``OneEuroFilter``); ``board_provider``:
    returns the current ``BoardPose`` (or a ``(BoardPose, BoardGeometry)`` pair, or None),
    used to prefer the person standing on the board; ``emit_2d_only``: return a ``2d_only``
    pose (True) or None (False) when no 3D is possible.

    ``process`` returns None when nobody is detected in any camera."""

    # Class-level defaults, so subclasses that bypass __init__ (e.g. test doubles) still work
    detector: Detector2D | None = None
    format: KeypointFormat | None = None
    provides_3d: bool = False
    min_score: float = 0.3
    reproj_threshold_px: float = 15.0
    min_views: int = 2
    smoothing: str | None = None
    smoothing_params: dict | None = None
    board_provider: Callable[[], object] | None = None
    emit_2d_only: bool = True

    def __init__(self, detector: Detector2D, *, min_score: float = 0.3,
                 reproj_threshold_px: float = 15.0, min_views: int = 2,
                 smoothing: str | None = None,
                 board_provider: Callable[[], BoardPose | tuple | None] | None = None,
                 smoothing_params: dict | None = None, emit_2d_only: bool = True):
        if smoothing in ("", "none"):
            smoothing = None
        if smoothing not in SMOOTHING:
            raise ValueError(f"smoothing must be one of {SMOOTHING}, not {smoothing!r}")
        self.detector = detector
        self.format = detector.format
        self.provides_3d = bool(detector.provides_3d)
        self.name = detector.key
        self.keypoint_names = list(detector.format.names)
        self.skeleton = list(detector.format.skeleton)
        self.min_score = float(min_score)
        self.reproj_threshold_px = float(reproj_threshold_px)
        self.min_views = max(2, int(min_views))
        self.smoothing = smoothing
        self.smoothing_params = dict(smoothing_params or {})
        self.board_provider = board_provider
        self.emit_2d_only = bool(emit_2d_only)

    # ------------------------------------------------------------------ helpers
    def _state(self) -> _State:
        st = self.__dict__.get("_mv_state")
        if st is None:
            st = self.__dict__["_mv_state"] = _State()
        return st

    def reset(self) -> None:
        """Forget the tracking and smoothing state (e.g. after the cameras changed)."""
        self.__dict__.pop("_mv_state", None)

    def info(self) -> dict:
        """Description for session.json."""
        det = self.detector
        d = det.info() if det is not None else {"backend": getattr(self, "name", None)}
        fmt = self.format
        if fmt is not None:
            d.update(keypoint_format=fmt.key, pose2sim_model=fmt.pose2sim_model)
        d.update(min_score=self.min_score, reproj_threshold_px=self.reproj_threshold_px,
                 min_views=self.min_views, smoothing=self.smoothing)
        return d

    def _detect(self, cam_name: str, t: float, image: np.ndarray) -> list[Person2D]:
        return list(self.detector.detect(image, t, cam_name) or [])

    def _board(self) -> tuple[BoardPose | None, BoardGeometry | None]:
        if self.board_provider is None:
            return None, None
        try:
            b = self.board_provider()
        except Exception:  # noqa: BLE001
            st = self._state()
            if not st.board_error_logged:
                st.board_error_logged = True
                log.exception("board_provider failed; subject selection ignores the board")
            return None, None
        if isinstance(b, tuple):
            return b[0], b[1] if len(b) > 1 else None
        return b, None

    def _reset_smoothing(self) -> None:
        st = self._state()
        if st.filter is not None:
            st.filter.reset()
        st.last_mode = None

    def _smooth(self, kps: np.ndarray, t: float, mode: str) -> np.ndarray:
        st = self._state()
        if self.smoothing != "one_euro":
            return kps
        if st.filter is None:
            st.filter = OneEuroFilter(**self.smoothing_params)
        if mode == MODE_2D_ONLY or st.last_mode != mode:
            st.filter.reset()  # no 3D, or another method: do not blend different estimates
        if mode == MODE_2D_ONLY:
            return kps
        return st.filter(kps, t)

    # ------------------------------------------------------------------ main
    def process(self, frames: dict[str, tuple[float, np.ndarray]],
                cams: dict[str, CameraCalibration]) -> Pose3D | None:
        fmt = self.format
        if fmt is None:
            raise RuntimeError("MultiViewEstimator has no keypoint format (no detector)")
        names = list(fmt.names)
        K = len(names)
        st = self._state()
        board, geo = self._board()
        notes: list[str] = []

        dets: dict[str, Person2D] = {}
        for name, (t, img) in frames.items():
            people = self._detect(name, t, img)
            poly = None
            if people and board is not None and name in cams:
                poly = board_polygon_px(board, cams[name], geo)
            subj = select_subject(people, board_polygon_px=poly,
                                  previous_bbox=st.prev_bbox.get(name), fmt=fmt)
            if subj is None:
                st.prev_bbox.pop(name, None)
                notes.append(f"{name}: no person detected" if not people else
                             f"{name}: nobody detected on the board (a person off the board "
                             "was ignored)")
                continue
            if len(subj.keypoints) != K:
                raise ValueError(f"{self.name}: the detector returned {len(subj.keypoints)} "
                                 f"keypoints, but its format {fmt.key} has {K}")
            dets[name] = subj
            box = subj.box()
            if box is not None:
                st.prev_bbox[name] = box
        camera_frames = {n: (float(t), None) for n, (t, _) in frames.items()}
        if not dets:
            self._reset_smoothing()
            return None
        per2d = {n: Pose2D(p.keypoints.copy(), p.scores.copy(), t=float(frames[n][0]))
                 for n, p in dets.items()}

        result = None  # (mode, keypoints, scores, views, error, t)
        wv = world_view(cams, self.world_camera)
        if wv is None:  # world = checkerboard: only cameras with extrinsics
            calibrated = [n for n in dets if n in cams and cams[n].has_extrinsics]
            for n in dets:
                if n not in cams:
                    notes.append(f"{n} not used for 3D: no calibration")
                elif n not in calibrated:
                    notes.append(f"{n} not used for 3D: no extrinsics (the world frame is the "
                                 "floor checkerboard)")
            if len(calibrated) >= 2:
                result = self._triangulate(calibrated, dets, frames, cams, notes)
                result = self._check_triangulation(result, calibrated, dets, frames, cams, notes)
            elif calibrated:
                result = self._single_view(calibrated[0], dets, frames, cams, notes)
            else:
                notes.append("no camera with extrinsics sees the subject: 2D only")
        else:  # no extrinsics: the world frame is the camera frame of wv
            for n in dets:
                if n != wv:
                    notes.append(f"{n} not used for 3D: without extrinsics the world frame is "
                                 f"the camera frame of {wv}")
            if wv in dets:
                result = self._single_view(wv, dets, frames, cams, notes)
            elif wv not in cams:
                notes.append(f"the world camera {wv} (the camera the board was registered "
                             "with) is not running: 2D only")
            else:
                notes.append(f"the world camera {wv} does not see the subject: 2D only")

        if result is None:
            if not self.emit_2d_only:
                self._reset_smoothing()
                return None
            t2 = float(np.mean([frames[n][0] for n in dets]))
            result = (MODE_2D_ONLY, np.full((K, 3), np.nan), np.zeros(K), [], None, t2)
        mode, kps, conf, views, err, t = result
        kps = self._smooth(kps, t, mode)
        st.last_mode = mode
        return Pose3D(t, names, kps, conf, per2d, mode=mode, views_used=views,
                      reproj_error_px=err, notes=notes, format_key=fmt.key,
                      camera_frames=camera_frames)

    def _triangulate(self, calibrated, dets, frames, cams, notes):
        tri = triangulate_robust([cams[n] for n in calibrated],
                                 [dets[n].keypoints for n in calibrated],
                                 [dets[n].scores for n in calibrated], self.min_score,
                                 self.reproj_threshold_px, self.min_views)
        rejected = tri.rejected
        views = []
        for v, n in enumerate(calibrated):
            if tri.used[v].any():
                views.append(n)
            r = int(rejected[v].sum())
            if r:
                notes.append(f"{n}: {r} of {int(tri.valid[v].sum())} keypoints rejected as "
                             f"outliers (reprojection error > {self.reproj_threshold_px:g} px)")
        err = tri.mean_error
        if err is not None and err > self.reproj_threshold_px:
            notes.append(f"mean reprojection error {err:.1f} px: check the calibration, the "
                         "camera synchronization and that all cameras see the same person")
        if not np.isfinite(tri.points).any():
            notes.append("no keypoint was seen confidently by two cameras")
        t = float(np.mean([frames[n][0] for n in calibrated]))
        return MODE_TRIANGULATED, tri.points, tri.scores, views, err, t

    def _check_triangulation(self, result, calibrated, dets, frames, cams, notes):
        """Too few triangulated keypoints: the single-view lift of the camera with the most
        confident keypoints (detectors with a 3D skeleton), else ``2d_only`` (None) when no
        keypoint could be triangulated."""
        n_ok = int(np.all(np.isfinite(result[1]), axis=1).sum())
        if n_ok >= MIN_TRIANGULATED_KEYPOINTS:
            return result
        if self.provides_3d:
            def confident(n):
                sc = np.nan_to_num(np.asarray(dets[n].scores, np.float64))
                return int((sc >= self.min_score).sum()), float(sc.sum())

            best = max(calibrated, key=confident)
            single = self._single_view(best, dets, frames, cams, notes)
            if single is not None:
                notes.append(f"only {n_ok} keypoints seen confidently by two cameras: "
                             f"single-view 3D from {best}")
                return single
        return None if n_ok == 0 else result  # nothing triangulated: 2D only

    def _single_view(self, name, dets, frames, cams, notes):
        p = dets[name]
        cam = cams.get(name)
        if cam is None:
            notes.append(f"{name}: no calibration: 2D only")
            return None
        if not self.provides_3d or p.keypoints_3d is None:
            label = getattr(self.detector, "label", None) or self.name
            notes.append(f"{name}: only one camera can be used and {label} gives no 3D skeleton: "
                         "2D only (use two or more cameras with extrinsics, or a backend with "
                         "3D such as MediaPipe)")
            return None
        pts = single_view_lift(cam, p, p.keypoints_3d, self.min_score)
        if pts is None:
            notes.append(f"{name}: single-view 3D failed (fewer than 6 confident keypoints): "
                         "2D only")
            return None
        mask = (p.scores >= self.min_score) & np.all(np.isfinite(p.keypoints), axis=1) \
            & np.all(np.isfinite(pts), axis=1)
        err = None
        if mask.any():
            proj = project_to_image(cam, pts[mask])
            err = float(np.linalg.norm(proj - p.keypoints[mask], axis=1).mean())
        return MODE_SINGLE_VIEW_3D, pts, p.scores.copy(), [name], err, float(frames[name][0])

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()
