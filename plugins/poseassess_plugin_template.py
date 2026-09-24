"""PoseAssess -> PoseBoard live pose plugin (TEMPLATE).

Use this file to run your own 3D pose code (e.g. PoseAssess) live inside PoseBoard, so the
pose is recorded on the same clock and in the same world frame as the Wii Balance Board.

How to use
----------
1. Copy this file, e.g. to ``plugins/poseassess_plugin.py`` (keep the template untouched).
2. Fill in the parts marked ``TODO``: ``_load_model`` and ONE of the two detection hooks
   (``detect_per_camera`` for case (a) or ``detect_3d_multiview`` for case (b), see below).
3. In PoseBoard: tab "4 Record" -> choose "Plugin (.py)" -> pick your file with "..." ->
   "Start Pose Estimation". Errors raised in ``process`` are shown next to the button.

As shipped, this template loads and runs but detects nothing (``process`` returns None), so
PoseBoard simply shows "(no person detected)". That makes it safe to test the wiring first.

Dependencies: the plugin runs inside PoseBoard's own Python interpreter. ``POSEASSESS_DIR``
only makes PoseAssess's source code importable; its third-party packages (torch, onnxruntime,
mmpose, ...) must be installed in PoseBoard's environment too, with ``pip install --no-deps``
followed by the missing packages so that only ONE OpenCV package is installed (see the README,
"Option A"). The packaged PoseBoard.exe cannot load extra packages: use the source install.

What PoseBoard gives you (every call to ``process``)
----------------------------------------------------
``frames``: ``{camera name: (t, bgr_image)}``
    The newest frame of every camera. ``t`` is ``time.perf_counter()`` taken when the frame
    was read (the same clock as the Wii samples); ``bgr_image`` is an OpenCV ``uint8``
    array of shape (H, W, 3) in BGR order (convert with ``cv2.cvtColor(img,
    cv2.COLOR_BGR2RGB)`` if your model expects RGB).
``cams``: ``{camera name: CameraCalibration}`` (see ``poseboard/calibration.py``)
    ``cam.K`` (3x3), ``cam.dist`` (distortion), ``cam.rvec``/``cam.tvec`` (world -> camera,
    ``X_cam = cam.R @ X_world + cam.t``), ``cam.has_extrinsics``, ``cam.image_size``,
    ``cam.project(pts_world)``, ``cam.center_world``. The world frame is the floor
    checkerboard: Z up, meters. It is the same frame in which the balance board was
    located, so a skeleton in this frame lines up with the center of pressure.

What you must return
--------------------
``None`` (no person) or a ``Pose3D``:
    ``t``          capture time of the frames used (mean of their ``t``), NOT the time
                   when your model finished; this is what keeps pose and force in sync;
    ``names``      keypoint names (``self.keypoint_names``);
    ``keypoints``  (K, 3) world coordinates in meters, NaN for missing points;
    ``scores``     (K,) confidences in 0..1;
    ``per_camera_2d`` optional ``{camera name: Pose2D}``, used to draw the 2D skeleton and
                   recorded in ``pose2d_<camera>.csv`` (and as OpenPose JSON if enabled);
    optional:      ``mode`` (``"triangulated"``, ``"single_view_3d"`` or ``"2d_only"``),
                   ``reproj_error_px``, ``notes`` (texts shown to the user) and
                   ``format_key`` (a key of ``poseboard.pose.formats.FORMATS`` if your
                   keypoints follow one of those layouts).

A 2D model can also be added as a pose backend instead of a plugin: implement a
``Detector2D`` (``poseboard/pose/detectors/base.py``) and wrap it in
``poseboard.pose.multiview.MultiViewEstimator``, which does the subject selection,
triangulation with outlier rejection, single-view lifting and smoothing for you.

``process`` runs in a background thread and always receives the newest frames, so a slow
model simply lowers the pose rate (frames are skipped, never queued). Raise an exception
to report a problem: PoseBoard logs it, shows it in the GUI and keeps running.

Keypoint names and the center of mass
-------------------------------------
PoseBoard computes the whole-body center of mass (COM) from the keypoints
(``poseboard/pose/com.py``). It matches names such as ``LShoulder``/``left_shoulder``
automatically. It needs the trunk (both shoulders and both hips) plus enough other segments
to reach 60% of the body mass: shoulders and hips alone are not enough; add both knees, or the
head (ears or nose) plus one knee or both elbows. Ankles, wrists and heels/toes make it more
accurate. Pick the keypoint set below that matches the ORDER of your model's output, or define
your own list.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from poseboard.calibration import CameraCalibration
from poseboard.pose.base import Pose2D, Pose3D, PoseEstimator
from poseboard.pose.mediapipe_backend import single_view_lift, world_view
from poseboard.pose.triangulation import triangulate_keypoints

log = logging.getLogger(__name__)

# =============================================================================
# USER SETTINGS (can also be overridden with keyword arguments to create_estimator)
# =============================================================================

# Folder that contains your PoseAssess code; it is added to sys.path so that
# ``import poseassess`` (or whatever your package is called) works inside this plugin.
POSEASSESS_DIR: str | None = None  # e.g. r"C:\Users\me\Projects\PoseAssess"

# Keypoint set: "halpe26" or "coco17". Must match the order of your model's output.
KEYPOINT_SET = "halpe26"

# Keypoints with a lower confidence are ignored for triangulation / single-view lifting.
MIN_SCORE = 0.3

# Case (b) only: 4x4 matrix that maps YOUR 3D frame to PoseBoard's world frame
# (X_world = T[:3, :3] @ X_yours + T[:3, 3]). Leave None when your 3D keypoints are
# already in the checkerboard world frame (Z up, meters).
# Example: Pose2Sim-style Y-up output -> use poseboard.analysis.POSE2SIM_YUP_TO_ZUP.
# If your units are millimeters, scale the rotation part by 0.001 or convert first.
WORLD_TRANSFORM: np.ndarray | None = None

# =============================================================================
# Keypoint definitions
# =============================================================================

# Halpe-26 order (AlphaPose / RTMPose body26 / Pose2Sim HALPE_26). Pose2Sim-style names,
# so exported TRC files use the same marker names as Pose2Sim.
HALPE_26 = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
    "Head", "Neck", "Hip",
    "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel",
]

# COCO-17 order (the first 17 Halpe-26 points).
COCO_17 = HALPE_26[:17]

_COCO_BONES = [
    ("LShoulder", "RShoulder"), ("LHip", "RHip"),
    ("LShoulder", "LHip"), ("RShoulder", "RHip"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
    ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("LHip", "LKnee"), ("LKnee", "LAnkle"),
    ("RHip", "RKnee"), ("RKnee", "RAnkle"),
    ("Nose", "LEye"), ("Nose", "REye"), ("LEye", "LEar"), ("REye", "REar"),
]
_HALPE_EXTRA_BONES = [
    ("Head", "Neck"), ("Neck", "Hip"),
    ("LAnkle", "LHeel"), ("LAnkle", "LBigToe"), ("LBigToe", "LSmallToe"), ("LHeel", "LBigToe"),
    ("RAnkle", "RHeel"), ("RAnkle", "RBigToe"), ("RBigToe", "RSmallToe"), ("RHeel", "RBigToe"),
]

KEYPOINT_SETS = {
    "halpe26": (HALPE_26, _COCO_BONES + _HALPE_EXTRA_BONES),
    "coco17": (COCO_17, _COCO_BONES),
}


# =============================================================================
# The estimator
# =============================================================================
class PoseAssessEstimator(PoseEstimator):
    """Wraps PoseAssess (or any other pose model) as a PoseBoard ``PoseEstimator``."""

    name = "poseassess"

    def __init__(self, keypoint_set: str = KEYPOINT_SET, min_score: float = MIN_SCORE,
                 poseassess_dir: str | None = POSEASSESS_DIR,
                 world_transform: np.ndarray | None = WORLD_TRANSFORM, **options):
        if keypoint_set not in KEYPOINT_SETS:
            raise ValueError(f"keypoint_set must be one of {sorted(KEYPOINT_SETS)}")
        names, bones = KEYPOINT_SETS[keypoint_set]
        # Instance attributes: PoseBoard reads these for the CSV header and the overlay.
        self.keypoint_names = list(names)
        self.skeleton = list(bones)
        self.min_score = float(min_score)
        self.world_transform = (None if world_transform is None
                                else np.asarray(world_transform, np.float64).reshape(4, 4))
        self.options = options  # any extra keyword arguments, for your own use
        self._warned: set[str] = set()

        if poseassess_dir:
            p = str(Path(poseassess_dir).expanduser().resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
        self.model = self._load_model()

    # ------------------------------------------------------------------ TODO 1
    def _load_model(self):
        """Load your model once. Return any object; it is stored as ``self.model``.

        TODO: replace with your PoseAssess initialisation, for example::

            from poseassess.api import PoseAssessor          # your package / class
            return PoseAssessor(weights=r"C:\\...\\model.pth", device="cuda")

        Returning None keeps the template in "no detection" mode.
        """
        return None

    # ------------------------------------------------------------------ TODO 2a
    def detect_per_camera(self, cam_name: str, t: float, image: np.ndarray,
                          cam: CameraCalibration | None) -> dict | None:
        """CASE (a): your code works on ONE image at a time.

        Return None when no person is found, otherwise a dict with:

        ``"kp2d"``   (K, 2) pixel coordinates in THIS image (same order as keypoint_names)
        ``"scores"`` (K,) confidences 0..1 (optional, default 1)
        ``"kp3d"``   (K, 3) metric 3D keypoints in meters (optional, see below)
        ``"kp3d_frame"`` where ``kp3d`` lives: ``"body"`` (default; any rigid frame such
                     as hip-centred), ``"camera"`` (this camera's frame, OpenCV axes:
                     x right, y down, z forward) or ``"world"`` (checkerboard frame).

        How PoseBoard turns this into 3D:
        * 2 or more cameras with extrinsics: the 2D keypoints are triangulated
          (``kp3d`` is not needed). This is the most accurate option.
        * a single camera: ``kp3d`` is required. "camera" is transformed with the
          extrinsics; "body" is placed into the world with PnP against ``kp2d``
          (the same method as the MediaPipe backend), so depth is approximate.

        If several people are detected, return only the subject (e.g. the largest box
        or the person standing on the board).

        TODO: call your PoseAssess 2D (or 2D + monocular 3D) detector here, e.g.::

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out = self.model.predict(rgb)                     # your API
            if out is None:
                return None
            return {"kp2d": out.keypoints_2d, "scores": out.scores,
                    "kp3d": out.keypoints_3d, "kp3d_frame": "body"}
        """
        return None  # TODO: replace with your PoseAssess call (see the docstring)

    # ------------------------------------------------------------------ TODO 2b
    def detect_3d_multiview(self, frames: dict[str, tuple[float, np.ndarray]],
                            cams: dict[str, CameraCalibration]
                            ) -> tuple[np.ndarray, np.ndarray] | None:
        """CASE (b): your code already produces 3D keypoints (e.g. its own multi-camera
        triangulation using the calibration exported from PoseBoard via
        File -> "Export Pose2Sim Calib.toml...").

        Return None, or ``(keypoints (K, 3) in meters, scores (K,))``. They must be in the
        checkerboard world frame, or set ``WORLD_TRANSFORM`` so they can be mapped into it.
        If this returns a result, ``detect_per_camera`` is not called.

        TODO: call your PoseAssess 3D pipeline here, e.g.::

            images = {n: img for n, (t, img) in frames.items()}
            out = self.model.predict_3d(images)               # your API
            return None if out is None else (out.keypoints_3d, out.scores)
        """
        return None  # TODO: replace with your PoseAssess call (see the docstring)

    # ------------------------------------------------------------ PoseBoard API
    def process(self, frames: dict[str, tuple[float, np.ndarray]],
                cams: dict[str, CameraCalibration]) -> Pose3D | None:
        if not frames:
            return None
        names = self.keypoint_names
        K = len(names)

        # ---- case (b): 3D directly --------------------------------------------
        res = self.detect_3d_multiview(frames, cams)
        if res is not None:
            kp3d, scores = res
            kp3d = self._check(kp3d, (K, 3), "3D keypoints")
            scores = self._scores(scores, K)
            t = float(np.mean([t for t, _ in frames.values()]))
            return Pose3D(t, list(names), self._to_world(kp3d), scores)

        # ---- case (a): per-camera detections -----------------------------------
        dets: dict[str, dict] = {}
        for cam_name, (t, image) in frames.items():
            d = self.detect_per_camera(cam_name, t, image, cams.get(cam_name))
            if d is not None:
                dets[cam_name] = d
        if not dets:
            return None

        per2d: dict[str, Pose2D] = {}
        for cam_name, d in dets.items():
            if d.get("kp2d") is not None:
                kp2d = self._check(d["kp2d"], (K, 2), f"2D keypoints of {cam_name}")
                per2d[cam_name] = Pose2D(kp2d, self._scores(d.get("scores"), K))

        # Never mix frames: with extrinsics (world = floor checkerboard) only calibrated views are
        # used; without extrinsics only PoseBoard's world camera (whose frame is the world frame).
        cam_name = world_view(cams, self.world_camera)
        if cam_name is None:
            calibrated = [n for n in per2d if n in cams and cams[n].has_extrinsics]
            if len(calibrated) >= 2:  # two or more calibrated views -> weighted triangulation
                t = float(np.mean([frames[n][0] for n in calibrated]))
                kps, conf = triangulate_keypoints([cams[n] for n in calibrated],
                                                  [per2d[n].keypoints for n in calibrated],
                                                  [per2d[n].scores for n in calibrated],
                                                  self.min_score)
                return Pose3D(t, list(names), kps, conf, per2d)
            calibrated = [n for n in dets if n in cams and cams[n].has_extrinsics]
            if not calibrated:
                return None  # only cameras without extrinsics see the person
            cam_name = calibrated[0]
        if cam_name not in dets:
            return None
        t = float(frames[cam_name][0])

        # Single view -> needs metric 3D from your model.
        d, cam = dets[cam_name], cams.get(cam_name)
        if d.get("kp3d") is None or cam is None:
            self._warn_once("single-view",
                            "Only one usable camera: return 'kp3d' from detect_per_camera, or "
                            "add a second calibrated camera so the 2D keypoints can be "
                            "triangulated.")
            return None
        kp3d = self._check(d["kp3d"], (K, 3), f"3D keypoints of {cam_name}")
        frame = d.get("kp3d_frame", "body")
        if frame == "world":
            pts = self._to_world(kp3d)
        elif frame == "camera":
            # camera -> world: X_world = R^T (X_cam - t). Without extrinsics the world
            # frame IS the camera frame (same convention as the rest of PoseBoard).
            pts = (kp3d - cam.t) @ cam.R if cam.has_extrinsics else kp3d
        elif frame == "body":
            if cam_name not in per2d:
                raise ValueError("kp3d_frame='body' needs 'kp2d' to place the skeleton")
            pts = single_view_lift(cam, per2d[cam_name], kp3d, self.min_score)
            if pts is None:  # too few confident keypoints for PnP
                return None
        else:
            raise ValueError(f"Unknown kp3d_frame {frame!r} (use 'body', 'camera' or 'world')")
        scores = (per2d[cam_name].scores.copy() if cam_name in per2d
                  else self._scores(d.get("scores"), K))
        return Pose3D(t, list(names), pts, scores, per2d)

    def close(self) -> None:
        """Called when pose estimation is stopped: release GPU memory, threads, files."""
        # TODO (optional): e.g. self.model.release()
        self.model = None

    # ----------------------------------------------------------------- helpers
    def _to_world(self, kp3d: np.ndarray) -> np.ndarray:
        if self.world_transform is None:
            return kp3d
        T = self.world_transform
        return kp3d @ T[:3, :3].T + T[:3, 3]

    @staticmethod
    def _check(a, shape: tuple[int, int], what: str) -> np.ndarray:
        a = np.asarray(a, np.float64)
        if a.shape != shape:
            raise ValueError(f"{what}: expected shape {shape}, got {a.shape}. Check that "
                             "KEYPOINT_SET matches the order/number of your model's keypoints.")
        return a

    @staticmethod
    def _scores(scores, K: int) -> np.ndarray:
        if scores is None:
            return np.ones(K)
        s = np.asarray(scores, np.float64).reshape(-1)
        if s.shape != (K,):
            raise ValueError(f"scores: expected {K} values, got {s.shape[0]}")
        return s

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg)


# =============================================================================
# Entry point required by PoseBoard (poseboard.pose.external.load_plugin)
# =============================================================================
def create_estimator(**kwargs) -> PoseEstimator:
    """Called once when "Start Pose Estimation" is pressed. The GUI passes no arguments,
    so the USER SETTINGS above are used; scripts may override them, e.g.
    ``load_plugin(path, keypoint_set="coco17", min_score=0.5)``."""
    return PoseAssessEstimator(**kwargs)
