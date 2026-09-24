"""PoseBoard main window.

Workflow: Devices → Camera calibration (checkerboard) → Board setup (click the 4 corners
+ center in the image) → Record.

Every device is optional: record cameras/pose only, the Wii Balance Board only, or both. With
"Auto-connect" (on by default) a paired board is picked up as soon as it is switched on and
reconnected after a Bluetooth drop; link changes during a recording go to events.csv.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

from poseboard.calibration import (CameraCalibration, CheckerboardSpec, IntrinsicCollector,
                                   approximate_calibration, check_world_checkerboard,
                                   extrinsics_from_checkerboard, find_checkerboard,
                                   load_calibrations, save_calibrations, save_pose2sim_toml)
from poseboard.camera import CameraStream
from poseboard.fusion import fuse
from poseboard.geometry import (LANDMARK_NAMES, BoardGeometry, BoardPose, register_board,
                                rotate_board_frame)
from poseboard.overlay import draw_board, draw_caption, draw_clicks, draw_cop, draw_pose
from poseboard.pose.base import (MODE_2D_ONLY, MODE_SINGLE_VIEW_3D, MODE_TRIANGULATED, Pose3D,
                                 PoseEstimator, attach_frame_info, check_pose3d)
from poseboard.pose.com import center_of_mass
from poseboard.pose.detectors import (BACKENDS, BackendUnavailable, backend_available,
                                      create_detector, list_backends)
from poseboard.pose.formats import FORMATS
from poseboard.pose.multiview import MultiViewEstimator
from poseboard.session import SessionRecorder, on_console_close
from poseboard.wii.device import (DEFAULT_COP_MIN_KG, BalanceBoardHID, ForceSource, SimulatedBoard,
                                  WiiAutoConnect)
from poseboard.wii.protocol import pressed_sensor
from poseboard.gui.widgets import CopView, VideoView

log = logging.getLogger(__name__)

MARK_EVENT_KEYS = ("F9", "M")  # "M" is ignored while typing in a text field
STALE_FORCE_S = 1.0  # do not draw a Wii sample older than this (board disconnected)
CAMERA_LOST_S = 2.0  # a camera without a new frame for this long is reported as lost
PREVIEW_DETECT_WIDTH = 640  # live checkerboard preview: search a downscaled copy ...
PREVIEW_DETECT_INTERVAL_S = 0.2  # ... in a worker thread, at most 5 times per second
CORNER_CHECK_TIMEOUT_S = 15.0
_UI_RESOLUTION = object()  # add_camera(size=...): take the Resolution box


def safe_slot(fn):
    """Decorator for GUI slots: an exception is logged and shown in a message box instead of
    escaping into the Qt event loop. Extra signal arguments (e.g. ``checked``) are dropped."""
    params = list(inspect.signature(fn).parameters.values())[1:]
    n_args = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)

    @functools.wraps(fn)
    def wrapper(self, *args):
        try:
            return fn(self, *args[:n_args])
        except Exception as e:  # noqa: BLE001
            log.exception("%s failed", fn.__name__)
            self._warn(f"{fn.__name__.strip('_').replace('_', ' ').capitalize()} failed: {e}")
            return None
    return wrapper


CLICK_HINTS = {
    "TL": "front-left corner TL (long edge OPPOSITE the power button, subject's left)",
    "TR": "front-right corner TR (edge opposite the power button, subject's right)",
    "BR": "back-right corner BR (power-button edge)",
    "BL": "back-left corner BL (power-button edge)",
    "C": "board center C",
}
CORNER_NAMES = {"TL": "front-left (edge opposite the power button, left)",
                "TR": "front-right (edge opposite the power button, right)",
                "BL": "back-left (power-button edge, left)",
                "BR": "back-right (power-button edge, right)"}
BOARD_INFO_HELP = ("To verify: connect the board and click Corner Check, then press the TL corner "
                   "(front-left: long edge opposite the power button). PoseBoard checks which "
                   "sensor responds and fixes a front/back swap by itself.")

# ------------------------------------------------------------------ pose backends (Record tab)
PLUGIN_KEY = "__plugin__"  # the "Plugin (.py)" entry of the backend list
PLUGIN_LABEL = "Plugin (.py)"
DEFAULT_MIN_SCORE = 0.3
RECOMMENDED_MIN_SCORE = {"mediapipe": 0.5}  # MediaPipe visibility: 0.5 as in earlier versions
OPTION_LABELS = {"model": "Model", "mode": "Mode", "device": "Device", "num_poses": "Max. persons",
                 "version": "Version", "input_size": "Input height (px)",
                 "person_detector": "Person detector", "prototxt": "Prototxt file",
                 "caffemodel": "Caffe model file", "model_path": "Model file (.onnx)",
                 "weights_path": "Weights file (.pth)", "pose_model": "Pose model file",
                 "det_model": "Person detector file", "detector_model": "Detector model"}
OPTION_TIPS = {
    "device": "auto = the NVIDIA GPU (CUDA) when the backend can use it, else the CPU",
    "mode": "lightweight = fastest, balanced, performance = most accurate (larger model)",
    "num_poses": "More than 1 lets PoseBoard pick the person standing on the board when "
                 "several people are in view",
    "model": "Model variant (larger = more accurate and slower)",
    "person_detector": "rtdetr: RT-DETR (transformers); yolo: Ultralytics YOLO (AGPL-3.0); "
                       "none: the whole image is one person",
    "prototxt": "OpenPose pose_deploy.prototxt (empty: downloaded from GitHub)",
    "caffemodel": "OpenPose pose_iter_*.caffemodel (never downloaded; empty: looked up in "
                  "models/openpose/body_25 or models/openpose/coco)",
    "model_path": "MoveNet .onnx file (empty: downloaded from GitHub into the models folder)",
    "weights_path": "keypointrcnn_resnet50_fpn_coco-fc266e95.pth (empty: models folder, the "
                    "PyTorch cache, or a download from download.pytorch.org)",
    "pose_model": "rtmlib pose model: an .onnx file or the mmdeploy .zip of download.openmmlab.com "
                  "(empty: rtmlib's model of the selected mode, downloaded on first use)",
    "det_model": "YOLOX person detector: an .onnx file or an mmdeploy .zip (empty: rtmlib's "
                 "detector of the selected mode). An 80-class COCO YOLOX (e.g. from the YOLOX "
                 "GitHub releases) works too: only persons are kept",
    "detector_model": "RT-DETR: a Hugging Face id or a local model folder; YOLO: a weights file "
                      "(empty: the default detector)",
}
# Options whose list also accepts a typed value (BackendSpec.editable)
EDIT_TIPS = {"model": " Type a Hugging Face id / MMPose model name, or a local model folder."}
FILE_FILTERS = {"prototxt": "Caffe network definition (*.prototxt);;All files (*)",
                "caffemodel": "Caffe weights (*.caffemodel);;All files (*)",
                "model_path": "ONNX model (*.onnx);;All files (*)",
                "weights_path": "PyTorch weights (*.pth *.pt);;All files (*)",
                "pose_model": "ONNX model or mmdeploy zip (*.onnx *.zip);;All files (*)",
                "det_model": "ONNX model or mmdeploy zip (*.onnx *.zip);;All files (*)"}
# Backends whose keypoint format depends on an option: backend -> (option, value -> format)
OPTION_FORMATS = {"openpose_dnn": ("model", {"body25": "body25", "body_25": "body25",
                                             "coco18": "coco18", "coco": "coco18",
                                             "coco_18": "coco18"})}
NOT_DOWNLOADED = {("openpose_dnn", "caffemodel")}  # file options that are never downloaded
MAX_NOTES_SHOWN = 4
# One estimator is created at a time: a start abandoned with Stop keeps loading (e.g. downloading
# its model) in the background, and the next start waits for it instead of writing the same
# model files at the same time
_LOAD_LOCK = threading.Lock()


def pose_mode_text(pose: Pose3D) -> str:
    """How the 3D keypoints were obtained, e.g. "triangulated from 3 views"."""
    mode = getattr(pose, "mode", None)
    views = list(getattr(pose, "views_used", None) or [])
    if mode == MODE_TRIANGULATED:
        return f"triangulated from {len(views)} views" if views else "3D"
    if mode == MODE_SINGLE_VIEW_3D:
        return f"single-view 3D ({views[0]})" if views else "single-view 3D"
    if mode == MODE_2D_ONLY:
        return "2D only (no 3D)"
    return str(mode or "")


def backend_format_key(key: str, options: dict | None = None) -> str:
    """Keypoint format of a registry backend with these options (e.g. OpenPose "coco18")."""
    spec = BACKENDS[key]
    opt, table = OPTION_FORMATS.get(key, (None, {}))
    value = (options or {}).get(opt) if opt else None
    if value is not None:
        return table.get(str(value).strip().lower().replace("-", "_"), spec.keypoint_format)
    return spec.keypoint_format


def backend_description(key: str, options: dict | None = None) -> str:
    """Keypoint format (for ``options``, e.g. the selected OpenPose model), single-camera 3D and
    license of a registry backend (one line)."""
    spec = BACKENDS[key]
    fk = backend_format_key(key, options)
    fmt = FORMATS.get(fk)
    kp = f"{fmt.label or fmt.key}, {len(fmt)} keypoints" if fmt else fk
    one = "yes" if spec.provides_3d else "no (needs 2+ calibrated cameras)"
    return f"{kp} | 3D from one camera: {one} | License: {spec.license}"


class PoseWorker:
    """Background thread: latest frame of each camera → pose estimation → keep the latest result
    and write it to the recording.

    The thread owns the estimator: it is closed by the thread itself when its loop ends, never
    while ``process`` runs. Frames more than ``max_skew_s`` older than the newest one (a camera
    that stopped delivering) are left out, so a frozen view is not mixed with live ones. With
    several cameras a pose is computed once every live camera has a new frame (or after
    ``sync_wait_s`` at the latest).

    With ``factory`` (and ``estimator=None``) the thread first creates the estimator, which may
    take long (a model download): ``loading`` is True meanwhile (``load_waiting`` while another
    worker's estimator is still being created: creations run one at a time), and a failure is
    reported in ``load_error`` (the thread then ends). ``on_ready(estimator)`` runs in the
    thread right after creation (e.g. to set ``world_camera``)."""

    def __init__(self, estimator: PoseEstimator | None, get_frames, get_cams,
                 recorder: SessionRecorder, max_skew_s: float = 0.25, *, factory=None,
                 on_ready=None, label: str = "", key: str | None = None,
                 sync_wait_s: float = 0.1):
        if estimator is None and factory is None:
            raise ValueError("PoseWorker needs an estimator or a factory")
        self.estimator = estimator
        self._factory = factory
        self._on_ready = on_ready
        self.label = label  # e.g. the backend's label (session.json "pose_backend")
        self.key = key  # the backend key (None for a plugin: its estimator's name is used)
        self.loading = estimator is None
        self.load_waiting = False  # waiting for another worker's estimator creation
        self.load_error: str | None = None
        self.load_started = time.perf_counter()
        self.ready_seen = False  # set by the GUI once it has handled the loaded estimator
        self.get_frames = get_frames
        self.get_cams = get_cams
        self.recorder = recorder
        self.max_skew_s = max_skew_s
        self.sync_wait_s = min(float(sync_wait_s), float(max_skew_s))
        self._waiting_since: float | None = None
        self.latest: Pose3D | None = None
        self.fps = 0.0
        self.error: str | None = None  # last error; cleared by the next successful frame
        self.stale_cameras: list[str] = []  # cameras left out (frame too old)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pose", daemon=True)

    def start(self):
        self._thread.start()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def load_crashed(self) -> bool:
        """The thread ended while ``loading`` was still set (it never reported a result)."""
        return self.loading and self._thread.ident is not None and not self._thread.is_alive()

    def stop(self, timeout: float = 3.0):
        """Ask the thread to stop and wait up to ``timeout`` s (0: do not wait). Results that
        arrive after this are dropped; the estimator is closed by the thread when it finishes.
        While the estimator is still being created (e.g. downloading a model) this returns at
        once: the thread closes it as soon as it exists."""
        self._stop.set()
        if self._thread.ident is None:  # never started
            self._close()
            return
        if self.loading or timeout <= 0:
            return
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            log.warning("pose estimation is still busy; it stops (and closes the estimator) when "
                        "the current frame is done")

    def _close(self):
        est = self.estimator
        if est is None:
            return
        try:
            est.close()
        except Exception:  # noqa: BLE001
            log.exception("closing the pose estimator failed")

    def _load(self) -> bool:
        """Create the estimator (worker thread); False if that failed or stop was requested."""
        est = None
        try:
            self.load_waiting = True
            with _LOAD_LOCK:  # one creation at a time (shared model files and downloads)
                self.load_waiting = False
                if self._stop.is_set():  # stopped while waiting: nothing to create
                    self.loading = False
                    return False
                est = self._factory()
            if not isinstance(est, PoseEstimator):
                raise TypeError(f"the pose backend returned {type(est).__name__}, not a "
                                "PoseEstimator")
            if self._on_ready is not None:
                self._on_ready(est)
        except BaseException as e:  # noqa: BLE001  (missing model, failed download; also
            # SystemExit, e.g. a plugin calling sys.exit() at import)
            log.exception("starting pose estimation failed")
            if est is not None and hasattr(est, "close"):
                try:
                    est.close()
                except Exception:  # noqa: BLE001
                    log.exception("closing the pose estimator failed")
            msg = str(e) or type(e).__name__
            if isinstance(e, SystemExit):
                msg = f"the pose backend called sys.exit({e.code!r})"
            self.load_error = self.error = msg
            self.loading = self.load_waiting = False
            if isinstance(e, (KeyboardInterrupt, GeneratorExit)):
                raise
            return False
        self.estimator = est  # before loading = False: the GUI never sees neither
        self.loading = False
        return not self._stop.is_set()

    def _run(self):
        try:
            if self.estimator is None and not self._load():
                return
            last_idx: dict[str, int] = {}
            while not self._stop.is_set():
                try:
                    if not self._step(last_idx):
                        time.sleep(0.003)
                except Exception as e:  # noqa: BLE001  (estimator, invalid result, disk full, ...)
                    log.exception("pose estimation failed")
                    self.error = str(e) or type(e).__name__
                    self._stop.wait(0.5)
        finally:
            self._close()

    def _step(self, last_idx: dict[str, int]) -> bool:
        frames = self.get_frames()
        new = {n for n, f in frames.items() if last_idx.get(n) != f.index}
        if not new:
            return False
        newest = max(f.t for f in frames.values())
        stale = sorted(n for n, f in frames.items() if newest - f.t > self.max_skew_s)
        # Several cameras: wait (at most sync_wait_s) until every live camera has a new frame,
        # so the model does not run twice on the same image and a video frame is not used
        # for two poses (pose2d_<cam>.csv / OpenPose JSON)
        if any(n not in new for n in frames if n not in stale):
            now = time.perf_counter()
            if self._waiting_since is None:
                self._waiting_since = now
            if now - self._waiting_since < self.sync_wait_s:
                return False
        self._waiting_since = None
        for n, f in frames.items():
            last_idx[n] = f.index
        self.stale_cameras = stale
        use = {n: (f.t, f.image) for n, f in frames.items() if n not in self.stale_cameras}
        t0 = time.perf_counter()
        pose = self.estimator.process(use, self.get_cams())
        if self._stop.is_set():
            return True  # stopped meanwhile: drop the result
        if pose is not None:
            pose = check_pose3d(pose)
            attach_frame_info(pose, {n: frames[n] for n in use})  # recorded video frame numbers
        dt = time.perf_counter() - t0
        self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-3) if self.fps else 1 / max(dt, 1e-3)
        self.latest = pose
        if pose is not None and self.recorder.recording:
            self.recorder.add_pose(pose)
        self.error = None
        return True


class MainWindow(QMainWindow):
    # (force source, new state, perf_counter time): emitted from the Wii reader thread and
    # delivered in the GUI thread (queued connection)
    wii_status = Signal(object, str, float)

    def __init__(self, auto_connect_wii: bool = True):
        super().__init__()
        self.setWindowTitle("PoseBoard — Synchronized 3D Pose + Wii Balance Board Recording")
        self.resize(1500, 900)

        self.streams: dict[str, CameraStream] = {}
        self.calibs: dict[str, CameraCalibration] = {}  # never overwritten by a resolution change
        self.collectors: dict[str, IntrinsicCollector] = {}
        self.clicks: dict[str, list[np.ndarray]] = {}
        self.click_sizes: dict[str, tuple[int, int]] = {}  # image size the clicks were made in
        self.geometry = BoardGeometry()
        self.board: BoardPose | None = None
        self.board_rotation = 0  # degrees applied after registration (Rotate / corner check)
        self.board_stale: str | None = None  # why the board pose is outdated (None: current)
        self._camera_sources: dict[str, str] = {}  # every camera name seen -> its source
        self._live_calibs: dict[str, CameraCalibration] = {}  # calibration used per running camera
        self._fallback_calibs: dict[str, CameraCalibration] = {}
        self._calib_mismatch: dict[str, tuple] = {}  # name -> (calibration size, frame size)
        self._cam_lost: set[str] = set()
        self._corner: dict | None = None  # running corner check
        self._detect_busy = threading.Event()
        self._detect_started = 0.0
        self._detect_result: tuple | None = None  # (camera, pattern, corners, time)
        self._pending_summary: Path | None = None  # recording being post-processed
        self.force: ForceSource | None = None
        self.wii_poll_interval_s = 1.0  # auto-connect: how often to look for a paired board
        self._wii_listener = None  # status listener registered on self.force
        self._wii_state = "stopped"  # last state reported by self.force
        self._wii_link: str | None = None  # last Wii link event written to the recording
        self._marks = 0  # "Mark Event" presses in the current recording
        self._tick_error: str | None = None
        self.pose_worker: PoseWorker | None = None
        # Workers stopped without waiting (still loading or finishing a frame in the background)
        self._abandoned_workers: list[PoseWorker] = []
        # Pose backend settings per backend key: {"options": {...}, "min_score": float}
        self._pose_settings: dict[str, dict] = {}
        self._shown_backend: str | None = None  # backend whose option widgets are shown
        self._opt_widgets: dict[str, tuple] = {}  # option -> (kind, widget, choices)
        self._opt_rows = 0  # option rows currently in the pose form
        self._pick_on_board = True  # read by the pose thread (board_provider)
        self.recorder = SessionRecorder(Path.cwd() / "recordings")
        self.click_mode = False
        self._rec_started = 0.0
        self._com_trail: deque = deque(maxlen=300)  # (t, x, y) in board coordinates

        self._build_ui()
        self.wii_status.connect(self._on_wii_status)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick_safe)
        self.timer.start(33)
        # Plug and play: start looking for a paired board right away (the Wii stays optional)
        self.chk_auto.setChecked(auto_connect_wii)

    # ================================================================== UI
    def _build_ui(self):
        self._build_menu()
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        top = QHBoxLayout()
        top.addWidget(QLabel("Show camera:"))
        self.cam_select = QComboBox()
        self.cam_select.currentTextChanged.connect(lambda _: self._update_click_hint())
        top.addWidget(self.cam_select, 1)
        self.chk_overlay = QCheckBox("Overlay board/COP")
        self.chk_overlay.setChecked(True)
        self.chk_skeleton = QCheckBox("Overlay skeleton/COM")
        self.chk_skeleton.setChecked(True)
        top.addWidget(self.chk_overlay)
        top.addWidget(self.chk_skeleton)
        lv.addLayout(top)
        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet("color: #b00; font-weight: bold")
        self.warn_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.warn_label.hide()
        lv.addWidget(self.warn_label)
        self.video = VideoView()
        self.video.clicked.connect(self._on_video_click)
        lv.addWidget(self.video, 1)
        self.cop_view = CopView()
        self.cop_view.setMaximumHeight(260)
        lv.addWidget(self.cop_view)
        splitter.addWidget(left)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_devices(), "1 Devices")
        self.tabs.addTab(self._tab_calibration(), "2 Camera Calibration")
        self.tabs.addTab(self._tab_board(), "3 Board Setup")
        self.tabs.addTab(self._tab_record(), "4 Record")
        # Keep every tab title fully visible (no scroll arrows / clipped "4 Rec..."): the panel is at
        # least as wide as the whole tab bar, whatever the platform font.
        bar = self.tabs.tabBar()
        bar.setUsesScrollButtons(False)
        bar.setElideMode(Qt.ElideNone)
        self.tabs.setMinimumWidth(max(420, self.tabs.minimumSizeHint().width()))
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.status = self.statusBar()

    def _build_menu(self):
        m = self.menuBar().addMenu("File")
        for text, fn in (("Save Project…", self.save_project), ("Load Project…", self.load_project)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            m.addAction(a)
        m.addSeparator()
        for text, fn in (("Import Camera Calibration (.json / Pose2Sim .toml)…", self.load_calib),
                         ("Export Camera Calibration (.json)…", self.save_calib),
                         ("Export Pose2Sim Calib.toml…", self.export_toml)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            m.addAction(a)

    def _tab_devices(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        g = QGroupBox("Cameras")
        f = QGridLayout(g)
        self.cam_source = QLineEdit("0")
        self.cam_source.setToolTip("Device index (0, 1, …), video file path, or rtsp/http URL")
        self.cam_res = QComboBox()
        self.cam_res.addItems(["Default", "640x480", "1280x720", "1920x1080"])
        self.cam_res.setCurrentText("1280x720")
        self.cam_res.setToolTip("Must be the resolution the camera was calibrated at")
        self.cam_name = QLineEdit()
        self.cam_name.setPlaceholderText("auto (cam0, cam1, ...)")
        self.cam_name.setToolTip("Camera name; the calibration and clicks are stored per name. "
                                 "By default a source that was used before gets its old name.")
        b_add = QPushButton("Add Camera")
        b_add.clicked.connect(self.add_camera)
        b_file = QPushButton("Video File…")
        b_file.clicked.connect(self._pick_video)
        self.cam_list = QListWidget()
        b_rm = QPushButton("Remove Selected")
        b_rm.clicked.connect(self.remove_camera)
        f.addWidget(QLabel("Source"), 0, 0)
        f.addWidget(self.cam_source, 0, 1)
        f.addWidget(b_file, 0, 2)
        f.addWidget(QLabel("Resolution"), 1, 0)
        f.addWidget(self.cam_res, 1, 1)
        f.addWidget(QLabel("Name"), 2, 0)
        f.addWidget(self.cam_name, 2, 1)
        f.addWidget(b_add, 2, 2)
        f.addWidget(self.cam_list, 3, 0, 1, 3)
        f.addWidget(b_rm, 4, 2)
        v.addWidget(g)

        g = QGroupBox("Wii Balance Board (optional)")
        f = QGridLayout(g)
        self.chk_auto = QCheckBox("Auto-connect (plug and play)")
        self.chk_auto.setToolTip("Connect as soon as a paired board is switched on and reconnect "
                                 "automatically after a Bluetooth drop.\nThe board must be paired "
                                 "in the Windows Bluetooth settings. With the Microsoft stack the "
                                 "pairing\nis usually lost when the board is switched off: pair "
                                 "it again (red SYNC button) at the start of a session.")
        self.chk_auto.toggled.connect(self.toggle_auto_wii)
        self.wii_devices = QComboBox()
        b_scan = QPushButton("Scan")
        b_scan.clicked.connect(self.scan_wii)
        b_conn = QPushButton("Connect")
        b_conn.clicked.connect(self.connect_wii)
        b_sim = QPushButton("Use Simulator")
        b_sim.clicked.connect(self.connect_sim)
        b_disc = QPushButton("Disconnect")
        b_disc.clicked.connect(self.disconnect_wii)
        self.b_tare = b_tare = QPushButton("Tare (board empty)")
        b_tare.setToolTip("Zero the four sensors (nobody on the board). Not possible while recording.")
        b_tare.clicked.connect(self.tare)
        self.min_kg = QDoubleSpinBox()
        self.min_kg.setRange(0, 50)
        self.min_kg.setValue(DEFAULT_COP_MIN_KG)
        self.min_kg.setSuffix(" kg")
        self.min_kg.valueChanged.connect(lambda v: setattr(self.force, "min_total_kg", v) if self.force else None)
        self.wii_label = QLabel("Not connected (optional)")
        self.wii_label.setStyleSheet("font-family: monospace")
        self.wii_label.setWordWrap(True)
        # Long error messages (e.g. HID paths) must not widen the whole panel
        self.wii_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        f.addWidget(self.chk_auto, 0, 0, 1, 3)
        f.addWidget(self.wii_devices, 1, 0, 1, 2)
        f.addWidget(b_scan, 1, 2)
        f.addWidget(b_conn, 2, 0)
        f.addWidget(b_sim, 2, 1)
        f.addWidget(b_disc, 2, 2)
        f.addWidget(b_tare, 3, 0, 1, 2)
        f.addWidget(QLabel("COP min. load"), 4, 0)
        f.addWidget(self.min_kg, 4, 1)
        f.addWidget(self.wii_label, 5, 0, 1, 3)
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _tab_calibration(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("Checkerboard")
        f = QFormLayout(g)
        self.cb_cols = QSpinBox()
        self.cb_cols.setRange(3, 30)
        self.cb_cols.setValue(9)
        self.cb_rows = QSpinBox()
        self.cb_rows.setRange(3, 30)
        self.cb_rows.setValue(6)
        self.cb_square = QDoubleSpinBox()
        self.cb_square.setRange(1, 500)
        self.cb_square.setValue(25.0)
        self.cb_square.setSuffix(" mm")
        self.chk_detect = QCheckBox("Show live checkerboard detection")
        f.addRow("Inner corners (cols)", self.cb_cols)
        f.addRow("Inner corners (rows)", self.cb_rows)
        f.addRow("Square size", self.cb_square)
        f.addRow(self.chk_detect)
        v.addWidget(g)

        g = QGroupBox("Intrinsics (displayed camera)")
        f = QVBoxLayout(g)
        f.addWidget(QLabel("Capture 15-30 frames with the checkerboard at different\n"
                           "positions/angles in the image, then compute."))
        h = QHBoxLayout()
        b = QPushButton("Capture Frame")
        b.clicked.connect(self.capture_intrinsic)
        h.addWidget(b)
        b = QPushButton("Clear")
        b.clicked.connect(lambda: self.collectors.pop(self.current_cam(), None))
        h.addWidget(b)
        b = QPushButton("Compute Intrinsics")
        b.clicked.connect(self.compute_intrinsic)
        h.addWidget(b)
        f.addLayout(h)
        v.addWidget(g)

        g = QGroupBox("Extrinsics / World Frame")
        f = QVBoxLayout(g)
        f.addWidget(QLabel("Lay the checkerboard flat on the floor (ideally near\n"
                           "the board) where all cameras can see it, then click\n"
                           "below: it becomes the world frame (Z up, meters).\n"
                           "It needs one odd and one even inner-corner count\n"
                           "(e.g. 9 x 6); a symmetric board is refused for\n"
                           "several cameras."))
        b = QPushButton("Set All Camera Extrinsics from Checkerboard")
        b.clicked.connect(self.compute_extrinsics_all)
        f.addWidget(b)
        v.addWidget(g)
        self.calib_info = QPlainTextEdit()
        self.calib_info.setReadOnly(True)
        v.addWidget(self.calib_info, 1)
        return w

    def _tab_board(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("Board Dimensions")
        f = QFormLayout(g)
        self.geo_spins = {}
        for key, label in (("length_mm", "Length (left-right)"), ("width_mm", "Width (front-back)"),
                           ("sensor_dx_mm", "Sensor spacing L-R"), ("sensor_dy_mm", "Sensor spacing F-B"),
                           ("height_mm", "Top height above checkerboard")):
            s = QDoubleSpinBox()
            s.setRange(0 if key == "height_mm" else 10, 2000)
            s.setSuffix(" mm")
            s.setValue(getattr(self.geometry, key))
            s.valueChanged.connect(self._geometry_changed)
            self.geo_spins[key] = s
            f.addRow(label, s)
        v.addWidget(g)

        g = QGroupBox("Click the Board in the Image")
        f = QVBoxLayout(g)
        f.addWidget(QLabel("Order: 1 TL front-left → 2 TR front-right →\n"
                           "3 BR back-right → 4 BL back-left → 5 C center.\n"
                           '"Front" = the long edge OPPOSITE the power button\n'
                           "(TL/TR sensors); the subject stands facing it.\n"
                           "Left/right = the subject's left/right, not the image's.\n"
                           "Left-click adds, right-click undoes. Clicking in several\n"
                           "cameras triangulates the board (more accurate)."))
        h = QHBoxLayout()
        self.b_click = QPushButton("Start Clicking")
        self.b_click.setCheckable(True)
        self.b_click.toggled.connect(self._toggle_click_mode)
        h.addWidget(self.b_click)
        b = QPushButton("Undo")
        b.clicked.connect(self.undo_click)
        h.addWidget(b)
        b = QPushButton("Clear This Camera")
        b.clicked.connect(self.clear_clicks)
        h.addWidget(b)
        f.addLayout(h)
        self.click_hint = QLabel("")
        self.click_hint.setStyleSheet("color: #c60; font-weight: bold")
        f.addWidget(self.click_hint)
        b = QPushButton("Compute Board Pose")
        b.clicked.connect(self.compute_board)
        f.addWidget(b)
        h = QHBoxLayout()
        b = QPushButton("Corner Check")
        b.setToolTip("Press the TL corner of the board: checks which sensor responds, and rotates "
                     "the frame by 180° if front and back were swapped")
        b.clicked.connect(self.start_corner_check)
        h.addWidget(b)
        b = QPushButton("Rotate Frame 180°")
        b.setToolTip("Front and back swapped (the corner check does this by itself)")
        b.clicked.connect(lambda: self.rotate_board(180))
        h.addWidget(b)
        f.addLayout(h)
        self.corner_label = QLabel("")
        self.corner_label.setWordWrap(True)
        self.corner_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        f.addWidget(self.corner_label)
        v.addWidget(g)
        self.board_info = QPlainTextEdit()
        self.board_info.setReadOnly(True)
        self.board_info.setPlainText(BOARD_INFO_HELP)
        v.addWidget(self.board_info, 1)
        return w

    def _tab_record(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(self._pose_group())

        g = QGroupBox("Recording")
        f = QFormLayout(g)
        self.subject = QLineEdit()
        self.notes = QLineEdit()
        self.out_dir = QLineEdit(str(self.recorder.root))
        f.addRow("Subject", self.subject)
        f.addRow("Notes", self.notes)
        h = QHBoxLayout()
        h.addWidget(self.out_dir)
        b = QPushButton("…")
        b.clicked.connect(self._pick_outdir)
        h.addWidget(b)
        f.addRow("Output folder", h)
        self.streams_label = QLabel("")
        self.streams_label.setWordWrap(True)
        self.streams_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        f.addRow("Streams", self.streams_label)
        self.b_rec = QPushButton("● Start Recording")
        self.b_rec.setCheckable(True)
        self.b_rec.setStyleSheet("QPushButton:checked { background: #d33; color: white }")
        self.b_rec.toggled.connect(self.toggle_record)
        f.addRow(self.b_rec)
        self.rec_label = QLabel("")
        f.addRow(self.rec_label)

        # Event markers (e.g. "sync" at a jump) for aligning with other recordings later
        self.event_label = QLineEdit()
        self.event_label.setPlaceholderText("label (optional), e.g. sync")
        self.event_label.setToolTip("Label written to events.csv; empty = mark_1, mark_2, ...")
        self.event_label.returnPressed.connect(self.mark_event)
        keys = " / ".join(MARK_EVENT_KEYS)
        self.b_mark = QPushButton(f"Mark Event ({keys})")
        self.b_mark.setToolTip("Write a time marker to events.csv (t, t_rel, t_unix, label). "
                               "Only while recording.")
        self.b_mark.clicked.connect(self.mark_event)
        self.act_mark = QAction("Mark Event", self)
        self.act_mark.setShortcuts([QKeySequence(k) for k in MARK_EVENT_KEYS])
        self.act_mark.triggered.connect(self.mark_event)
        self.addAction(self.act_mark)
        h = QHBoxLayout()
        h.addWidget(self.event_label, 1)
        h.addWidget(self.b_mark)
        f.addRow("Event", h)
        self.events_info = QLabel("Events: -")
        f.addRow(self.events_info)
        self._set_marking(False)
        v.addWidget(g)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(80)
        v.addWidget(self.summary, 1)
        scroll = QScrollArea()  # the pose options make the tab tall: scroll on small screens
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(w)
        # As wide as the content needs (the tab panel is sized from its tabs' minimum widths)
        scroll.setMinimumWidth(w.minimumSizeHint().width()
                               + scroll.verticalScrollBar().sizeHint().width() + 2)
        return scroll

    def _pose_group(self) -> QGroupBox:
        """"3D Pose Estimation": backend list (every registered 2D backend, unavailable ones
        greyed out with the reason and install command, plus Plugin (.py)), the backend's
        options, 3D / subject / output settings, Start button and status."""
        g = QGroupBox("3D Pose Estimation")
        f = self.pose_form = QFormLayout(g)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.pose_backend = QComboBox()
        self.pose_backend.setToolTip("2D pose backend. 3D comes from triangulation (2+ cameras "
                                     "with extrinsics) or, with one camera, from backends with "
                                     "a 3D skeleton (MediaPipe, RTMPose3D).")
        model = self.pose_backend.model()
        first_ok = None
        for spec in list_backends():
            ok, why = backend_available(spec.key)
            text = spec.label if ok else (
                f"{spec.label}  (not installed)" if " needs " in why else f"{spec.label}  (unavailable)")
            self.pose_backend.addItem(text, spec.key)
            item = model.item(self.pose_backend.count() - 1)
            tip = f"{spec.label}\n{backend_description(spec.key)}\n{spec.notes}"
            if not ok:
                tip += f"\n\nNot available: {why}"
                item.setEnabled(False)
            item.setToolTip(tip)
            if ok and first_ok is None:
                first_ok = spec.key
        self.pose_backend.addItem(PLUGIN_LABEL, PLUGIN_KEY)
        model.item(self.pose_backend.count() - 1).setToolTip(
            "Your own PoseEstimator: a .py file with create_estimator(**kwargs)")
        self.pose_backend.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.pose_backend.setMinimumContentsLength(18)
        f.addRow("Backend", self.pose_backend)
        self.backend_info = QLabel("")
        self.backend_info.setWordWrap(True)
        self.backend_info.setStyleSheet("color: #555")
        self.backend_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        f.addRow(self.backend_info)

        self.plugin_path = QLineEdit()
        self.plugin_path.setPlaceholderText("Plugin file, e.g. plugins/poseassess_plugin.py")
        self.b_pick = b_pick = QPushButton("…")
        b_pick.setFixedWidth(32)
        b_pick.clicked.connect(self._pick_plugin)
        self.plugin_row = QWidget()
        h = QHBoxLayout(self.plugin_row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.plugin_path, 1)
        h.addWidget(b_pick)
        f.addRow("Plugin file", self.plugin_row)

        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(0.0, 1.0)
        self.min_score.setSingleStep(0.05)
        self.min_score.setDecimals(2)
        self.min_score.setValue(DEFAULT_MIN_SCORE)
        self.min_score.setToolTip("Keypoints with a lower confidence are not used for 3D (and "
                                  "not drawn). Scores are not comparable between backends, so "
                                  "this is stored per backend.")
        self.min_score.valueChanged.connect(self._pose_param_changed)
        f.addRow("Min keypoint confidence", self.min_score)
        self.reproj_thr = QDoubleSpinBox()
        self.reproj_thr.setRange(1.0, 500.0)
        self.reproj_thr.setSingleStep(1.0)
        self.reproj_thr.setDecimals(1)
        self.reproj_thr.setValue(15.0)
        self.reproj_thr.setToolTip("Triangulation (2+ cameras): per keypoint, while its "
                                   "reprojection error exceeds this in some camera and at least 2 "
                                   "cameras remain, the camera that fits the others worst is "
                                   "dropped (e.g. a left/right swap in one view)")
        self.reproj_thr.valueChanged.connect(self._pose_param_changed)
        f.addRow("Reprojection outlier threshold (px)", self.reproj_thr)
        self.chk_smooth = QCheckBox("Temporal smoothing (One-Euro)")
        self.chk_smooth.setToolTip("Smooth the 3D keypoints over time (One-Euro filter, 2 Hz "
                                   "cutoff rising with speed): about one frame of lag for fast "
                                   "movements, roughly half the jitter when still; a 1 Hz sway "
                                   "keeps about 90 % of its amplitude")
        self.chk_smooth.toggled.connect(self._pose_param_changed)
        f.addRow(self.chk_smooth)
        self.chk_pick_board = QCheckBox("Pick the person standing on the board")
        self.chk_pick_board.setChecked(True)
        self.chk_pick_board.setToolTip("With several people in a camera, use the one whose feet "
                                       "are on the Balance Board (needs the board pose, tab 3); "
                                       "otherwise the tracked or largest person")
        self.chk_pick_board.toggled.connect(self._pose_param_changed)
        f.addRow(self.chk_pick_board)
        self.chk_save2d = QCheckBox("Save 2D keypoints")
        self.chk_save2d.setChecked(True)
        self.chk_save2d.setToolTip("Record pose2d_<camera>.csv: the subject's 2D keypoints in "
                                   "every processed frame, with the video frame number")
        self.chk_json = QCheckBox("Also save OpenPose JSON (Pose2Sim)")
        self.chk_json.setToolTip("Also write pose2d_json/<camera>/<camera>_<set>_keypoints.json "
                                 "(one numbered set per pose in every camera folder, plus "
                                 "sets.csv and Calib.toml) so the recording can be "
                                 "re-triangulated with Pose2Sim (set pose_model to "
                                 "session.json's pose2sim_model)")
        self.chk_save2d.toggled.connect(lambda _: self._update_pose_controls())
        h = QHBoxLayout()
        h.addWidget(self.chk_save2d)
        h.addWidget(self.chk_json)
        h.addStretch(1)
        f.addRow(h)

        self.b_pose = QPushButton("Start Pose Estimation")
        self.b_pose.setCheckable(True)
        self.b_pose.toggled.connect(self.toggle_pose)
        f.addRow(self.b_pose)
        self.pose_label = QLabel("Not running")
        self.pose_label.setWordWrap(True)
        self.pose_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        f.addRow(self.pose_label)

        self.pose_backend.currentIndexChanged.connect(lambda _: self._backend_changed())
        default = "mediapipe" if backend_available("mediapipe")[0] else first_ok
        i = self.pose_backend.findData(default or PLUGIN_KEY)
        self.pose_backend.setCurrentIndex(i)
        self._backend_changed()
        return g

    # ------------------------------------------------------- pose options
    def current_backend(self) -> str:
        """Key of the selected pose backend (``PLUGIN_KEY`` for Plugin (.py))."""
        return self.pose_backend.currentData() or PLUGIN_KEY

    def _settings(self, key: str) -> dict:
        st = self._pose_settings.get(key)
        if st is None:
            spec = BACKENDS.get(key)
            st = self._pose_settings[key] = {
                "options": dict(spec.defaults) if spec else {},
                "min_score": RECOMMENDED_MIN_SCORE.get(key, DEFAULT_MIN_SCORE)}
        return st

    def _store_shown_options(self) -> None:
        """Remember the values of the option widgets of the backend shown now."""
        key = self._shown_backend
        if key is None or key == PLUGIN_KEY:
            return
        st = self._settings(key)
        st["options"].update(self.backend_options())
        st["min_score"] = round(self.min_score.value(), 4)

    def backend_options(self) -> dict:
        """The option values of the shown backend, as passed to ``create_detector``."""
        out = {}
        for opt, (kind, wdg, choices) in self._opt_widgets.items():
            if kind == "combo":
                i = wdg.currentIndex()
                out[opt] = choices[i] if 0 <= i < len(choices) else None
            elif kind == "edit_combo":  # a listed value or a typed one
                text = wdg.currentText().strip()
                out[opt] = next((c for c in choices if str(c) == text), text) or None
            else:
                out[opt] = wdg.text().strip()
        return {k: v for k, v in out.items() if v is not None}

    def _refresh_backend_info(self) -> None:
        """The line below the backend list: keypoint format for the current options (e.g. the
        OpenPose model), 3D, license and notes."""
        key = self._shown_backend
        if key is None or key == PLUGIN_KEY or key not in BACKENDS:
            return
        ok, why = backend_available(key)
        info = f"{backend_description(key, self.backend_options())}\n{BACKENDS[key].notes}"
        if not ok:
            info = f"NOT AVAILABLE: {why}\n{info}"
        self.backend_info.setText(info)
        self.backend_info.setToolTip(info)

    def _backend_changed(self) -> None:
        """Show the options of the selected backend (keeping the values chosen before for each
        backend) and its description."""
        self._store_shown_options()
        key = self.current_backend()
        f = self.pose_form
        for _ in range(self._opt_rows):
            f.removeRow(f.getWidgetPosition(self.plugin_row)[0] + 1)
        self._opt_widgets, self._opt_rows = {}, 0
        is_plugin = key == PLUGIN_KEY
        f.setRowVisible(self.plugin_row, is_plugin)
        self._shown_backend = key
        if is_plugin:
            info = ("Your own pose estimator: a Python file defining create_estimator(**kwargs) "
                    "-> PoseEstimator (see plugins/poseassess_plugin_template.py).")
            self.backend_info.setText(info)
            self.backend_info.setToolTip(info)
            self._update_pose_controls()
            return
        spec = BACKENDS[key]
        st = self._settings(key)
        row = f.getWidgetPosition(self.plugin_row)[0] + 1
        for opt, choices in spec.options.items():
            value = st["options"].get(opt, spec.defaults.get(opt))
            tip = OPTION_TIPS.get(opt, "")
            if opt in spec.needs_files or not choices:
                edit = QLineEdit("" if value is None else str(value))
                edit.setPlaceholderText("select the file (or put it into models/)"
                                        if (key, opt) in NOT_DOWNLOADED
                                        else "empty = automatic (download / default)")
                field = QWidget()
                h = QHBoxLayout(field)
                h.setContentsMargins(0, 0, 0, 0)
                h.addWidget(edit, 1)
                if opt in spec.needs_files:
                    b = QPushButton("…")
                    b.setFixedWidth(32)
                    b.setToolTip("Select the file")
                    b.clicked.connect(functools.partial(self._pick_option_file, opt, edit))
                    h.addWidget(b)
                    self._opt_widgets[opt] = ("file", edit, ())
                else:
                    self._opt_widgets[opt] = ("text", edit, ())
                edit.setToolTip(tip)
                f.insertRow(row, OPTION_LABELS.get(opt, opt.replace("_", " ").capitalize()), field)
            else:
                combo = QComboBox()
                combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
                combo.setMinimumContentsLength(10)
                vals = list(choices)
                if value is not None and value not in vals:
                    vals.append(value)  # e.g. a custom model name from a project file
                for c in vals:
                    combo.addItem(str(c))
                combo.setCurrentIndex(vals.index(value) if value in vals else 0)
                editable = opt in spec.editable
                if editable:  # e.g. a Hugging Face id or a local model folder
                    combo.setEditable(True)
                    combo.setInsertPolicy(QComboBox.NoInsert)
                    tip += EDIT_TIPS.get(opt, " Type another value if needed.")
                    combo.editTextChanged.connect(lambda _: self._refresh_backend_info())
                combo.currentIndexChanged.connect(lambda _: self._refresh_backend_info())
                combo.setToolTip(tip)
                self._opt_widgets[opt] = ("edit_combo" if editable else "combo", combo, vals)
                f.insertRow(row, OPTION_LABELS.get(opt, opt.replace("_", " ").capitalize()), combo)
            row += 1
            self._opt_rows += 1
        self._refresh_backend_info()
        self.min_score.blockSignals(True)
        self.min_score.setValue(float(st["min_score"]))
        self.min_score.blockSignals(False)
        self._update_pose_controls()

    def _pick_option_file(self, opt: str, edit: QLineEdit) -> None:
        from poseboard.pose import mediapipe_backend as mpb  # MODEL_DIR is set by the exe

        start = edit.text().strip() or str(mpb.MODEL_DIR)
        p, _ = QFileDialog.getOpenFileName(self, f"Select {OPTION_LABELS.get(opt, opt)}", start,
                                           FILE_FILTERS.get(opt, "All files (*)"))
        if p:
            edit.setText(p)

    def _pose_param_changed(self, *_):
        """Min score / outlier threshold / smoothing / board pick: applied to the running
        estimator at once (they are plain attributes read for every frame)."""
        self._pick_on_board = self.chk_pick_board.isChecked()
        if self._shown_backend and self._shown_backend != PLUGIN_KEY:
            self._settings(self._shown_backend)["min_score"] = round(self.min_score.value(), 4)
        pw = self.pose_worker
        est = pw.estimator if pw is not None else None
        if isinstance(est, MultiViewEstimator):
            est.min_score = float(self.min_score.value())
            est.reproj_threshold_px = float(self.reproj_thr.value())
            smoothing = "one_euro" if self.chk_smooth.isChecked() else None
            if smoothing != est.smoothing:
                est.smoothing = smoothing
                est.reset()  # no blending with an old filter state

    def _update_pose_controls(self, recording: bool | None = None) -> None:
        """Backend and options: fixed while pose estimation runs or a recording runs (they
        define pose3d.csv); thresholds and outputs: fixed while recording."""
        if recording is None:
            recording = self.recorder.recording
        free = not recording and self.pose_worker is None
        self.pose_backend.setEnabled(free)
        self.plugin_path.setEnabled(free)
        self.b_pick.setEnabled(free)
        for kind, wdg, _ in self._opt_widgets.values():
            wdg.setEnabled(free)
            parent = wdg.parentWidget()
            if kind == "file" and parent is not None:
                parent.setEnabled(free)
        builtin = self.current_backend() != PLUGIN_KEY  # a plugin does its own 3D
        for wdg in (self.min_score, self.reproj_thr, self.chk_smooth, self.chk_pick_board):
            wdg.setEnabled(not recording and builtin)
        self.chk_save2d.setEnabled(not recording)
        self.chk_json.setEnabled(not recording and self.chk_save2d.isChecked())

    # ============================================================ helpers
    def current_cam(self) -> str | None:
        n = self.cam_select.currentText()
        return n or None

    def checker_spec(self) -> CheckerboardSpec:
        return CheckerboardSpec(self.cb_cols.value(), self.cb_rows.value(), self.cb_square.value())

    def _warn(self, msg: str):
        QMessageBox.warning(self, "PoseBoard", msg)

    def _latest_frames(self):
        out = {}
        for n, s in list(self.streams.items()):
            f = s.latest()
            if f is not None:
                out[n] = f
        return out

    def _cams_snapshot(self) -> dict[str, CameraCalibration]:
        """Calibrations valid for the running cameras' current frames (read by the pose thread)."""
        return dict(self._live_calibs)

    def _update_live_calibs(self) -> dict[str, CameraCalibration]:
        self._live_calibs = {n: self._ensure_calib(n, f) for n, f in self._latest_frames().items()}
        return self._live_calibs

    # ============================================================ devices
    def _pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if p:
            self.cam_source.setText(p)

    def _auto_camera_name(self, src: str) -> str:
        """A source used before (e.g. in the loaded project) gets its old name back, so its
        calibration and clicks apply again; otherwise the first unused camN."""
        for n, known in self._camera_sources.items():
            if known == src and n not in self.streams:
                return n
        i = 0
        while f"cam{i}" in self.streams or f"cam{i}" in self._camera_sources:
            i += 1
        return f"cam{i}"

    def add_camera(self, source: str | None = None, name: str | None = None,
                   size=_UI_RESOLUTION):
        """Add a camera. ``size``: (width, height), None (the camera's default) or, by default,
        the Resolution box."""
        from_ui = not isinstance(source, str)
        src = self.cam_source.text().strip() if from_ui else source
        if not src:
            return
        if not (isinstance(name, str) and name.strip()):
            name = self.cam_name.text().strip() if from_ui else ""
        name = name.strip() or self._auto_camera_name(src)
        if name in self.streams:
            self._warn(f"A camera named {name} is already running; choose another name")
            return
        if size is _UI_RESOLUTION:
            wh = self.cam_res.currentText()
            size = tuple(int(v) for v in wh.split("x")) if "x" in wh else None
        width, height = size if size else (None, None)
        s = CameraStream(name, src, width, height)
        try:
            s.start()
        except Exception as e:  # noqa: BLE001
            self._warn(str(e))
            return
        self.streams[name] = s
        self._camera_sources[name] = src
        item = QListWidgetItem(f"{name}  ←  {src}" + (f"  ({width}x{height})" if width else ""))
        item.setData(Qt.UserRole, name)
        self.cam_list.addItem(item)
        self.cam_select.addItem(name)
        self.cam_select.setCurrentText(name)
        if from_ui:
            self.cam_name.clear()

    def remove_camera(self):
        row = self.cam_list.currentRow()
        if row < 0:
            return
        item = self.cam_list.item(row)
        name = item.data(Qt.UserRole) or item.text().split()[0]
        s = self.streams.pop(name, None)
        if s:
            self._log_event(f"camera_removed {name}")  # its video ends here
            s.stop()
        self.cam_list.takeItem(row)
        self.cam_select.removeItem(self.cam_select.findText(name))
        # Its calibration and clicks are kept (for re-adding it), but no longer used
        for d in (self._live_calibs, self._calib_mismatch, self._fallback_calibs):
            d.pop(name, None)
        self._cam_lost.discard(name)

    @staticmethod
    def _is_approximate(c: CameraCalibration) -> bool:
        a = approximate_calibration(c.name, *c.image_size)
        return np.allclose(c.K, a.K) and not np.any(np.asarray(c.dist))

    def _ensure_calib(self, name: str, frame) -> CameraCalibration:
        """Calibration to use for this camera's current frames.

        Without a calibration, approximate intrinsics are used (and stored). A calibration made at
        another resolution is never overwritten, since it may hold calibrated intrinsics and the
        extrinsics: it stays in ``self.calibs`` and a warning is shown until the camera runs at
        that resolution again. Meanwhile approximate intrinsics are used together with its
        extrinsics (the camera pose does not depend on the resolution), so the pose stays in the
        checkerboard frame. K is not rescaled: webcams crop differently in different modes."""
        c = self.calibs.get(name)
        h, w = frame.image.shape[:2]
        if c is not None and tuple(c.image_size) == (w, h):
            self._calib_mismatch.pop(name, None)
            return c
        if c is None or (not c.has_extrinsics and self._is_approximate(c)):
            c = approximate_calibration(name, w, h)  # nothing worth keeping
            self.calibs[name] = c
            self._calib_mismatch.pop(name, None)
            return c
        if name not in self._calib_mismatch:
            log.warning("%s: frames are %dx%d but the calibration is for %dx%d; keeping it, using "
                        "approximate intrinsics meanwhile", name, w, h, *c.image_size)
        self._calib_mismatch[name] = (tuple(c.image_size), (w, h))
        fb = self._fallback_calibs.get(name)
        if fb is None or tuple(fb.image_size) != (w, h):
            fb = self._fallback_calibs[name] = approximate_calibration(name, w, h)
        fb.rvec, fb.tvec = c.rvec, c.tvec
        return fb

    @safe_slot
    def scan_wii(self):
        self.wii_devices.clear()
        try:
            devs = BalanceBoardHID.list_devices()
        except Exception as e:  # noqa: BLE001  (the message says how to fix it)
            self._warn(f"Cannot enumerate HID devices: {e}")
            return
        for d in devs:
            self.wii_devices.addItem(f"{d.get('product_string') or 'Balance Board'}  {d['path']!r}", d["path"])
        if not devs:
            self.status.showMessage("No Wii Balance Board found. Pair it via Bluetooth (red SYNC "
                                    "button in the battery compartment; usually needed again after "
                                    "it was switched off) and switch it on.", 8000)

    def _set_force(self, src: ForceSource):
        """Make ``src`` the only running force source (stops the previous one). The tare of each
        board is carried over, so reconnecting the same board keeps its zero."""
        old = self.force
        self._stop_force()
        src.adopt_tares(old)
        src.sensor_dx_m = self.geometry.sensor_dx_mm / 1000
        src.sensor_dy_m = self.geometry.sensor_dy_mm / 1000
        src.min_total_kg = self.min_kg.value()

        def listener(state: str, src=src):  # reader thread -> GUI thread
            self.wii_status.emit(src, state, time.perf_counter())

        src.add_status_listener(listener)
        self._wii_listener = listener
        self.force = src
        src.start()

    def _stop_force(self):
        """Stop the current force source, if any, without touching the Auto-connect checkbox."""
        src, self.force = self.force, None
        self._wii_state = "stopped"
        if src is None:
            return
        if self._wii_listener is not None:
            src.remove_status_listener(self._wii_listener)
            self._wii_listener = None
        if self.recorder.recording:
            self.recorder.attach_force(None)
            self._log_wii_link("wii_disconnected", time.perf_counter())
        src.stop()
        self.cop_view.set_data([], [], 0.0)

    def _set_auto_checked(self, on: bool):
        """Tick/untick Auto-connect without starting or stopping anything."""
        self.chk_auto.blockSignals(True)
        self.chk_auto.setChecked(on)
        self.chk_auto.blockSignals(False)

    def _make_auto_source(self, path=None) -> WiiAutoConnect:
        return WiiAutoConnect(path, poll_interval_s=self.wii_poll_interval_s)

    @safe_slot
    def toggle_auto_wii(self, on: bool):
        """Auto-connect on: replace any other source by a plug-and-play reader that waits for a
        paired board and reconnects after drops. Off: stop the plug-and-play reader."""
        if on:
            if not isinstance(self.force, WiiAutoConnect):
                self._set_force(self._make_auto_source())
        elif isinstance(self.force, WiiAutoConnect):
            self._stop_force()

    def _wii_connected(self) -> bool:
        return self.force is not None and self.force.status == "connected"

    @safe_slot
    def connect_wii(self):
        """Connect the board selected after Scan (or the first one found). With Auto-connect on
        it stays plug and play (reconnects after drops), limited to that board."""
        path = self.wii_devices.currentData()
        if self.chk_auto.isChecked():
            self._set_force(self._make_auto_source(path))
        else:
            self._set_force(BalanceBoardHID(path))

    @safe_slot
    def connect_sim(self):
        self._set_auto_checked(False)
        self._set_force(SimulatedBoard())

    @safe_slot
    def disconnect_wii(self):
        """Release the board and switch Auto-connect off (otherwise it would reconnect at once)."""
        self._set_auto_checked(False)
        self._stop_force()

    @safe_slot
    def _on_wii_status(self, src, state: str, t: float):
        """Connection state change of ``src`` (GUI thread). During a recording, a board that
        (re)connects is added to it and link changes are written to events.csv."""
        if src is not self.force:
            return  # late message from a source that has been replaced
        prev, self._wii_state = self._wii_state, state
        connected = state == "connected"
        if self.recorder.recording:
            if connected:
                self.recorder.attach_force(src)
                self.recorder.log_force_source(src, "connected")  # device and tare in effect
            self._log_wii_link("wii_connected" if connected else "wii_disconnected", t)
        if connected and prev != "connected":
            self.status.showMessage("Wii Balance Board connected", 5000)
        elif prev == "connected" and not connected:
            again = " - waiting for it to reconnect" if isinstance(src, WiiAutoConnect) else ""
            self.status.showMessage(f"Wii Balance Board connection lost{again}", 8000)

    def _log_wii_link(self, label: str, t: float):
        """Write wii_connected / wii_disconnected to events.csv when the link state changes."""
        if label != self._wii_link and self.recorder.recording:
            self._wii_link = label
            if self.recorder.add_event(label, t) is not None:
                self._refresh_events_info()

    def _wii_text(self, sample) -> str:
        """Text for the Wii panel: connection state, sensor values, battery."""
        src = self.force
        if src is None:
            return "Not connected (optional)"
        auto = isinstance(src, WiiAutoConnect)
        kind = ("Simulator" if isinstance(src, SimulatedBoard)
                else "Auto-connect" if auto else "Board")
        st = src.status
        if st == "searching":
            txt = "Auto-connect: waiting for a paired board.\nSwitch the board on (power button)."
            last = getattr(src, "last_error", None) or ""
            if last.startswith("board not responding"):
                txt += ("\nA paired board is listed but does not answer: switch it on, or pair it "
                        "again (usually needed after it was switched off).")
            elif "not a Balance Board" in last:
                txt += f"\n{last}"
            return txt
        if st == "connecting":
            return f"{kind}: connecting..."
        if st.startswith("error"):
            return f"{kind} error{' (retrying)' if auto else ''}: {st[len('error: '):]}"
        if st != "connected":
            return f"{kind}: {st}"
        bat = getattr(src, "battery", None)
        head = f"{kind}: connected" + (f", battery {bat}" if bat is not None else "")
        if sample is None:
            return head + "\nwaiting for data..."
        k, cop = sample.kg, sample.cop_board
        return (f"{head}\nTL {k[2]:6.2f}   TR {k[0]:6.2f}\nBL {k[3]:6.2f}   BR {k[1]:6.2f} kg\n"
                f"Total {sample.total_kg:6.2f} kg\nCOP x={cop[0] * 1000:7.1f} y={cop[1] * 1000:7.1f} mm")

    @safe_slot
    def tare(self):
        if not self.force:
            return
        if self.recorder.recording:
            self._warn("Tare is not possible during a recording (it would change the zero of "
                       "wii.csv in the middle of the file). Tare before starting.")
            return
        try:
            t = self.force.do_tare(1.0)
            self.status.showMessage(f"Tare done: {np.round(t, 2)} kg", 5000)
        except Exception as e:  # noqa: BLE001
            self._warn(str(e))

    # ======================================================== calibration
    def capture_intrinsic(self):
        name = self.current_cam()
        f = self.streams[name].latest() if name in self.streams else None
        if f is None:
            return
        col = self.collectors.get(name)
        if col is None or col.spec != self.checker_spec():
            col = self.collectors[name] = IntrinsicCollector(self.checker_spec())
        if col.add(f.image) is None:
            self.status.showMessage("Checkerboard not detected", 3000)
        else:
            self.status.showMessage(f"{name}: {len(col)} frames captured", 3000)

    def compute_intrinsic(self):
        name = self.current_cam()
        col = self.collectors.get(name)
        if not col:
            self._warn("Capture checkerboard images first")
            return
        try:
            c = col.calibrate(name)
        except Exception as e:  # noqa: BLE001
            self._warn(str(e))
            return
        had_ext = self.calibs.get(name) is not None and self.calibs[name].has_extrinsics
        self.calibs[name] = c
        self._refresh_calib_info()
        if had_ext:
            self.status.showMessage(f"{name}: intrinsics computed. Set the extrinsics again "
                                    "(they depend on the intrinsics).", 10000)
        self._recompute_board(f"{name}: intrinsics changed")

    @safe_slot
    def compute_extrinsics_all(self):
        spec, msgs = self.checker_spec(), []
        frames = self._latest_frames()
        note = check_world_checkerboard(spec, len(frames))  # refuses a symmetric board (2+ cams)
        if note:
            msgs.append(note)
        changed = False
        for name, f in frames.items():
            c = self._ensure_calib(name, f)
            if name in self._calib_mismatch:
                msgs.append(f"{name}: skipped (its calibration is for another resolution)")
                continue
            try:
                err = extrinsics_from_checkerboard(c, f.image, spec)
                changed = True
                msgs.append(f"{name}: reprojection error {err:.2f}px, camera position {np.round(c.center_world, 3)} m")
            except Exception as e:  # noqa: BLE001
                msgs.append(f"{name}: {e}")
        self.status.showMessage("; ".join(msgs), 10000)
        self._refresh_calib_info()
        if changed:
            self._recompute_board("camera extrinsics changed")

    def _refresh_calib_info(self):
        lines = []
        for n, c in self.calibs.items():
            lines.append(f"[{n}] {c.image_size[0]}x{c.image_size[1]}  "
                         f"intrinsic RMS={c.intrinsic_rms if c.intrinsic_rms is not None else 'uncalibrated (approx.)'}")
            lines.append(f"  fx={c.K[0, 0]:.1f} fy={c.K[1, 1]:.1f} cx={c.K[0, 2]:.1f} cy={c.K[1, 2]:.1f}")
            if c.has_extrinsics:
                lines.append(f"  extrinsic error={c.extrinsic_rms if c.extrinsic_rms is None else round(c.extrinsic_rms, 3)}px"
                             f"  camera position (m)={np.round(c.center_world, 3)}")
            else:
                lines.append("  extrinsics: not set")
        for n, col in self.collectors.items():
            lines.append(f"{n}: {len(col)} checkerboard frames captured")
        self.calib_info.setPlainText("\n".join(lines))

    def load_calib(self):
        p, _ = QFileDialog.getOpenFileName(self, "Import Camera Calibration", "", "Calibration (*.json *.toml)")
        if not p:
            return
        cams = load_calibrations(p)
        names = list(self.streams)
        for i, c in enumerate(cams):
            # If the names don't match, map to the added cameras in order
            if c.name not in self.streams and i < len(names):
                c.name = names[i]
            self.calibs[c.name] = c
        self._refresh_calib_info()
        self._recompute_board("camera calibration imported")

    def save_calib(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export Camera Calibration", "calibration.json", "JSON (*.json)")
        if p:
            save_calibrations(p, list(self.calibs.values()))

    @safe_slot
    def export_toml(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export Pose2Sim Calibration", "Calib.toml", "TOML (*.toml)")
        if p:
            warnings = save_pose2sim_toml(p, list(self.calibs.values()))
            if warnings:
                self._warn("Calib.toml written.\n\n" + "\n".join(warnings))

    # ============================================================== board
    def _geometry_changed(self):
        self.geometry = BoardGeometry(**{k: s.value() for k, s in self.geo_spins.items()})
        if self.board is not None:
            self.board_stale = "board dimensions changed: click Compute Board Pose"
            self._refresh_board_info()
        self.cop_view.board_mm = (self.geometry.length_mm, self.geometry.width_mm)
        self.cop_view.sensor_mm = (self.geometry.sensor_dx_mm, self.geometry.sensor_dy_mm)
        if self.force:
            self.force.sensor_dx_m = self.geometry.sensor_dx_mm / 1000
            self.force.sensor_dy_m = self.geometry.sensor_dy_mm / 1000

    def _toggle_click_mode(self, on: bool):
        self.click_mode = on
        self.video.crosshair = on
        self.b_click.setText("Stop Clicking" if on else "Start Clicking")
        self._update_click_hint()

    def _next_click_hint(self, name: str | None) -> str | None:
        if not self.click_mode or name is None:
            return None
        n = len(self.clicks.get(name, []))
        if n >= 5:
            return 'All 5 points done - click "Compute Board Pose"'
        return f"Click {n + 1}/5: {CLICK_HINTS[LANDMARK_NAMES[n]]}"

    def _update_click_hint(self):
        self.click_hint.setText(self._next_click_hint(self.current_cam()) or "")

    def _on_video_click(self, x: float, y: float, button: int):
        name = self.current_cam()
        if not self.click_mode or name is None:
            return
        frame = self.streams[name].latest() if name in self.streams else None
        if frame is None:
            return  # nothing shown for this camera: a click would refer to another image
        size = (frame.image.shape[1], frame.image.shape[0])
        pts = self.clicks.setdefault(name, [])
        if pts and self.click_sizes.get(name) not in (None, size):
            pts.clear()  # old clicks were made at another resolution
        if button == 2:
            if pts:
                pts.pop()
        elif len(pts) < 5:
            pts.append(np.array([x, y]))
            self.click_sizes[name] = size
        self._update_click_hint()

    def undo_click(self):
        pts = self.clicks.get(self.current_cam() or "")
        if pts:
            pts.pop()
        self._update_click_hint()

    def clear_clicks(self):
        self.clicks.pop(self.current_cam(), None)
        self.click_sizes.pop(self.current_cam(), None)
        self._update_click_hint()

    def _register_from_clicks(self) -> BoardPose:
        """Register the board from the clicks of the running cameras (clicks made at another
        resolution, or of cameras that were removed, are not used)."""
        frames = self._latest_frames()
        names, skipped = [], []
        for n, pts in self.clicks.items():
            if len(pts) != 5:
                continue
            if n not in frames:
                skipped.append(f"{n} (not running)")
                continue
            size = self.click_sizes.get(n)
            h, w = frames[n].image.shape[:2]
            if size is not None and tuple(size) != (w, h):
                skipped.append(f"{n} (clicked at {size[0]}x{size[1]}, now {w}x{h})")
                continue
            names.append(n)
        if not names:
            raise ValueError("Click all 5 points in at least one running camera"
                             + (f" (not usable: {', '.join(skipped)})" if skipped else ""))
        cams = [self._ensure_calib(n, frames[n]) for n in names]
        board = register_board(self.geometry, cams, [np.array(self.clicks[n]) for n in names])
        if skipped:
            board.notes.append("Clicks not used: " + ", ".join(skipped))
        return board

    @safe_slot
    def compute_board(self):
        board = self._register_from_clicks()
        self.board, self.board_rotation, self.board_stale = board, 0, None
        self.b_click.setChecked(False)
        self._refresh_board_info()
        self._sync_world_camera()
        if board.warnings:
            self._warn("Board pose computed, but please check:\n\n" + "\n\n".join(board.warnings))

    def _recompute_board(self, reason: str) -> None:
        """After a calibration change, register the board again from the stored clicks (keeping
        a 180° rotation applied before); if that fails, mark the board pose as outdated."""
        if self.board is None:
            return
        try:
            board = self._register_from_clicks()
        except Exception as e:  # noqa: BLE001
            self.board_stale = f"{reason}; the board could not be recomputed ({e})"
        else:
            if self.board_rotation:
                board = rotate_board_frame(board, self.board_rotation)
            self.board, self.board_stale = board, None
            self.status.showMessage(f"Board pose recomputed ({reason})", 6000)
        self._refresh_board_info()
        self._sync_world_camera()

    def _sync_world_camera(self) -> None:
        """Without extrinsics the pose must come from the camera the board was registered with."""
        if self.pose_worker is not None and self.pose_worker.estimator is not None:
            self._set_world_camera(self.pose_worker.estimator)

    def _set_world_camera(self, est: PoseEstimator) -> None:
        """Called in the GUI thread, and in the pose thread right after the estimator was
        created."""
        try:
            est.world_camera = self.board.world_camera if self.board else None
        except AttributeError:  # a plugin that defines it read-only
            log.warning("pose estimator %s does not accept world_camera", getattr(est, "name", est))

    def rotate_board(self, deg: int = 180):
        if self.board:
            self.board = rotate_board_frame(self.board, deg)
            self.board_rotation = (self.board_rotation + deg) % 360
            self._refresh_board_info()

    def _board_stale_text(self) -> str | None:
        if self.board is None:
            return None
        if self.board_stale:
            return self.board_stale
        bad = [n for n in self.board.reproj_error_px if n in self._calib_mismatch]
        if bad:
            return (f"{', '.join(bad)} runs at another resolution than its calibration, so the "
                    "board pose does not match the camera(s) until that is fixed")
        return None

    def _refresh_board_info(self):
        b = self.board
        if b is None:
            self.board_info.setPlainText(BOARD_INFO_HELP)
            return
        T = b.board_to_world
        method = "multi-camera triangulation" if b.method == "triangulation" else "single-camera PnP"
        lines = [f"Method: {method}" + (", fitted flat on the floor" if b.floor_constrained else ""),
                 "Reprojection error (px): " + ", ".join(f"{k}={v:.2f}" for k, v in b.reproj_error_px.items()),
                 f"Board center (world, m): {np.round(T.t, 4)}",
                 f"Board normal (world): {np.round(b.up_world, 3)}"]
        if b.tilt_deg is not None:
            lines.append(f"Tilt of the unconstrained fit: {b.tilt_deg:.1f}°"
                         + (" (removed: board assumed flat on the floor)" if b.floor_constrained else ""))
        if self.board_rotation:
            lines.append(f"Frame rotated {self.board_rotation}° after registration.")
        stale = self._board_stale_text()
        if stale:
            lines.append(f"OUTDATED: {stale}.")
        lines += [f"Warning: {w}" for w in b.warnings] + [f"Note: {n}" for n in b.notes]
        lines.append("\n" + BOARD_INFO_HELP)
        self.board_info.setPlainText("\n".join(lines))

    # ------------------------------------------------------- corner check
    @safe_slot
    def start_corner_check(self):
        """Press the TL corner: the sensor that responds shows whether the board frame matches
        the sensors (TL), is turned 180° (BR: fixed automatically) or is mirrored (TR / BL)."""
        src = self.force
        samples = src.recent(0.5) if src is not None else []
        if not samples:
            self._warn("The corner check needs the Wii Balance Board: connect it first (tab 1).")
            return
        self._corner = {"t": time.perf_counter(), "src": src,
                        "baseline": np.mean([s.kg for s in samples], axis=0)}
        where = "the corner labelled TL in the video" if self.board is not None else "the TL corner"
        self.corner_label.setText(f"Press firmly with one hand on {where}: front-left, i.e. on the "
                                  "long edge OPPOSITE the power button, on the left when facing "
                                  "that edge. Keep pressing for a second...")

    def _update_corner_check(self, now: float) -> None:
        c = self._corner
        if c is None:
            return
        if c["src"] is not self.force:
            self._corner = None
            self.corner_label.setText("Corner check cancelled (the board was disconnected).")
            return
        if now - c["t"] > CORNER_CHECK_TIMEOUT_S:
            self._corner = None
            self.corner_label.setText("Corner check: no clear press detected. Try again and press "
                                      "harder (at least 3 kg).")
            return
        recent = self.force.recent(0.3)
        if now - c["t"] < 0.3 or not recent:
            return
        name = pressed_sensor(c["baseline"], np.mean([s.kg for s in recent], axis=0))
        if name is not None:
            self._corner = None
            self._corner_result(name)

    def _corner_result(self, name: str) -> None:
        if self.board is None:
            msg = (f"The {name} sensor responded: the corner you pressed is the "
                   f"{CORNER_NAMES[name]} one.")
        elif name == "TL":
            msg = ("OK: the TL sensor responded, the board frame matches the sensors (+y = front = "
                   "edge opposite the power button, +x = right).")
        elif name == "BR":
            self.rotate_board(180)
            msg = ("The BR sensor responded: front and back were swapped in the clicks. The board "
                   "frame was rotated 180° to match the sensors. The subject must stand facing the "
                   "TL/TR edge (opposite the power button). Run the check again to confirm.")
        else:
            msg = (f"The {name} sensor responded: the clicks do not match the sensors (left/right "
                   "mirrored or turned 90°). Redo the clicks: TL/TR is the long edge opposite the "
                   "power button, left/right as seen by the subject facing that edge.")
        self.corner_label.setText(msg)
        self.status.showMessage(f"Corner check: {name} sensor responded", 6000)

    # =============================================================== pose
    @staticmethod
    def _plugins_dir() -> str:
        """The plugins folder: next to the working directory, the packaged build's bundle, or the
        source tree."""
        cands = [Path.cwd() / "plugins"]
        if getattr(sys, "_MEIPASS", None):
            cands.append(Path(sys._MEIPASS) / "plugins")
        cands.append(Path(__file__).resolve().parents[2] / "plugins")
        return str(next((c for c in cands if c.is_dir()), cands[0]))

    def _pick_plugin(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Pose Plugin", self._plugins_dir(), "Python (*.py)")
        if p:
            self.plugin_path.setText(p)
            self.pose_backend.setCurrentIndex(self.pose_backend.findData(PLUGIN_KEY))

    def _subject_board(self):
        """board_provider of the estimator (called in the pose thread): the board pose and
        geometry when "Pick the person standing on the board" is on."""
        if not self._pick_on_board or self.board is None:
            return None
        return self.board, self.geometry

    def _estimator_factory(self):
        """(factory, label, key) for the selected backend; ``factory()`` creates the estimator
        (in the pose thread: it may download a model). Raises with a clear message when the
        selection cannot work (backend not installed, plugin or required file missing)."""
        key = self.current_backend()
        if key == PLUGIN_KEY:
            path = self.plugin_path.text().strip()
            if not path:
                raise ValueError("Select the plugin file (...) first")
            if not Path(path).is_file():
                raise FileNotFoundError(f"Plugin file not found: {path}")
            from poseboard.pose.external import load_plugin

            return functools.partial(load_plugin, path), PLUGIN_LABEL, None
        ok, why = backend_available(key)
        if not ok:
            raise BackendUnavailable(why)
        spec = BACKENDS[key]
        self._store_shown_options()
        opts = self.backend_options()
        params = dict(min_score=float(self.min_score.value()),
                      reproj_threshold_px=float(self.reproj_thr.value()),
                      smoothing="one_euro" if self.chk_smooth.isChecked() else None,
                      board_provider=self._subject_board)

        def make() -> MultiViewEstimator:
            det = create_detector(key, **opts)
            try:
                return MultiViewEstimator(det, **params)
            except Exception:
                det.close()
                raise

        return make, spec.label, key

    def toggle_pose(self, on: bool):
        """Start/stop pose estimation. The estimator is created in the pose thread (the first
        start of a backend may download its model), so the window stays responsive; the status
        line shows the progress, and a failure is shown when it happens."""
        if not on:
            pw = self.pose_worker
            if pw is not None:
                # never block the window: a frame being processed (or a model being loaded) is
                # finished in the background, and the thread closes its estimator itself
                pw.stop(timeout=0)
                self._abandoned_workers = [w for w in self._abandoned_workers if w.alive]
                if pw.alive:
                    self._abandoned_workers.append(pw)
            self.pose_worker = None
            self.b_pose.setText("Start Pose Estimation")
            self._set_text(self.pose_label, "Not running")
            self._update_pose_controls()
            return
        if self.pose_worker is not None:
            return
        try:
            factory, label, key = self._estimator_factory()
        except Exception as e:  # noqa: BLE001
            self.b_pose.setChecked(False)
            self._warn(f"Cannot start pose estimation: {e}")
            return
        self._pick_on_board = self.chk_pick_board.isChecked()
        self._update_live_calibs()
        fps = min((s.fps for s in self.streams.values()), default=30.0)
        self.pose_worker = PoseWorker(None, self._latest_frames, self._cams_snapshot, self.recorder,
                                      max_skew_s=max(0.25, 3.0 / max(fps, 1.0)), factory=factory,
                                      on_ready=self._set_world_camera, label=label, key=key)
        self.pose_worker.start()
        self.b_pose.setText("Stop Pose Estimation")
        self._set_text(self.pose_label, f"Loading {label}...")
        self._update_pose_controls()
        if self.recorder.recording and not self.recorder.meta.get("pose_backend"):
            self.recorder.meta["pose_backend"] = label  # pose started during the recording
            self.recorder.meta["pose_backend_key"] = key

    def _poll_pose_worker(self) -> None:
        """GUI side of the estimator creation: report a failure (and reset the Start button),
        or complete the recording's metadata once the estimator exists."""
        pw = self.pose_worker
        if pw is None or (pw.loading and not pw.load_crashed):
            return
        if pw.load_error is not None or pw.loading:
            if pw.load_error is None:
                pw.load_error = "the pose thread ended while loading the model"
            self.pose_worker = None
            self.b_pose.blockSignals(True)
            self.b_pose.setChecked(False)
            self.b_pose.blockSignals(False)
            self.b_pose.setText("Start Pose Estimation")
            self._set_text(self.pose_label, f"Not running: {pw.load_error}")
            self._update_pose_controls()
            self._warn(f"Cannot start pose estimation ({pw.label}):\n{pw.load_error}")
            return
        if not pw.ready_seen:
            pw.ready_seen = True
            if pw.key is None:
                pw.key = getattr(pw.estimator, "name", None)
            # values changed while loading, and a board registered meanwhile: before pose_info
            self._pose_param_changed()
            self._sync_world_camera()
            meta = self.recorder.meta if self.recorder.recording else None
            if meta is not None and meta.get("pose_backend") == pw.label:
                meta["pose_backend_key"] = meta.get("pose_backend_key") or pw.key
                meta["pose_info"] = meta.get("pose_info") or self._pose_info(pw.estimator)

    @staticmethod
    def _pose_info(est) -> dict | None:
        try:
            return est.info() if hasattr(est, "info") else None
        except Exception:  # noqa: BLE001
            log.exception("pose estimator info failed")
            return None

    def _pose_status_text(self, pw: PoseWorker, pose: Pose3D | None) -> str:
        """Status line of the Record tab: rate, how 3D was obtained, reprojection error, notes."""
        if pw.loading and pw.load_waiting:
            return (f"Loading {pw.label}: waiting for the previous model load to finish "
                    f"({time.perf_counter() - pw.load_started:.0f} s)")
        if pw.loading and not pw.load_crashed:
            return (f"Loading {pw.label}... {time.perf_counter() - pw.load_started:.0f} s "
                    "(the first start of a backend downloads its model; this can take a few "
                    "minutes)")
        if not pw.alive:
            return f"STOPPED: {pw.error or 'the pose thread ended'} (click Stop, then Start)"
        parts = [f"{pw.fps:.1f} fps"]
        if pose is None:
            parts.append("no person detected")
        else:
            parts.append(pose_mode_text(pose))
            err = getattr(pose, "reproj_error_px", None)
            if err is not None and np.isfinite(err):
                parts.append(f"reprojection error {err:.1f} px")
        txt = " | ".join(parts)
        if pw.error:
            txt += f"\nError: {pw.error}"
        if pw.stale_cameras:
            txt += f"\nNo recent frames from {', '.join(pw.stale_cameras)}: not used"
        notes = list(getattr(pose, "notes", None) or []) if pose is not None else []
        if notes:
            shown = notes[:MAX_NOTES_SHOWN]
            txt += "\n" + "\n".join(shown)
            if len(notes) > len(shown):
                txt += f"\n(+{len(notes) - len(shown)} more)"
        return txt

    # ============================================================= record
    def _pick_outdir(self):
        p = QFileDialog.getExistingDirectory(self, "Output Folder", self.out_dir.text())
        if p:
            self.out_dir.setText(p)

    @safe_slot
    def toggle_record(self, on: bool):
        """Start/stop a recording of whatever is available: cameras (+ pose), the Wii, or both.
        A board that connects later (auto-connect) is added to the running recording."""
        if on:
            wii = self.force if self._wii_connected() else None
            if not self.streams and wii is None:
                self._warn("Nothing to record: add a camera (tab 1), or connect the Wii Balance "
                           "Board (with Auto-connect on, just switch it on and wait until it shows "
                           "as connected).")
                self.b_rec.setChecked(False)
                return
            calibs = self._update_live_calibs()  # the calibrations used for these streams
            self.recorder.root = Path(self.out_dir.text())
            pw = self.pose_worker
            est = pw.estimator if pw is not None else None
            fmt = getattr(est, "format", None)
            try:
                folder = self.recorder.start(
                    cams=list(self.streams.values()), calibrations=dict(calibs),
                    force=wii, board=self.board, geometry=self.geometry,
                    pose_backend=pw.label if pw is not None else None,
                    pose_backend_key=(pw.key or getattr(est, "name", None)) if pw else None,
                    keypoint_format=getattr(fmt, "key", None),
                    pose_info=self._pose_info(est) if est is not None else None,
                    save_pose2d=self.chk_save2d.isChecked(),
                    save_openpose_json=self.chk_save2d.isChecked() and self.chk_json.isChecked(),
                    subject=self.subject.text().strip(), notes=self.notes.text())
            except Exception as e:  # noqa: BLE001
                self.b_rec.setChecked(False)
                self._warn(f"Cannot start recording: {e}")
                return
            self._wii_link = "wii_connected" if wii is not None else "wii_disconnected"
            self._marks = 0
            self._rec_started = time.perf_counter()
            self.b_rec.setText("■ Stop Recording")
            self._set_marking(True)
            self._refresh_events_info()
            for n in sorted(self._cam_lost):  # already without frames at the start
                self._log_event(f"camera_lost {n}")
            self.status.showMessage(f"Recording to {folder}")
        else:
            self._set_marking(False)
            elapsed = time.perf_counter() - self._rec_started
            folder = self.recorder.stop(background=True)  # post-processing off the GUI thread
            self.b_rec.setText("● Start Recording")
            if folder:
                c = self.recorder.counts
                self.rec_label.setText(f"Stopped after {elapsed:.1f}s   Wii {c['wii']}  "
                                       f"Pose {c['pose']}  Events {c.get('events', 0)}")
                self.status.showMessage(f"Saved to {folder}")
                self.summary.setPlainText(f"Saved: {folder}\n\nPost-processing (fused.csv, "
                                          "summary.json)...")
                self._pending_summary = folder

    def _show_summary_when_done(self) -> None:
        folder = self._pending_summary
        if folder is None or self.recorder.post_processing:
            return
        self._pending_summary = None
        s = folder / "summary.json"
        text = s.read_text(encoding="utf-8") if s.exists() else "(post-processing failed, see the log)"
        self.summary.setPlainText(f"Saved: {folder}\n\n{text}")

    def _set_marking(self, on: bool):
        """Event markers are only possible while recording; tare and the pose settings cannot be
        changed while recording (they would change wii.csv / pose3d.csv mid-file)."""
        for w in (self.b_mark, self.act_mark, self.event_label):
            w.setEnabled(on)
        if hasattr(self, "b_tare"):
            self.b_tare.setEnabled(not on)
        if hasattr(self, "pose_backend"):
            self._update_pose_controls(recording=on)

    def _log_event(self, label: str) -> None:
        """Write an automatic event (camera lost/recovered/removed) while recording."""
        if self.recorder.recording and self.recorder.add_event(label) is not None:
            self._refresh_events_info()

    @safe_slot
    def mark_event(self):
        """Write an event marker (label from the text field, default mark_<n>) to events.csv."""
        if not self.recorder.recording:
            return
        t = time.perf_counter()
        label = self.event_label.text().strip() or f"mark_{self._marks + 1}"
        ev = self.recorder.add_event(label, t)
        if ev is None:
            return
        self._marks += 1
        self._refresh_events_info(ev)
        self.status.showMessage(f"Event '{ev['label']}' at {ev['t_rel']:.3f} s", 4000)

    def _refresh_events_info(self, last: dict | None = None):
        n = self.recorder.counts.get("events", 0)
        txt = f"Events: {n} ({self._marks} marked)"
        if last is not None:
            txt += f"   last: '{last['label']}' at {last['t_rel']:.2f} s"
        self.events_info.setText(txt)

    def _streams_text(self) -> str:
        """Which streams a recording contains (or would contain if started now)."""
        if self.recorder.recording:
            cams = list(self.recorder.meta.get("camera_names", []))
        else:
            cams = list(self.streams)
        items = cams + (["pose"] if self.pose_worker is not None and cams else [])
        wii = self._wii_connected()
        if wii:
            items.append("Wii")
        if not items:
            if self.force is not None:
                return "Waiting for the Wii Balance Board (switch it on), or add a camera"
            return "Nothing to record: add a camera or connect the Wii Balance Board"
        txt = ", ".join(items)
        if not wii:
            later = " (the board is added when it connects)" if self.force is not None else ""
            txt += f"\nWii not connected — recording video/pose only{later}"
        elif not cams:
            txt += "\nNo cameras — recording Wii only"
        return txt

    # ============================================================ project
    @safe_slot
    def save_project(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save Project", "poseboard_project.json", "JSON (*.json)")
        if not p:
            return
        data = {
            # width/height = the requested resolution (null = camera default)
            "cameras": [{"name": n, "source": str(s.source), "width": s.req[0], "height": s.req[1]}
                        for n, s in self.streams.items()],
            "calibrations": [c.to_dict() for c in self.calibs.values()],
            "checkerboard": {"cols": self.cb_cols.value(), "rows": self.cb_rows.value(),
                             "square_mm": self.cb_square.value()},
            "board_geometry": self.geometry.to_dict(),
            "board_pose": None if self.board is None else self.board.to_dict(),
            "board_rotation_deg": self.board_rotation,
            "clicks": {n: [list(map(float, q)) for q in pts] for n, pts in self.clicks.items()},
            "click_image_size": {n: list(sz) for n, sz in self.click_sizes.items()},
            "pose": self.pose_settings(),
        }
        Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def pose_settings(self) -> dict:
        """The Record tab's pose settings (saved in the project file)."""
        self._store_shown_options()
        return {"backend": self.current_backend(), "plugin": self.plugin_path.text().strip(),
                "backends": {k: {"options": dict(v["options"]), "min_score": v["min_score"]}
                             for k, v in self._pose_settings.items()},
                "reproj_threshold_px": self.reproj_thr.value(),
                "smoothing": self.chk_smooth.isChecked(),
                "pick_person_on_board": self.chk_pick_board.isChecked(),
                "save_pose2d": self.chk_save2d.isChecked(),
                "save_openpose_json": self.chk_json.isChecked()}

    def apply_pose_settings(self, d: dict) -> None:
        """Restore ``pose_settings()`` (unknown backends/options are ignored; not while pose
        estimation or a recording runs)."""
        if not d or self.pose_worker is not None or self.recorder.recording:
            return
        for key, v in (d.get("backends") or {}).items():
            spec = BACKENDS.get(key)
            if spec is None or not isinstance(v, dict):
                continue
            st = self._settings(key)
            st["options"].update({k: val for k, val in (v.get("options") or {}).items()
                                  if k in spec.options})
            if isinstance(v.get("min_score"), (int, float)):
                st["min_score"] = float(v["min_score"])
        if d.get("plugin"):
            self.plugin_path.setText(str(d["plugin"]))
        for spin, k in ((self.reproj_thr, "reproj_threshold_px"),):
            if isinstance(d.get(k), (int, float)):
                spin.setValue(float(d[k]))
        for chk, k in ((self.chk_smooth, "smoothing"), (self.chk_pick_board, "pick_person_on_board"),
                       (self.chk_save2d, "save_pose2d"), (self.chk_json, "save_openpose_json")):
            if isinstance(d.get(k), bool):
                chk.setChecked(d[k])
        i = self.pose_backend.findData(d.get("backend"))
        self._shown_backend = None  # the stored values win over the widgets shown now
        if i >= 0 and i != self.pose_backend.currentIndex():
            self.pose_backend.setCurrentIndex(i)  # rebuilds the option widgets
        else:
            self._backend_changed()

    @safe_slot
    def load_project(self, path: str | None = None):
        if not isinstance(path, str):
            path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "JSON (*.json)")
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        calib_dicts = {d["name"]: d for d in data.get("calibrations", [])}
        for cam in data.get("cameras", []):
            self._camera_sources.setdefault(cam["name"], str(cam["source"]))
            if cam["name"] in self.streams:
                continue
            if "width" in cam:  # the resolution the camera ran at when the project was saved
                size = (cam["width"], cam["height"]) if cam.get("width") else None
            elif cam["name"] in calib_dicts:  # older project: the calibration's resolution
                size = tuple(calib_dicts[cam["name"]]["image_size"])
            else:
                size = _UI_RESOLUTION
            self.add_camera(str(cam["source"]), cam["name"], size=size)
        self.calibs = {n: CameraCalibration.from_dict(d) for n, d in calib_dicts.items()}
        self._fallback_calibs.clear()
        self._calib_mismatch.clear()
        cb = data.get("checkerboard")
        if cb:
            self.cb_cols.setValue(cb["cols"])
            self.cb_rows.setValue(cb["rows"])
            self.cb_square.setValue(cb["square_mm"])
        for k, v in data.get("board_geometry", {}).items():
            if k in self.geo_spins:
                self.geo_spins[k].setValue(v)
        self.board = BoardPose.from_dict(data["board_pose"]) if data.get("board_pose") else None
        self.board_rotation = int(data.get("board_rotation_deg", 0))
        self.board_stale = None
        self.clicks = {n: [np.array(q) for q in pts] for n, pts in data.get("clicks", {}).items()}
        sizes = data.get("click_image_size")
        if sizes is None:  # older project: clicks were made with the calibration's resolution
            sizes = {n: calib_dicts[n]["image_size"] for n in self.clicks if n in calib_dicts}
        self.click_sizes = {n: tuple(int(v) for v in sz) for n, sz in sizes.items()}
        self.apply_pose_settings(data.get("pose") or {})
        self._refresh_calib_info()
        self._refresh_board_info()
        self._sync_world_camera()

    # =============================================================== tick
    def _tick_safe(self):
        """Timer slot: a failing view update is logged (once per message) and shown in the
        status bar instead of a message box every 33 ms."""
        try:
            self._tick()
            self._tick_error = None
        except Exception as e:  # noqa: BLE001
            msg = f"Display update failed: {e}"
            if msg != self._tick_error:
                log.exception("display update failed")
                self._tick_error = msg
            self.status.showMessage(msg, 2000)

    def _tick(self):
        name = self.current_cam()
        stream = self.streams.get(name) if name else None
        frame = stream.latest() if stream else None
        now = time.perf_counter()
        self._update_live_calibs()
        self._check_cameras(now)
        force = self.force.latest() if self.force else None
        if force is not None and now - force.t > STALE_FORCE_S:
            force = None  # board disconnected: do not show its last sample as live
        self._poll_pose_worker()
        pw = self.pose_worker
        pose = pw.latest if pw is not None and pw.alive else None
        fused = fuse(self.board, force, pose)
        com_world = (fused.com_world if fused is not None
                     else center_of_mass(pose.keypoints, pose.names) if pose is not None else None)

        if frame is not None:
            img = frame.image.copy()
            cam = self._live_calibs.get(name) or self._ensure_calib(name, frame)
            if self.chk_detect.isChecked() and self.tabs.currentIndex() == 1:
                self._preview_checkerboard(name, frame.image, img, now)
            if self.board is not None and self.chk_overlay.isChecked():
                draw_board(img, cam, self.board, self.geometry)
                if force is not None:
                    draw_cop(img, cam, self.board, force.cop_board, force.total_kg)
            if pose is not None and self.chk_skeleton.isChecked():
                est = pw.estimator
                draw_pose(img, cam, name, pose, getattr(est, "skeleton", None) or None,
                          self.board, com_world,
                          min_score=float(getattr(est, "min_score", 0.0) or 0.0))
                draw_caption(img, f"Pose: {pose_mode_text(pose)}"
                             + ("" if name in pose.per_camera_2d or name in pose.views_used
                                else f" (no subject in {name})"))
            if self.click_mode or (self.tabs.currentIndex() == 2 and self.clicks.get(name)):
                draw_clicks(img, self.clicks.get(name, []), self._next_click_hint(name))
            if name in self._cam_lost:
                cv2.putText(img, f"NO NEW FRAMES for {now - frame.t:.0f} s", (12, img.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            self.video.set_image(img)
        else:
            self.video.placeholder = (f"No frames from {name} yet" if stream is not None
                                      else "No camera selected")
            self.video.set_image(None)

        # Wii status and top-down view (COP from the Wii, COM from the pose: each optional)
        self._set_text(self.wii_label, self._wii_text(force))
        self._update_corner_check(now)
        if self.force is not None or pw is not None:
            cop_trail = [s.cop_board for s in self.force.recent(5.0)] if self.force else []
            if pose is not None and com_world is not None and self.board is not None:
                if not self._com_trail or self._com_trail[-1][0] != pose.t:
                    self._com_trail.append((pose.t, *self.board.world_to_board.apply(com_world)[:2]))
            com_trail = [(x, y) for t, x, y in self._com_trail if now - t <= 5.0]
            self.cop_view.set_data(cop_trail, com_trail, force.total_kg if force else 0.0,
                                   "" if fused is None or fused.com_minus_cop is None else
                                   f"COM−COP = {np.round(fused.com_minus_cop * 1000, 1)} mm")

        if pw is not None:
            self._set_text(self.pose_label, self._pose_status_text(pw, pose))

        self._set_text(self.streams_label, self._streams_text())
        self._update_warnings(now)
        self._show_summary_when_done()
        if self.recorder.recording:
            c = self.recorder.counts
            self.rec_label.setText(f"Recording {now - self._rec_started:6.1f}s   "
                                   f"Wii {c['wii']}  Pose {c['pose']}  Events {c.get('events', 0)}")
        cams_fps = "  ".join(self._camera_status(n, s, now) for n, s in self.streams.items())
        if cams_fps and not self.status.currentMessage():
            self.status.showMessage(cams_fps, 1000)

    def _preview_checkerboard(self, name: str, image: np.ndarray, out: np.ndarray, now: float):
        """Live checkerboard preview: detection runs on a downscaled copy in a worker thread (a
        few times per second), so the window stays responsive; the latest result is drawn."""
        spec = self.checker_spec()
        if not self._detect_busy.is_set() and now - self._detect_started >= PREVIEW_DETECT_INTERVAL_S:
            self._detect_busy.set()
            self._detect_started = now

            def work():
                try:
                    c = find_checkerboard(image, spec, fast=True, max_width=PREVIEW_DETECT_WIDTH)
                except Exception:  # noqa: BLE001
                    c = None
                self._detect_result = (name, spec.pattern_size, c, time.perf_counter())
                self._detect_busy.clear()

            threading.Thread(target=work, name="checkerboard-preview", daemon=True).start()
        res = self._detect_result
        if (res is not None and res[0] == name and res[1] == spec.pattern_size
                and res[2] is not None and now - res[3] < 1.0):
            cv2.drawChessboardCorners(out, spec.pattern_size,
                                      res[2].reshape(-1, 1, 2).astype(np.float32), True)

    def _check_cameras(self, now: float) -> None:
        """Detect cameras that stopped delivering frames (unplugged, driver stalled) and log
        camera_lost / camera_recovered events while recording."""
        for n, s in list(self.streams.items()):
            age = s.frame_age(now)
            lost = age is not None and age > max(CAMERA_LOST_S, 5.0 / max(s.fps, 1.0))
            if lost and n not in self._cam_lost:
                self._cam_lost.add(n)
                log.warning("camera %s: no new frames for %.1f s", n, age)
                self._log_event(f"camera_lost {n}")
            elif not lost and n in self._cam_lost:
                self._cam_lost.discard(n)
                self._log_event(f"camera_recovered {n}")

    def _camera_status(self, n: str, s: CameraStream, now: float) -> str:
        age = s.frame_age(now)
        if age is None:
            return f"{n}: no frames yet"
        if n in self._cam_lost:
            return f"{n}: NO FRAMES for {age:.0f} s"
        txt = f"{n}:{s.measured_fps:.0f}fps"
        if s.fps and s.measured_fps and s.measured_fps < 0.6 * s.fps:
            txt += f" (camera reports {s.fps:.0f}: low light or USB bandwidth?)"
        if s.error and s.error.startswith("recording"):
            txt += f" [{s.error}]"
        return txt

    def _update_warnings(self, now: float) -> None:
        """Persistent warnings above the video: calibration/resolution mismatch, outdated board
        pose, cameras without frames, stopped pose estimation."""
        msgs = []
        for n, (cal, got) in sorted(self._calib_mismatch.items()):
            msgs.append(f"{n}: the camera delivers {got[0]}x{got[1]} but its calibration is for "
                        f"{cal[0]}x{cal[1]}. The calibration is kept, but only approximate "
                        f"intrinsics can be used until the camera runs at {cal[0]}x{cal[1]}: "
                        "remove it and add it again with that Resolution.")
        stale = self._board_stale_text()
        if stale:
            msgs.append(f"Board pose outdated: {stale}.")
        for n in sorted(self._cam_lost):
            age = self.streams[n].frame_age(now) if n in self.streams else None
            if age is not None:
                msgs.append(f"{n}: no new frames for {age:.0f} s (unplugged or driver stalled?).")
        pw = self.pose_worker
        if pw is not None and not pw.alive and not pw.loading and pw.load_error is None:
            msgs.append(f"Pose estimation stopped: {pw.error or 'the pose thread ended'}.")
        wc = self.board.world_camera if self.board is not None else None
        if pw is not None and wc and wc not in self.streams:
            msgs.append(f"The board was registered in {wc} (no extrinsics: the world frame is "
                        f"{wc}'s camera frame), which is not running: no 3D pose until {wc} is "
                        "added again, or set the camera extrinsics and compute the board pose.")
        text = "\n".join(msgs)
        self._set_text(self.warn_label, text)
        self.warn_label.setVisible(bool(text))

    @staticmethod
    def _set_text(label: QLabel, text: str):
        if label.text() != text:  # avoid relayouts at 30 Hz
            label.setText(text)

    def closeEvent(self, e):
        if self.recorder.recording:
            self.recorder.stop(background=True)  # the process waits for it before exiting
        if self.pose_worker:
            self.pose_worker.stop(timeout=0.5)  # a busy thread is a daemon: it ends with the app
        self._stop_force()
        for s in self.streams.values():
            s.stop()
        super().closeEvent(e)


def _install_excepthook() -> None:
    """Show otherwise unhandled exceptions (e.g. from a Qt slot) in a message box instead of
    losing them in a console that may not exist (pythonw / packaged exe). The previous hook
    (console or log file) still receives the traceback."""
    showing = [False]
    previous = sys.excepthook

    def hook(etype, value, tb):
        previous(etype, value, tb)
        if (showing[0] or issubclass(etype, KeyboardInterrupt)
                or threading.current_thread() is not threading.main_thread()):
            return
        showing[0] = True
        try:
            QMessageBox.critical(QApplication.activeWindow(), "PoseBoard",
                                 f"Unexpected error: {etype.__name__}: {value}\n\n"
                                 "The app keeps running; details are in the log.")
        finally:
            showing[0] = False

    sys.excepthook = hook


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    _install_excepthook()
    w = MainWindow()
    # Closing the console window of run_poseboard.bat ends the process without closeEvent:
    # close the recording's files first (post-processing can be run later)
    w._console_handler = on_console_close(lambda: w.recorder.stop(analyze=False))
    w.show()
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        w.load_project(sys.argv[1])  # errors (e.g. file not found) are shown, the app keeps running
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
