"""PoseBoard 主界面。

流程：设备 → 相机标定（棋盘格）→ Wii 定位（在画面中点选四角 + 中心）→ 采集。
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from poseboard.calibration import (CameraCalibration, CheckerboardSpec, IntrinsicCollector,
                                   approximate_calibration, extrinsics_from_checkerboard,
                                   find_checkerboard, load_calibrations, save_calibrations,
                                   save_pose2sim_toml)
from poseboard.camera import CameraStream
from poseboard.fusion import fuse
from poseboard.geometry import (LANDMARK_NAMES, BoardGeometry, BoardPose, register_board,
                                rotate_board_frame)
from poseboard.overlay import draw_board, draw_clicks, draw_cop, draw_pose
from poseboard.pose.base import Pose3D, PoseEstimator
from poseboard.session import SessionRecorder
from poseboard.wii.device import BalanceBoardHID, ForceSource, SimulatedBoard
from poseboard.gui.widgets import CopView, VideoView

log = logging.getLogger(__name__)

CLICK_HINTS = {
    "TL": "左前角 TL（靠近 TL 传感器的板角）",
    "TR": "右前角 TR",
    "BR": "右后角 BR",
    "BL": "左后角 BL",
    "C": "板面中心 C",
}


class PoseWorker:
    """后台线程：取各相机最新帧 → 姿态估计 → 保存最新结果并写入录制。"""

    def __init__(self, estimator: PoseEstimator, get_frames, get_cams, recorder: SessionRecorder):
        self.estimator = estimator
        self.get_frames = get_frames
        self.get_cams = get_cams
        self.recorder = recorder
        self.latest: Pose3D | None = None
        self.fps = 0.0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pose", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        self.estimator.close()

    def _run(self):
        last_idx: dict[str, int] = {}
        while not self._stop.is_set():
            frames = self.get_frames()
            fresh = {n: f for n, f in frames.items() if last_idx.get(n) != f.index}
            if not fresh:
                time.sleep(0.003)
                continue
            for n, f in frames.items():
                last_idx[n] = f.index
            t0 = time.perf_counter()
            try:
                pose = self.estimator.process({n: (f.t, f.image) for n, f in frames.items()},
                                              self.get_cams())
            except Exception as e:  # noqa: BLE001
                log.exception("pose estimation failed")
                self.error = str(e)
                time.sleep(0.5)
                continue
            dt = time.perf_counter() - t0
            self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-3) if self.fps else 1 / max(dt, 1e-3)
            self.latest = pose
            if pose is not None and self.recorder.recording:
                self.recorder.add_pose(pose)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PoseBoard — 3D 姿态 + Wii Balance Board 同步采集")
        self.resize(1500, 900)

        self.streams: dict[str, CameraStream] = {}
        self.calibs: dict[str, CameraCalibration] = {}
        self.collectors: dict[str, IntrinsicCollector] = {}
        self.clicks: dict[str, list[np.ndarray]] = {}
        self.geometry = BoardGeometry()
        self.board: BoardPose | None = None
        self.force: ForceSource | None = None
        self.pose_worker: PoseWorker | None = None
        self.recorder = SessionRecorder(Path.cwd() / "recordings")
        self.click_mode = False
        self._rec_started = 0.0
        self._com_trail: deque = deque(maxlen=300)  # (t, x, y) 板坐标

        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    # ================================================================== UI
    def _build_ui(self):
        self._build_menu()
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        top = QHBoxLayout()
        top.addWidget(QLabel("显示相机:"))
        self.cam_select = QComboBox()
        self.cam_select.currentTextChanged.connect(lambda _: self._update_click_hint())
        top.addWidget(self.cam_select, 1)
        self.chk_overlay = QCheckBox("叠加平衡板/COP")
        self.chk_overlay.setChecked(True)
        self.chk_skeleton = QCheckBox("叠加骨架/COM")
        self.chk_skeleton.setChecked(True)
        top.addWidget(self.chk_overlay)
        top.addWidget(self.chk_skeleton)
        lv.addLayout(top)
        self.video = VideoView()
        self.video.clicked.connect(self._on_video_click)
        lv.addWidget(self.video, 1)
        self.cop_view = CopView()
        self.cop_view.setMaximumHeight(260)
        lv.addWidget(self.cop_view)
        splitter.addWidget(left)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_devices(), "1 设备")
        self.tabs.addTab(self._tab_calibration(), "2 相机标定")
        self.tabs.addTab(self._tab_board(), "3 Wii 定位")
        self.tabs.addTab(self._tab_record(), "4 采集")
        self.tabs.setMinimumWidth(420)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.status = self.statusBar()

    def _build_menu(self):
        m = self.menuBar().addMenu("文件")
        for text, fn in (("保存配置…", self.save_project), ("加载配置…", self.load_project)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            m.addAction(a)
        m.addSeparator()
        for text, fn in (("导入相机标定 (.json / Pose2Sim .toml)…", self.load_calib),
                         ("导出相机标定 (.json)…", self.save_calib),
                         ("导出 Pose2Sim Calib.toml…", self.export_toml)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            m.addAction(a)

    def _tab_devices(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        g = QGroupBox("相机")
        f = QGridLayout(g)
        self.cam_source = QLineEdit("0")
        self.cam_source.setToolTip("设备序号 (0,1,…)、视频文件路径或 rtsp/http 地址")
        self.cam_res = QComboBox()
        self.cam_res.addItems(["默认", "640x480", "1280x720", "1920x1080"])
        self.cam_res.setCurrentText("1280x720")
        b_add = QPushButton("添加相机")
        b_add.clicked.connect(self.add_camera)
        b_file = QPushButton("视频文件…")
        b_file.clicked.connect(self._pick_video)
        self.cam_list = QListWidget()
        b_rm = QPushButton("移除所选")
        b_rm.clicked.connect(self.remove_camera)
        f.addWidget(QLabel("来源"), 0, 0)
        f.addWidget(self.cam_source, 0, 1)
        f.addWidget(b_file, 0, 2)
        f.addWidget(QLabel("分辨率"), 1, 0)
        f.addWidget(self.cam_res, 1, 1)
        f.addWidget(b_add, 1, 2)
        f.addWidget(self.cam_list, 2, 0, 1, 3)
        f.addWidget(b_rm, 3, 2)
        v.addWidget(g)

        g = QGroupBox("Wii Balance Board")
        f = QGridLayout(g)
        self.wii_devices = QComboBox()
        b_scan = QPushButton("扫描")
        b_scan.clicked.connect(self.scan_wii)
        b_conn = QPushButton("连接")
        b_conn.clicked.connect(self.connect_wii)
        b_sim = QPushButton("使用模拟器")
        b_sim.clicked.connect(self.connect_sim)
        b_disc = QPushButton("断开")
        b_disc.clicked.connect(self.disconnect_wii)
        b_tare = QPushButton("去皮（板上无人时）")
        b_tare.clicked.connect(self.tare)
        self.min_kg = QDoubleSpinBox()
        self.min_kg.setRange(0, 50)
        self.min_kg.setValue(5.0)
        self.min_kg.setSuffix(" kg")
        self.min_kg.valueChanged.connect(lambda v: setattr(self.force, "min_total_kg", v) if self.force else None)
        self.wii_label = QLabel("未连接")
        self.wii_label.setStyleSheet("font-family: monospace")
        f.addWidget(self.wii_devices, 0, 0, 1, 2)
        f.addWidget(b_scan, 0, 2)
        f.addWidget(b_conn, 1, 0)
        f.addWidget(b_sim, 1, 1)
        f.addWidget(b_disc, 1, 2)
        f.addWidget(b_tare, 2, 0, 1, 2)
        f.addWidget(QLabel("COP 最小总重"), 3, 0)
        f.addWidget(self.min_kg, 3, 1)
        f.addWidget(self.wii_label, 4, 0, 1, 3)
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _tab_calibration(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("棋盘格")
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
        self.chk_detect = QCheckBox("实时显示棋盘格检测")
        f.addRow("内角点（列）", self.cb_cols)
        f.addRow("内角点（行）", self.cb_rows)
        f.addRow("方格边长", self.cb_square)
        f.addRow(self.chk_detect)
        v.addWidget(g)

        g = QGroupBox("内参（对当前显示的相机）")
        f = QVBoxLayout(g)
        f.addWidget(QLabel("把棋盘格放在画面不同位置/角度，逐张采集 15~30 张后计算。"))
        h = QHBoxLayout()
        b = QPushButton("采集当前帧")
        b.clicked.connect(self.capture_intrinsic)
        h.addWidget(b)
        b = QPushButton("清空")
        b.clicked.connect(lambda: self.collectors.pop(self.current_cam(), None))
        h.addWidget(b)
        b = QPushButton("计算内参")
        b.clicked.connect(self.compute_intrinsic)
        h.addWidget(b)
        f.addLayout(h)
        v.addWidget(g)

        g = QGroupBox("外参 / 世界坐标系")
        f = QVBoxLayout(g)
        f.addWidget(QLabel("把棋盘格平放在地面（最好靠近平衡板），所有相机都能看到，\n"
                           "点击下方按钮：棋盘格即成为世界坐标系（Z 朝上，单位米）。"))
        b = QPushButton("用棋盘格设置所有相机外参")
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
        g = QGroupBox("平衡板尺寸")
        f = QFormLayout(g)
        self.geo_spins = {}
        for key, label in (("length_mm", "板面长（左右）"), ("width_mm", "板面宽（前后）"),
                           ("sensor_dx_mm", "传感器间距 左右"), ("sensor_dy_mm", "传感器间距 前后")):
            s = QDoubleSpinBox()
            s.setRange(10, 2000)
            s.setSuffix(" mm")
            s.setValue(getattr(self.geometry, key))
            s.valueChanged.connect(self._geometry_changed)
            self.geo_spins[key] = s
            f.addRow(label, s)
        v.addWidget(g)

        g = QGroupBox("在画面中点选平衡板")
        f = QVBoxLayout(g)
        f.addWidget(QLabel("顺序：1 TL 左前 → 2 TR 右前 → 3 BR 右后 → 4 BL 左后 → 5 C 中心\n"
                           "“前”= 人站在板上面朝的方向（TL/TR 传感器一侧）。\n"
                           "左键添加，右键撤销。多台相机可分别点选（会三角化，更准）。"))
        h = QHBoxLayout()
        self.b_click = QPushButton("开始点选")
        self.b_click.setCheckable(True)
        self.b_click.toggled.connect(self._toggle_click_mode)
        h.addWidget(self.b_click)
        b = QPushButton("撤销")
        b.clicked.connect(self.undo_click)
        h.addWidget(b)
        b = QPushButton("清除本相机")
        b.clicked.connect(lambda: (self.clicks.pop(self.current_cam(), None), self._update_click_hint()))
        h.addWidget(b)
        f.addLayout(h)
        self.click_hint = QLabel("")
        self.click_hint.setStyleSheet("color: #c60; font-weight: bold")
        f.addWidget(self.click_hint)
        b = QPushButton("计算平衡板位置")
        b.clicked.connect(self.compute_board)
        f.addWidget(b)
        h = QHBoxLayout()
        for deg in (90, 180, -90):
            b = QPushButton(f"坐标系旋转 {deg}°")
            b.clicked.connect(lambda _=False, d=deg: self.rotate_board(d))
            h.addWidget(b)
        f.addLayout(h)
        v.addWidget(g)
        self.board_info = QPlainTextEdit()
        self.board_info.setReadOnly(True)
        self.board_info.setPlainText("验证方法：站到板的某一角，画面中 COP 点应出现在同一角；\n"
                                     "若方向不对，使用“坐标系旋转”修正或重新点选。")
        v.addWidget(self.board_info, 1)
        return w

    def _tab_record(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("3D 姿态估计")
        f = QGridLayout(g)
        self.pose_backend = QComboBox()
        self.pose_backend.addItems(["MediaPipe (full)", "MediaPipe (lite)", "MediaPipe (heavy)",
                                    "插件 (.py)"])
        self.plugin_path = QLineEdit()
        self.plugin_path.setPlaceholderText("插件文件，例如 plugins/poseassess_plugin.py")
        b_pick = QPushButton("…")
        b_pick.clicked.connect(self._pick_plugin)
        self.b_pose = QPushButton("启动姿态估计")
        self.b_pose.setCheckable(True)
        self.b_pose.toggled.connect(self.toggle_pose)
        self.pose_label = QLabel("未启动")
        f.addWidget(self.pose_backend, 0, 0, 1, 3)
        f.addWidget(self.plugin_path, 1, 0, 1, 2)
        f.addWidget(b_pick, 1, 2)
        f.addWidget(self.b_pose, 2, 0, 1, 3)
        f.addWidget(self.pose_label, 3, 0, 1, 3)
        v.addWidget(g)

        g = QGroupBox("录制")
        f = QFormLayout(g)
        self.subject = QLineEdit()
        self.notes = QLineEdit()
        self.out_dir = QLineEdit(str(self.recorder.root))
        f.addRow("受试者", self.subject)
        f.addRow("备注", self.notes)
        h = QHBoxLayout()
        h.addWidget(self.out_dir)
        b = QPushButton("…")
        b.clicked.connect(self._pick_outdir)
        h.addWidget(b)
        f.addRow("保存目录", h)
        self.b_rec = QPushButton("● 开始录制")
        self.b_rec.setCheckable(True)
        self.b_rec.setStyleSheet("QPushButton:checked { background: #d33; color: white }")
        self.b_rec.toggled.connect(self.toggle_record)
        f.addRow(self.b_rec)
        self.rec_label = QLabel("")
        f.addRow(self.rec_label)
        v.addWidget(g)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        v.addWidget(self.summary, 1)
        return w

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
        return dict(self.calibs)

    # ============================================================ devices
    def _pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if p:
            self.cam_source.setText(p)

    def add_camera(self, source: str | None = None, name: str | None = None):
        src = source if isinstance(source, str) else self.cam_source.text().strip()
        if not src:
            return
        name = name or f"cam{len(self.streams)}"
        while name in self.streams:
            name += "_"
        wh = self.cam_res.currentText()
        width = height = None
        if "x" in wh:
            width, height = (int(v) for v in wh.split("x"))
        s = CameraStream(name, src, width, height)
        try:
            s.start()
        except Exception as e:  # noqa: BLE001
            self._warn(str(e))
            return
        self.streams[name] = s
        self.cam_list.addItem(f"{name}  ←  {src}")
        self.cam_select.addItem(name)
        self.cam_select.setCurrentText(name)

    def remove_camera(self):
        row = self.cam_list.currentRow()
        if row < 0:
            return
        name = self.cam_list.item(row).text().split()[0]
        s = self.streams.pop(name, None)
        if s:
            s.stop()
        self.cam_list.takeItem(row)
        self.cam_select.removeItem(self.cam_select.findText(name))

    def _ensure_calib(self, name: str, frame) -> CameraCalibration:
        c = self.calibs.get(name)
        h, w = frame.image.shape[:2]
        if c is None or tuple(c.image_size) != (w, h):
            if c is not None:
                log.warning("%s 分辨率与标定不一致，使用近似内参", name)
            c = approximate_calibration(name, w, h)
            self.calibs[name] = c
        return c

    def scan_wii(self):
        self.wii_devices.clear()
        try:
            devs = BalanceBoardHID.list_devices()
        except Exception as e:  # noqa: BLE001
            self._warn(f"无法枚举 HID 设备：{e}\n请安装 hidapi（pip install hidapi）")
            return
        for d in devs:
            self.wii_devices.addItem(f"{d.get('product_string') or 'Balance Board'}  {d['path']!r}", d["path"])
        if not devs:
            self.status.showMessage("未找到 Wii Balance Board，请先蓝牙配对（按电池仓里的红色 SYNC 键）", 8000)

    def _set_force(self, src: ForceSource):
        self.disconnect_wii()
        src.sensor_dx_m = self.geometry.sensor_dx_mm / 1000
        src.sensor_dy_m = self.geometry.sensor_dy_mm / 1000
        src.min_total_kg = self.min_kg.value()
        self.force = src
        src.start()

    def connect_wii(self):
        path = self.wii_devices.currentData()
        self._set_force(BalanceBoardHID(path))

    def connect_sim(self):
        self._set_force(SimulatedBoard())

    def disconnect_wii(self):
        if self.force:
            self.force.stop()
        self.force = None

    def tare(self):
        if not self.force:
            return
        try:
            t = self.force.do_tare(1.0)
            self.status.showMessage(f"去皮完成：{np.round(t, 2)} kg", 5000)
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
            self.status.showMessage("未检测到棋盘格", 3000)
        else:
            self.status.showMessage(f"{name}: 已采集 {len(col)} 张", 3000)

    def compute_intrinsic(self):
        name = self.current_cam()
        col = self.collectors.get(name)
        if not col:
            self._warn("请先采集棋盘格图像")
            return
        try:
            c = col.calibrate(name)
        except Exception as e:  # noqa: BLE001
            self._warn(str(e))
            return
        self.calibs[name] = c
        self._refresh_calib_info()

    def compute_extrinsics_all(self):
        spec, msgs = self.checker_spec(), []
        for name, f in self._latest_frames().items():
            c = self._ensure_calib(name, f)
            try:
                err = extrinsics_from_checkerboard(c, f.image, spec)
                msgs.append(f"{name}: 重投影误差 {err:.2f}px, 相机位置 {np.round(c.center_world, 3)} m")
            except Exception as e:  # noqa: BLE001
                msgs.append(f"{name}: {e}")
        self.status.showMessage("; ".join(msgs), 10000)
        self._refresh_calib_info()

    def _refresh_calib_info(self):
        lines = []
        for n, c in self.calibs.items():
            lines.append(f"[{n}] {c.image_size[0]}x{c.image_size[1]}  "
                         f"内参 RMS={c.intrinsic_rms if c.intrinsic_rms is not None else '未标定(近似)'}")
            lines.append(f"  fx={c.K[0, 0]:.1f} fy={c.K[1, 1]:.1f} cx={c.K[0, 2]:.1f} cy={c.K[1, 2]:.1f}")
            if c.has_extrinsics:
                lines.append(f"  外参误差={c.extrinsic_rms if c.extrinsic_rms is None else round(c.extrinsic_rms, 3)}px"
                             f"  相机位置(m)={np.round(c.center_world, 3)}")
            else:
                lines.append("  外参：未设置")
        for n, col in self.collectors.items():
            lines.append(f"{n}: 已采集 {len(col)} 张棋盘格")
        self.calib_info.setPlainText("\n".join(lines))

    def load_calib(self):
        p, _ = QFileDialog.getOpenFileName(self, "导入相机标定", "", "Calibration (*.json *.toml)")
        if not p:
            return
        cams = load_calibrations(p)
        names = list(self.streams)
        for i, c in enumerate(cams):
            # 名称对不上时按顺序对应到已添加的相机
            if c.name not in self.streams and i < len(names):
                c.name = names[i]
            self.calibs[c.name] = c
        self._refresh_calib_info()

    def save_calib(self):
        p, _ = QFileDialog.getSaveFileName(self, "导出相机标定", "calibration.json", "JSON (*.json)")
        if p:
            save_calibrations(p, list(self.calibs.values()))

    def export_toml(self):
        p, _ = QFileDialog.getSaveFileName(self, "导出 Pose2Sim 标定", "Calib.toml", "TOML (*.toml)")
        if p:
            save_pose2sim_toml(p, list(self.calibs.values()))

    # ============================================================== board
    def _geometry_changed(self):
        self.geometry = BoardGeometry(**{k: s.value() for k, s in self.geo_spins.items()})
        self.cop_view.board_mm = (self.geometry.length_mm, self.geometry.width_mm)
        self.cop_view.sensor_mm = (self.geometry.sensor_dx_mm, self.geometry.sensor_dy_mm)
        if self.force:
            self.force.sensor_dx_m = self.geometry.sensor_dx_mm / 1000
            self.force.sensor_dy_m = self.geometry.sensor_dy_mm / 1000

    def _toggle_click_mode(self, on: bool):
        self.click_mode = on
        self.video.crosshair = on
        self.b_click.setText("结束点选" if on else "开始点选")
        self._update_click_hint()

    def _next_click_hint(self, name: str | None) -> str | None:
        if not self.click_mode or name is None:
            return None
        n = len(self.clicks.get(name, []))
        if n >= 5:
            return "5 个点已完成，可点击“计算平衡板位置”"
        return f"请点击 {n + 1}/5：{CLICK_HINTS[LANDMARK_NAMES[n]]}"

    def _update_click_hint(self):
        self.click_hint.setText(self._next_click_hint(self.current_cam()) or "")

    def _on_video_click(self, x: float, y: float, button: int):
        name = self.current_cam()
        if not self.click_mode or name is None:
            return
        pts = self.clicks.setdefault(name, [])
        if button == 2:
            if pts:
                pts.pop()
        elif len(pts) < 5:
            pts.append(np.array([x, y]))
        self._update_click_hint()

    def undo_click(self):
        pts = self.clicks.get(self.current_cam() or "")
        if pts:
            pts.pop()
        self._update_click_hint()

    def compute_board(self):
        frames = self._latest_frames()
        names = [n for n, p in self.clicks.items() if len(p) == 5]
        if not names:
            self._warn("请至少在一台相机中点完 5 个点")
            return
        cams = []
        for n in names:
            if n in frames:
                self._ensure_calib(n, frames[n])
            if n not in self.calibs:
                self._warn(f"{n} 没有标定信息")
                return
            cams.append(self.calibs[n])
        try:
            self.board = register_board(self.geometry, cams, [np.array(self.clicks[n]) for n in names])
        except Exception as e:  # noqa: BLE001
            self._warn(str(e))
            return
        self.b_click.setChecked(False)
        self._refresh_board_info()

    def rotate_board(self, deg: int):
        if self.board:
            self.board = rotate_board_frame(self.board, deg)
            self._refresh_board_info()

    def _refresh_board_info(self):
        b = self.board
        if b is None:
            return
        T = b.board_to_world
        lines = [f"方法：{'多相机三角化' if b.method == 'triangulation' else '单相机 PnP'}",
                 "重投影误差(px)：" + ", ".join(f"{k}={v:.2f}" for k, v in b.reproj_error_px.items()),
                 f"板中心（世界, m）：{np.round(T.t, 4)}",
                 f"板法向（世界）：{np.round(b.up_world, 3)}"]
        cams_wo_ext = [n for n in self.clicks if n in self.calibs and not self.calibs[n].has_extrinsics]
        if cams_wo_ext:
            lines.append("注意：相机未设置外参，世界坐标系 = 相机坐标系。")
        lines.append("\n验证：站到板的某一角，画面中 COP 点应出现在同一角。")
        self.board_info.setPlainText("\n".join(lines))

    # =============================================================== pose
    def _pick_plugin(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择姿态插件", "plugins", "Python (*.py)")
        if p:
            self.plugin_path.setText(p)
            self.pose_backend.setCurrentText("插件 (.py)")

    def toggle_pose(self, on: bool):
        if not on:
            if self.pose_worker:
                self.pose_worker.stop()
            self.pose_worker = None
            self.b_pose.setText("启动姿态估计")
            return
        backend = self.pose_backend.currentText()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if backend.startswith("MediaPipe"):
                from poseboard.pose.mediapipe_backend import MediaPipePose

                est = MediaPipePose(backend.split("(")[1].rstrip(")"))
            else:
                from poseboard.pose.external import load_plugin

                est = load_plugin(self.plugin_path.text())
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            self.b_pose.setChecked(False)
            self._warn(f"无法启动姿态估计：{e}")
            return
        QApplication.restoreOverrideCursor()
        for n, f in self._latest_frames().items():
            self._ensure_calib(n, f)
        self.pose_worker = PoseWorker(est, self._latest_frames, self._cams_snapshot, self.recorder)
        self.pose_worker.start()
        self.b_pose.setText("停止姿态估计")

    # ============================================================= record
    def _pick_outdir(self):
        p = QFileDialog.getExistingDirectory(self, "保存目录", self.out_dir.text())
        if p:
            self.out_dir.setText(p)

    def toggle_record(self, on: bool):
        if on:
            if not self.streams and not self.force:
                self._warn("没有可录制的设备")
                self.b_rec.setChecked(False)
                return
            for n, f in self._latest_frames().items():
                self._ensure_calib(n, f)
            self.recorder.root = Path(self.out_dir.text())
            try:
                folder = self.recorder.start(
                    cams=list(self.streams.values()), calibrations=dict(self.calibs),
                    force=self.force, board=self.board, geometry=self.geometry,
                    pose_backend=self.pose_backend.currentText() if self.pose_worker else None,
                    subject=self.subject.text().strip(), notes=self.notes.text())
            except Exception as e:  # noqa: BLE001
                self.b_rec.setChecked(False)
                self._warn(f"无法开始录制：{e}")
                return
            self._rec_started = time.perf_counter()
            self.b_rec.setText("■ 停止录制")
            self.status.showMessage(f"录制到 {folder}")
        else:
            folder = self.recorder.stop()
            self.b_rec.setText("● 开始录制")
            if folder:
                s = folder / "summary.json"
                text = s.read_text(encoding="utf-8") if s.exists() else ""
                self.summary.setPlainText(f"已保存：{folder}\n\n{text}")

    # ============================================================ project
    def save_project(self):
        p, _ = QFileDialog.getSaveFileName(self, "保存配置", "poseboard_project.json", "JSON (*.json)")
        if not p:
            return
        data = {
            "cameras": [{"name": n, "source": str(s.source)} for n, s in self.streams.items()],
            "calibrations": [c.to_dict() for c in self.calibs.values()],
            "checkerboard": {"cols": self.cb_cols.value(), "rows": self.cb_rows.value(),
                             "square_mm": self.cb_square.value()},
            "board_geometry": self.geometry.to_dict(),
            "board_pose": None if self.board is None else self.board.to_dict(),
            "clicks": {n: [list(map(float, q)) for q in pts] for n, pts in self.clicks.items()},
        }
        Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_project(self, path: str | None = None):
        if not isinstance(path, str):
            path, _ = QFileDialog.getOpenFileName(self, "加载配置", "", "JSON (*.json)")
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for cam in data.get("cameras", []):
            if cam["name"] not in self.streams:
                self.add_camera(cam["source"], cam["name"])
        self.calibs = {d["name"]: CameraCalibration.from_dict(d) for d in data.get("calibrations", [])}
        cb = data.get("checkerboard")
        if cb:
            self.cb_cols.setValue(cb["cols"])
            self.cb_rows.setValue(cb["rows"])
            self.cb_square.setValue(cb["square_mm"])
        for k, v in data.get("board_geometry", {}).items():
            if k in self.geo_spins:
                self.geo_spins[k].setValue(v)
        self.board = BoardPose.from_dict(data["board_pose"]) if data.get("board_pose") else None
        self.clicks = {n: [np.array(q) for q in pts] for n, pts in data.get("clicks", {}).items()}
        self._refresh_calib_info()
        self._refresh_board_info()

    # =============================================================== tick
    def _tick(self):
        name = self.current_cam()
        stream = self.streams.get(name) if name else None
        frame = stream.latest() if stream else None
        force = self.force.latest() if self.force else None
        pose = self.pose_worker.latest if self.pose_worker else None
        fused = fuse(self.board, force, pose)

        if frame is not None:
            img = frame.image.copy()
            cam = self._ensure_calib(name, frame)
            if self.chk_detect.isChecked() and self.tabs.currentIndex() == 1:
                c = find_checkerboard(img, self.checker_spec(), fast=True)
                if c is not None:
                    cv2.drawChessboardCorners(img, self.checker_spec().pattern_size,
                                              c.reshape(-1, 1, 2).astype(np.float32), True)
            if self.board is not None and self.chk_overlay.isChecked():
                draw_board(img, cam, self.board, self.geometry)
                if force is not None:
                    draw_cop(img, cam, self.board, force.cop_board, force.total_kg)
            if pose is not None and self.chk_skeleton.isChecked():
                sk = getattr(self.pose_worker.estimator, "skeleton", [])
                draw_pose(img, cam, name, pose, sk, self.board,
                          fused.com_world if fused is not None else None)
            if self.click_mode or (self.tabs.currentIndex() == 2 and self.clicks.get(name)):
                draw_clicks(img, self.clicks.get(name, []), self._next_click_hint(name))
            self.video.set_image(img)
        elif not self.streams:
            self.video.set_image(None)

        # Wii 状态与俯视图
        if self.force is not None:
            if self.force.error:
                self.wii_label.setText(f"错误：{self.force.error}")
            elif force is not None:
                k = force.kg
                cop = force.cop_board
                bat = getattr(self.force, "battery", None)
                self.wii_label.setText(
                    f"TL {k[2]:6.2f}   TR {k[0]:6.2f}\nBL {k[3]:6.2f}   BR {k[1]:6.2f} kg\n"
                    f"总重 {force.total_kg:6.2f} kg\nCOP x={cop[0] * 1000:7.1f} y={cop[1] * 1000:7.1f} mm"
                    + (f"\n电量 {bat}" if bat is not None else ""))
            trail = self.force.recent(5.0)
            cop_trail = [s.cop_board for s in trail]
            now = time.perf_counter()
            if fused is not None and fused.com_board is not None and pose is not None:
                if not self._com_trail or self._com_trail[-1][0] != pose.t:
                    self._com_trail.append((pose.t, *fused.com_board[:2]))
            com_trail = [(x, y) for t, x, y in self._com_trail if now - t <= 5.0]
            self.cop_view.set_data(cop_trail, com_trail, force.total_kg if force else 0.0,
                                   "" if fused is None or fused.com_minus_cop is None else
                                   f"COM−COP = {np.round(fused.com_minus_cop * 1000, 1)} mm")

        if self.pose_worker:
            txt = f"{self.pose_worker.fps:.1f} fps"
            if self.pose_worker.error:
                txt += f"  错误：{self.pose_worker.error}"
            if pose is None:
                txt += "  （未检测到人）"
            self.pose_label.setText(txt)

        if self.recorder.recording:
            self.rec_label.setText(f"录制中 {time.perf_counter() - self._rec_started:6.1f}s   "
                                   f"Wii {self.recorder.counts['wii']}  姿态 {self.recorder.counts['pose']}")
        cams_fps = "  ".join(f"{n}:{s.measured_fps:.0f}fps" for n, s in self.streams.items())
        if cams_fps and not self.status.currentMessage():
            self.status.showMessage(cams_fps, 1000)

    def closeEvent(self, e):
        if self.recorder.recording:
            self.recorder.stop()
        if self.pose_worker:
            self.pose_worker.stop()
        self.disconnect_wii()
        for s in self.streams.values():
            s.stop()
        super().closeEvent(e)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    w = MainWindow()
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        w.load_project(sys.argv[1])
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
