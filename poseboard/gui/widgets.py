"""GUI widgets: clickable video view and top-down board view with COP/COM trails."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


def bgr_to_qimage(img: np.ndarray) -> QImage:
    h, w = img.shape[:2]
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class VideoView(QWidget):
    """Shows a BGR image (aspect ratio preserved) and maps mouse clicks to source-image pixel coordinates."""

    clicked = Signal(float, float, int)  # x, y (image pixels), mouse button (1 = left / 2 = right)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img: QImage | None = None
        self._size = (0, 0)
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.crosshair = False
        self._mouse: QPointF | None = None
        self.placeholder = "No camera selected"

    def set_image(self, img: np.ndarray | None) -> None:
        if img is None:
            self._img = None
            self._size = (0, 0)  # no image: clicks are ignored (not mapped to an old image)
        else:
            self._img = bgr_to_qimage(img)
            self._size = (img.shape[1], img.shape[0])
        self.update()

    def _target(self) -> QRectF:
        w, h = self._size
        if not w or not h:
            return QRectF()
        s = min(self.width() / w, self.height() / h)
        tw, th = w * s, h * s
        return QRectF((self.width() - tw) / 2, (self.height() - th) / 2, tw, th)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        if self._img is None:
            p.setPen(QColor(160, 160, 160))
            p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            return
        r = self._target()
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(r, self._img)
        if self.crosshair and self._mouse is not None and r.contains(self._mouse):
            p.setPen(QPen(QColor(0, 255, 255, 180), 1, Qt.DashLine))
            p.drawLine(QPointF(r.left(), self._mouse.y()), QPointF(r.right(), self._mouse.y()))
            p.drawLine(QPointF(self._mouse.x(), r.top()), QPointF(self._mouse.x(), r.bottom()))

    def mouseMoveEvent(self, e):
        self._mouse = e.position()
        if self.crosshair:
            self.update()

    def mousePressEvent(self, e):
        r = self._target()
        pos = e.position()
        if r.isEmpty() or not r.contains(pos):
            return
        x = (pos.x() - r.left()) / r.width() * self._size[0]
        y = (pos.y() - r.top()) / r.height() * self._size[1]
        self.clicked.emit(x, y, 2 if e.button() == Qt.RightButton else 1)


class CopView(QWidget):
    """Top-down board view: outline, sensors, COP trail (blue) and projected COM trail (orange)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self.board_mm = (511.0, 316.0)
        self.sensor_mm = (433.0, 238.0)
        self.cop_trail = np.zeros((0, 2))
        self.com_trail = np.zeros((0, 2))
        self.total_kg = 0.0
        self.text = ""

    def set_data(self, cop_trail, com_trail, total_kg, text=""):
        self.cop_trail = np.asarray(cop_trail, float).reshape(-1, 2)
        self.com_trail = np.asarray(com_trail, float).reshape(-1, 2)
        self.total_kg = total_kg
        self.text = text
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(250, 250, 250))
        bw, bh = self.board_mm[0] / 1000, self.board_mm[1] / 1000
        margin = 20
        s = min((self.width() - 2 * margin) / bw, (self.height() - 2 * margin - 16) / bh)
        cx, cy = self.width() / 2, (self.height() + 16) / 2

        def pt(x, y):  # board coords (meters, y forward) -> screen (front facing up)
            return QPointF(cx + x * s, cy - y * s)

        p.setPen(QPen(QColor(90, 90, 90), 2))
        p.setBrush(QColor(235, 235, 235))
        p.drawRoundedRect(QRectF(pt(-bw / 2, bh / 2), pt(bw / 2, -bh / 2)), 12, 12)
        p.setPen(QPen(QColor(200, 200, 200), 1, Qt.DashLine))
        p.drawLine(pt(-bw / 2, 0), pt(bw / 2, 0))
        p.drawLine(pt(0, -bh / 2), pt(0, bh / 2))
        sx, sy = self.sensor_mm[0] / 2000, self.sensor_mm[1] / 2000
        p.setPen(QColor(120, 120, 120))
        for name, (x, y) in {"TL": (-sx, sy), "TR": (sx, sy), "BL": (-sx, -sy), "BR": (sx, -sy)}.items():
            p.setBrush(QColor(150, 150, 150))
            p.drawEllipse(pt(x, y), 5, 5)
            p.drawText(pt(x, y) + QPointF(-8, -9 if y > 0 else 20), name)
        p.drawText(QPointF(8, 14), "Front ↑ = TL/TR edge (opposite the power button)")
        p.drawText(pt(0, -bh / 2) + QPointF(-40, 14), "power button")

        for trail, color in ((self.com_trail, QColor(255, 140, 0)), (self.cop_trail, QColor(30, 100, 220))):
            ok = trail[np.all(np.isfinite(trail), axis=1)]
            if len(ok) > 1:
                c = QColor(color)
                c.setAlpha(140)
                p.setPen(QPen(c, 1.5))
                for a, b in zip(ok[:-1], ok[1:]):
                    p.drawLine(pt(*a), pt(*b))
            if len(ok):
                p.setPen(Qt.NoPen)
                p.setBrush(color)
                p.drawEllipse(pt(*ok[-1]), 6, 6)
        p.setPen(QColor(30, 30, 30))
        p.drawText(QPointF(8, self.height() - 8),
                   f"{self.total_kg:6.1f} kg   {self.text}   ● COP (blue)  ● COM projection (orange)")
