"""Wii Balance Board device reader (hidapi) and simulator.

Both implement ``ForceSource``: a background thread reads continuously and stamps each sample
with ``time.perf_counter()``, the same clock used by the camera/pose threads, so everything
can be aligned on a single timeline.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from poseboard.wii import protocol as P

log = logging.getLogger(__name__)


@dataclass
class ForceSample:
    t: float  # time.perf_counter()
    kg: np.ndarray  # (4,) TR, BR, TL, BL (tared)
    total_kg: float
    cop_board: tuple[float, float]  # board frame (x, y), meters; NaN when nobody is standing
    raw: np.ndarray | None = None


class ForceSource:
    """Base class for a force data source with a background thread."""

    name = "force"

    def __init__(self, sensor_dx_m: float = 0.433, sensor_dy_m: float = 0.238,
                 min_total_kg: float = 1.0, buffer_seconds: float = 30.0, rate_hint: float = 100.0):
        self.sensor_dx_m = sensor_dx_m
        self.sensor_dy_m = sensor_dy_m
        self.min_total_kg = min_total_kg
        self.tare = np.zeros(4)
        self.buffer: deque[ForceSample] = deque(maxlen=int(buffer_seconds * rate_hint))
        self._listeners: list[Callable[[ForceSample], None]] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.error: str | None = None

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run_safe, name=self.name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_safe(self) -> None:
        try:
            self._run()
        except Exception as e:  # noqa: BLE001
            log.exception("force source stopped")
            self.error = str(e)

    def _run(self) -> None:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    # ------------------------------------------------------------- listeners
    def add_listener(self, fn: Callable[[ForceSample], None]) -> None:
        with self._lock:
            self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[ForceSample], None]) -> None:
        with self._lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def _emit_kg(self, t: float, kg_raw: np.ndarray, raw: np.ndarray | None = None) -> ForceSample:
        kg = np.asarray(kg_raw, float) - self.tare
        total = float(kg.sum())
        cop = P.center_of_pressure(kg, self.sensor_dx_m, self.sensor_dy_m, self.min_total_kg)
        s = ForceSample(t, kg, total, cop, raw)
        with self._lock:
            self.buffer.append(s)
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(s)
            except Exception:  # noqa: BLE001
                log.exception("force listener failed")
        return s

    def latest(self) -> ForceSample | None:
        with self._lock:
            return self.buffer[-1] if self.buffer else None

    def recent(self, seconds: float) -> list[ForceSample]:
        now = time.perf_counter()
        with self._lock:
            return [s for s in self.buffer if now - s.t <= seconds]

    def do_tare(self, seconds: float = 1.0) -> np.ndarray:
        """Use the mean reading over the last ``seconds`` seconds (board empty) as the zero point."""
        samples = self.recent(seconds)
        if not samples:
            raise RuntimeError("No data available for taring")
        self.tare = self.tare + np.mean([s.kg for s in samples], axis=0)
        return self.tare

    def info(self) -> dict:
        return {"type": type(self).__name__, "tare_kg": self.tare.tolist(),
                "sensor_dx_m": self.sensor_dx_m, "sensor_dy_m": self.sensor_dy_m}


class BalanceBoardHID(ForceSource):
    """Read a Bluetooth-paired Wii Balance Board directly via hidapi.

    Windows: first pair "Nintendo RVL-WBC-01" under "Bluetooth devices" (press the red SYNC
    button in the battery compartment, leave the PIN empty). Once the device shows up in the
    HID device list it can be connected.
    """

    name = "wii-balance-board"

    def __init__(self, path: bytes | None = None, **kw):
        super().__init__(**kw)
        self.path = path
        self.calibration: P.Calibration | None = None
        self.battery: int | None = None
        self._dev = None

    @staticmethod
    def list_devices() -> list[dict]:
        import hid

        return [d for d in hid.enumerate(P.VENDOR_ID, 0) if d.get("product_id") in P.PRODUCT_IDS]

    # ------------------------------------------------------------------ I/O
    def _open(self):
        import hid

        dev = hid.device()
        if self.path is not None:
            dev.open_path(self.path)
        else:
            devs = self.list_devices()
            if not devs:
                raise RuntimeError("Wii Balance Board not found (pair it via Bluetooth first)")
            dev.open_path(devs[0]["path"])
        dev.set_nonblocking(False)
        return dev

    def _write(self, report: bytes) -> None:
        n = self._dev.write(list(report))
        if n < 0:
            raise RuntimeError("Failed to write HID report")

    def _read(self, timeout_ms: int = 100) -> bytes:
        data = self._dev.read(P.OUTPUT_REPORT_LEN, timeout_ms)
        return bytes(data) if data else b""

    def _read_memory(self, address: int, size: int, timeout: float = 2.0) -> bytes:
        self._write(P.read_memory_report(address, size))
        out = bytearray()
        deadline = time.perf_counter() + timeout
        while len(out) < size and time.perf_counter() < deadline:
            r = self._read(200)
            if r and r[0] == P.INPUT_READ_DATA:
                _, data, err = P.parse_read_data(r)
                if err:
                    raise RuntimeError(f"Register read error 0x{err:x}")
                out += data
        if len(out) < size:
            raise RuntimeError("Timed out reading calibration data")
        return bytes(out[:size])

    def _initialize(self) -> None:
        self._write(P.write_memory_report(P.ADDR_EXT_INIT_1, b"\x55"))
        time.sleep(0.05)
        self._write(P.write_memory_report(P.ADDR_EXT_INIT_2, b"\x00"))
        time.sleep(0.05)
        self.calibration = P.Calibration.from_bytes(
            self._read_memory(P.ADDR_CALIBRATION, P.CALIBRATION_LEN))
        self._write(P.led_report(True))
        self._write(P.set_mode_report())

    def _run(self) -> None:
        self._dev = self._open()
        try:
            self._initialize()
            last_status = time.perf_counter()
            while not self._stop.is_set():
                r = self._read(100)
                if not r:
                    continue
                t = time.perf_counter()
                if r[0] == P.INPUT_BUTTONS_EXT8:
                    raw = P.parse_sensor_raw(r)
                    if raw is not None:
                        self._emit_kg(t, self.calibration.to_kg(raw), raw)
                elif r[0] == P.INPUT_STATUS:
                    self.battery = r[6]
                    # After a status report the device falls back to the default reporting mode; set it again
                    self._write(P.set_mode_report())
                if t - last_status > 10.0:
                    self._write(P.status_request_report())
                    last_status = t
        finally:
            try:
                self._write(P.led_report(False))
            except Exception:  # noqa: BLE001
                pass
            self._dev.close()
            self._dev = None

    def info(self) -> dict:
        d = super().info()
        d["calibration"] = None if self.calibration is None else self.calibration.to_dict()
        d["battery"] = self.battery
        return d


class SimulatedBoard(ForceSource):
    """Simulate a 70 kg person slowly swaying on the board, for testing the UI and workflow without a device."""

    name = "wii-simulator"

    def __init__(self, mass_kg: float = 70.0, rate_hz: float = 100.0, **kw):
        super().__init__(rate_hint=rate_hz, **kw)
        self.mass_kg = mass_kg
        self.rate_hz = rate_hz

    def kg_at(self, t: float) -> np.ndarray:
        # COP follows a Lissajous path, amplitude ±6 cm (left/right) / ±4 cm (front/back)
        x = 0.06 * math.sin(2 * math.pi * 0.23 * t)
        y = 0.04 * math.sin(2 * math.pi * 0.31 * t + 0.7)
        hx, hy = self.sensor_dx_m / 2, self.sensor_dy_m / 2
        m = self.mass_kg * (1 + 0.01 * math.sin(2 * math.pi * 1.1 * t))
        # Distribute bilinearly to the four corners so that COP = (x, y)
        u, v = (x / hx + 1) / 2, (y / hy + 1) / 2  # 0..1, right/front
        tr, br, tl, bl = u * v, u * (1 - v), (1 - u) * v, (1 - u) * (1 - v)
        return m * np.array([tr, br, tl, bl])

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        t0 = time.perf_counter()
        next_t = t0
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
                continue
            self._emit_kg(now, self.kg_at(now - t0) + np.random.normal(0, 0.05, 4))
            next_t += period
