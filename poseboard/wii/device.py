"""Wii Balance Board device readers (hidapi), plug-and-play auto-connect, and simulator.

All implement ``ForceSource``: a background thread reads continuously and stamps each sample
with ``time.perf_counter()``, the same clock used by the camera/pose threads, so everything
can be aligned on a single timeline.

* ``BalanceBoardHID``  connects once to a given (or the first) board; a lost link ends it with
  an error.
* ``WiiAutoConnect``   waits for a paired board, connects as soon as it appears and reconnects
  after a Bluetooth drop, like a stand-alone Wii recorder app.
* ``SimulatedBoard``   synthetic data for testing without hardware.
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

HIDAPI_MISSING = "hidapi is not installed (pip install hidapi)"
HID_SHADOWED = ("the installed 'hid' module is not hidapi "
                "(pip uninstall hid; pip install --force-reinstall hidapi)")
DEFAULT_COP_MIN_KG = 5.0  # below this total load the COP is undefined (NaN), GUI and CLI alike


class BoardDisconnected(RuntimeError):
    """The board stopped responding: Bluetooth link lost, board switched off, or HID I/O error."""


class HidapiUnavailable(RuntimeError):
    """hidapi is missing or shadowed by another 'hid' package: retrying cannot help."""


class NotABalanceBoard(RuntimeError):
    """The HID device answered but is not a Balance Board (e.g. a Wii Remote with a Nunchuk)."""


def _import_hid():
    """Import hidapi's ``hid`` module, with an actionable message when it is unavailable."""
    try:
        import hid
    except ImportError as e:
        raise HidapiUnavailable(HIDAPI_MISSING) from e
    if not hasattr(hid, "device") or not hasattr(hid, "enumerate"):
        raise HidapiUnavailable(HID_SHADOWED)
    return hid


def _path_str(path) -> str | None:
    if path is None:
        return None
    return path.decode("utf-8", "replace") if isinstance(path, (bytes, bytearray)) else str(path)


def sort_boards(devs: list[dict]) -> list[dict]:
    """Drop entries that are clearly Wii Remotes (same vendor/product ID as the board) and
    list Balance Boards ("RVL-WBC") first."""
    out = [d for d in devs if "RVL-CNT" not in (d.get("product_string") or "")]
    return sorted(out, key=lambda d: "WBC" not in (d.get("product_string") or ""))


@dataclass
class ForceSample:
    t: float  # time.perf_counter()
    kg: np.ndarray  # (4,) TR, BR, TL, BL (tared)
    total_kg: float
    cop_board: tuple[float, float]  # board frame (x, y), meters; NaN when nobody is standing
    raw: np.ndarray | None = None


class ForceSource:
    """Base class for a force data source with a background thread.

    ``status`` is one of 'stopped' | 'searching' | 'connecting' | 'connected' |
    'error: <message>' and is safe to read from any thread.
    """

    name = "force"
    _start_state = "connecting"

    def __init__(self, sensor_dx_m: float = 0.433, sensor_dy_m: float = 0.238,
                 min_total_kg: float = DEFAULT_COP_MIN_KG, buffer_seconds: float = 30.0,
                 rate_hint: float = 100.0):
        self.sensor_dx_m = sensor_dx_m
        self.sensor_dy_m = sensor_dy_m
        self.min_total_kg = min_total_kg
        self.tare = np.zeros(4)
        self._tares: dict[str, np.ndarray] = {}  # tare per device (device_key)
        self.buffer: deque[ForceSample] = deque(maxlen=int(buffer_seconds * rate_hint))
        self._listeners: list[Callable[[ForceSample], None]] = []
        self._status_listeners: list[Callable[[str], None]] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state = "stopped"
        self.error: str | None = None

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.error = None
        self._set_state(self._start_state)
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

    @property
    def status(self) -> str:
        """'stopped' | 'searching' | 'connecting' | 'connected' | 'error: <message>'."""
        err = self.error
        if err:
            return f"error: {err}"
        if not self.running:
            return "stopped"
        return self._state

    @property
    def connected(self) -> bool:
        return self.status == "connected"

    def _set_state(self, state: str) -> None:
        with self._lock:
            if state == self._state:
                return
            self._state = state
            listeners = list(self._status_listeners)
        for fn in listeners:
            try:
                fn(state)
            except Exception:  # noqa: BLE001
                log.exception("status listener failed")

    def _run_safe(self) -> None:
        try:
            self._run()
        except Exception as e:  # noqa: BLE001
            log.exception("force source stopped")
            self.error = str(e)
        finally:
            self._set_state("stopped")

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

    def add_status_listener(self, fn: Callable[[str], None]) -> None:
        """Call ``fn(state)`` whenever the connection state changes (usually from the reader
        thread, so GUI code must hand it over to its own thread)."""
        with self._lock:
            self._status_listeners.append(fn)

    def remove_status_listener(self, fn: Callable[[str], None]) -> None:
        with self._lock:
            if fn in self._status_listeners:
                self._status_listeners.remove(fn)

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

    @property
    def device_key(self) -> str | None:
        """Identifies the physical device the tare belongs to (None: not known yet)."""
        return type(self).__name__

    def do_tare(self, seconds: float = 1.0) -> np.ndarray:
        """Use the mean reading over the last ``seconds`` seconds (board empty) as the zero point.
        The tare is remembered for this device (``device_key``)."""
        samples = self.recent(seconds)
        if not samples:
            raise RuntimeError("No data available for taring")
        self.tare = self.tare + np.mean([s.kg for s in samples], axis=0)
        key = self.device_key
        if key is not None:
            self._tares[key] = self.tare.copy()
        return self.tare

    def adopt_tares(self, other: "ForceSource | None") -> None:
        """Take over the tare offsets remembered by ``other`` (a source this one replaces, e.g.
        after Connect or toggling Auto-connect): the same board keeps its tare, a different
        board does not inherit it."""
        if other is None:
            return
        self._tares.update({k: np.array(v, float) for k, v in other._tares.items()})
        self._restore_tare()

    def _restore_tare(self) -> None:
        """Set the tare of the device now in use (zero if it was never tared)."""
        key = self.device_key
        self.tare = self._tares[key].copy() if key in self._tares else np.zeros(4)

    def info(self) -> dict:
        return {"type": type(self).__name__, "tare_kg": self.tare.tolist(),
                "sensor_dx_m": self.sensor_dx_m, "sensor_dy_m": self.sensor_dy_m,
                "min_total_kg": self.min_total_kg}


class BalanceBoardHID(ForceSource):
    """Read a Bluetooth-paired Wii Balance Board directly via hidapi.

    Windows: first pair "Nintendo RVL-WBC-01" under "Bluetooth devices" (press the red SYNC
    button in the battery compartment, leave the PIN empty). Once the device shows up in the
    HID device list it can be connected.

    Connects once; if the link is lost (HID read/write error, or no report at all for
    ``silence_timeout_s``) the reader stops and ``error`` / ``status`` say so. Use
    ``WiiAutoConnect`` to reconnect automatically.
    """

    name = "wii-balance-board"

    def __init__(self, path: bytes | None = None, silence_timeout_s: float = 5.0, **kw):
        super().__init__(**kw)
        self.path = path
        self.silence_timeout_s = silence_timeout_s
        self.device_info: dict = {}
        self.calibration: P.Calibration | None = None
        self.battery: int | None = None
        self._dev = None

    @staticmethod
    def list_devices() -> list[dict]:
        """hidapi device dicts ("path", "product_string", "serial_number", ...) of paired boards."""
        hid = _import_hid()
        return sort_boards([d for d in hid.enumerate(P.VENDOR_ID, 0)
                            if d.get("product_id") in P.PRODUCT_IDS])

    @staticmethod
    def open_device(path: bytes):
        """Open one HID device path and return the raw (blocking) hidapi handle."""
        hid = _import_hid()
        dev = hid.device()
        dev.open_path(path)
        dev.set_nonblocking(False)
        return dev

    # ------------------------------------------------------------------ I/O
    def _write(self, report: bytes) -> None:
        try:
            n = self._dev.write(list(report))
        except (OSError, ValueError) as e:
            raise BoardDisconnected(f"HID write failed: {e}") from e
        if n is None or n < 0:
            raise BoardDisconnected("HID write failed")

    def _read(self, timeout_ms: int = 100) -> bytes:
        try:
            data = self._dev.read(P.OUTPUT_REPORT_LEN, timeout_ms)
        except (OSError, ValueError) as e:
            raise BoardDisconnected(f"HID read failed: {e}") from e
        return bytes(data) if data else b""

    def _read_memory(self, address: int, size: int, timeout: float = 2.0,
                     retries: int = 1) -> bytes:
        """Read ``size`` bytes of register memory. Error 0x7 (nothing there yet, e.g. right after
        the board was switched on) is retried ``retries`` times after a short pause."""
        for attempt in range(retries + 1):
            self._write(P.read_memory_report(address, size))
            out = bytearray()
            deadline = time.perf_counter() + timeout
            err = 0
            while len(out) < size:
                if self._stop.is_set():
                    raise BoardDisconnected("stopped while initializing")
                if time.perf_counter() > deadline:
                    raise BoardDisconnected("timed out reading register data (board not responding)")
                r = self._read(100)
                if r and r[0] == P.INPUT_READ_DATA:
                    _, data, err = P.parse_read_data(r)
                    if err:
                        break
                    out += data
            if not err:
                return bytes(out[:size])
            if err == P.REGISTER_ERROR_NO_EXTENSION and attempt < retries:
                self._stop.wait(0.3)
                continue
            raise NotABalanceBoard(f"register read error 0x{err:x}: no Balance Board extension "
                                   "(is this a Wii Remote?)")
        raise AssertionError("unreachable")  # pragma: no cover

    def _initialize(self) -> None:
        self._write(P.write_memory_report(P.ADDR_EXT_INIT_1, b"\x55"))
        time.sleep(0.05)
        self._write(P.write_memory_report(P.ADDR_EXT_INIT_2, b"\x00"))
        time.sleep(0.05)
        ext = self._read_memory(P.ADDR_EXT_TYPE, P.EXT_TYPE_LEN)
        if not P.is_balance_board(ext):
            raise NotABalanceBoard(f"not a Balance Board (extension type {ext.hex(' ')}; a Wii "
                                   "Remote with a Nunchuk or other accessory?)")
        self.calibration = P.Calibration.from_bytes(
            self._read_memory(P.ADDR_CALIBRATION, P.CALIBRATION_LEN))
        self._write(P.led_report(True))
        self._write(P.set_mode_report())

    # ------------------------------------------------- one connection (shared)
    def _connect(self, path: bytes, device_info: dict | None = None) -> None:
        """Open ``path`` and initialize the board (calibration, LED, continuous reporting)."""
        self._dev = self.open_device(path)
        try:
            self._initialize()
        except BaseException:
            self._disconnect()
            raise
        self.path = path
        self.device_info = dict(device_info or {})
        self._restore_tare()  # the tare of this board (a different board starts at zero)

    @property
    def device_key(self) -> str | None:
        return _path_str(self.path)

    def _disconnect(self) -> None:
        dev, self._dev = self._dev, None
        if dev is None:
            return
        try:
            dev.write(list(P.led_report(False)))
        except Exception:  # noqa: BLE001
            pass
        try:
            dev.close()
        except Exception:  # noqa: BLE001
            pass

    def _stream(self) -> None:
        """Emit samples until stop() is called; raise BoardDisconnected when the link is lost."""
        last_report = last_data = last_request = time.perf_counter()
        while not self._stop.is_set():
            r = self._read(100)
            t = time.perf_counter()
            if r:
                last_report = t
                if r[0] == P.INPUT_BUTTONS_EXT8:
                    raw = P.parse_sensor_raw(r)
                    if raw is not None:
                        last_data = t
                        self._emit_kg(t, self.calibration.to_kg(raw), raw)
                elif r[0] == P.INPUT_STATUS and len(r) > 6:
                    self.battery = r[6]
                    # After a status report the device falls back to the default reporting mode; set it again
                    self._write(P.set_mode_report())
            if t - last_report > self.silence_timeout_s:
                raise BoardDisconnected(f"no data from the board for {self.silence_timeout_s:g} s")
            # Status request every 10 s (battery); more often while the data stream has stalled
            stall = min(1.0, self.silence_timeout_s / 3)
            if t - last_request > (stall if t - last_data > stall else 10.0):
                self._write(P.status_request_report())
                last_request = t

    def _run(self) -> None:
        path, dev_info = self.path, None
        if path is None:
            devs = self.list_devices()
            if not devs:
                raise RuntimeError("Wii Balance Board not found (pair it via Bluetooth first)")
            dev_info = devs[0]
            path = dev_info["path"]
        try:
            self._connect(path, dev_info)
        except BoardDisconnected as e:
            raise RuntimeError(f"Board not responding (is it switched on?): {e}") from e
        try:
            self._set_state("connected")
            self._stream()
        except BoardDisconnected as e:
            raise RuntimeError(f"Board disconnected: {e}") from e
        finally:
            self._disconnect()

    def info(self) -> dict:
        d = super().info()
        d["calibration"] = None if self.calibration is None else self.calibration.to_dict()
        d["battery"] = self.battery
        d["device_path"] = _path_str(self.path if self.path is not None
                                     else self.device_info.get("path"))
        d["product_string"] = self.device_info.get("product_string")
        d["serial_number"] = self.device_info.get("serial_number")
        return d


class WiiAutoConnect(BalanceBoardHID):
    """Plug-and-play Balance Board source, like a stand-alone Wii recorder app.

    While no board is connected it polls the HID device list every ``poll_interval_s``
    seconds. As soon as a paired board appears (switched on) it is opened, initialized, and its
    samples flow through the normal ForceSource listener/buffer API, so recorders and the GUI
    need no special handling. When the link is lost (HID error, or no report for
    ``silence_timeout_s``) the handle is closed and polling resumes. The tare offset is kept
    per board across reconnects (a different board starts untared); calibration is re-read from
    the board on every connect. Devices that turn out not to be a Balance Board (e.g. a Wii
    Remote) are skipped. ``fatal_error`` is set when hidapi itself is unusable.

    ``path`` restricts it to one HID device path; by default the first board found is used.
    ``status``: 'searching' | 'connecting' | 'connected' | 'error: <message>' (retrying) |
    'stopped'. ``path`` is the connected device path (None while not connected).
    """

    name = "wii-auto-connect"
    _start_state = "searching"

    def __init__(self, path: bytes | None = None, poll_interval_s: float = 2.0,
                 retry_failed_s: float = 10.0, **kw):
        super().__init__(path=None, **kw)
        self.target_path = path
        self.poll_interval_s = poll_interval_s
        self.retry_failed_s = retry_failed_s
        self.connections = 0  # successful connects
        self.disconnects = 0  # link losses after being connected
        self.last_error: str | None = None
        self.fatal_error: str | None = None  # hidapi unusable: waiting cannot help
        self._retry_at: dict = {}
        self._not_boards: set = set()  # device paths that are not Balance Boards

    def _find_board(self) -> dict | None:
        """Pick a device to try now, or None (and update the status accordingly)."""
        try:
            devs = self.list_devices()
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)
            if isinstance(e, HidapiUnavailable):
                self.fatal_error = str(e)
            self._set_state(f"error: {e}")
            return None
        if self.target_path is not None:
            devs = [d for d in devs if d.get("path") == self.target_path]
        devs = [d for d in devs if d.get("path") not in self._not_boards]
        if not devs:
            self._set_state("searching")
            return None
        now = time.perf_counter()
        ready = [d for d in devs if self._retry_at.get(d.get("path"), 0.0) <= now]
        return ready[0] if ready else None

    def _run(self) -> None:
        self._retry_at = {}
        while not self._stop.is_set():
            dev = self._find_board()
            if dev is None:
                self._stop.wait(self.poll_interval_s)
                continue
            path = dev["path"]
            self._set_state("connecting")
            try:
                self._connect(path, dev)
            except BoardDisconnected as e:
                if self._stop.is_set():
                    break
                # Typical for a board that is paired but switched off: its HID entry can remain
                # listed (Windows) while writes fail. Keep waiting quietly.
                self.last_error = f"board not responding: {e}"
                log.debug("%s: %s", _path_str(path), self.last_error)
                self._retry_at[path] = time.perf_counter() + self.poll_interval_s
                self._set_state("searching")
                self._stop.wait(self.poll_interval_s)
                continue
            except NotABalanceBoard as e:
                self.last_error = f"{_path_str(path)} skipped: {e}"
                log.warning(self.last_error)
                self._not_boards.add(path)
                self._set_state("searching")
                continue
            except Exception as e:  # noqa: BLE001
                self.last_error = f"cannot connect to {_path_str(path)}: {e}"
                log.warning(self.last_error)
                self._retry_at[path] = time.perf_counter() + self.retry_failed_s
                self._set_state(f"error: {e}")
                self._stop.wait(self.poll_interval_s)
                continue
            self.connections += 1
            log.info("Balance Board connected: %s", _path_str(path))
            self._set_state("connected")
            try:
                self._stream()
            except Exception as e:  # noqa: BLE001
                self.disconnects += 1
                self.last_error = f"board disconnected: {e}"
                log.warning("%s; waiting for it to reconnect", self.last_error)
            finally:
                self._disconnect()
                self.path = None
                self.battery = None
            if not self._stop.is_set():
                self._set_state("searching")

    def info(self) -> dict:
        d = super().info()
        d["auto_connect"] = True
        d["status"] = self.status
        d["connections"] = self.connections
        d["disconnects"] = self.disconnects
        return d


class SimulatedBoard(ForceSource):
    """Simulate a 70 kg person slowly swaying on the board, for testing the UI and workflow without a device."""

    name = "wii-simulator"

    def __init__(self, mass_kg: float = 70.0, rate_hz: float = 100.0, **kw):
        super().__init__(rate_hint=rate_hz, **kw)
        self.mass_kg = mass_kg
        self.rate_hz = rate_hz

    @property
    def device_key(self) -> str:
        return "simulator"

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
        self._set_state("connected")
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
