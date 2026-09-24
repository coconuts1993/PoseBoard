"""Plug-and-play Wii Balance Board connection and the Wii-only CLI, with a fake hidapi device."""

import json
import sys
import time
from collections import deque

import numpy as np
import pytest

from poseboard.analysis import read_csv_columns
from poseboard.wii import protocol as P
from poseboard.wii.device import BalanceBoardHID, SimulatedBoard, WiiAutoConnect, sort_boards
from poseboard.wii.record import main as record_main

KG0 = [1000, 1100, 1200, 1300]
KG17 = [2700, 2800, 2900, 3000]
KG34 = [4400, 4500, 4600, 4700]
CAL_BYTES = bytes(4) + b"".join(v.to_bytes(2, "big") for v in KG0 + KG17 + KG34) + bytes(4)
CAL = P.Calibration.from_bytes(CAL_BYTES)
PATH = b"fake-board-path"
BOARD = {"path": PATH, "product_id": 0x0306, "product_string": "Nintendo RVL-WBC-01",
         "serial_number": "00191d000000"}


def sensor_report(raw) -> bytes:
    return bytes([P.INPUT_BUTTONS_EXT8, 0, 0]) + b"".join(int(v).to_bytes(2, "big") for v in raw) + bytes(11)


class FakeHid:
    """Stand-in for a hidapi ``hid.device`` connected to a Balance Board."""

    def __init__(self, raw, write_ok=True, silent=False, ext_id=P.BALANCE_BOARD_EXT_ID):
        self.raw = list(raw)
        self.write_ok = write_ok
        self.silent = silent  # never sends sensor data (link stalled)
        self.ext_id = ext_id  # extension type at 0xA400FA (None: no extension -> error 0x7)
        self.broken = False  # set to simulate a Bluetooth drop
        self.closed = False
        self.streaming = False
        self.writes: list[bytes] = []
        self.queue: deque[bytes] = deque()

    def write(self, data):
        if self.closed:
            raise ValueError("not open")
        r = bytes(data)
        self.writes.append(r)
        if not self.write_ok or self.broken:
            return -1
        if r[0] == P.REPORT_READ_MEMORY:
            addr = (r[2] << 16) | (r[3] << 8) | r[4]
            size = (r[5] << 8) | r[6]
            if addr == P.ADDR_EXT_TYPE and self.ext_id is None:
                a = addr & 0xFFFF
                self.queue.append(bytes([P.INPUT_READ_DATA, 0, 0, 0x07, a >> 8, a & 0xFF]) + bytes(16))
                return len(r)
            data = (bytes(self.ext_id) if addr == P.ADDR_EXT_TYPE
                    else CAL_BYTES[addr - P.ADDR_CALIBRATION:])[:size]
            for off in range(0, len(data), 16):
                chunk = data[off:off + 16]
                a = (addr + off) & 0xFFFF
                self.queue.append(bytes([P.INPUT_READ_DATA, 0, 0, (len(chunk) - 1) << 4, a >> 8, a & 0xFF])
                                  + chunk.ljust(16, b"\0"))
        elif r[0] == P.REPORT_MODE:
            self.streaming = True
        elif r[0] == P.REPORT_STATUS_REQUEST and not self.silent:
            self.queue.append(bytes([P.INPUT_STATUS, 0, 0, 0, 0, 0, 0xC0]) + bytes(15))
        return len(r)

    def read(self, n, timeout_ms):
        if self.closed:
            raise ValueError("not open")
        if self.broken:
            raise OSError("read error")
        if self.queue:
            return list(self.queue.popleft())
        time.sleep(0.002 if self.streaming and not self.silent else timeout_ms / 1000)
        if self.streaming and not self.silent:
            return list(sensor_report(self.raw))
        return []

    def close(self):
        self.closed = True


class FakeBus:
    """Monkeypatched device list + opener: boards can appear, drop out and come back."""

    def __init__(self, monkeypatch):
        self.present = False
        self.next_device = lambda: FakeHid([1850, 1950, 2050, 2150])
        self.opened: list[FakeHid] = []
        monkeypatch.setattr(BalanceBoardHID, "list_devices", staticmethod(self.list_devices))
        monkeypatch.setattr(BalanceBoardHID, "open_device", staticmethod(self.open_device))

    def list_devices(self):
        return [dict(BOARD)] if self.present else []

    def open_device(self, path):
        assert path == PATH
        dev = self.next_device()
        self.opened.append(dev)
        return dev


def wait_until(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_sort_boards_prefers_balance_board_and_drops_remotes():
    devs = [{"path": b"a", "product_string": "Nintendo RVL-CNT-01"},
            {"path": b"b", "product_string": ""},
            {"path": b"c", "product_string": "Nintendo RVL-WBC-01"}]
    assert [d["path"] for d in sort_boards(devs)] == [b"c", b"b"]


def test_auto_connect_reconnect_and_tare(monkeypatch):
    bus = FakeBus(monkeypatch)
    src = WiiAutoConnect(poll_interval_s=0.05)
    states: list[str] = []
    src.add_status_listener(states.append)
    src.start()
    try:
        assert wait_until(lambda: src.status == "searching")
        time.sleep(0.15)
        assert not bus.opened and src.latest() is None

        # Board switched on -> found and connected without any user action
        bus.present = True
        assert wait_until(lambda: src.connected and len(src.recent(1.0)) > 20)
        assert src.path == PATH and src.connections == 1
        np.testing.assert_allclose(src.calibration.kg17, KG17)
        s = src.latest()
        np.testing.assert_allclose(s.kg, CAL.to_kg(np.array([1850, 1950, 2050, 2150])))
        np.testing.assert_allclose(s.kg, 8.5)
        assert s.total_kg == pytest.approx(34.0)
        assert s.cop_board == pytest.approx(P.center_of_pressure(s.kg, src.sensor_dx_m, src.sensor_dy_m))
        dev1 = bus.opened[0]
        assert P.set_mode_report() in dev1.writes and P.led_report(True) in dev1.writes
        info = src.info()
        assert info["device_path"] == PATH.decode() and info["product_string"] == BOARD["product_string"]
        json.dumps(info)  # goes into session.json

        # Tare with the board "empty"
        time.sleep(0.1)
        tare = src.do_tare(0.1).copy()
        np.testing.assert_allclose(tare, 8.5, atol=1e-9)

        # Bluetooth drop: reads fail and the device disappears from the list
        bus.present = False
        dev1.broken = True
        assert wait_until(lambda: src.status == "searching")
        assert dev1.closed and src.path is None and src.disconnects == 1
        assert "disconnected" in src.last_error
        assert src.error is None  # not fatal: still running and waiting
        assert src.running

        # Board comes back with a different load -> reconnects, tare kept
        bus.next_device = lambda: FakeHid(KG17)  # 17 kg on every sensor
        n_before = len(src.buffer)
        bus.present = True
        assert wait_until(lambda: src.connected and len(src.buffer) > n_before + 20)
        assert src.connections == 2 and len(bus.opened) == 2
        np.testing.assert_allclose(src.tare, tare)
        np.testing.assert_allclose(src.latest().kg, 17.0 - 8.5)
    finally:
        t = time.perf_counter()
        src.stop()
        assert time.perf_counter() - t < 1.0
    assert src.status == "stopped"
    assert bus.opened[-1].closed and bus.opened[-1].writes[-1] == P.led_report(False)
    expected = ["searching", "connecting", "connected", "searching", "connecting", "connected", "stopped"]
    it = iter(states)
    assert all(e in it for e in expected), states  # ordered subsequence


def test_auto_connect_board_paired_but_off(monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    bus.next_device = lambda: FakeHid(KG0, write_ok=False)  # listed, but writes fail
    src = WiiAutoConnect(poll_interval_s=0.02)
    src.start()
    try:
        assert wait_until(lambda: len(bus.opened) >= 3)
        assert src.status in ("searching", "connecting")
        assert "not responding" in src.last_error and src.connections == 0
        assert all(d.closed for d in bus.opened[:-1])
    finally:
        src.stop()


def test_auto_connect_detects_stalled_link(monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    first = FakeHid(KG0, silent=True)
    devices = iter([first])
    bus.next_device = lambda: next(devices, None) or FakeHid(KG0)
    src = WiiAutoConnect(poll_interval_s=0.02, silence_timeout_s=0.3)
    src.start()
    try:
        assert wait_until(lambda: src.disconnects == 1)
        assert first.closed and "no data" in src.last_error
        assert P.status_request_report() in first.writes  # tried to wake the stream first
        assert wait_until(lambda: src.connected and src.latest() is not None)
    finally:
        src.stop()


def test_auto_connect_without_hidapi(monkeypatch):
    monkeypatch.setitem(sys.modules, "hid", None)  # "import hid" raises ImportError
    src = WiiAutoConnect(poll_interval_s=0.02)
    src.start()
    try:
        assert wait_until(lambda: src.status.startswith("error:"))
        assert "pip install hidapi" in src.status and src.running
    finally:
        src.stop()


def test_balance_board_hid_surfaces_disconnect(monkeypatch):
    bus = FakeBus(monkeypatch)
    src = BalanceBoardHID(PATH)
    src.start()
    try:
        assert wait_until(lambda: src.connected and src.latest() is not None)
        bus.opened[0].broken = True
        assert wait_until(lambda: not src.running)
        assert "disconnected" in src.error.lower() and src.status.startswith("error:")
        assert bus.opened[0].closed
    finally:
        src.stop()


def test_simulator_status():
    sim = SimulatedBoard()
    assert sim.status == "stopped"
    sim.start()
    assert wait_until(lambda: sim.connected)
    sim.stop()
    assert sim.status == "stopped"


# ------------------------------------------------------------------ CLI
def only_session(root):
    folders = [p for p in root.iterdir() if p.is_dir()]
    assert len(folders) == 1
    return folders[0]


def test_cli_simulate(tmp_path, capsys):
    assert record_main(["--simulate", "--seconds", "1", "--out", str(tmp_path), "--subject", "cli",
                        "--notes", "wii only"]) == 0
    folder = only_session(tmp_path)
    assert folder.name.endswith("_cli")
    assert str(folder) in capsys.readouterr().out
    wii = read_csv_columns(folder / "wii.csv")
    assert len(wii["t"]) > 50
    assert 60 < np.nanmean(wii["total_kg"]) < 80
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["force_source"]["type"] == "SimulatedBoard" and meta["notes"] == "wii only"
    assert not list(folder.glob("*.mp4")) and not list(folder.glob("*.mkv"))


def test_cli_auto_connect_with_tare(tmp_path, monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    assert record_main(["--seconds", "0.5", "--tare", "0.2", "--out", str(tmp_path)]) == 0
    folder = only_session(tmp_path)
    wii = read_csv_columns(folder / "wii.csv")
    assert len(wii["t"]) > 20
    np.testing.assert_allclose(wii["total_kg"], 0.0, atol=1e-6)  # tared before recording
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["force_source"]["type"] == "WiiAutoConnect"
    np.testing.assert_allclose(meta["force_source"]["tare_kg"], 8.5)


def test_cli_wait_timeout_and_list(tmp_path, monkeypatch, capsys):
    FakeBus(monkeypatch)  # no board present
    t = time.monotonic()
    assert record_main(["--wait", "0.3", "--out", str(tmp_path / "rec")]) == 2
    assert time.monotonic() - t < 3.0
    assert not (tmp_path / "rec").exists()
    assert record_main(["--list"]) == 0
    assert "No Wii Balance Board found" in capsys.readouterr().out


# ------------------------------------------------------------ device identity and tare
NUNCHUK_ID = bytes([0x00, 0x00, 0xA4, 0x20, 0x00, 0x00])


def test_wii_remote_with_accessory_is_not_used_as_a_board(monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    bus.next_device = lambda: FakeHid(KG0, ext_id=NUNCHUK_ID)
    src = WiiAutoConnect(poll_interval_s=0.02)
    src.start()
    try:
        assert wait_until(lambda: "not a Balance Board" in (src.last_error or ""))
        time.sleep(0.2)
        assert len(bus.opened) == 1  # skipped for good, not retried
        assert src.connections == 0 and src.latest() is None and src.status == "searching"
        assert bus.opened[0].closed
    finally:
        src.stop()
    one_shot = BalanceBoardHID(PATH)
    one_shot.start()
    try:
        assert wait_until(lambda: not one_shot.running)
        assert "not a Balance Board" in one_shot.error
    finally:
        one_shot.stop()


def test_extension_read_error_0x7_is_retried_once(monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    dev = FakeHid(KG17, ext_id=None)  # the first read of the extension type fails with 0x7
    real_write = dev.write

    def write(data):
        n = real_write(data)
        if bytes(data)[0] == P.REPORT_READ_MEMORY and dev.ext_id is None:
            dev.ext_id = P.BALANCE_BOARD_EXT_ID  # answers the retry
        return n

    dev.write = write
    bus.next_device = lambda: dev
    src = BalanceBoardHID(PATH)
    src.start()
    try:
        assert wait_until(lambda: src.connected and src.latest() is not None)
    finally:
        src.stop()


def test_tare_is_kept_per_board(monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    src = WiiAutoConnect(poll_interval_s=0.02)
    src.start()
    try:
        assert wait_until(lambda: src.connected and len(src.recent(0.2)) > 5)
        tare = src.do_tare(0.1).copy()
        np.testing.assert_allclose(tare, 8.5)
    finally:
        src.stop()
    # Replaced by a new source object (Connect / Auto-connect toggled): same board, same tare
    again = WiiAutoConnect(poll_interval_s=0.02)
    again.adopt_tares(src)
    again.start()
    try:
        assert wait_until(lambda: again.connected and again.latest() is not None)
        np.testing.assert_allclose(again.tare, tare)
        np.testing.assert_allclose(again.latest().kg, 0.0, atol=1e-9)
    finally:
        again.stop()
    # A different board does not inherit it
    other = {**BOARD, "path": b"other-board"}
    monkeypatch.setattr(BalanceBoardHID, "list_devices", staticmethod(lambda: [other]))
    monkeypatch.setattr(BalanceBoardHID, "open_device", staticmethod(lambda p: FakeHid(KG17)))
    third = WiiAutoConnect(poll_interval_s=0.02)
    third.adopt_tares(again)
    third.start()
    try:
        assert wait_until(lambda: third.connected and third.latest() is not None)
        np.testing.assert_allclose(third.tare, 0.0)
        np.testing.assert_allclose(third.latest().kg, 17.0)
        assert third.info()["min_total_kg"] == 5.0
    finally:
        third.stop()
