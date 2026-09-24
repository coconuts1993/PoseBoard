"""Wii Balance Board HID protocol (pure functions, easy to test).

Based on the wiibrew.org Wiimote / Balance Board docs and the WiimoteLib implementation:
* Output report 0x12 sets the data reporting mode; 0x32 = buttons + 8 bytes of extension data.
* Output report 0x16 writes a register, 0x17 reads one; read data comes back in input report 0x21.
* Extension init: write 0x55 to 0xA400F0, then 0x00 to 0xA400FB.
* Calibration data is 32 bytes starting at 0xA40020: [4:12] 0 kg, [12:20] 17 kg, [20:28] 34 kg,
  each group holding 4 sensors in TR, BR, TL, BL order, big-endian 16-bit.
* The 8 extension bytes in a 0x32 report are also in TR, BR, TL, BL order, big-endian 16-bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VENDOR_ID = 0x057E
PRODUCT_IDS = (0x0306, 0x0330)
OUTPUT_REPORT_LEN = 22

REPORT_LEDS = 0x11
REPORT_MODE = 0x12
REPORT_STATUS_REQUEST = 0x15
REPORT_WRITE_MEMORY = 0x16
REPORT_READ_MEMORY = 0x17

INPUT_STATUS = 0x20
INPUT_READ_DATA = 0x21
INPUT_ACK = 0x22
INPUT_BUTTONS_EXT8 = 0x32

REGISTER_SPACE = 0x04
ADDR_EXT_INIT_1 = 0xA400F0
ADDR_EXT_INIT_2 = 0xA400FB
ADDR_CALIBRATION = 0xA40020
CALIBRATION_LEN = 32

KG_TO_N = 9.80665


def pad(report: list[int] | bytes) -> bytes:
    b = bytes(report)
    return b + bytes(max(0, OUTPUT_REPORT_LEN - len(b)))


def set_mode_report(mode: int = INPUT_BUTTONS_EXT8, continuous: bool = True) -> bytes:
    return pad([REPORT_MODE, 0x04 if continuous else 0x00, mode])


def led_report(on: bool = True) -> bytes:
    return pad([REPORT_LEDS, 0x10 if on else 0x00])


def status_request_report() -> bytes:
    return pad([REPORT_STATUS_REQUEST, 0x00])


def write_memory_report(address: int, data: bytes) -> bytes:
    if len(data) > 16:
        raise ValueError("At most 16 bytes can be written per report")
    payload = bytes(data) + bytes(16 - len(data))
    return pad([REPORT_WRITE_MEMORY, REGISTER_SPACE, (address >> 16) & 0xFF, (address >> 8) & 0xFF,
                address & 0xFF, len(data)] + list(payload))


def read_memory_report(address: int, size: int) -> bytes:
    return pad([REPORT_READ_MEMORY, REGISTER_SPACE, (address >> 16) & 0xFF, (address >> 8) & 0xFF,
                address & 0xFF, (size >> 8) & 0xFF, size & 0xFF])


def parse_read_data(report: bytes | list[int]) -> tuple[int, bytes, int]:
    """Parse a 0x21 report; return (low 16 bits of address, data, error code)."""
    r = bytes(report)
    se = r[3]
    size = (se >> 4) + 1
    err = se & 0x0F
    offset = (r[4] << 8) | r[5]
    return offset, r[6:6 + size], err


def be16(b: bytes, i: int) -> int:
    return (b[i] << 8) | b[i + 1]


@dataclass
class Calibration:
    """Raw reading of each sensor at 0 / 17 / 34 kg, in TR, BR, TL, BL order."""

    kg0: np.ndarray
    kg17: np.ndarray
    kg34: np.ndarray

    @classmethod
    def from_bytes(cls, data: bytes) -> "Calibration":
        """``data`` is the 32 bytes read from 0xA40020."""
        if len(data) < 28:
            raise ValueError("Calibration data too short")
        vals = [be16(data, 4 + 2 * i) for i in range(12)]
        return cls(np.array(vals[0:4], float), np.array(vals[4:8], float), np.array(vals[8:12], float))

    def to_kg(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw readings to kg using WiimoteLib's piecewise-linear interpolation."""
        raw = np.asarray(raw, float)
        low = 17.0 * (raw - self.kg0) / np.maximum(self.kg17 - self.kg0, 1e-9)
        high = 17.0 + 17.0 * (raw - self.kg17) / np.maximum(self.kg34 - self.kg17, 1e-9)
        return np.where(raw < self.kg17, low, high)

    def to_dict(self) -> dict:
        return {"kg0": self.kg0.tolist(), "kg17": self.kg17.tolist(), "kg34": self.kg34.tolist()}


def parse_sensor_raw(report: bytes | list[int]) -> np.ndarray | None:
    """Extract the 4 raw sensor values (TR, BR, TL, BL) from a 0x32 report."""
    r = bytes(report)
    if not r or r[0] != INPUT_BUTTONS_EXT8 or len(r) < 11:
        return None
    ext = r[3:11]
    return np.array([be16(ext, 2 * i) for i in range(4)], float)


def center_of_pressure(kg: np.ndarray, sensor_dx_m: float, sensor_dy_m: float,
                       min_total_kg: float = 1.0) -> tuple[float, float]:
    """Center of pressure (x, y) in board coordinates, in meters, from the four sensor
    forces (order TR, BR, TL, BL).

    +x points right (TR/BR side), +y points forward (TL/TR side). Returns NaN when the
    total weight is below the threshold.
    """
    tr, br, tl, bl = [float(v) for v in kg]
    total = tr + br + tl + bl
    if total < min_total_kg:
        return float("nan"), float("nan")
    x = (sensor_dx_m / 2.0) * ((tr + br) - (tl + bl)) / total
    y = (sensor_dy_m / 2.0) * ((tr + tl) - (br + bl)) / total
    return x, y
