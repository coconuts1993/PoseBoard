"""Wii Balance Board HID 协议（纯函数，便于测试）。

参考 wiibrew.org 的 Wiimote / Balance Board 文档以及 WiimoteLib 的实现：
* 输出报告 0x12 设置数据上报模式，0x32 = 按键 + 8 字节扩展数据。
* 输出报告 0x16 写寄存器，0x17 读寄存器；读回的数据通过输入报告 0x21 返回。
* 扩展初始化：向 0xA400F0 写 0x55，再向 0xA400FB 写 0x00。
* 校准数据位于 0xA40020 起的 32 字节：[4:12] 0kg，[12:20] 17kg，[20:28] 34kg，
  每组 4 个传感器按 TR, BR, TL, BL 顺序，大端 16 位。
* 0x32 报告中扩展数据的 8 字节同样按 TR, BR, TL, BL 顺序，大端 16 位。
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
        raise ValueError("每次最多写 16 字节")
    payload = bytes(data) + bytes(16 - len(data))
    return pad([REPORT_WRITE_MEMORY, REGISTER_SPACE, (address >> 16) & 0xFF, (address >> 8) & 0xFF,
                address & 0xFF, len(data)] + list(payload))


def read_memory_report(address: int, size: int) -> bytes:
    return pad([REPORT_READ_MEMORY, REGISTER_SPACE, (address >> 16) & 0xFF, (address >> 8) & 0xFF,
                address & 0xFF, (size >> 8) & 0xFF, size & 0xFF])


def parse_read_data(report: bytes | list[int]) -> tuple[int, bytes, int]:
    """解析 0x21 报告，返回 (地址低 16 位, 数据, 错误码)。"""
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
    """每个传感器在 0 / 17 / 34 kg 时的原始读数，顺序 TR, BR, TL, BL。"""

    kg0: np.ndarray
    kg17: np.ndarray
    kg34: np.ndarray

    @classmethod
    def from_bytes(cls, data: bytes) -> "Calibration":
        """data 为从 0xA40020 读取的 32 字节。"""
        if len(data) < 28:
            raise ValueError("校准数据长度不足")
        vals = [be16(data, 4 + 2 * i) for i in range(12)]
        return cls(np.array(vals[0:4], float), np.array(vals[4:8], float), np.array(vals[8:12], float))

    def to_kg(self, raw: np.ndarray) -> np.ndarray:
        """按 WiimoteLib 的分段线性插值把原始读数换算为 kg。"""
        raw = np.asarray(raw, float)
        low = 17.0 * (raw - self.kg0) / np.maximum(self.kg17 - self.kg0, 1e-9)
        high = 17.0 + 17.0 * (raw - self.kg17) / np.maximum(self.kg34 - self.kg17, 1e-9)
        return np.where(raw < self.kg17, low, high)

    def to_dict(self) -> dict:
        return {"kg0": self.kg0.tolist(), "kg17": self.kg17.tolist(), "kg34": self.kg34.tolist()}


def parse_sensor_raw(report: bytes | list[int]) -> np.ndarray | None:
    """从 0x32 报告中取出 4 个传感器原始值（TR, BR, TL, BL）。"""
    r = bytes(report)
    if not r or r[0] != INPUT_BUTTONS_EXT8 or len(r) < 11:
        return None
    ext = r[3:11]
    return np.array([be16(ext, 2 * i) for i in range(4)], float)


def center_of_pressure(kg: np.ndarray, sensor_dx_m: float, sensor_dy_m: float,
                       min_total_kg: float = 1.0) -> tuple[float, float]:
    """由四个传感器的力（顺序 TR, BR, TL, BL）计算板坐标系中的压力中心 (x, y)，米。

    x 向右（TR/BR 一侧）为正，y 向前（TL/TR 一侧）为正。总重量低于阈值时返回 NaN。
    """
    tr, br, tl, bl = [float(v) for v in kg]
    total = tr + br + tl + bl
    if total < min_total_kg:
        return float("nan"), float("nan")
    x = (sensor_dx_m / 2.0) * ((tr + br) - (tl + bl)) / total
    y = (sensor_dy_m / 2.0) * ((tr + tl) - (br + bl)) / total
    return x, y
