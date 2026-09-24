"""基于节段模型的人体重心（COM）估计。

节段质量比例与重心位置采用 Winter (2009) *Biomechanics and Motor Control of Human
Movement* 表 4.1。关键点名称做了归一化匹配，兼容 MediaPipe（left_hip）、
OpenPose/Halpe/Pose2Sim（LHip, LBigToe）等常见命名。
"""

from __future__ import annotations

import re

import numpy as np

# (质量比例, 起点, 终点, COM 距起点的比例)
# 起点/终点为 "部位" 名称，由 _locate() 解析（l/r 前缀表示左右）
SEGMENTS = [
    ("head", 0.081, "head", "head", 0.0),
    ("trunk", 0.497, "mid_shoulder", "mid_hip", 0.5),
    ("l_upperarm", 0.028, "l_shoulder", "l_elbow", 0.436),
    ("r_upperarm", 0.028, "r_shoulder", "r_elbow", 0.436),
    ("l_forearm_hand", 0.022, "l_elbow", "l_wrist", 0.682),
    ("r_forearm_hand", 0.022, "r_elbow", "r_wrist", 0.682),
    ("l_thigh", 0.100, "l_hip", "l_knee", 0.433),
    ("r_thigh", 0.100, "r_hip", "r_knee", 0.433),
    ("l_shank", 0.0465, "l_knee", "l_ankle", 0.433),
    ("r_shank", 0.0465, "r_knee", "r_ankle", 0.433),
    ("l_foot", 0.0145, "l_heel", "l_toe", 0.5),
    ("r_foot", 0.0145, "r_heel", "r_toe", 0.5),
]

_PART_ALIASES = {
    "shoulder": ["shoulder", "sho", "sh"],
    "elbow": ["elbow", "elb"],
    "wrist": ["wrist", "wri"],
    "hip": ["hip"],
    "knee": ["knee"],
    "ankle": ["ankle", "ank"],
    "heel": ["heel"],
    "toe": ["footindex", "bigtoe", "toe", "foot"],
    "ear": ["ear"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class KeypointIndex:
    """把任意命名的关键点映射到 COM 模型需要的部位。"""

    def __init__(self, names: list[str]):
        self.names = list(names)
        self._idx = {_norm(n): i for i, n in enumerate(names)}

    def find(self, side: str | None, part: str) -> int | None:
        prefixes = [""] if side is None else (["left", "l"] if side == "l" else ["right", "r"])
        for alias in _PART_ALIASES.get(part, [part]):
            for p in prefixes:
                i = self._idx.get(p + alias)
                if i is not None:
                    return i
        return None


def _point(kp: np.ndarray, idx: KeypointIndex, token: str) -> np.ndarray | None:
    def get(side, part):
        i = idx.find(side, part)
        if i is None:
            return None
        p = kp[i]
        return None if np.any(~np.isfinite(p)) else p

    def mid(a, b):
        return None if a is None or b is None else (a + b) / 2

    if token == "mid_shoulder":
        return mid(get("l", "shoulder"), get("r", "shoulder"))
    if token == "mid_hip":
        return mid(get("l", "hip"), get("r", "hip"))
    if token == "head":
        ears = mid(get("l", "ear"), get("r", "ear"))
        if ears is not None:
            return ears
        for n in ("head", "nose"):
            p = get(None, n)
            if p is not None:
                return p
        return None
    side, part = token.split("_", 1)
    p = get(side, part)
    if p is None and part == "heel":  # 没有足跟点时用踝关节代替
        p = get(side, "ankle")
    return p


def center_of_mass(keypoints: np.ndarray, names: list[str],
                   return_segments: bool = False):
    """由 3D 关键点估计全身重心。缺失的节段按剩余节段的质量重新归一化。

    至少需要躯干（双肩 + 双髋）才返回结果，否则返回 None。
    """
    idx = KeypointIndex(names)
    kp = np.asarray(keypoints, np.float64)
    total_m, acc = 0.0, np.zeros(kp.shape[1])
    segs = {}
    for seg, m, a, b, r in SEGMENTS:
        pa, pb = _point(kp, idx, a), _point(kp, idx, b)
        if pa is None or pb is None:
            continue
        c = pa + r * (pb - pa)
        segs[seg] = c
        acc += m * c
        total_m += m
    if "trunk" not in segs or total_m < 0.6:
        return (None, segs) if return_segments else None
    com = acc / total_m
    return (com, segs) if return_segments else com
