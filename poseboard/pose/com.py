"""Whole-body center of mass (COM) estimation from a segment model.

Segment mass fractions and COM locations follow Winter (2009) *Biomechanics and Motor
Control of Human Movement*, Table 4.1. Keypoint names are matched after normalization,
so common naming schemes such as MediaPipe (left_hip) and OpenPose/Halpe/Pose2Sim
(LHip, LBigToe) are supported.
"""

from __future__ import annotations

import re

import numpy as np

# (mass fraction, proximal point, distal point, COM position as a fraction from proximal)
# Proximal/distal are "part" names resolved by _locate() (l/r prefix = left/right)
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
    """Map arbitrarily named keypoints to the body parts required by the COM model."""

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
    if p is None and part == "heel":  # fall back to the ankle when there is no heel point
        p = get(side, "ankle")
    return p


def center_of_mass(keypoints: np.ndarray, names: list[str],
                   return_segments: bool = False):
    """Estimate the whole-body COM from 3D keypoints. Missing segments are handled by
    renormalizing over the mass of the remaining segments.

    At least the trunk (both shoulders + both hips) is required; otherwise returns None.
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
