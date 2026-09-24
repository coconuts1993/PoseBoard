"""Keypoint formats: consistent names and skeletons, and a center of mass for every format."""

import numpy as np
import pytest

from poseboard.pose.com import KeypointIndex, center_of_mass
from poseboard.pose.formats import FORMATS, KeypointFormat, get_format
from poseboard.pose.mediapipe_backend import MP_NAMES, MP_SKELETON
from tests.fakes import standing_person

POSE2SIM_MODELS = {"COCO_17", "HALPE_26", "BODY_25", "COCO", "COCO_133", "BLAZEPOSE"}
EXPECTED = {"coco17": 17, "halpe26": 26, "body25": 25, "coco18": 18, "wholebody133": 133,
            "mediapipe33": 33}
WITH_FEET = {"halpe26", "body25", "wholebody133", "mediapipe33"}


def test_all_formats_registered():
    assert set(FORMATS) == set(EXPECTED)
    for key, n in EXPECTED.items():
        fmt = FORMATS[key]
        assert fmt.key == key and len(fmt) == n and get_format(key) is fmt
        assert fmt.pose2sim_model in POSE2SIM_MODELS and fmt.label
    with pytest.raises(KeyError, match="Unknown keypoint format"):
        get_format("nope")


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_names_unique_and_skeleton_valid(key):
    fmt = FORMATS[key]
    assert isinstance(fmt, KeypointFormat)
    assert len(set(fmt.names)) == len(fmt.names)
    for n in fmt.names:
        assert n.isascii() and n == n.lower() and " " not in n and "-" not in n
    edges = set()
    for a, b in fmt.skeleton:
        assert a in fmt.names and b in fmt.names and a != b
        assert frozenset((a, b)) not in edges, (a, b)
        edges.add(frozenset((a, b)))
    assert len(fmt.skeleton_indices()) == len(fmt.skeleton)
    # every keypoint except the face points is drawn (MediaPipe keeps its classic skeleton)
    if key != "mediapipe33":
        linked = {n for e in fmt.skeleton for n in e}
        assert {n for n in fmt.names if not n.startswith("face_")} <= linked


def test_layout_spot_checks():
    c17 = FORMATS["coco17"].names
    assert c17[:5] == ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
    assert c17[-2:] == ("left_ankle", "right_ankle")
    h26 = FORMATS["halpe26"].names
    assert h26[:17] == c17 and h26[17:20] == ("head", "neck", "hip")
    assert h26[24:] == ("left_heel", "right_heel")
    b25 = FORMATS["body25"]
    assert b25.index("neck") == 1 and b25.index("mid_hip") == 8 and b25.index("right_heel") == 24
    c18 = FORMATS["coco18"]
    assert c18.index("right_shoulder") == 2 and c18.index("left_ear") == 17
    wb = FORMATS["wholebody133"]
    assert wb.names[:17] == c17 and wb.index("left_big_toe") == 17 and wb.index("right_heel") == 22
    assert wb.index("face_0") == 23 and wb.index("face_67") == 90
    assert wb.index("left_hand_0") == 91 and wb.index("right_hand_0") == 112
    assert wb.index("right_hand_20") == 132
    mp = FORMATS["mediapipe33"]
    assert list(mp.names) == MP_NAMES and list(mp.skeleton) == MP_SKELETON


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_com_parts_recognized(key):
    fmt = FORMATS[key]
    idx = KeypointIndex(list(fmt.names))
    for side, word in (("l", "left"), ("r", "right")):
        for part in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
            assert idx.find(side, part) == fmt.index(f"{word}_{part}"), (key, side, part)
        if key in WITH_FEET:
            assert idx.find(side, "heel") == fmt.index(f"{word}_heel")
            toe = f"{word}_foot_index" if key == "mediapipe33" else f"{word}_big_toe"
            assert idx.find(side, "toe") == fmt.index(toe)
    assert idx.find("l", "ear") == fmt.index("left_ear")


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_com_from_standing_skeleton_in_every_format(key):
    offset = np.array([0.8, 0.5, 0.053])
    kp = standing_person(key, offset=offset)
    assert kp.shape == (len(FORMATS[key]), 3) and np.all(np.isfinite(kp))
    com, segs = center_of_mass(kp, list(FORMATS[key].names), return_segments=True)
    assert com is not None
    expected = {"head", "trunk", "l_upperarm", "r_upperarm", "l_forearm_hand", "r_forearm_hand",
                "l_thigh", "r_thigh", "l_shank", "r_shank"}
    if key in WITH_FEET:
        expected |= {"l_foot", "r_foot"}
    assert set(segs) == expected  # every segment the format can describe is found
    assert com[0] == pytest.approx(offset[0], abs=1e-6)  # left-right symmetric
    assert 0.85 < com[2] - offset[2] < 1.1
    if key in WITH_FEET:  # a real foot segment: heel -> toe, not collapsed onto the ankle
        fmt = FORMATS[key]
        assert not np.allclose(segs["l_foot"], kp[fmt.index("left_ankle")])


def test_com_same_in_all_formats():
    """The formats describe the same body: their COMs agree within a few centimeters."""
    coms = {k: center_of_mass(standing_person(k), list(FORMATS[k].names)) for k in FORMATS}
    ref = coms["coco17"]
    for k, c in coms.items():
        assert np.linalg.norm(c - ref) < 0.03, k
