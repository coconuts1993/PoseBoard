"""Keypoint layouts (formats) of the 2D pose backends.

Every backend reports its keypoints in one of these layouts. Names are canonical snake_case
(``left_shoulder``, ``right_big_toe``, ...), so the center-of-mass model
(``poseboard.pose.com``), the overlays and the recorded files work the same for all of them.
``pose2sim_model`` is the matching Pose2Sim ``pose_model`` name, stored in ``session.json`` so
the OpenPose-style JSON files written during a recording can be re-triangulated with Pose2Sim.

Formats (``FORMATS`` keys):

* ``coco17``        COCO body, 17 points (RTMPose, RTMO, YOLO-pose, ViTPose, Keypoint R-CNN, MoveNet)
* ``halpe26``       Halpe body with head, neck, mid-hip and feet, 26 points (RTMPose BodyWithFeet)
* ``body25``        OpenPose BODY_25 (with mid-hip and feet), OpenPose order
* ``coco18``        OpenPose COCO-18 (with neck), OpenPose order
* ``wholebody133``  COCO-WholeBody: body 17 + feet 6 + face 68 + hands 2 x 21 (RTMW, DWPose)
* ``mediapipe33``   MediaPipe / BlazePose, 33 points
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeypointFormat:
    key: str
    names: tuple[str, ...]
    skeleton: tuple[tuple[str, str], ...]
    pose2sim_model: str | None  # Pose2Sim ``pose_model`` for this layout (None: no equivalent)
    label: str = ""  # human-readable name, e.g. "COCO-17"

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        """Index of keypoint ``name`` (ValueError if the format has no such keypoint)."""
        return self.names.index(name)

    def find(self, name: str) -> int | None:
        """Index of keypoint ``name``, or None."""
        try:
            return self.names.index(name)
        except ValueError:
            return None

    def skeleton_indices(self) -> list[tuple[int, int]]:
        """The skeleton as index pairs."""
        return [(self.names.index(a), self.names.index(b)) for a, b in self.skeleton]


def _chain(names: list[str], closed: bool = False) -> list[tuple[str, str]]:
    pairs = list(zip(names[:-1], names[1:]))
    if closed:
        pairs.append((names[-1], names[0]))
    return pairs


# ------------------------------------------------------------------ COCO-17
COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
_COCO17_BODY = (
    ("left_ankle", "left_knee"), ("left_knee", "left_hip"), ("right_ankle", "right_knee"),
    ("right_knee", "right_hip"), ("left_hip", "right_hip"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
)
_COCO17_FACE = (
    ("left_eye", "right_eye"), ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"), ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
)
COCO17 = KeypointFormat("coco17", COCO17_NAMES, _COCO17_BODY + _COCO17_FACE, "COCO_17", "COCO-17")

# ------------------------------------------------------------------ Halpe-26
_FEET_LINKS = (
    ("left_ankle", "left_big_toe"), ("left_ankle", "left_small_toe"), ("left_ankle", "left_heel"),
    ("right_ankle", "right_big_toe"), ("right_ankle", "right_small_toe"),
    ("right_ankle", "right_heel"),
)
HALPE26 = KeypointFormat(
    "halpe26",
    COCO17_NAMES + ("head", "neck", "hip", "left_big_toe", "right_big_toe", "left_small_toe",
                    "right_small_toe", "left_heel", "right_heel"),
    (("left_ankle", "left_knee"), ("left_knee", "left_hip"), ("left_hip", "hip"),
     ("right_ankle", "right_knee"), ("right_knee", "right_hip"), ("right_hip", "hip"),
     ("head", "neck"), ("neck", "hip"), ("neck", "left_shoulder"),
     ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("neck", "right_shoulder"),
     ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"))
    + _COCO17_FACE + _FEET_LINKS,
    "HALPE_26", "Halpe-26 (body with feet)")

# ------------------------------------------------------------------ OpenPose BODY_25 / COCO-18
BODY25 = KeypointFormat(
    "body25",
    ("nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
     "left_elbow", "left_wrist", "mid_hip", "right_hip", "right_knee", "right_ankle", "left_hip",
     "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe",
     "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"),
    (("neck", "mid_hip"), ("neck", "right_shoulder"), ("neck", "left_shoulder"),
     ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
     ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("mid_hip", "right_hip"),
     ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("mid_hip", "left_hip"),
     ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("neck", "nose"),
     ("nose", "right_eye"), ("right_eye", "right_ear"), ("nose", "left_eye"),
     ("left_eye", "left_ear"), ("left_ankle", "left_big_toe"), ("left_big_toe", "left_small_toe"),
     ("left_ankle", "left_heel"), ("right_ankle", "right_big_toe"),
     ("right_big_toe", "right_small_toe"), ("right_ankle", "right_heel")),
    "BODY_25", "OpenPose BODY_25")

COCO18 = KeypointFormat(
    "coco18",
    ("nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
     "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip",
     "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"),
    (("neck", "right_shoulder"), ("neck", "left_shoulder"), ("right_shoulder", "right_elbow"),
     ("right_elbow", "right_wrist"), ("left_shoulder", "left_elbow"),
     ("left_elbow", "left_wrist"), ("neck", "right_hip"), ("right_hip", "right_knee"),
     ("right_knee", "right_ankle"), ("neck", "left_hip"), ("left_hip", "left_knee"),
     ("left_knee", "left_ankle"), ("neck", "nose"), ("nose", "right_eye"),
     ("right_eye", "right_ear"), ("nose", "left_eye"), ("left_eye", "left_ear")),
    "COCO", "OpenPose COCO-18")

# ------------------------------------------------------------------ COCO-WholeBody 133
_FACE = [f"face_{i}" for i in range(68)]
_LHAND = [f"left_hand_{i}" for i in range(21)]
_RHAND = [f"right_hand_{i}" for i in range(21)]


def _face_links() -> list[tuple[str, str]]:
    """iBUG 68-point face contours."""
    f = _FACE
    return (_chain(f[0:17]) + _chain(f[17:22]) + _chain(f[22:27]) + _chain(f[27:31])
            + _chain(f[31:36]) + _chain(f[36:42], True) + _chain(f[42:48], True)
            + _chain(f[48:60], True) + _chain(f[60:68], True))


def _hand_links(hand: list[str], wrist: str) -> list[tuple[str, str]]:
    """Hand root (index 0) to the 4 joints of each finger; the body wrist to the hand root."""
    links = [(wrist, hand[0])]
    for finger in range(5):
        joints = [hand[0]] + hand[1 + 4 * finger: 5 + 4 * finger]
        links += _chain(joints)
    return links


WHOLEBODY133 = KeypointFormat(
    "wholebody133",
    COCO17_NAMES + ("left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
                    "right_small_toe", "right_heel") + tuple(_FACE) + tuple(_LHAND) + tuple(_RHAND),
    _COCO17_BODY + _COCO17_FACE + _FEET_LINKS + tuple(_face_links())
    + tuple(_hand_links(_LHAND, "left_wrist")) + tuple(_hand_links(_RHAND, "right_wrist")),
    "COCO_133", "COCO-WholeBody 133")

# ------------------------------------------------------------------ MediaPipe / BlazePose 33
MEDIAPIPE33 = KeypointFormat(
    "mediapipe33",
    ("nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
     "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
     "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
     "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
     "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
     "right_heel", "left_foot_index", "right_foot_index"),
    (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
     ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
     ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
     ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
     ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
     ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
     ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"),
     ("left_ankle", "left_foot_index"), ("right_ankle", "right_heel"),
     ("right_heel", "right_foot_index"), ("right_ankle", "right_foot_index"),
     ("nose", "left_ear"), ("nose", "right_ear")),
    "BLAZEPOSE", "MediaPipe / BlazePose 33")

FORMATS: dict[str, KeypointFormat] = {f.key: f for f in (
    COCO17, HALPE26, BODY25, COCO18, WHOLEBODY133, MEDIAPIPE33)}


def get_format(key: str | KeypointFormat) -> KeypointFormat:
    """The format with this key (a ``KeypointFormat`` is returned unchanged)."""
    if isinstance(key, KeypointFormat):
        return key
    try:
        return FORMATS[key]
    except KeyError:
        raise KeyError(f"Unknown keypoint format {key!r}; known: {', '.join(FORMATS)}") from None
