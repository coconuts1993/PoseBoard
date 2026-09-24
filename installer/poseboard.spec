# PyInstaller spec: builds dist/PoseBoard/ with
#   PoseBoard.exe      - the GUI (windowed)
#   PoseBoard-Wii.exe  - Wii-only console recorder
# sharing one set of libraries.
#
# Build (from the repository root):
#   pyinstaller installer/poseboard.spec --noconfirm
# Put models/pose_landmarker_full.task in place first to bundle the MediaPipe model
# (otherwise it is downloaded next to the exe on first use).
#
# Pose backends in the exe: MediaPipe (always), OpenPose (OpenCV DNN) and, when rtmlib and
# ONNX Runtime are installed in the build environment, the rtmlib backends (RTMPose, RTMW,
# RTMO, ViTPose ONNX, RTMPose3D) and MoveNet. The PyTorch stack (torch, torchvision,
# Ultralytics, transformers, MMPose) is always left out (several GB): those backends are shown
# as "not installed" in the exe and need the source install.

import importlib.util
import json
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
INSTALLER = ROOT / "installer"


def installed(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")

datas = list(mp_datas)
binaries = list(mp_binaries)
for model in (ROOT / "models").glob("pose_landmarker_*.task"):
    datas.append((str(model), "models"))
if (ROOT / "plugins").is_dir():
    datas.append((str(ROOT / "plugins"), "plugins"))

# Pose backends are imported by name (poseboard.pose.detectors.create_detector), which the
# import analysis cannot see: bundle every backend module. A backend whose packages (PyTorch,
# ONNX Runtime, ...) are not bundled is reported as unavailable at run time.
hidden = mp_hidden + [
    "hid",
    "poseboard.pose.mediapipe_backend",
    "poseboard.pose.multiview",
    "poseboard.pose.external",
    "poseboard.analysis",
    "poseboard.pose.detectors",
] + sorted(f"poseboard.pose.detectors.{p.stem}"
           for p in (ROOT / "poseboard" / "pose" / "detectors").glob("*.py") if p.stem != "__init__")

# Torch-free extra backends: rtmlib + ONNX Runtime (the contrib hook collects ONNX Runtime's
# native libraries).
bundled = ["mediapipe", "openpose_dnn"]
if installed("onnxruntime"):
    hidden += ["onnxruntime"]
    bundled.append("movenet")
    if installed("rtmlib"):
        hidden += collect_submodules("rtmlib")
        bundled += ["rtmpose", "rtmpose_halpe26", "rtmw_wholebody", "rtmo", "vitpose_onnx",
                    "rtmpose3d"]

# Written into the bundle: the selftest checks that these backends are available in the exe.
build_info = Path(workpath) / "bundled_backends.json"
build_info.parent.mkdir(parents=True, exist_ok=True)
build_info.write_text(json.dumps({"backends": bundled}), encoding="utf-8")
datas.append((str(build_info), "."))

# matplotlib must stay: mediapipe.tasks.python.vision imports it (drawing_utils).
excludes = [
    "tkinter", "IPython", "pytest",
    # PyTorch stack of the torchvision / Ultralytics / transformers / MMPose backends (too large;
    # their detector modules import them only when used)
    "torch", "torchvision", "torchaudio", "ultralytics", "ultralytics_thop", "transformers",
    "accelerate", "timm", "mmpose", "mmcv", "mmengine", "mmdet", "tensorflow", "tensorboard",
    "onnxruntime.transformers", "onnxruntime.quantization", "onnxruntime.tools",
]


def analysis(script):
    return Analysis(
        [str(INSTALLER / script)],
        pathex=[str(ROOT), str(INSTALLER)],
        binaries=binaries,
        datas=datas,
        hiddenimports=hidden,
        excludes=excludes,
        noarchive=False,
    )


a_gui = analysis("entry_gui.py")
a_wii = analysis("entry_wii.py")

exe_gui = EXE(
    PYZ(a_gui.pure),
    a_gui.scripts,
    [],
    exclude_binaries=True,
    name="PoseBoard",
    console=False,
    upx=False,
)
exe_wii = EXE(
    PYZ(a_wii.pure),
    a_wii.scripts,
    [],
    exclude_binaries=True,
    name="PoseBoard-Wii",
    console=True,
    upx=False,
)

coll = COLLECT(
    exe_gui, a_gui.binaries, a_gui.datas,
    exe_wii, a_wii.binaries, a_wii.datas,
    upx=False,
    name="PoseBoard",
)
