# PyInstaller spec: builds dist/PoseBoard/ with
#   PoseBoard.exe      - the GUI (windowed)
#   PoseBoard-Wii.exe  - Wii-only console recorder
# sharing one set of libraries.
#
# Build (from the repository root):
#   pyinstaller installer/poseboard.spec --noconfirm
# Put models/pose_landmarker_full.task in place first to bundle the MediaPipe model
# (otherwise it is downloaded next to the exe on first use).

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent
INSTALLER = ROOT / "installer"

mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")

datas = list(mp_datas)
for model in (ROOT / "models").glob("pose_landmarker_*.task"):
    datas.append((str(model), "models"))
if (ROOT / "plugins").is_dir():
    datas.append((str(ROOT / "plugins"), "plugins"))

# Pose backends are imported by name (poseboard.pose.detectors.create_detector), which the
# import analysis cannot see: bundle every backend module. A backend whose packages (PyTorch,
# ONNX Runtime, ...) are not installed at build time is reported as unavailable at run time.
hidden = mp_hidden + [
    "hid",
    "poseboard.pose.mediapipe_backend",
    "poseboard.pose.multiview",
    "poseboard.pose.external",
    "poseboard.analysis",
    "poseboard.pose.detectors",
] + sorted(f"poseboard.pose.detectors.{p.stem}"
           for p in (ROOT / "poseboard" / "pose" / "detectors").glob("*.py") if p.stem != "__init__")
# matplotlib must stay: mediapipe.tasks.python.vision imports it (drawing_utils).
excludes = ["tkinter", "IPython", "pytest"]


def analysis(script):
    return Analysis(
        [str(INSTALLER / script)],
        pathex=[str(ROOT), str(INSTALLER)],
        binaries=mp_binaries,
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
