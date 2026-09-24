"""PyInstaller entry point for PoseBoard.exe (GUI).

``PoseBoard.exe --selftest`` runs a headless smoke test (used by CI): it opens the main
window offscreen, streams the simulated Wii board, runs MediaPipe on a blank frame (directly
and through the pose backend registry), checks that the backends bundled into the exe are
available (and the PyTorch ones are not), runs ONNX Runtime and the rtmlib detector class when
they are bundled, and exits with code 0 on success. It needs no network: the real rtmlib models
are only run when they are already in rtmlib's download cache (or with
``POSEBOARD_SELFTEST_DOWNLOAD=1``).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

# A tiny Caffe network (ReLU) converted to ONNX by PoseBoard's own converter: runs in ONNX
# Runtime without any model file
_TINY_PROTOTXT = """
name: "selftest"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 4
layer { name: "relu" type: "ReLU" bottom: "data" top: "out" }
"""


def _bundled_backends() -> list[str]:
    """Backends the build bundled (written by installer/poseboard.spec)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    f = base / "bundled_backends.json"
    if not f.is_file():
        return ["mediapipe"]
    return list(json.loads(f.read_text(encoding="utf-8")).get("backends", []))


def _rtmlib_cached(mode: str = "lightweight") -> bool:
    """True when rtmlib's person detector and RTMPose models of ``mode`` are already downloaded
    (so the real models can be run without network access)."""
    import rtmlib
    from rtmlib.tools.file import _get_rtmhub_dir

    m = rtmlib.Body.MODE[mode]
    folder = Path(_get_rtmhub_dir()) / "checkpoints"
    for url in (m["det"], m["pose"]):
        name = Path(url.split("?")[0]).name
        if not ((folder / name).is_file() or (folder / (name.split(".")[0] + ".onnx")).is_file()):
            return False
    return True


def _selftest_real_rtmpose(real, img) -> str:
    """Run both real rtmlib models: YOLOX on a black image (no person), then RTMPose (ONNX
    session and SimCC decoding) on a forced whole-image box, directly and through PoseBoard's
    detector (box and keypoint mapping)."""
    import numpy as np

    assert real.detect(img, 0.0, "selftest") == []  # YOLOX: nobody in a black image
    h, w = img.shape[:2]
    box = [0.0, 0.0, float(w), float(h)]
    kp, sc = real.pose_model(real._pose_image(img), bboxes=[box])[:2]
    kp, sc = np.asarray(kp, np.float64), np.asarray(sc, np.float64)
    assert kp.shape == (1, 17, 2) and sc.shape == (1, 17), (kp.shape, sc.shape)
    assert np.isfinite(kp).all() and np.isfinite(sc).all(), "RTMPose output is not finite"
    det_model = real.det_model
    real.det_model = lambda image: np.array([box])  # one person box: the whole image
    try:
        people = real.detect(img, 0.0, "selftest")
    finally:
        real.det_model = det_model
    assert len(people) <= 1, people  # one box -> at most one person (none if all scores <= 0)
    for p in people:
        found = np.isfinite(p.keypoints).all(axis=1)
        assert p.keypoints.shape == (17, 2) and found.any(), p.keypoints
        # mapped back to image pixels: within the (1.25x padded) crop of the box
        lo, hi = -0.25 * np.array([w, h]), 1.25 * np.array([w, h])
        inside = ((p.keypoints[found] >= lo) & (p.keypoints[found] <= hi)).all()
        assert inside, f"RTMPose keypoints not mapped back to the image: {p.keypoints[found]}"
    return (f"rtmlib real models (lightweight YOLOX + RTMPose): OK ({len(people)} person with "
            f"{int(np.isfinite(people[0].keypoints).all(axis=1).sum()) if people else 0} "
            "keypoints on the forced box)")


def selftest_backends() -> list[str]:
    """Checks of the pose backend registry in the exe; returns report lines."""
    import numpy as np

    from poseboard.pose.detectors import BACKENDS, backend_available

    lines = []
    bundled = _bundled_backends()
    for key in BACKENDS:
        ok, why = backend_available(key)
        lines.append(f"  {key}: {'available' if ok else 'not available (' + why + ')'}")
        if key in bundled:
            assert ok, f"{key} was bundled but is not available: {why}"
    if getattr(sys, "frozen", False):
        for key in ("yolo_pose", "keypoint_rcnn", "vitpose_hf"):  # PyTorch stack left out
            assert not backend_available(key)[0], f"{key} should not be in the exe"
    if "rtmpose" not in bundled:
        lines.append("rtmlib: not bundled")
        return lines

    # ONNX Runtime's native libraries load and run a model
    import onnxruntime as ort

    from poseboard.pose.detectors import openpose_dnn_det, rtmlib_det

    sess = ort.InferenceSession(openpose_dnn_det.caffe_to_onnx(_TINY_PROTOTXT, {}),
                                providers=["CPUExecutionProvider"])
    (y,) = sess.run(None, {"data": np.array([-1, 2, -3, 4], np.float32).reshape(1, 1, 1, 4)})
    assert y.ravel().tolist() == [0.0, 2.0, 0.0, 4.0], y
    lines.append(f"onnxruntime {ort.__version__}: OK ({', '.join(ort.get_available_providers())})")

    # The rtmlib detector class with stand-in models: one person box -> one COCO-17 person
    from rtmlib.version import __version__ as rtmlib_version

    img = np.zeros((240, 320, 3), np.uint8)
    kp = np.column_stack([np.linspace(100, 200, 17), np.linspace(40, 200, 17)])[None]
    det = rtmlib_det.RTMLibDetector(
        "rtmpose", pose_model=lambda image, bboxes: (kp, np.full((1, 17), 0.9)),
        det_model=lambda image: np.array([[90.0, 30.0, 210.0, 210.0]]))
    people = det.detect(img, 0.0, "selftest")
    assert len(people) == 1 and people[0].keypoints.shape == (17, 2), people
    det.close()
    lines.append(f"rtmlib {rtmlib_version}: detector class OK")

    # The real models, when available offline (or downloads are allowed)
    if _rtmlib_cached() or os.environ.get("POSEBOARD_SELFTEST_DOWNLOAD") == "1":
        from poseboard.pose.detectors import create_detector

        try:
            real = create_detector("rtmpose", mode="lightweight", device="cpu")
        except rtmlib_det.ModelUnavailable as e:
            lines.append(f"rtmlib real models: not available ({e})")
        else:
            try:
                lines.append(_selftest_real_rtmpose(real, img))
            finally:
                real.close()
    else:
        lines.append("rtmlib real models: not in the download cache (not run; offline test)")
    return lines


def selftest() -> int:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from poseboard.gui.app import MainWindow
    from poseboard.pose.mediapipe_backend import MediaPipePose

    app = QApplication([])
    w = MainWindow()
    w.show()
    w.connect_sim()
    end = time.perf_counter() + 1.0
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.01)
    sample = w.force.latest() if w.force else None
    assert sample is not None and sample.total_kg > 10, "simulated board produced no data"
    w.close()

    est = MediaPipePose("full")
    est.detect("selftest", 0.0, np.zeros((240, 320, 3), np.uint8))
    est.close()

    # The backend registry finds its (lazily imported) modules in the frozen app
    from poseboard.pose.detectors import backend_available, create_detector

    ok, why = backend_available("mediapipe")
    assert ok, why
    det = create_detector("mediapipe", model="full")
    assert det.detect(np.zeros((240, 320, 3), np.uint8), 0.0, "selftest") == []
    det.close()

    print("Pose backends:")
    for line in selftest_backends():
        print(line)

    import hid  # noqa: F401  (hidapi must be bundled for the real board)

    print("PoseBoard selftest OK")
    return 0


def _log_to_file_when_windowed() -> None:
    """A windowed exe has no console: send stdout/stderr, logging and crashes to a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    from frozen_setup import app_dir

    log = open(app_dir() / "poseboard.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log

    def hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb, file=log)

    sys.excepthook = hook


def main() -> None:
    from frozen_setup import configure  # bundled next to this script

    _log_to_file_when_windowed()
    configure()
    if "--selftest" in sys.argv:
        try:
            code = selftest()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            code = 1
        sys.exit(code)
    from poseboard.gui.app import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
