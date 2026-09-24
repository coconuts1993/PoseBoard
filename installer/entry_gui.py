"""PyInstaller entry point for PoseBoard.exe (GUI).

``PoseBoard.exe --selftest`` runs a headless smoke test (used by CI): it opens the main
window offscreen, streams the simulated Wii board, runs MediaPipe on a blank frame and
exits with code 0 on success.
"""

from __future__ import annotations

import os
import sys
import time
import traceback


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
