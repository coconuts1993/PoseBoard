"""PyInstaller entry point for PoseBoard-Wii.exe (console Wii-only recorder)."""

from __future__ import annotations

import sys


def main() -> None:
    from frozen_setup import configure  # bundled next to this script

    configure()
    from poseboard.wii.record import main as record_main

    sys.exit(record_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
