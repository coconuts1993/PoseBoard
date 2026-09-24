# PoseBoard

**Synchronized 3D human pose + Wii Balance Board recording, in one coordinate frame and on one clock.**

PoseBoard is a Windows desktop app (Python, PySide6, OpenCV) for balance and posture
research. It records:

* **force and center of pressure (COP)** from a Nintendo Wii Balance Board over Bluetooth, and
* **video and 3D body keypoints** from one or more ordinary webcams (MediaPipe built in, or
  your own model such as PoseAssess through a plugin),

stamps every sample with the same high-resolution clock (`time.perf_counter`), and places the
board, the COP, the skeleton and the whole-body **center of mass (COM)** in a single metric
world frame defined by a checkerboard on the floor. After each recording it writes a fused
table (COP and COM side by side) and standard sway metrics.

The Wii Balance Board is **optional**. You can use PoseBoard as a plain Balance Board logger
(connect the board, tare, record), as a camera/pose recorder, or both together. Recordings
share the same file layout either way, so data collected with the board alone can be aligned
with pose data later.

---

## Contents

1. [Features](#features)
2. [Hardware](#hardware)
3. [Installation (Windows)](#installation-windows)
4. [Pairing the Wii Balance Board on Windows](#pairing-the-wii-balance-board-on-windows)
5. [Quick start without hardware](#quick-start-without-hardware)
6. [Step-by-step workflow](#step-by-step-workflow)
7. [Coordinate conventions](#coordinate-conventions)
8. [Output files](#output-files)
9. [Post-processing CLI](#post-processing-cli)
10. [Integrating PoseAssess (or any other 3D pose tool)](#integrating-poseassess-or-any-other-3d-pose-tool)
11. [Accuracy: single camera vs. multiple cameras](#accuracy-single-camera-vs-multiple-cameras)
12. [Limitations](#limitations)
13. [Troubleshooting](#troubleshooting)
14. [Running the tests](#running-the-tests)
15. [Project layout](#project-layout)
16. [References](#references)

---

## Features

* **Wii Balance Board driver in pure Python** (`hidapi`): reads the board's factory
  calibration (0 / 17 / 34 kg per sensor), converts the four load cells to kg, computes the
  COP, and supports tare, battery status and a configurable minimum load for COP. No
  WiimoteLib or other driver needed.
* **Built-in simulator** (a 70 kg person swaying at 100 Hz) so you can try the whole
  workflow without a board.
* **Any number of cameras**: USB webcams (DirectShow on Windows), video files (looped, for
  testing) or network streams (RTSP/HTTP). Each camera is recorded to MP4 together with a
  per-frame timestamp file.
* **Camera calibration**: checkerboard intrinsics, and extrinsics from a checkerboard on the
  floor that defines the shared world frame (Z up, meters). The corner ordering is made
  canonical, so all cameras agree on the same origin and axes. Import/export JSON and
  Pose2Sim `Calib.toml`.
* **Board registration by clicking**: click the 4 board corners + center in the image. One
  camera: PnP. Two or more calibrated cameras: triangulation + Kabsch fit.
* **3D pose**: MediaPipe Pose Landmarker (lite/full/heavy). One camera: MediaPipe's metric
  skeleton is placed in the world with PnP. Two or more cameras: weighted triangulation.
  **Plugin interface** for your own estimator, and **offline import** of TRC/CSV files.
* **Center of mass** from 3D keypoints with Winter's segment table; works with MediaPipe,
  COCO, Halpe/Pose2Sim/OpenPose-style names.
* **Live view**: board outline, sensors and axes, COP marker and force arrow, skeleton and
  COM (with its projection on the board) drawn over the video; top-down board view with COP
  and COM trails and the COM-COP distance.
* **Post-processing**: `fused.csv` (force interpolated at pose times), `summary.json` with
  COP and COM sway metrics, TRC export, and fusion of external pose files.
* **Project files**: save/load the full setup (cameras, calibration, checkerboard, board size
  and pose, clicks) and reopen it next time.

---

## Hardware

| Item | Notes |
|---|---|
| **Wii Balance Board** (RVL-WBC-01) | Optional. Connected over Bluetooth. Use fresh AA batteries. |
| **Bluetooth adapter** | The PC's built-in Bluetooth usually works. |
| **1 or more webcams** | 720p at 30 fps is plenty for balance tasks. Mount them on tripods; they must not move after calibration. Turn off auto-focus if the camera allows it (focusing changes the intrinsics). Two or more cameras give much better 3D (see [Accuracy](#accuracy-single-camera-vs-multiple-cameras)). |
| **Printed checkerboard** | Flat and rigid (glue it on foam board). Enter the number of **inner corners** and the exact **square size in mm**. A board with **one odd and one even inner-corner count** is required, e.g. **9 x 6** inner corners (= 10 x 7 squares). |

**Why odd + even?** OpenCV may list the corners of a checkerboard starting from any corner.
PoseBoard removes the mirror ambiguity by requiring the camera to be on the +Z side, and
removes the 180° ambiguity by picking the origin whose diagonally outward square is black.
That second test only works if the two opposite corner squares have different colors, which
is the case exactly when *cols + rows* is odd (9 x 6 yes, 8 x 6 or 9 x 7 no). With a
symmetric board, different cameras could pick different origins and the world frame would be
inconsistent.

**Size tips.** For intrinsics an A4/A3 print with ~25 mm squares held in the hand is fine.
For the floor (world frame) use a larger board, e.g. 9 x 6 with 60-80 mm squares, so every
camera sees it sharply. You can use two different boards: change the checkerboard settings
between the intrinsics and extrinsics steps.

---

## Installation (Windows)

1. Install **Python 3.10 or newer, 64-bit** from python.org (3.10-3.12 recommended, since
   MediaPipe publishes wheels for those; tick *"Add python.exe to PATH"*).
2. Open a terminal (PowerShell or cmd) in the PoseBoard folder and create a virtual
   environment (recommended):

   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Alternatively install it as a package: `pip install -e .` (adds the `poseboard` and
   `poseboard-analysis` commands).
3. Start the app:

   ```bat
   python -m poseboard
   ```

   or double-click **`run_poseboard.bat`** (it activates `.venv` if present and keeps the
   window open if something goes wrong). Pass a saved project to open it directly:
   `run_poseboard.bat my_setup.json` / `python -m poseboard my_setup.json`.

The first time you start MediaPipe pose estimation, the model file
(`models/pose_landmarker_<variant>.task`, ~5-30 MB) is downloaded automatically. Without
internet access, download it manually from the URL listed in
`poseboard/pose/mediapipe_backend.py` (`MODEL_URLS`) and put it in `models/`.

Dependencies (`requirements.txt`): `numpy`, `opencv-contrib-python` (the OpenCV build that
MediaPipe itself depends on; installing `opencv-python` as well would give two copies of
`cv2`), `PySide6`, `hidapi`, `mediapipe`, and `tomli` on Python 3.10. For tests:
`pip install -r requirements-dev.txt`.

---

## Pairing the Wii Balance Board on Windows

1. Open **Settings -> Bluetooth & devices -> Add device -> Bluetooth**.
2. Open the battery cover under the board and press the small red **SYNC** button. The
   power LED on the board starts blinking.
3. Select **"Nintendo RVL-WBC-01"**. If Windows asks for a PIN, **leave it empty** and
   continue (on older dialogs choose *"Skip"* / *"Pair without using a code"*).
4. In PoseBoard, tab **1 Devices -> Scan**, choose the board, **Connect**. When it works the
   LED stays on and the sensor values update. If the board goes to sleep before you connect,
   press SYNC again.

Notes:

* PoseBoard talks to the board through **hidapi** (`pip install hidapi`, included in
  `requirements.txt`); no extra driver is needed. The board appears as an HID device with
  Nintendo's vendor ID `0x057E`.
* With the **Microsoft Bluetooth stack**, pairing without a PIN is often only temporary: if
  the board was paired before but now fails to connect (or connects and immediately stops),
  **remove the device in Windows Bluetooth settings and pair it again** (steps 1-3). This is
  normal and usually needed after the board has been switched off.
* **Close any other program** that talks to the board (for example a previous Wii app)
  before connecting in PoseBoard.
* If "Scan" says that HID devices cannot be enumerated with *"module 'hid' has no attribute
  ..."*, a different package called `hid` is shadowing hidapi: run `pip uninstall hid` and
  `pip install --force-reinstall hidapi`.

---

## Quick start without hardware

1. `python -m poseboard`
2. Tab **1 Devices**: click **Use Simulator** (fake board). Optionally add a webcam
   (source `0`, **Add Camera**) or a video file (**Video File...**).
3. The top-down view at the bottom left shows the simulated COP trail.
4. Tab **4 Record**: enter a subject name, **Start Recording**, wait a few seconds, **Stop
   Recording**. The summary appears in the panel and the files are in `recordings/`.

---

## Step-by-step workflow

The window has the live video on the left (choose the displayed camera with **Show camera**,
toggle the overlays with **Overlay board/COP** and **Overlay skeleton/COM**), a top-down board
view below it (blue = COP trail, orange = COM projection, both over the last 5 s), and four
tabs on the right. Work through the tabs in order.

### Tab 1 - Devices

**Cameras**

* **Source**: a device index (`0`, `1`, ...), a video file (**Video File...**) or an
  `rtsp://`/`http://` URL. Choose a **Resolution** and click **Add Camera**. Cameras are
  named `cam0`, `cam1`, ...
* Keep the same resolution for calibration and recording: a calibration only applies to the
  resolution it was made at (otherwise PoseBoard falls back to approximate intrinsics).

**Wii Balance Board**

* **Scan** -> select -> **Connect**, or **Use Simulator**. **Disconnect** releases it.
* The panel shows the four sensors (TL, TR, BL, BR in kg), total weight, COP (mm, board
  frame) and battery.
* **Tare (board empty)**: with nobody on the board, wait a second and click it. The mean of
  the last 1 s becomes the zero point of each sensor. Tare at the start of every session.
* **COP min. load** (default 5 kg): below this total weight the COP is undefined (empty in
  the files), which avoids noisy COP values when nobody is on the board.

### Tab 2 - Camera Calibration

1. **Checkerboard**: inner corners (cols, rows) and square size in mm. Tick **Show live
   checkerboard detection** to see whether it is detected.
2. **Intrinsics** (per camera; select it with *Show camera*): hold the checkerboard in
   15-30 different positions, distances and tilts covering the whole image and click
   **Capture Frame** each time, then **Compute Intrinsics**. An RMS below ~0.5-1 px is good.
   You can skip this for a quick trial; PoseBoard then assumes a 60° horizontal field of
   view without lens distortion (less accurate).
3. **Extrinsics / world frame**: lay the (floor) checkerboard flat on the floor next to the
   balance board, fully visible in **all** cameras at once, and click **Set All Camera
   Extrinsics from Checkerboard**. The panel lists the reprojection error and each camera's
   position in meters; camera heights (Z) should be positive and plausible. After this the
   checkerboard can be removed, but **the cameras must not move** any more.

Calibrations can be saved and loaded via the **File** menu (JSON, or Pose2Sim `Calib.toml`
import/export).

### Tab 3 - Board Setup

1. **Board Dimensions**: defaults are the official Wii Balance Board (surface 511 x 316 mm,
   sensor spacing 433 x 238 mm). Measure your board and edit if needed.
2. Decide which long side is the **front**: the side the subject faces when standing on the
   board = the side of the TL/TR sensors (the "top" in Wii terms). Left and right are the
   **subject's** left and right when facing the front, not left/right in the image.
3. Click **Start Clicking** and click, in this exact order, on the corners of the **top
   (standing) surface**:

   | # | Point | Meaning |
   |---|---|---|
   | 1 | **TL** | front-left corner |
   | 2 | **TR** | front-right corner |
   | 3 | **BR** | back-right corner |
   | 4 | **BL** | back-left corner |
   | 5 | **C** | center of the top surface |

   Left-click adds a point, right-click (or **Undo**) removes the last one, **Clear This
   Camera** starts over. The board has rounded corners: click where the straight edges would
   meet. Small pieces of tape on the corners and the center make clicking easier and more
   precise.
4. Optional: select another calibrated camera and click the same 5 points there. With 2 or
   more cameras the board is triangulated (more accurate).
5. Click **Compute Board Pose**. Check the result:
   * reprojection error of a few pixels or less;
   * **board normal** close to `[0, 0, 1]` (board and checkerboard lie on the same floor);
   * **board center** Z close to the board height (about 0.05 m).
6. **Verify the orientation**: connect the board and stand on (or press down on) one corner,
   e.g. front-left. The COP dot in the video and in the top-down view must appear at that
   same corner. If it appears at the diagonally opposite corner, click **Rotate Frame 180°**
   (the front was chosen the wrong way round); the 90° buttons exist for other mix-ups.
   If it is still wrong, redo the clicks.

Save the setup with **File -> Save Project...**. As long as the cameras and the board stay in
place you can simply load the project next time.

**The board must not move after registration.** If it is moved, click the 5 points again.

### Tab 4 - Record

1. **3D Pose Estimation** (optional): choose **MediaPipe (full / lite / heavy)** or
   **Plugin (.py)** (select the file with "..."), then **Start Pose Estimation**. The label
   shows the pose rate and "(no person detected)" when nobody is found. The skeleton and COM
   are drawn on the video; the COM projection appears in the top-down view.
2. **Recording**: enter **Subject** and **Notes**, choose the **Output folder** (default
   `recordings/`) and click **Start Recording**. At least one camera or the board must be
   connected; everything that is running is recorded (Wii only, cameras only, cameras + pose,
   or all).
3. **Stop Recording** writes `fused.csv` and `summary.json`; the summary is shown in the
   panel.

Tip: sway metrics are computed over the whole recording. Start recording once the subject is
standing still, or trim the data afterwards, if you want clean quiet-standing metrics.

---

## Coordinate conventions

All lengths are in **meters** in files and APIs (sway metrics are reported in mm).

**World frame = the floor checkerboard.**

* Origin: the inner corner whose diagonally outward square is black (chosen automatically,
  identical for all cameras).
* X along the checkerboard's *cols* direction, Y along its *rows* direction, **Z up**
  (pointing toward the cameras).
* Camera extrinsics are world -> camera, `X_cam = R @ X_world + t` (`rvec` Rodrigues, `tvec`
  in meters), the same convention as OpenCV and Pose2Sim.
* If a camera has no extrinsics, the world frame is that camera's frame.

**Board frame** (top view, subject standing on the board facing the front):

```
        TL ----------- TR          +Y  front ("top" in Wii terms)
        |               |           ^
        |       C       |           +--> +X  right
        |               |
        BL ----------- BR
```

* Origin **C** = center of the top surface; **X right, Y front, Z up** (board normal).
* Sensors (Wii data order **TR, BR, TL, BL**) at (±216.5, ±119) mm with the default
  433 x 238 mm spacing; surface corners at (±255.5, ±158) mm for 511 x 316 mm.
* COP in the board frame: `x = (dx/2)·((TR+BR) - (TL+BL)) / total`,
  `y = (dy/2)·((TR+TL) - (BR+BL)) / total`, so **x = medio-lateral (+ right)** and
  **y = antero-posterior (+ front)**.
* COM in the board frame: `com_z_board` is the height above the board surface;
  `com_minus_cop_x/y` is the COM projected on the board plane minus the COP.

**Clock**: every sample carries `t = time.perf_counter()` in seconds, taken in the same
process for the Wii, all cameras and the pose. `t0` (in `session.json`) is the start of the
recording and `t_rel = t - t0`.

---

## Output files

Each recording creates `recordings/<YYYYMMDD_HHMMSS>_<subject>/`:

| File | Content |
|---|---|
| `session.json` | Metadata: `created`, `subject`, `notes`, `clock`, `t0`, `world_frame`, `board_geometry`, `board_pose` (`board_to_world` R/t, `method` pnp/triangulation, `reproj_error_px`; `null` if not registered), `cameras` (full calibrations), `streams` (name, source, fps, video), `force_source` (type, tare, sensor spacing, board calibration, battery), `pose_backend`, `duration_s`, `samples` (counts). |
| `wii.csv` | Every Balance Board sample (only if a board/simulator is connected). |
| `camN.mp4` | Video of camera `camN` (MPEG-4). The frame rate in the file header is nominal; use the timestamps. |
| `camN_timestamps.csv` | `frame`, `t`: capture time (perf_counter, s) of every video frame. `t - t0` gives `t_rel`. |
| `pose3d.csv` | One row per pose result (only if pose estimation ran during the recording). |
| `fused.csv` | Written at stop: Wii data linearly interpolated at the pose timestamps, next to the COM (needs `wii.csv` and `pose3d.csv`). |
| `summary.json` | Written at stop: sway metrics (see below). |
| `pose3d.trc` | Optional, from `--trc` (see CLI). |
| `fused_<name>.csv` | Optional, from `--external <name>.trc/.csv` (see CLI). |

**`wii.csv` columns**

| Column | Meaning |
|---|---|
| `t`, `t_rel` | perf_counter time (s) and time since recording start (s) |
| `TR_kg`, `BR_kg`, `TL_kg`, `BL_kg` | tared load per sensor (kg) |
| `total_kg` | sum of the four sensors (kg; multiply by 9.80665 for N) |
| `cop_x_board`, `cop_y_board` | COP in the board frame (m); empty below *COP min. load* |
| `cop_x_world`, `cop_y_world`, `cop_z_world` | COP in the world frame (m); empty if the board is not registered |

**`pose3d.csv` columns**

| Column | Meaning |
|---|---|
| `t`, `t_rel` | capture time of the frame(s) used for this pose (s) |
| `<name>_x`, `<name>_y`, `<name>_z`, `<name>_score` | for every keypoint (e.g. MediaPipe `left_shoulder_x`, or `LShoulder_x` from a Halpe-26 plugin): world position (m) and confidence; empty if missing |
| `com_x`, `com_y`, `com_z` | whole-body COM in the world frame (m) |
| `com_x_board`, `com_y_board`, `com_z_board` | COM in the board frame (m) |

**`fused.csv` columns**: `t`, `t_rel`, `TR_kg`, `BR_kg`, `TL_kg`, `BL_kg`, `total_kg`,
`cop_x_board`, `cop_y_board`, `cop_x_world`, `cop_y_world`, `cop_z_world`, `com_x`, `com_y`,
`com_z`, `com_x_board`, `com_y_board`, `com_z_board`, `com_minus_cop_x`, `com_minus_cop_y`.
Force values are empty where the nearest Wii sample is more than 0.1 s away.

**`summary.json`**

* `session`, `mean_total_kg`, `wii_rate_hz`, `pose_rate_hz`
* `cop` and `com`: sway metrics in the board frame (ML = x, AP = y), each with
  `duration_s`, `samples`, `mean_x_mm`, `mean_y_mm`, `range_ml_mm`, `range_ap_mm`,
  `rms_ml_mm`, `rms_ap_mm`, `path_length_mm`, `mean_velocity_mm_s` and
  `ellipse95_area_mm2` (95 % confidence ellipse, `π · 5.991 · sqrt(λ1 λ2)`)
* `com_cop_distance_rms_mm`: RMS horizontal distance between COM and COP

---

## Post-processing CLI

```bat
python -m poseboard.analysis <session folder> [--external FILE.trc|FILE.csv] [--offset SECONDS] [--pose2sim-yup] [--trc]
```

| Option | Effect |
|---|---|
| *(none)* | Recompute `fused.csv` and `summary.json` and print the summary. |
| `--external FILE` | Fuse an external 3D pose file (TRC or CSV) with this session's Wii data -> `fused_<FILE stem>.csv`. |
| `--offset S` | Time offset (s): the external file's time 0 corresponds to `t0 + S`. Use `0` when the file's time column is already `t_rel`. |
| `--pose2sim-yup` | The external TRC is in Pose2Sim's Y-up convention (Pose2Sim writes `(X', Y', Z') = (Y, Z, X)`); convert it back to the checkerboard Z-up frame. |
| `--trc` | Export `pose3d.csv` as `pose3d.trc` (mm, time = `t_rel`, same Z-up world frame; rotate to Y-up yourself if a tool such as OpenSim expects it). |

External file formats:

* **TRC** (Pose2Sim/OpenSim): units taken from the header (`mm`, `cm` or `m`), marker names
  from the 4th line, `Time` column in seconds.
* **CSV**: a time column named `t`, `time` or `timestamp` (s), and `<name>_x`, `<name>_y`,
  `<name>_z` columns in meters.

From Python you can also pass an arbitrary 4x4 transform (external frame -> PoseBoard world):

```python
import numpy as np
from poseboard.analysis import fuse_external

T = np.eye(4)  # replace with your transform
fuse_external("recordings/20260924_153000_S01", "poseassess.trc", offset_s=0.0, world_transform=T)
```

---

## Integrating PoseAssess (or any other 3D pose tool)

There are two ways to combine an existing 3D pose application with PoseBoard's Wii data. In
both cases the key requirements are the **same clock** and the **same world frame**.

### Option A - live plugin (recommended when PoseAssess is Python code)

PoseBoard loads a Python file that defines `create_estimator(**kwargs)` returning a
`PoseEstimator` (see `poseboard/pose/base.py`). Start from
**`plugins/poseassess_plugin_template.py`**:

1. Copy it (e.g. to `plugins/poseassess_plugin.py`) and set `POSEASSESS_DIR` (added to
   `sys.path`) and `KEYPOINT_SET` (`"halpe26"` or `"coco17"`, matching your model's output
   order).
2. Load your model in `_load_model()`.
3. Fill in **one** hook:
   * **Case (a), 2D per camera** - `detect_per_camera(cam_name, t, image, cam)` returns the
     2D keypoints (and scores) of one image. With 2 or more calibrated cameras PoseBoard
     triangulates them. With a single camera, also return a metric 3D skeleton (`kp3d`,
     `kp3d_frame="body"` or `"camera"`); PoseBoard then places it in the world frame (PnP
     against the 2D keypoints, or directly via the extrinsics).
   * **Case (b), 3D directly** - `detect_3d_multiview(frames, cams)` returns 3D keypoints
     that are already in the checkerboard world frame. If your frame differs, set
     `WORLD_TRANSFORM` (4x4). To make PoseAssess use PoseBoard's calibration, export it via
     **File -> Export Pose2Sim Calib.toml...**.
4. Tab **4 Record** -> **Plugin (.py)** -> choose the file -> **Start Pose Estimation**.

As shipped the template runs but detects nothing, which is handy for checking the wiring.
The plugin receives `frames = {camera name: (t, BGR image)}` and
`cams = {camera name: CameraCalibration}` and must return a `Pose3D` whose `t` is the
**capture time** of the frames (not the time the model finished), which is what keeps pose and
force aligned. Exceptions are shown in the GUI and do not stop the app. `tests/test_plugin.py`
shows how to test a plugin without cameras.

### Option B - offline import (when PoseAssess runs as a separate program)

1. **Share the world frame.** Either export PoseBoard's calibration
   (**File -> Export Pose2Sim Calib.toml...**) and use it in PoseAssess, or import PoseAssess's
   calibration into PoseBoard (**File -> Import Camera Calibration (.json / Pose2Sim .toml)...**,
   meters, world -> camera, same cameras and resolution) *before* clicking the board, so the
   board is registered in PoseAssess's frame. Otherwise supply a 4x4 transform (`fuse_external(..., world_transform=T)`).
2. **Record with PoseBoard** (cameras + Wii). PoseBoard saves `camN.mp4` and
   `camN_timestamps.csv`.
3. **Run PoseAssess on PoseBoard's videos.** This is the easiest way to get exact timing: the
   time of frame *i* is `t` from `camN_timestamps.csv` minus `t0` from `session.json`. Write
   that as the TRC `Time` column (or the CSV `t` column) and use `--offset 0`. Do not use
   `frame / fps`, because webcams drop frames and their fps is not exact.
   If PoseAssess records its own videos at the same time instead, the clocks differ: start
   the trial with a sync event (e.g. a small jump or a heel drop, visible both as a spike in
   `total_kg` and as a vertical jump of the ankles/COM), measure the time difference, and
   pass it as `--offset`.
4. Fuse:

   ```bat
   python -m poseboard.analysis recordings\20260924_153000_S01 --external poseassess.trc --offset 0
   ```

   Add `--pose2sim-yup` if the TRC was written by Pose2Sim (Y-up). The result is
   `fused_poseassess.csv` with the same columns as `fused.csv`.

Keypoint names only need to be recognizable for the COM model (`poseboard/pose/com.py`):
`left_shoulder`, `LShoulder`, `l_shoulder` etc. all work. At least both shoulders and both
hips are required; knees, ankles, elbows, wrists, ears/nose and heels/toes improve the COM.

---

## Accuracy: single camera vs. multiple cameras

| | One camera | Two or more calibrated cameras |
|---|---|---|
| Board registration | PnP on 5 coplanar clicked points. Good if intrinsics are calibrated and the board is seen at an oblique angle (not edge-on). | 5 points triangulated, rigid fit to the board model. More robust to click errors. |
| Pose | MediaPipe's metric skeleton placed with PnP. Positions **across** the image are good; **depth along the camera's line of sight is approximate** (errors of several cm are possible), and body proportions come from the model. | Weighted triangulation of 2D keypoints (confidence >= 0.5). Metric and consistent in all directions. |
| Recommendation | Put the camera so that the sway direction you care about moves across the image (e.g. a side view for antero-posterior sway), 2-3 m away, whole body and board visible. | Place cameras 60-120° apart around the subject, all seeing the board and the floor checkerboard. |

General tips: calibrate the intrinsics of each camera, keep the resolution fixed, click the
board corners carefully (tape marks help) and check the reprojection errors and the corner
test described in [Tab 3](#tab-3---board-setup).

---

## Limitations

* **Monocular depth is approximate.** With one camera, COM positions along the viewing
  direction should be interpreted with care; use two or more cameras for quantitative 3D.
* **Wii sample rate is about 60-100 Hz and not perfectly uniform** (Bluetooth). Each sample is
  stamped when it arrives; `wii.csv` contains all raw samples, and `fused.csv` interpolates
  them at the pose times.
* **Timestamps are arrival times.** Webcam frames are stamped when the app receives them,
  which includes the camera's internal latency (typically tens of ms for USB webcams), and
  Bluetooth adds its own small delay to the Wii data. For fast movements, check the alignment
  with a sync event (see Option B).
* **Multiple cameras are not hardware-synchronized.** The pose uses the newest frame of each
  camera (and their mean time). This is fine for standing balance; for fast movements expect
  some triangulation error.
* **Pose rate = inference rate.** Live pose runs as fast as the model allows and skips frames;
  the videos contain every frame, so you can re-process them offline (Option B).
* **The board and the cameras must not move** after board registration and extrinsic
  calibration. If they do, recalibrate/re-click.
* **The Balance Board is not a laboratory force plate.** It measures only the vertical load
  (four load cells, no shear forces or moments), has more noise and drift, and is rated for
  about 150 kg. Tare before each session and use fresh batteries.
* The COM model uses generic segment parameters (Winter) and assumes a single subject.

---

## Troubleshooting

| Problem | What to try |
|---|---|
| **Scan finds no board** | Pair it first (see [Pairing](#pairing-the-wii-balance-board-on-windows)); press SYNC again; remove and re-pair in Windows; close other Wii programs. |
| **"Cannot enumerate HID devices"** | `pip install hidapi`; if the error mentions `hid.device`, run `pip uninstall hid` then `pip install --force-reinstall hidapi`. |
| **Connects, then "Error: ..." in the Wii panel** | Board went to sleep or the pairing expired: remove and re-pair, then Connect again. Check the batteries. |
| **Camera cannot be opened** | Another program uses it; try another index; try "Default" resolution. |
| **Checkerboard not detected** | Check cols/rows (inner corners!), lighting, glare, and that the whole board is visible and sharp. |
| **COP dot at the wrong corner** | Rotate Frame 180° (front/back swapped), or redo the clicks in the order TL, TR, BR, BL, C using the subject's left/right. |
| **Large board reprojection error / tilted board normal** | Click more carefully, calibrate intrinsics, check the board dimensions, use a second camera. |
| **MediaPipe model download fails** | Download the `.task` file manually (URL in `poseboard/pose/mediapipe_backend.py`) into `models/`. |
| **"(no person detected)"** | The whole body should be visible; improve lighting; try "MediaPipe (heavy)". |

---

## Running the tests

```bat
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests need no hardware: they use synthetic camera scenes, the board simulator and an
offscreen Qt window. On a headless Linux machine set `QT_QPA_PLATFORM=offscreen`
(`QT_QPA_PLATFORM=offscreen python -m pytest -q tests`).

---

## Project layout

```
PoseBoard/
├── poseboard/
│   ├── __main__.py            python -m poseboard  -> GUI
│   ├── calibration.py         intrinsics, checkerboard world frame, JSON / Pose2Sim TOML I/O
│   ├── geometry.py            board model, board registration (PnP / triangulation + Kabsch)
│   ├── camera.py              camera threads, MP4 + per-frame timestamp recording
│   ├── fusion.py              COP -> world, COM vs. COP
│   ├── overlay.py             drawing board, COP, skeleton, COM on the video
│   ├── session.py             synchronized recorder (session folder layout)
│   ├── analysis.py            fused.csv, sway metrics, TRC export, external import (CLI)
│   ├── wii/
│   │   ├── protocol.py        Balance Board HID protocol (pure functions)
│   │   └── device.py          hidapi reader, simulator, tare
│   ├── pose/
│   │   ├── base.py            Pose2D / Pose3D / PoseEstimator interface
│   │   ├── com.py             center of mass (Winter segment table)
│   │   ├── mediapipe_backend.py  MediaPipe backend, single-view lifting
│   │   ├── triangulation.py   multi-camera keypoint triangulation
│   │   └── external.py        plugin loader, TRC / CSV readers
│   └── gui/
│       ├── app.py             main window (tabs 1-4)
│       └── widgets.py         video view, top-down board view
├── plugins/
│   └── poseassess_plugin_template.py   template for your own pose estimator
├── tests/                     pytest suite (no hardware needed)
├── models/                    MediaPipe model files (downloaded on first use)
├── recordings/                default output folder (created on first recording)
├── requirements.txt / requirements-dev.txt / pyproject.toml
└── run_poseboard.bat          Windows launcher
```

---

## References

* Wii Balance Board protocol: [wiibrew.org - Wii Balance Board](https://wiibrew.org/wiki/Wii_Balance_Board)
  and WiimoteLib.
* Clark, R. A. et al. (2010). *Validity and reliability of the Nintendo Wii Balance Board for
  assessment of standing balance.* Gait & Posture, 31(3), 307-310.
* Winter, D. A. (2009). *Biomechanics and Motor Control of Human Movement*, 4th ed., Table 4.1
  (segment parameters).
* Pagnon, D. et al. (2022). *Pose2Sim: an open-source Python package for multiview markerless
  kinematics.* Journal of Open Source Software, 7(77), 4362.
* MediaPipe Pose Landmarker: <https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker>
