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
(no cameras needed), as a camera/pose recorder (no board needed), or both together. Inside
PoseBoard the board is **plug and play**: as soon as Windows has a Bluetooth link to it,
PoseBoard connects by itself (no Scan/Connect clicks) and reconnects after a Bluetooth drop.
Windows itself usually needs the board to be paired again after it was switched off (see
[Pairing](#pairing-the-wii-balance-board-on-windows)). There is also a small command-line
recorder for Wii-only sessions.

Every file stores both the shared monotonic clock and **wall-clock Unix time** (`t_unix`), and
you can drop **event markers** while recording, so PoseBoard data can be aligned later with
recordings from other programs (your own 3D pose app, the original Wii app, EMG, ...). See
[Aligning with other recordings](#aligning-with-other-recordings).

---

## Contents

1. [Features](#features)
2. [Hardware](#hardware)
3. [Installation (Windows)](#installation-windows)
4. [Pairing the Wii Balance Board on Windows](#pairing-the-wii-balance-board-on-windows)
5. [Quick start without hardware](#quick-start-without-hardware)
6. [Step-by-step workflow](#step-by-step-workflow)
7. [Wii-only recording from the command line](#wii-only-recording-from-the-command-line)
8. [Coordinate conventions and clocks](#coordinate-conventions-and-clocks)
9. [Output files](#output-files)
10. [Post-processing CLI](#post-processing-cli)
11. [Aligning with other recordings](#aligning-with-other-recordings)
12. [Integrating PoseAssess (or any other 3D pose tool)](#integrating-poseassess-or-any-other-3d-pose-tool)
13. [Accuracy: single camera vs. multiple cameras](#accuracy-single-camera-vs-multiple-cameras)
14. [Limitations](#limitations)
15. [Troubleshooting](#troubleshooting)
16. [Running the tests](#running-the-tests)
17. [Project layout](#project-layout)
18. [References](#references)

---

## Features

* **Wii Balance Board driver in pure Python** (`hidapi`): reads the board's factory
  calibration (0 / 17 / 34 kg per sensor), converts the four load cells to kg, computes the
  COP, and supports tare, battery status and a configurable minimum load for COP. No
  WiimoteLib or other driver needed.
* **Optional board, plug and play**: record with cameras only, the board only, or both.
  With **Auto-connect** (on by default) a board that Windows has paired and connected is
  opened as soon as it is switched on and reconnected automatically after a Bluetooth drop; a
  board that connects while a recording is running is added to it. Each board keeps its own
  tare across reconnects.
* **Wii-only command-line recorder** (`python -m poseboard.wii.record`): no cameras, no
  GUI, same session format.
* **Alignment with other recordings**: every row has `t` (monotonic), `t_rel` and `t_unix`
  (wall clock); **Mark Event** (F9) writes labelled markers; step-on/off, jumps and stomps
  are detected in the force signal (`--events`); external files with Unix times can be fused
  directly (`--time-base unix`).
* **Built-in simulator** (a 70 kg person swaying at 100 Hz) so you can try the whole
  workflow without a board.
* **Any number of cameras**: USB webcams (DirectShow with MJPG on Windows), video files
  (looped, for testing) or network streams (RTSP/HTTP). Each camera is recorded to MKV
  (MPEG-4 video; still readable, up to the last few seconds, after a crash or power loss) together with a per-frame
  timestamp file.
* **Camera calibration**: checkerboard intrinsics, and extrinsics from a checkerboard on the
  floor that defines the shared world frame (Z up, meters). The corner ordering is made
  canonical, so all cameras agree on the same origin and axes. Import/export JSON and
  Pose2Sim `Calib.toml`.
* **Board registration by clicking**: click the 4 board corners + center in the image. One
  camera: PnP. Two or more calibrated cameras: triangulation. With the floor checkerboard the
  board is then fitted flat on the floor (position and heading only), which avoids a tilt
  from click noise. Mirrored click orders are refused, and a **Corner Check** (press one corner)
  verifies the orientation against the sensors.
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
* **Project files**: save/load the full setup (cameras with their resolution, calibration,
  checkerboard, board size and pose, clicks) and reopen it next time.

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

### Option 0 - packaged build (no Python needed)

Each build of the repository produces **`PoseBoard-windows-x64.zip`** (GitHub: *Actions* ->
the latest *Build Windows exe* run -> *Artifacts*, or the *Releases* page for tagged versions).
It contains `PoseBoard.exe` (the GUI), `PoseBoard-Wii.exe` (the Wii-only console recorder,
see below) and the MediaPipe *full* model, so pose estimation works offline.

* Unzip it to a **writable folder** such as `C:\PoseBoard` or your Documents, not under
  `C:\Program Files`: the exe works in its own folder, so `recordings\` and relative paths such
  as `--out` end up next to the exe.
* The plugin template is inside the package (`_internal\plugins`, which the **...** button
  opens). Plugins that need extra Python packages (PyTorch, ONNX Runtime, ...) do **not** work
  with the packaged build: use the source install below.

### Option 1 - from source

1. Get the PoseBoard folder: `git clone <repository URL>`, or on GitHub *Code -> Download ZIP*
   and unzip it.
2. Install **Python 3.10 or newer, 64-bit** from python.org (3.10-3.12 recommended, since
   MediaPipe publishes wheels for those; tick *"Add python.exe to PATH"*).
3. Open a terminal in the PoseBoard folder and create a virtual environment. Call the
   environment's Python directly; no "activate" step is needed (in PowerShell,
   `.venv\Scripts\activate` is blocked by the default script policy, and packages would then
   silently go into the global Python):

   ```bat
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

   Alternatively install it as a package: `.venv\Scripts\python -m pip install -e .` (adds the
   `poseboard`, `poseboard-analysis` and `poseboard-wii` commands in `.venv\Scripts`).
4. Start the app:

   ```bat
   .venv\Scripts\python -m poseboard
   ```

   or double-click **`run_poseboard.bat`** (it uses `.venv` if present and keeps the window
   open if something goes wrong). Pass a saved project to open it directly:
   `run_poseboard.bat my_setup.json` (a relative path is taken from the folder you run it
   from) / `.venv\Scripts\python -m poseboard my_setup.json`. In the commands below, `python`
   means `.venv\Scripts\python` if you use the virtual environment.

The first time you start MediaPipe pose estimation, the model file
(`models/pose_landmarker_<variant>.task`, ~5-30 MB) is downloaded automatically (the window
stays responsive; a stalled download gives up after 30 s without data). Without internet
access, or if `storage.googleapis.com` is blocked, download the file on any computer and put
it into the `models` folder of PoseBoard (or copy it from the packaged build or another PC):

| Variant | File | URL |
|---|---|---|
| lite | `pose_landmarker_lite.task` | <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task> |
| full | `pose_landmarker_full.task` | <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task> |
| heavy | `pose_landmarker_heavy.task` | <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task> |

The error message after a failed download names the exact file path expected.

Dependencies (`requirements.txt`): `numpy`, `opencv-contrib-python` (the OpenCV build that
MediaPipe itself depends on; installing `opencv-python` as well would give two copies of
`cv2`), `PySide6`, `hidapi`, `mediapipe`, and `tomli` on Python 3.10. For tests:
`python -m pip install -r requirements-dev.txt`.

---

## Pairing the Wii Balance Board on Windows

1. Open **Settings -> Bluetooth & devices -> Add device -> Bluetooth**.
2. Open the battery cover under the board and press the small red **SYNC** button. The
   power LED on the board starts blinking.
3. Select **"Nintendo RVL-WBC-01"**. If Windows asks for a PIN, **leave it empty** and
   continue (on older dialogs choose *"Skip"* / *"Pair without using a code"*).
4. Start PoseBoard. With **Auto-connect** ticked (the default, tab **1 Devices**) it finds
   the board by itself within about a second: the board's LED stays on and the sensor values
   update. Nothing else to click.

**What "plug and play" means here.** Whenever Windows has the board paired and connected,
PoseBoard (the GUI with Auto-connect, or `python -m poseboard.wii.record`) opens it by itself:
no Scan/Connect needed, and if the Bluetooth link drops during a session PoseBoard keeps
waiting and reconnects as soon as the board is available again (in a recording, the gap is
logged in `events.csv`). What PoseBoard cannot do is restore the Windows pairing itself: with
the Microsoft Bluetooth stack and an empty-PIN pairing (the only method Windows offers for the
board), the pairing is usually **lost when the board is switched off**.

**Daily checklist** (Microsoft stack):

1. Switch the board on. If Windows lists "Nintendo RVL-WBC-01" as *Paired* but PoseBoard says
   *"A paired board is listed but does not answer"*, **remove the device** in Windows Bluetooth
   settings and pair it again (steps 1-3; press SYNC). This is expected, not a fault.
2. PoseBoard picks the board up within a second, without any click.
3. Tare (board empty), then record.

Third-party Bluetooth tools that store a permanent pairing for the board (the "permanent
sync" method used by Wii-remote utilities) avoid step 1 but are outside PoseBoard.

The manual buttons are still there: **Scan** lists the paired boards, **Connect** connects the
selected one (with Auto-connect ticked it stays plug and play but only for that board; with
Auto-connect unticked it is a one-time connection that stops with an error if the link
drops). **Use Simulator** and **Disconnect** switch Auto-connect off; only one board source is
ever active.

Notes:

* PoseBoard talks to the board through **hidapi** (`pip install hidapi`, included in
  `requirements.txt`); no extra driver is needed. The board appears as an HID device with
  Nintendo's vendor ID `0x057E`.
* With the **Microsoft Bluetooth stack**, pairing without a PIN is often only temporary: if
  the board was paired before but now fails to connect (or connects and immediately stops),
  **remove the device in Windows Bluetooth settings and pair it again** (steps 1-3). This is
  normal and usually needed after the board has been switched off (see the checklist above).
* PoseBoard checks that the device really is a Balance Board (its extension type) before
  using it; a Wii Remote that shows up in the same list is skipped with a message.
* **Close any other program** that talks to the board (for example a previous Wii app)
  before connecting in PoseBoard: only one program at a time can read the board. To combine
  data from another program with PoseBoard's, see
  [Aligning with other recordings](#aligning-with-other-recordings).
* If "Scan" (or the Wii panel) says *"the installed 'hid' module is not hidapi"*, a different
  package called `hid` is shadowing hidapi: run `python -m pip uninstall hid` and
  `python -m pip install --force-reinstall hidapi`. *"hidapi is not installed"* means
  `python -m pip install hidapi`.

---

## Quick start without hardware

1. `python -m poseboard`
2. Tab **1 Devices**: click **Use Simulator** (fake board; this switches Auto-connect off).
   Optionally add a webcam (source `0`, **Add Camera**) or a video file (**Video File...**).
3. The top-down view at the bottom left shows the simulated COP trail.
4. Tab **4 Record**: enter a subject name, **Start Recording**, press **F9** once or twice to
   drop event markers, wait a few seconds, **Stop Recording**. The summary appears in the
   panel and the files are in `recordings/`.

Without the GUI: `python -m poseboard.wii.record --simulate --seconds 5`.

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
  named `cam0`, `cam1`, ... (or type a **Name**). Calibration and clicks are stored per name;
  a source that was used before (e.g. in the loaded project) gets its old name back.
* **Keep the same resolution for calibration and recording**: a calibration only applies to
  the resolution it was made at. Projects store each camera's resolution and reopen it with
  it. If a camera nevertheless delivers another size, PoseBoard keeps the calibration (it is
  never overwritten) but cannot use its intrinsics: a red warning above the video says so,
  approximate intrinsics (with the calibrated camera position) are used meanwhile, and the
  board pose is marked outdated. Remove the camera and add it again with the resolution named
  in the warning.
* On Windows, webcams are opened with DirectShow in **MJPG** mode, so USB2 webcams reach
  30 fps at 720p/1080p. The status bar shows the measured frame rate of each camera and a hint
  when it is far below the rate the camera reports; a camera that stops delivering frames is
  shown in red (*no new frames*) and logged as `camera_lost` / `camera_recovered` in
  `events.csv` while recording.

**Wii Balance Board (optional)**

* **Auto-connect (plug and play)**, ticked by default: PoseBoard looks for a paired board
  every second, connects as soon as one is switched on and reconnects after a drop. The
  panel shows the state: *waiting for a paired board*, *connecting...*, *connected* (with
  battery), or an error (it keeps retrying). "A paired board is listed but does not answer"
  means Windows still lists the board but it is switched off or its pairing has expired.
* Manual alternative: **Scan** -> select -> **Connect**, or **Use Simulator**. **Disconnect**
  releases the board and switches Auto-connect off (see
  [Pairing](#pairing-the-wii-balance-board-on-windows)).
* The panel shows the four sensors (TL, TR, BL, BR in kg), total weight, COP (mm, board
  frame) and battery.
* You can skip the board entirely: recordings then contain video and pose only.
* **Tare (board empty)**: with nobody on the board, wait a second and click it. The mean of
  the last 1 s becomes the zero point of each sensor. Tare at the start of every session. The
  tare belongs to that board: it is kept when the board reconnects or you click Connect again,
  and a different board starts untared. Tare is disabled while recording (it would change the
  zero in the middle of `wii.csv`); `session.json` lists the tare in effect for each connection
  (`force_source_history`).
* **COP min. load** (default 5 kg, the same in the Wii-only command-line recorder): below this
  total weight the COP is undefined (empty in the files), which avoids noisy COP values when
  nobody is on the board.

### Tab 2 - Camera Calibration

1. **Checkerboard**: inner corners (cols, rows) and square size in mm. Tick **Show live
   checkerboard detection** to see whether it is detected (it runs a few times per second on
   a reduced image, so the window stays responsive).
2. **Intrinsics** (per camera; select it with *Show camera*): hold the checkerboard in
   15-30 different positions, distances and tilts covering the whole image and click
   **Capture Frame** each time, then **Compute Intrinsics**. An RMS below ~0.5-1 px is good.
   The model uses the distortion coefficients k1, k2, p1, p2 (k3 fixed at 0, as in Pose2Sim),
   so the calibration can be exported to Pose2Sim unchanged. You can skip this for a quick
   trial; PoseBoard then assumes a 60° horizontal field of view without lens distortion (less
   accurate).
3. **Extrinsics / World Frame**: lay the (floor) checkerboard flat on the floor next to the
   balance board, fully visible in **all** cameras at once, and click **Set All Camera
   Extrinsics from Checkerboard**. The panel lists the reprojection error and each camera's
   position in meters; camera heights (Z) should be positive and plausible. After this the
   checkerboard can be removed, but **the cameras must not move** any more. A symmetric
   checkerboard (cols + rows even, e.g. 8 x 6) is refused when two or more cameras are
   calibrated, because they could pick different origins.

Calibrations can be saved and loaded via the **File** menu (JSON, or Pose2Sim `Calib.toml`
import/export). The Pose2Sim export only contains cameras with extrinsics (the others are
listed in a message); on import, all-zero rotation/translation entries count as "no
extrinsics". After a calibration change (intrinsics, extrinsics, import) an already registered
board is recomputed from its stored clicks automatically.

### Tab 3 - Board Setup

1. **Board Dimensions**: defaults are the official Wii Balance Board (surface 511 x 316 mm,
   sensor spacing 433 x 238 mm, top surface 53 mm above the floor). Measure your board and edit
   if needed. **Top height above checkerboard** is used to place the board on the floor with a
   single camera: if the checkerboard lies on a mat or foam board, subtract its thickness.
2. The **front** is the long edge **opposite the power button** (the blue LED): that edge
   carries the TL/TR sensors (the "top" in Wii terms), and the power-button edge carries BL/BR.
   The subject stands facing away from the power button (as in Wii Fit, where the power button
   points away from the TV). Left and right are the **subject's** left and right when facing
   the front, not left/right in the image.
3. Click **Start Clicking** and click, in this exact order, on the corners of the **top
   (standing) surface**:

   | # | Point | Meaning |
   |---|---|---|
   | 1 | **TL** | front-left corner (edge opposite the power button) |
   | 2 | **TR** | front-right corner (edge opposite the power button) |
   | 3 | **BR** | back-right corner (power-button edge) |
   | 4 | **BL** | back-left corner (power-button edge) |
   | 5 | **C** | center of the top surface |

   Left-click adds a point, right-click (or **Undo**) removes the last one, **Clear This
   Camera** starts over. The board has rounded corners: click where the straight edges would
   meet. Small pieces of tape on the corners and the center make clicking easier and more
   precise.
4. Optional: select another calibrated camera and click the same 5 points there. With 2 or
   more cameras the board is triangulated (more accurate).
5. Click **Compute Board Pose**. With the floor checkerboard (extrinsics set), the board is
   fitted **flat on the floor**: only its position and heading are taken from the clicks, the
   normal is exactly `[0, 0, 1]`. A free (6-DoF) fit from 5 noisy clicks is typically tilted
   by 0.5-2°, which would shift the COM (about 1 m above the board) by 1-4 cm in board
   coordinates; the panel shows that discarded tilt. Check the result:
   * reprojection error of a few pixels or less;
   * **tilt of the unconstrained fit** of a few degrees at most (above 10° the floor fit is not
     applied and a warning explains why: click order, board dimensions, or a checkerboard that
     is not on the same floor);
   * **board center** Z = the board height (about 0.05 m).

   A **mirrored** click order (left/right or front/back swapped) is refused with a message,
   since it would put the board upside down. Clicks in a camera without extrinsics are not
   combined with calibrated cameras. Without any extrinsics the world frame is the camera
   frame of the (first) clicked camera, and the pose is taken from that camera only.
6. **Verify the orientation with the Corner Check**: connect the board, click **Corner
   Check** and press firmly with one hand on the corner labelled **TL** in the video
   (front-left: edge opposite the power button, on the left). PoseBoard looks at which sensor
   responds:
   * **TL**: correct.
   * **BR**: front and back were swapped in the clicks; the frame is rotated by 180°
     automatically. The subject must then stand facing the TL/TR edge (opposite the power
     button): the board frame always follows the sensors, so `x` = subject's right and `y` =
     subject's front only hold for a subject facing that edge. If the subject faces the other
     way, turn the subject (or turn the board and click again).
   * **TR** or **BL**: the clicks are turned by 90° or mirrored: redo them.

   **Rotate Frame 180°** does the same as the automatic fix (there are no 90° buttons: the
   sensor axis is always the long axis of the board, so a 90° turn is never right). Without a
   registered board (e.g. Wii only), the corner check just reports which corner you pressed;
   the Wii-only recorder shows the nearest sensor next to the COP.

Save the setup with **File -> Save Project...**. As long as the cameras and the board stay in
place you can simply load the project next time (the cameras are reopened at their saved
resolution, so their calibrations apply).

**The board must not move after registration.** If it is moved, click the 5 points again.
If you change the board dimensions, click **Compute Board Pose** again.

### Tab 4 - Record

1. **3D Pose Estimation** (optional): choose **MediaPipe (full / lite / heavy)** or
   **Plugin (.py)** (select the file with "..."), then **Start Pose Estimation**. The label
   shows the pose rate, "(no person detected)" when nobody is found, the last error (cleared
   by the next good frame) and **STOPPED** if the pose thread ended. The skeleton and COM are
   drawn on the video; the COM projection appears in the top-down view. A camera whose newest
   frame is much older than the others' (it stopped delivering) is left out of the pose. The
   backend cannot be changed while recording.
2. **Recording**: enter **Subject** and **Notes**, choose the **Output folder** (default
   `recordings/`) and click **Start Recording**. The **Streams** line shows what will be
   recorded, e.g. `cam0, cam1, pose, Wii`, `Wii not connected — recording video/pose only`,
   or `No cameras — recording Wii only`. At least one camera or a connected board is needed;
   everything that is running is recorded (Wii only, cameras only, cameras + pose, or all).
   If the board connects (or reconnects) while recording, its data are added to the same
   `wii.csv`, and `wii_connected` / `wii_disconnected` markers are written to `events.csv`.
3. **Mark Event** (button, **F9**, or **M** when no text field has the focus; only while
   recording) writes a time marker to `events.csv`. Type a label first (e.g. `sync`,
   `eyes closed`), or leave it empty for `mark_1`, `mark_2`, ... Pressing Enter in the label
   field also marks. The line below shows the number of events and the last one.
4. **Stop Recording** closes the files at once; `fused.csv` (only when both Wii and pose data
   exist) and `summary.json` are then written in the background ("Post-processing...") and
   the summary appears in the panel when done. If the app is closed meanwhile it finishes this
   before exiting. If the console window of `run_poseboard.bat` is closed during a recording,
   the files are still closed properly, without post-processing: run
   `python -m poseboard.analysis <folder>` afterwards.

Tip: sway metrics are computed over the whole recording. Start recording once the subject is
standing still, or trim the data afterwards, if you want clean quiet-standing metrics (a sync
jump inside the recording also ends up in the metrics; use `events.csv` / `--events` to cut
it out).

---

## Wii-only recording from the command line

For sessions with the board only (no cameras, no GUI) there is a small console recorder. It
behaves like a stand-alone Wii Balance Board app: start it, switch the board on (paired in
Windows, see the [daily checklist](#pairing-the-wii-balance-board-on-windows)), and it connects
by itself and reconnects after a Bluetooth drop. It writes the same session folder format as
the GUI (`session.json`, `wii.csv`, `events.csv`, `summary.json`).

```bat
python -m poseboard.wii.record --subject S01 --tare 2
python -m poseboard.wii --list                    (same program: list the paired boards)
python -m poseboard.wii --simulate --seconds 5    (no hardware)
poseboard-wii --subject S01 --seconds 60          (after pip install -e .)
```

In the packaged Windows build the same program is `PoseBoard-Wii.exe`.

| Option | Effect |
|---|---|
| `--out DIR` | Output root folder (default `recordings`). |
| `--subject NAME` | Subject name, used in the session folder name. |
| `--notes TEXT` | Free-text notes stored in `session.json`. |
| `--seconds N` | Stop after N seconds (default: record until Enter on an empty line or Ctrl+C). |
| `--wait S` | Give up if no board connects within S seconds (default: wait forever). |
| `--tare S` | Zero the board over S seconds before recording (the board must be **empty**). Without it, a warning is printed when the empty board reads more than 1 kg. |
| `--min-kg KG` | COP is left empty below this total load (default 5 kg, as in the GUI; stored in `session.json`). |
| `--device N` or `--device "PATH"` | Only use this board: its number `N` from `--list` (easiest), or its HID path in **double quotes** (Windows paths contain `&`, `#` and `{}`, which cmd and PowerShell would otherwise interpret). A path that matches no listed board gives a warning. Default: the first board found. |
| `--simulate` | Use the built-in simulator instead of a board. |
| `--list` | List the paired boards and exit. |

While it records it prints the total load, COP (with the nearest sensor, e.g. `(TL)`: press a
corner to check the orientation) and connection state once per second. **Type a label and
press Enter** to add an event marker to `events.csv`; press Enter on an empty line (or Ctrl+C)
to stop. Anything typed before the line `Recording to ...` is ignored, so an early Enter does
not end the recording. Bluetooth drops and reconnects are logged as `wii_disconnected` /
`wii_connected` events. The files are flushed about once per second, and closing the console
window (or Ctrl+Break) still closes them properly (the post-processing is then skipped; run
`python -m poseboard.analysis <folder>` later).

Exit codes: 0 = recorded, 2 = no board connected (`--wait` expired, or hidapi is not usable),
3 = error before the recording started (e.g. taring failed because the board sent no data),
4 = the recording contains no force samples, 130 = interrupted before the recording started.

---

## Coordinate conventions and clocks

All lengths are in **meters** in files and APIs (sway metrics are reported in mm).

**World frame = the floor checkerboard.**

* Origin: the inner corner whose diagonally outward square is black (chosen automatically,
  identical for all cameras).
* X along the checkerboard's *cols* direction, Y along its *rows* direction, **Z up**
  (pointing toward the cameras).
* Camera extrinsics are world -> camera, `X_cam = R @ X_world + t` (`rvec` Rodrigues, `tvec`
  in meters), the same convention as OpenCV and Pose2Sim.
* If a camera has no extrinsics, the world frame is that camera's frame.

**Board frame** (top view, subject standing on the board facing the front, i.e. facing away
from the power button):

```
        TL ----------- TR          +Y  front ("top" in Wii terms)
        |               |           ^
        |       C       |           +--> +X  right
        |               |
        BL ----------- BR
          [power button]
```

* Origin **C** = center of the top surface; **X right, Y front, Z up** (board normal).
* Sensors (Wii data order **TR, BR, TL, BL**) at (±216.5, ±119) mm with the default
  433 x 238 mm spacing; surface corners at (±255.5, ±158) mm for 511 x 316 mm.
* COP in the board frame: `x = (dx/2)·((TR+BR) - (TL+BL)) / total`,
  `y = (dy/2)·((TR+TL) - (BR+BL)) / total`, so **x = medio-lateral (+ right)** and
  **y = antero-posterior (+ front)** for a subject facing the TL/TR edge. The board frame is
  tied to the sensors: a subject standing the other way round has these signs reversed.
* COM in the board frame: `com_z_board` is the height above the board surface;
  `com_minus_cop_x/y` is the COM projected on the board plane minus the COP.

**Clocks.** Every row in every file has three time columns (`camN_timestamps.csv` also has
the frame number):

| Column | Meaning | Use it for |
|---|---|---|
| `t` | `time.perf_counter()` in seconds: a monotonic high-resolution clock, taken in the same process for the Wii, all cameras, the pose and the event markers | aligning streams *within* a PoseBoard session |
| `t_rel` | `t - t0`, seconds since the recording started | plotting, trimming, `--offset` |
| `t_unix` | wall-clock time, Unix seconds (UTC, e.g. `1790000000.123456`) = `t + clock_offset_unix` | aligning with *other* programs or devices |

`session.json` stores `t0` (perf_counter at the start), `t0_unix` (wall-clock time at the
same instant), `clock_offset_unix = t0_unix - t0` and a readable `start_time_iso` (local time
with time zone). The offset is measured **once** at the start, so `t_unix` never jumps during a
recording, even if Windows adjusts the system clock. To convert a `t_unix` value to local time
in Python: `datetime.fromtimestamp(t_unix)`; in Excel: `=t_unix/86400 + DATE(1970,1,1)` (UTC).

---

## Output files

Each recording creates `recordings/<YYYYMMDD_HHMMSS>_<subject>/` (the folder name is the
local start time; a second recording started within the same second gets `_2`, `_3`, ...).
The same layout is used by the GUI and the Wii-only recorder; files of streams that were not
recorded are simply missing. All CSV files are flushed about once per second while recording,
so a crash loses at most the last second.

| File | Content |
|---|---|
| `session.json` | Metadata: `created`, `subject`, `notes`, `clock` (description), **clock fields** `t0`, `t0_unix`, `clock_offset_unix`, `start_time_iso`, and at stop `t_stop`, `t_stop_unix`, `stop_time_iso`, `duration_s`; **stream flags** `has_wii`, `has_pose`, `has_video`, `camera_names`; `world_frame`, `board_geometry`, `board_pose` (`board_to_world` R/t, `method` pnp/triangulation, `reproj_error_px`, `floor_constrained`, `tilt_deg`, `world_camera` (camera whose frame is the world when no extrinsics were set), `warnings`, `notes`; `null` if not registered), `cameras` (the calibrations used for the recorded cameras), `streams` (name, source, fps, video file), `force_source` (at the end of the recording: type, tare, COP min. load, sensor spacing, board calibration, battery, device path; for auto-connect also the number of connections/disconnects; `null` without a board), `force_source_history` (device and tare in effect from each start/connection, with `t_rel`), `pose_backend` (label of the pose source), `pose_backend_key` (backend key, e.g. `mediapipe`), `keypoint_format` (e.g. `coco17`, `halpe26`, `mediapipe33`) and `pose2sim_model` (the matching Pose2Sim `pose_model`, e.g. `COCO_17`, `HALPE_26`, `BLAZEPOSE`), `pose_info` (backend details, if known), `pose2d` (`cameras` with a `pose2d_<camera>.csv`, `openpose_json`, `json_dir`, `json_sets_written`, `json_sets_skipped`), `samples` (counts of `wii`, `pose`, `events` rows). |
| `wii.csv` | Every Balance Board sample (only if a board or the simulator was connected during the recording). |
| `camN.mkv` | Video of camera `camN` (MPEG-4 in a Matroska container; plays in VLC, the Windows media apps and OpenCV; if the recording ends abruptly, e.g. a crash or power loss, the file stays readable up to the last few seconds, where an MP4 would be lost completely). Convert to MP4 without re-encoding if needed: `ffmpeg -i cam0.mkv -c copy cam0.mp4`. If the MKV writer is not available, `camN.avi` (MJPG) is written instead (`streams[].video` names the file). The frame rate in the file header is nominal; use the timestamps. |
| `camN_timestamps.csv` | `frame`, `t`, `t_rel`, `t_unix`: capture time of every stored video frame (one row per frame in the video). |
| `pose3d.csv` | One row per pose result (only if pose estimation ran during the recording). |
| `pose2d_<camera>.csv` | The subject's 2D keypoints in every processed frame of that camera (see below). |
| `pose2d_json/<camera>/` | Optional (recorder option `save_openpose_json`): one OpenPose-format file per processed frame, `<camera>_<frame:012d>_keypoints.json`, for re-triangulating offline with Pose2Sim (see below). |
| `events.csv` | Event markers: `t`, `t_rel`, `t_unix`, `label`. Written when the first marker is added: **Mark Event** / F9 in the GUI, a typed label in the Wii-only recorder, and automatic markers: `wii_connected` / `wii_disconnected` when the board link changes, `camera_lost <name>` / `camera_recovered <name>` / `camera_removed <name>`, and `pose_keypoints_changed`. UTF-8 with a byte order mark, so Excel shows non-ASCII labels correctly when the file is double-clicked. |
| `fused.csv` | Written at stop: Wii data linearly interpolated at the pose timestamps, next to the COM (needs `wii.csv` and `pose3d.csv`). |
| `summary.json` | Written at stop: sway metrics (see below). |
| `events_detected.csv` | Optional, from `--events`: force events found in `wii.csv` (see [Aligning](#aligning-with-other-recordings)). |
| `pose3d.trc` | Optional, from `--trc` (see CLI). |
| `fused_<name>.csv` | Optional, from `--external <name>.trc/.csv` (see CLI). |

**`wii.csv` columns**

| Column | Meaning |
|---|---|
| `t`, `t_rel`, `t_unix` | perf_counter time (s), time since recording start (s), wall-clock Unix time (s) |
| `TR_kg`, `BR_kg`, `TL_kg`, `BL_kg` | tared load per sensor (kg) |
| `total_kg` | sum of the four sensors (kg; multiply by 9.80665 for N) |
| `cop_x_board`, `cop_y_board` | COP in the board frame (m); empty below *COP min. load* |
| `cop_x_world`, `cop_y_world`, `cop_z_world` | COP in the world frame (m); empty if the board is not registered |

**`pose3d.csv` columns**

| Column | Meaning |
|---|---|
| `t`, `t_rel`, `t_unix` | capture time of the frame(s) used for this pose (perf_counter, since start, Unix; s) |
| `<name>_x`, `<name>_y`, `<name>_z`, `<name>_score` | for every keypoint (e.g. MediaPipe `left_shoulder_x`, or `LShoulder_x` from a Halpe-26 plugin): world position (m) and confidence; empty if missing. The columns are fixed by the first pose of the recording; later poses are stored by keypoint name. |
| `com_x`, `com_y`, `com_z` | whole-body COM in the world frame (m) |
| `com_x_board`, `com_y_board`, `com_z_board` | COM in the board frame (m) |
| `mode` | how the 3D keypoints were obtained: `triangulated` (two or more cameras with extrinsics), `single_view_3d` (one camera, the backend's 3D skeleton placed with PnP) or `2d_only` (no 3D possible, e.g. one camera and a 2D-only backend: keypoints and COM are empty) |
| `reproj_error_px` | mean reprojection error of the 3D keypoints in the cameras used (pixels) |

**`pose2d_<camera>.csv` columns** (one row per processed frame of that camera; a frame in which
no subject was found has empty keypoint values)

| Column | Meaning |
|---|---|
| `t`, `t_rel`, `t_unix` | capture time of that camera frame (as in `camN_timestamps.csv`) |
| `frame` | the frame number in `camN.mkv` (the `frame` column of `camN_timestamps.csv`); empty if the frame is not in the video (e.g. captured just before the recording started) |
| `<name>_x`, `<name>_y`, `<name>_score` | pixel coordinates in the original image and confidence (0-1) of every keypoint of the backend's format |

**OpenPose JSON** (`pose2d_json/<camera>/<camera>_<frame:012d>_keypoints.json`, `frame` = video
frame number): `{"version": 1.3, "people": [{"person_id": [-1], "pose_keypoints_2d": [x1, y1,
c1, x2, ...], "face_keypoints_2d": [], "hand_left_keypoints_2d": [], "hand_right_keypoints_2d":
[], "pose_keypoints_3d": [], ...}]}` with all keypoints of the format in `pose_keypoints_2d`
(missing = `0, 0, 0`), and an empty `people` list when no subject was found. The files of all
cameras are written together for every processed multi-camera frame set (a set with a frame that
is not in the videos is skipped in all cameras), so the n-th file of each camera folder belongs
to the same instant, as Pose2Sim expects. Use the `pose2sim_model` of `session.json` as Pose2Sim's
`pose_model` and export the calibration with **File -> Export Pose2Sim Calib.toml...**.

**`fused.csv` columns**: `t`, `t_rel`, `t_unix`, `TR_kg`, `BR_kg`, `TL_kg`, `BL_kg`, `total_kg`,
`cop_x_board`, `cop_y_board`, `cop_x_world`, `cop_y_world`, `cop_z_world`, `com_x`, `com_y`,
`com_z`, `com_x_board`, `com_y_board`, `com_z_board`, `com_minus_cop_x`, `com_minus_cop_y`.
Force values are empty where the nearest Wii sample is more than 0.1 s away (`t_unix` is
empty for sessions recorded before PoseBoard stored the wall clock).

**`summary.json`**

* `session`, `t0`, `t0_unix`, `start_time_iso`, `has_wii`, `has_pose`
* `events_marked` (rows in `events.csv`), `force_events_detected` (see `--events`)
* `mean_total_kg`, `wii_rate_hz`, `pose_rate_hz`
* `cop` and `com`: sway metrics in the board frame (ML = x, AP = y), each with
  `duration_s`, `samples`, `mean_x_mm`, `mean_y_mm`, `range_ml_mm`, `range_ap_mm`,
  `rms_ml_mm`, `rms_ap_mm`, `path_length_mm`, `mean_velocity_mm_s` and
  `ellipse95_area_mm2` (95 % confidence ellipse, `π · 5.991 · sqrt(λ1 λ2)`)
* `com_cop_distance_rms_mm`: RMS horizontal distance between COM and COP

---

## Post-processing CLI

```bat
python -m poseboard.analysis <session folder> [--events] [--external FILE.trc|FILE.csv] [--time-base rel|unix] [--offset SECONDS] [--pose2sim-yup] [--trc]
```

| Option | Effect |
|---|---|
| *(none)* | Recompute `fused.csv` and `summary.json` and print the summary. |
| `--events` | Detect step-on/step-off, jumps (take-off and landing) and stomps in `wii.csv`, print them with `t_rel`, `t_unix` and local time, and write `events_detected.csv` (see [Aligning](#aligning-with-other-recordings)). |
| `--external FILE` | Fuse an external 3D pose file (TRC or CSV) with this session's Wii data -> `fused_<FILE stem>.csv` (force columns are empty if the session has no `wii.csv`). |
| `--time-base rel` | (default) The external time column counts seconds from the recording start: its time 0 is `t0 + offset`. |
| `--time-base unix` | The external times are wall-clock Unix seconds (for a CSV, a `t_unix` column is used when present, otherwise the time column); they are mapped with the session's `clock_offset_unix`. Needs a session recorded with this version. |
| `--offset S` | Time offset (s) added to the external times. With `rel`: the external file's time 0 corresponds to `t0 + S` (use `0` when the file's time column is already `t_rel`). With `unix`: a known clock difference, e.g. between two computers. |
| `--pose2sim-yup` | The external TRC is in Pose2Sim's Y-up convention (Pose2Sim writes `(X', Y', Z') = (Y, Z, X)`); convert it back to the checkerboard Z-up frame. |
| `--trc` | Export `pose3d.csv` as `pose3d.trc` (mm, time = `t_rel`, same Z-up world frame; rotate to Y-up yourself if a tool such as OpenSim expects it). |

External file formats:

* **TRC** (Pose2Sim/OpenSim): units taken from the header (`mm`, `cm` or `m`), marker names
  from the 4th line, `Time` column in seconds.
* **CSV**: a time column in seconds, named `t`, `time`, `timestamp`, `time_s` or `t_rel`
  (case and a unit in brackets are ignored, e.g. `Time (s)`), and `<name>_x`, `<name>_y`,
  `<name>_z` columns in meters. A `t_unix` column (Unix seconds) is used with
  `--time-base unix`; a CSV whose only time column is `t_unix` works with `--time-base unix`
  and is refused with `rel`. Missing values may be empty or `nan`.
* In TRC files, missing markers may be empty fields (as written by Pose2Sim/pandas);
  every marker keeps its column.

From Python you can also pass an arbitrary 4x4 transform (external frame -> PoseBoard world):

```python
import numpy as np
from poseboard.analysis import fuse_external

T = np.eye(4)  # replace with your transform
fuse_external("recordings/20260924_153000_S01", "poseassess.trc", offset_s=0.0, world_transform=T)
```

---

## Aligning with other recordings

Inside a PoseBoard session all streams (Wii, video frames, pose, markers) are already on one
clock. To combine a session with data from **another program or device** (PoseAssess running
on its own, the original Wii app, a motion capture system, EMG, a phone video, ...) use one or
more of the methods below. Combining two (e.g. wall clock + one sync jump) lets you check the
result.

### 1. Wall clock (`t_unix`): easiest when the other data has absolute timestamps

Every row in every PoseBoard file has `t_unix` (Unix seconds), and `session.json` has
`t0_unix`, `clock_offset_unix` and `start_time_iso`. If the other program also logs
wall-clock time, both recordings can be put on one axis directly:

* **Same computer**: both read the same system clock; what remains is the other program's own
  timestamping delay plus about 1 ms.
* **Two computers**: synchronize both clocks right before the session (Windows: *Settings ->
  Time & language -> Date & time -> Sync now*, or `w32tm /resync` as administrator). The
  remaining difference is typically 1-50 ms; measure it once with a sync jump (method 2) and
  pass it as `--offset`.

Fuse an external 3D pose file whose times are Unix seconds:

```bat
python -m poseboard.analysis recordings\20260924_153000_S01 --external poseassess.csv --time-base unix
```

In your own scripts: PoseBoard time `t = t_unix_other - clock_offset_unix`, and
`t_rel = t_unix_other - t0_unix`. A local time string converts to Unix seconds with
`datetime.fromisoformat("2026-09-24T15:30:00.123+08:00").timestamp()` in Python.

### 2. A sync movement seen by both systems (`--events`)

A small **jump** on the board (or a firm **stomp** / heel drop) at the start, and ideally
again at the end, of a trial is easy to find in both the force signal and the video/pose (the
feet leave the floor, the COM rises and falls). PoseBoard finds these events in `wii.csv`:

```bat
python -m poseboard.analysis recordings\20260924_153000_S01 --events
```

It prints the events with `t_rel`, `t_unix` and local time and writes `events_detected.csv`
(`type`, `t`, `t_rel`, `t_unix`, `flight_s`, `peak_kg`):

| Type | Detected when |
|---|---|
| `step_on` / `step_off` | the total load crosses 50 % of body weight (with hysteresis) |
| `takeoff` / `landing` | a jump: the load drops below 5 kg for at most 1 s between two standing phases (`flight_s` = flight time, `peak_kg` = landing impact) |
| `stomp` | a sharp peak above 1.5 x body weight that is not part of a jump |

Times are interpolated between samples; with the board's 60-100 Hz they are typically within
10-20 ms of the true instant. Take-off and landing are the sharpest markers. Find the same
events in the other recording (e.g. the video frame in which the feet touch down) and compute
the offset; with several events, `estimate_time_offset` pairs them up robustly:

```python
from poseboard.analysis import detect_force_events, estimate_time_offset

events = detect_force_events("recordings/20260924_153000_S01")
wii = [e["t_rel"] for e in events if e["type"] == "landing"]
other = [12.84, 45.10, 80.37]  # the same landings in the other recording (its own seconds)
offset, n_matched = estimate_time_offset(wii, other)  # wii t_rel = other time + offset
```

Then fuse with `--external FILE --offset <offset>` (default `--time-base rel`).

### 3. Event markers (`events.csv`)

Press **Mark Event** / **F9** in the GUI, or type a label and press Enter in the Wii-only
recorder, when you start the other system or give a cue (`start`, `eyes closed`, ...). Each
marker has `t`, `t_rel`, `t_unix` and its label. Markers are as precise as the operator's
key press (roughly 0.1-0.3 s), so use them to label trial phases and for coarse alignment, and
a sync jump for precise alignment.

### Using PoseBoard together with the original Wii app

Only one program can read the Balance Board at a time. Either record the board in PoseBoard
(GUI, or `python -m poseboard.wii.record` for a Wii-only session: `wii.csv` has the four
sensors in kg, the total and the COP, plus the three time columns), or let the other app
record the board and run PoseBoard with cameras only; then align the two with the wall clock
(if the other app logs timestamps) or with a sync jump (visible in the other app's force data
and in PoseBoard's video/pose).

### Comparing with WiimoteLib-based apps

Many Windows Balance Board programs are built on **WiimoteLib**, whose conventions differ from
PoseBoard's:

| | WiimoteLib | PoseBoard (`wii.csv`) |
|---|---|---|
| COP units | cm | m |
| COP y (antero-posterior) | `CenterOfGravity.Y`, **+ toward the power button** (BL/BR) | `cop_y_board`, + toward TL/TR (front) |
| COP x half-span | 21 cm (integer division of 43/2) | 216.5 mm (433/2) |
| Per-sensor load | `SensorValuesKg.*` = 4 x the load on that sensor | `TR_kg` ... = load on that sensor |

Conversion of a WiimoteLib export to PoseBoard's convention:

```
cop_x_board = CenterOfGravity.X / 100 * (216.5 / 210)
cop_y_board = -CenterOfGravity.Y / 100 * (238 / 240)
TR_kg = SensorValuesKg.TopRight / 4      (same for BR, TL, BL)
```

Otherwise the anterior-posterior sway appears inverted and about 3 % smaller in x. (This
applies only if the other app uses WiimoteLib; check its documentation.)

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
   **Dependencies:** the plugin runs inside PoseBoard's Python, so `POSEASSESS_DIR` only makes
   PoseAssess's own code importable; its third-party packages (PyTorch, ONNX Runtime, mmpose,
   ...) must be installed in PoseBoard's environment too. Keep a **single OpenCV package**:
   PoseBoard uses `opencv-contrib-python` (needed by MediaPipe), and a second `opencv-python`
   breaks `cv2`. Install PoseAssess's requirements without their dependencies and add the
   missing packages by hand, e.g.
   `.venv\Scripts\python -m pip install --no-deps -r <PoseAssess>\requirements.txt`, then
   `.venv\Scripts\python -m pip install <each missing package>` (leave out `opencv-python`).
   The same applies the other way round: `pip install -e <PoseBoard>` inside PoseAssess's
   environment would pull in `opencv-contrib-python` and MediaPipe; use `--no-deps` there. The
   packaged PoseBoard.exe cannot load extra packages: use the source install for such plugins.
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
2. **Record with PoseBoard** (cameras + Wii). PoseBoard saves `camN.mkv` and
   `camN_timestamps.csv`.
3. **Run PoseAssess on PoseBoard's videos.** This is the easiest way to get exact timing: the
   time of frame *i* is `t_rel` in row *i* of `camN_timestamps.csv` (= `t` minus `t0` from
   `session.json`; there is one row per frame in the video). Write that as the TRC `Time`
   column (or the CSV `t` column) and use `--offset 0`. Do not use
   `frame / fps`, because webcams drop frames and their fps is not exact.
   If PoseAssess records its own videos at the same time instead, the clocks differ. If
   PoseAssess writes wall-clock timestamps, use them directly (`--time-base unix`, see
   [Aligning](#aligning-with-other-recordings)). Otherwise start the trial with a sync event
   (e.g. a small jump or a heel drop, visible both in `total_kg` and as a vertical jump of the
   ankles/COM), find it with `--events`, measure the time difference, and pass it as
   `--offset`.
4. Fuse:

   ```bat
   python -m poseboard.analysis recordings\20260924_153000_S01 --external poseassess.trc --offset 0
   ```

   Add `--pose2sim-yup` if the TRC was written by Pose2Sim (Y-up). The result is
   `fused_poseassess.csv` with the same columns as `fused.csv`.

Keypoint names only need to be recognizable for the COM model (`poseboard/pose/com.py`):
`left_shoulder`, `LShoulder`, `l_shoulder` etc. all work, and Human3.6M / BVH skeletons whose
ankle joint is called `LFoot` / `LeftFoot` are handled. The COM needs the trunk (both
shoulders and both hips) **plus enough other segments to reach 60 % of the body mass**:
shoulders and hips alone are only about 50 %, so add both knees, or the head (ears or nose)
plus one knee or both elbows. Ankles, wrists and heels/toes make it more accurate. Frames with
too few keypoints get empty COM columns; the log says once which segments were missing.

---

## Accuracy: single camera vs. multiple cameras

| | One camera | Two or more calibrated cameras |
|---|---|---|
| Board registration | Clicks intersected with the floor plane (board height above the checkerboard), then position and heading fitted; a few mm with careful clicks. Good if intrinsics are calibrated and the board is seen at an oblique angle (not edge-on). | 5 points triangulated, position and heading fitted flat on the floor. More robust to click errors. |
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
  with a sync event (see [Aligning](#aligning-with-other-recordings)).
* **`t_unix` is only as good as the system clock.** The wall-clock offset is measured once at
  the start (to about 1 ms) and then follows the monotonic clock, so it never jumps. If
  Windows adjusts the system clock during a long recording, `t_unix` and the system clock of
  other programs can drift apart by a few ms; for long sessions confirm the alignment with a
  sync event at the end as well. Between two computers, synchronize the clocks first.
* **Bluetooth drops leave gaps.** Auto-connect reconnects, but `wii.csv` has no samples while
  the board was disconnected (a drop is detected after up to 5 s without data); the gap is
  marked with `wii_disconnected` / `wii_connected` in `events.csv`.
* **Multiple cameras are not hardware-synchronized.** The pose uses the newest frame of each
  camera (and their mean time); a camera whose newest frame is more than about 0.25 s older
  than the others' is left out. This is fine for standing balance; for fast movements expect
  some triangulation error.
* **Pose rate = inference rate.** Live pose runs as fast as the model allows and skips frames;
  the videos contain every frame, so you can re-process them offline (Option B).
* **The board and the cameras must not move** after board registration and extrinsic
  calibration. If they do, recalibrate/re-click.
* **The Balance Board is not a laboratory force plate.** It measures only the vertical load
  (four load cells, no shear forces or moments), has more noise and drift, and is rated for
  about 150 kg. Tare before each session and use fresh batteries.
* The COM model uses generic segment parameters (Winter) and assumes a single subject.
* **Board on the floor.** With the floor checkerboard the board is assumed to lie flat on the
  same floor (tilt removed). For a board on a ramp or platform, register it without the floor
  constraint from Python (`register_board(..., floor=False)`).
* **Not verified on hardware here:** the DirectShow MJPG setting, the console-close handling of
  the Wii-only recorder and the Balance Board extension check follow the OpenCV, Windows and
  Wii documentation but were tested only with simulated devices. If a webcam ignores MJPG, the
  log says so and the status bar shows its low frame rate (try a lower resolution).

---

## Troubleshooting

| Problem | What to try |
|---|---|
| **Scan finds no board / Auto-connect keeps "waiting for a paired board"** | Switch the board on (power button); pair it (see the [daily checklist](#pairing-the-wii-balance-board-on-windows)); press SYNC again; remove and re-pair in Windows; close other Wii programs. |
| **"A paired board is listed but does not answer"** | Windows still lists the board but cannot reach it: it is off, asleep, or the pairing expired (normal after switching it off). Remove it in Windows and pair it again (SYNC); Auto-connect then connects by itself. |
| **"... skipped: not a Balance Board"** | A Wii Remote (or remote with an accessory) is paired: PoseBoard ignores it. Pair the Balance Board itself. |
| **Board connected but recording says "Wii not connected"** | The Streams line is updated live; wait until the Wii panel shows "connected" before starting, or just start: the board is added to the recording as soon as it connects. |
| **"Cannot enumerate HID devices"** | *"hidapi is not installed"*: `python -m pip install hidapi`. *"the installed 'hid' module is not hidapi"*: `python -m pip uninstall hid`, then `python -m pip install --force-reinstall hidapi`. |
| **Connects, then "Error: ..." in the Wii panel** | Board went to sleep or the pairing expired: remove and re-pair, then Connect again. Check the batteries. |
| **Camera cannot be opened** | Another program uses it; try another index; try "Default" resolution. |
| **Red warning "the camera delivers WxH but its calibration is for ..."** | The camera runs at another resolution than it was calibrated at. Remove it (tab 1) and add it again with the Resolution named in the warning; the calibration was kept and applies again. |
| **Low camera frame rate (e.g. 5-10 fps at 720p)** | USB2 bandwidth without MJPG, or low light (long exposure). Use more light, a lower resolution, or another USB port; see the log for "did not accept MJPG". |
| **Checkerboard not detected** | Check cols/rows (inner corners!), lighting, glare, and that the whole board is visible and sharp. |
| **COP dot at the wrong corner** | Run the **Corner Check** (tab 3): it fixes a front/back swap by itself (the subject must then face the edge opposite the power button). Otherwise redo the clicks in the order TL, TR, BR, BL, C (TL/TR = edge opposite the power button, subject's left/right). |
| **"The click order is mirrored"** | Left/right (or front/back) were swapped while clicking: redo the clicks as described. |
| **Large board reprojection error / large tilt of the unconstrained fit** | Click more carefully, calibrate intrinsics, check the board dimensions and height, make sure the checkerboard lay on the same floor as the board, use a second camera. |
| **MediaPipe model download fails** | Download the `.task` file on any computer (URLs in [Installation](#installation-windows)) into the `models` folder named in the error message. |
| **"(no person detected)"** | The whole body should be visible; improve lighting; try "MediaPipe (heavy)". |

---

## Running the tests

```bat
python -m pip install -r requirements-dev.txt
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
│   ├── geometry.py            board model, board registration (PnP / triangulation, floor fit)
│   ├── camera.py              camera threads, MKV video + per-frame timestamp recording
│   ├── fusion.py              COP -> world, COM vs. COP
│   ├── overlay.py             drawing board, COP, skeleton, COM on the video
│   ├── session.py             synchronized recorder (session folder layout, clocks, events)
│   ├── analysis.py            fused.csv, sway metrics, force events, TRC export, external import (CLI)
│   ├── wii/
│   │   ├── protocol.py        Balance Board HID protocol (pure functions)
│   │   ├── device.py          hidapi reader, plug-and-play auto-connect, simulator, tare
│   │   └── record.py          Wii-only command-line recorder (python -m poseboard.wii)
│   ├── pose/
│   │   ├── base.py            Pose2D / Pose3D / PoseEstimator interface
│   │   ├── com.py             center of mass (Winter segment table)
│   │   ├── formats.py         keypoint formats (COCO-17, Halpe-26, BODY_25, COCO-18, WholeBody-133, MediaPipe-33)
│   │   ├── detectors/         2D pose backends: registry (__init__.py), Detector2D/Person2D (base.py), MediaPipe
│   │   ├── multiview.py       any 2D backend -> 3D: triangulation with outlier rejection, single-view lifting
│   │   ├── subject.py         picks the person standing on the board in each camera
│   │   ├── filters.py         One-Euro keypoint smoothing
│   │   ├── mediapipe_backend.py  MediaPipe model files, MediaPipePose (compatibility)
│   │   ├── triangulation.py   multi-camera keypoint triangulation (weighted DLT, robust)
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
