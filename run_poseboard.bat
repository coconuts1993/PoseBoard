@echo off
REM PoseBoard launcher for Windows. A project file can be passed,
REM e.g.  run_poseboard.bat my_project.json  (relative to the folder you run it from)

REM Make the project path absolute before changing to the PoseBoard folder
set "POSEBOARD_PROJECT="
if not "%~1"=="" set "POSEBOARD_PROJECT=%~f1"
cd /d "%~dp0"

REM Use the local virtual environment if there is one (no activation needed).
set "PY=python"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if defined POSEBOARD_PROJECT (
    "%PY%" -m poseboard "%POSEBOARD_PROJECT%"
) else (
    "%PY%" -m poseboard
)
if errorlevel 1 (
    echo.
    echo PoseBoard exited with an error. See the messages above.
    echo If Python or a package is missing, run:  "%PY%" -m pip install -r requirements.txt
    pause
)
