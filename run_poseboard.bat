@echo off
REM PoseBoard launcher for Windows. Extra arguments are passed on,
REM e.g.  run_poseboard.bat my_project.json
cd /d "%~dp0"

REM Use the local virtual environment if there is one.
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"

python -m poseboard %*
if errorlevel 1 (
    echo.
    echo PoseBoard exited with an error. See the messages above.
    echo If Python or a package is missing, run:  pip install -r requirements.txt
    pause
)
