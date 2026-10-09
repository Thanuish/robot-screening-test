@echo off
rem One-time setup on Windows: creates .venv and installs the Python packages.
rem Needs Python 3.12 from https://www.python.org/downloads/ (tick "Add python.exe to PATH").
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3.12) || (set PY=python)
%PY% --version || (echo Python not found. Install Python 3.12 from python.org first. & pause & exit /b 1)
if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment...
    %PY% -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || (echo Package install failed & pause & exit /b 1)
if not exist assets\universal_robots_ur5e\ur5e.xml .venv\Scripts\python setup_assets.py
echo.
echo Setup done. Start the screening test with TEST_TELEOP\start_screening.bat
pause
