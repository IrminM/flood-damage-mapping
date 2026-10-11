@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==============================================================
echo  FloodIntel: Satellite Flood Damage & Connectivity Intelligence
echo ==============================================================
echo.

:: 1. Verify Python installation
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in your system PATH.
    echo Please install Python 3.10 or higher from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: 2. Check or create virtual environment
if not exist ".venv" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [*] Installing dependencies from requirements.txt...
    call .\.venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Some dependencies failed to install. Retrying...
        pip install -r requirements.txt
    )
) else (
    call .\.venv\Scripts\activate.bat
)

:: 3. Open dashboard in default web browser after short delay
echo.
echo [*] Starting FloodIntel FastAPI Backend on http://127.0.0.1:8000...
echo [*] Note: 127.0.0.1 is this local computer running the browser.
echo [*] Keep this Command Prompt window open while using the dashboard.
echo.

start "" http://127.0.0.1:8000/floodintel/

:: 4. Run backend server
python scripts/run_server.py %*

pause
