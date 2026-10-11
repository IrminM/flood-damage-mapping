@echo off
setlocal
cd /d "%~dp0"

echo [*] Activating Python virtual environment...
if exist ".venv\Scripts\activate.bat" (
    call .\.venv\Scripts\activate.bat
) else (
    echo [!] No .venv found. Running with system Python...
)

echo [*] Starting FloodIntel FastAPI Backend on http://127.0.0.1:8000...
echo [*] Note: 127.0.0.1 refers to this local computer running the browser.
echo [*] Keep this terminal open while using the dashboard.
echo.

python scripts/run_server.py %*

pause
