@echo off
REM Starts the KCC-Sahayak demo and opens it in your browser.
REM First start takes ~30-60 s while the models load.
cd /d "%~dp0"

REM make ffmpeg (needed for microphone input) visible if it was installed with winget
for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*") do (
  for /d %%E in ("%%D\ffmpeg-*") do set "PATH=%PATH%;%%E\bin"
)

if not exist ".venv\Scripts\python.exe" (
  echo No .venv found. Run:  python -m venv .venv ^&^& .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
REM first run on a fresh clone: rebuilds the two big retrieval files (10-20 min); instant afterwards
".venv\Scripts\python.exe" scripts\setup_models.py

start "" cmd /c "timeout /t 45 >nul & start http://localhost:8420"
".venv\Scripts\python.exe" -m uvicorn app:app --app-dir backend --port 8420
pause
