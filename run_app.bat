@echo off
REM Starts the KCC-Sahayak demo and opens it in your browser.
REM First start takes ~30-60 s while the models load.
cd /d "%~dp0"

REM make ffmpeg (needed for microphone input) visible if it was installed with winget
for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*") do (
  for /d %%E in ("%%D\ffmpeg-*") do set "PATH=%PATH%;%%E\bin"
)

start "" cmd /c "timeout /t 45 >nul & start http://localhost:8420"
".venv\Scripts\python.exe" -m uvicorn app:app --app-dir backend --port 8420
pause
