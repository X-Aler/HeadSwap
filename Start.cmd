@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\.dependencies-ready" (
  echo First run Install.cmd, then open Start.cmd again.
  pause
  exit /b 1
)
if "%~1"=="" (
  start "HeadSwap" ".venv\Scripts\pythonw.exe" "app.py"
) else (
  start "HeadSwap" ".venv\Scripts\pythonw.exe" "app.py" --photo "%~1"
)
