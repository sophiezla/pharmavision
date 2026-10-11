@echo off
setlocal
title PharmaChecked GPU kit - step 2 of 2 (run)
set "KIT=%~dp0"
cd /d "%KIT%pharmavision" || goto fail
if not exist .venv\Scripts\python.exe (
  echo Run 1_SETUP.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe scripts\kit_run.py --kit "%KIT%." || goto fail
pause
exit /b 0

:fail
echo.
echo ============================================================
echo  The run stopped early. Double-click 2_RUN.bat again: finished
echo  runs are kept and the rest continue. If it stops again, take a
echo  photo of this window and bring it back with any RESULTS zip.
echo ============================================================
pause
exit /b 1
