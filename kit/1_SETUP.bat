@echo off
setlocal
title PharmaChecked GPU kit - step 1 of 2 (setup)
set "KIT=%~dp0"
set "REPO=%KIT%pharmavision"
set "UV=%KIT%tools\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%KIT%tools\python"
set "UV_CACHE_DIR=%KIT%tools\uv-cache"
set "UV_PYTHON_PREFERENCE=only-managed"
cd /d "%REPO%" || goto fail

echo ============================================================
echo  Step 1 of 2: installing Python and the packages.
echo  This needs internet and takes about 10-30 minutes.
echo  You can leave it running.
echo ============================================================
echo.
"%UV%" venv --python 3.13 .venv || goto fail
"%UV%" pip install --python .venv\Scripts\python.exe --index-url __TORCH_INDEX__ -c constraints\cpu.txt torch torchvision || goto fail
"%UV%" pip install --python .venv\Scripts\python.exe -c constraints\cpu.txt -e .[anomaly] || goto fail
.venv\Scripts\python.exe scripts\kit_check.py || goto fail

echo.
echo ============================================================
echo  Setup finished. Now double-click 2_RUN.bat
echo ============================================================
pause
exit /b 0

:fail
echo.
echo ============================================================
echo  SETUP DID NOT FINISH. Take a photo of this window and bring
echo  it back to the laptop. Nothing else needs to be done here.
echo ============================================================
pause
exit /b 1
