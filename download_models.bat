@echo off
title LTX-2.5 Model Downloader
cd /d "%~dp0"

set "PYTHON_EXE=C:\ComfyUI_portable\python_embeded\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo ========================================================
echo  LTX-2.5 Video Restoration - Model Downloader
echo ========================================================
echo.
echo Options:
echo   1. Download INT8 Models + IC-LoRAs (Recommended for 16GB VRAM, ~40GB total)
echo   2. Download Only Restore ^& Refine IC-LoRAs (Public, ~2.8GB)
echo   3. Download Full BF16 Models (~80GB total)
echo   4. Set / Update Hugging Face Access Token
echo   5. Check Model Status Only
echo   6. Download NVIDIA SoL-Refiner Models (1-Step Refiner, ~23GB)
echo.
echo * Note: Lightricks/LTX-2.5 base models are gated.
echo   Accept license: https://huggingface.co/Lightricks/LTX-2.5
echo   Get free token: https://huggingface.co/settings/tokens
echo.
set /p choice="Enter choice (1-6): "

if "%choice%"=="1" (
    echo Downloading INT8 model stack directly to ComfyUI models...
    "%PYTHON_EXE%" restore_cli.py download
) else if "%choice%"=="2" (
    echo Downloading IC-LoRAs only...
    "%PYTHON_EXE%" restore_cli.py download --loras-only
) else if "%choice%"=="3" (
    echo Downloading BF16 model stack...
    "%PYTHON_EXE%" restore_cli.py download --bf16
) else if "%choice%"=="4" (
    "%PYTHON_EXE%" restore_cli.py token
) else if "%choice%"=="6" (
    echo Downloading NVIDIA SoL-Refiner models...
    "%PYTHON_EXE%" restore_cli.py download --sol
) else (
    "%PYTHON_EXE%" restore_cli.py status
)

pause
