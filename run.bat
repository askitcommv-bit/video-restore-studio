@echo off
setlocal enabledelayedexpansion
title Video Restore Studio
cd /d "%~dp0"

echo ====================================================================
echo                        Video Restore Studio
echo ====================================================================
echo.

:: 1. Setup Python Environment & Dependencies
set "DEP_DIR=%USERPROFILE%\Documents\VisoMaster\dependencies"
if exist "%DEP_DIR%\Python\python.exe" (
    set "PATH=%DEP_DIR%;%DEP_DIR%\CUDA\bin;%DEP_DIR%\TensorRt\lib;%PATH%"
    set "PYTHON_EXE=%DEP_DIR%\Python\python.exe"
    echo [OK] Using VisoMaster environment: %DEP_DIR%\Python\python.exe
) else if exist "C:\ComfyUI_portable\python_embeded\python.exe" (
    set "PYTHON_EXE=C:\ComfyUI_portable\python_embeded\python.exe"
    echo [OK] Using ComfyUI environment: C:\ComfyUI_portable\python_embeded\python.exe
) else (
    set "PYTHON_EXE=python"
    echo [!] Falling back to system python.
)

:: 2. Check ComfyUI Status
echo [..] Checking ComfyUI server status (127.0.0.1:8188)...
powershell -NoProfile -Command "$client = New-Object System.Net.Sockets.TcpClient; try { $client.Connect('127.0.0.1', 8188); exit 0 } catch { exit 1 } finally { $client.Close() }" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] ComfyUI server is online and ready!
) else (
    echo [!] ComfyUI is not currently responding on port 8188.
    if exist "C:\ComfyUI_portable\run_nvidia_gpu.bat" (
        echo [..] Starting ComfyUI server in the background...
        start "ComfyUI Server" /min "C:\ComfyUI_portable\run_nvidia_gpu.bat"
        echo [..] Waiting 5 seconds for ComfyUI to initialize...
        timeout /t 5 /nobreak >nul
    ) else (
        echo [!] Note: Please ensure ComfyUI is started on port 8188 for AI rendering.
    )
)

echo.
echo Launching Video Restore Studio...
echo ====================================================================
echo.

:: 3. Launch LTX-2.5 Video Restoration Studio
"%PYTHON_EXE%" main.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ====================================================================
    echo [!] Application exited with an error code: %ERRORLEVEL%
    echo Press any key to close this terminal...
    pause >nul
)
