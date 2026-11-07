@echo off
title Environment Diagnostic Check
cd /d "%~dp0"
"C:\ComfyUI_portable\python_embeded\python.exe" restore_cli.py status
pause
