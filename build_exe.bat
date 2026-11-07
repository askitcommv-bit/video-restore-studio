@echo off
:: Builds dist\VideoRestoreStudio\ (VideoRestoreStudio.exe + _internal\) with PyInstaller in an
:: isolated build venv. Copy both into the project folder (next to workflows\ and third_party\)
:: or run it from there: the exe treats its own folder as the project folder.
setlocal
cd /d "%~dp0"

where uv >nul 2>&1 || (echo [!] uv is required: https://docs.astral.sh/uv/ & exit /b 1)
if not exist ".build\venv\Scripts\python.exe" (
    uv venv .build\venv --python 3.10 || exit /b 1
)
uv pip install --python .build\venv\Scripts\python.exe -r requirements-app.txt pyinstaller==6.16.0 || exit /b 1

.build\venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --windowed ^
    --name VideoRestoreStudio --icon "%~dp0app\ui\assets\logo.ico" ^
    --distpath dist --workpath .build\work --specpath .build ^
    --add-data "%~dp0app\ui\assets;app\ui\assets" ^
    --exclude-module torch --exclude-module transformers --exclude-module tkinter ^
    main.py || exit /b 1

echo.
echo Built dist\VideoRestoreStudio\VideoRestoreStudio.exe
