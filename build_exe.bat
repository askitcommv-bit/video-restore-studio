@echo off
:: Builds dist\VideoRestoreStudio.exe, a single-file build (PyInstaller --onefile), in an isolated
:: build venv. Run it from the project folder or any subfolder of it (dist\ works): it uses the nearest
:: folder at or above its own that holds relayvsr_worker.py as the project folder.
setlocal
cd /d "%~dp0"

where uv >nul 2>&1 || (echo [!] uv is required: https://docs.astral.sh/uv/ & exit /b 1)
if not exist ".build\venv\Scripts\python.exe" (
    uv venv .build\venv --python 3.10 || exit /b 1
)
uv pip install --python .build\venv\Scripts\python.exe -r requirements-app.txt pyinstaller==6.16.0 || exit /b 1

.build\venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --windowed --onefile ^
    --name VideoRestoreStudio --icon "%~dp0app\ui\assets\logo.ico" ^
    --distpath dist --workpath .build\work --specpath .build ^
    --add-data "%~dp0app\ui\assets;app\ui\assets" ^
    --exclude-module torch --exclude-module transformers --exclude-module tkinter ^
    main.py || exit /b 1
:: The old folder build, if any, would sit next to the new exe and confuse which one to run
if exist dist\VideoRestoreStudio rmdir /s /q dist\VideoRestoreStudio

echo.
echo Built dist\VideoRestoreStudio.exe
