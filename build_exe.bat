@echo off
:: Builds dist\VideoRestoreStudio-<version>.exe (version from version.py), a single-file build
:: (PyInstaller --onefile), in an isolated build venv. Run it from the project folder or any subfolder of
:: it (dist\ works): it uses the nearest folder at or above its own that holds relayvsr_worker.py as the
:: project folder. Anywhere else it uses %%LOCALAPPDATA%%\VideoRestoreStudio and offers the setup.
setlocal
cd /d "%~dp0"

where uv >nul 2>&1 || (echo [!] uv is required: https://docs.astral.sh/uv/ & exit /b 1)
if not exist ".build\venv\Scripts\python.exe" (
    uv venv .build\venv --python 3.10 || exit /b 1
)
uv pip install --python .build\venv\Scripts\python.exe -r requirements-app.txt pyinstaller==6.16.0 || exit /b 1

for /f %%v in ('.build\venv\Scripts\python.exe -c "from version import __version__; print(__version__)"') do set "VERSION=%%v"
if not defined VERSION (echo [!] Could not read version.py & exit /b 1)
set "NAME=VideoRestoreStudio-%VERSION%"

.build\venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --windowed --onefile ^
    --name %NAME% --icon "%~dp0app\ui\assets\logo.ico" ^
    --distpath dist --workpath .build\work --specpath .build ^
    --add-data "%~dp0app\ui\assets;app\ui\assets" ^
    --add-data "%~dp0relayvsr_worker.py;." --add-data "%~dp0relayvsr_requirements.txt;." ^
    --exclude-module torch --exclude-module transformers --exclude-module tkinter ^
    main.py || exit /b 1
:: Older builds (the folder build, unversioned or other versions) would sit next to the new exe and
:: confuse which one to run
if exist dist\VideoRestoreStudio rmdir /s /q dist\VideoRestoreStudio
for %%f in (dist\VideoRestoreStudio*.exe) do if /i not "%%~nxf"=="%NAME%.exe" del "%%f"

echo.
echo Built dist\%NAME%.exe
