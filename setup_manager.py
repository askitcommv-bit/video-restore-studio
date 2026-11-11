"""
In-app setup: what each engine needs, whether it is installed, and installing it.

Required for the default engine (RelayVSR), on a PC that has nothing but the exe:
  FFmpeg          tools/ffmpeg/bin            reads and writes video
  RelayVSR engine third_party/RelayVSR(/.venv) pinned code + Python 3.13 + torch 2.10 (CUDA 12.8);
                                              uv (tools/uv) builds it and fetches Python itself
  RelayVSR models config.RELAYVSR_WEIGHTS_DIR wan + flash weights (public)
Optional, offered when ComfyUI is installed (comfy_models): the LTX-2.5 and SoL-Refiner packs.

Each part is skipped when present, downloads resume after an interruption, and everything goes
into the project folder (config.PROJECT_DIR).
"""

import filecmp
import os
import shutil
import subprocess
import threading
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import comfy_models
import config
import downloads
import gpu
import processes
from downloads import Cancelled, Progress, check_cancel  # noqa: F401  (Cancelled is part of this API)

# gyan.dev's essentials build (NVENC, QSV and AMF encoders) from its GitHub mirror: gyan.dev itself
# served it at ~0.1 MB/s in testing
FFMPEG_VERSION = "9.0.2"
FFMPEG_URL = (f"https://github.com/GyanD/codexffmpeg/releases/download/{FFMPEG_VERSION}/"
              f"ffmpeg-{FFMPEG_VERSION}-essentials_build.zip")
UV_VERSION = "0.12.20"
UV_URL = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-pc-windows-msvc.zip"
RELAYVSR_COMMIT = "e856773350f53036d825404c9ea9228e6dffadca"
RELAYVSR_ZIP_URL = f"https://github.com/kopperx/RelayVSR/archive/{RELAYVSR_COMMIT}.zip"
PYTHON_VERSION = "3.13"
TORCH_SPEC = "torch==2.10.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
REQUIREMENTS_FILE = "relayvsr_requirements.txt"
WEIGHTS = {"flash.safetensors": 127_759_784, "wan.safetensors": 10_066_399_024}
ENV_PACKAGES = ("torch", "diffusers", "accelerate", "einops", "av", "safetensors")


@dataclass
class Component:
    key: str
    name: str
    detail: str
    download_gb: float
    disk_gb: float  # peak space needed while installing
    needs_gpu: bool
    optional: bool  # offered, not needed for the default engine
    is_ready: Callable[[], bool]
    install: Callable[[Progress, Optional[threading.Event]], None]

    @property
    def ready(self) -> bool:
        return self.is_ready()


# ---------------------------------------------------------------------------
# What is installed
# ---------------------------------------------------------------------------
def ffmpeg_ready() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def engine_ready() -> bool:
    site = config.RELAYVSR_DIR / ".venv" / "Lib" / "site-packages"
    return ((config.RELAYVSR_DIR / "refvsr_inference").is_dir() and config.RELAYVSR_WORKER.is_file()
            and config.RELAYVSR_PYTHON.is_file() and all((site / p).is_dir() for p in ENV_PACKAGES))


def weights_ready() -> bool:
    return all((config.RELAYVSR_WEIGHTS_DIR / f).is_file() and (config.RELAYVSR_WEIGHTS_DIR / f).stat().st_size == n
               for f, n in WEIGHTS.items())


def relayvsr_missing() -> List[str]:
    """What RelayVSR still needs, for people (empty when it can run)."""
    return [name for name, ok in (("FFmpeg", ffmpeg_ready()), ("RelayVSR engine", engine_ready()),
                                  ("RelayVSR models", weights_ready())) if not ok]


def relayvsr_ready() -> bool:
    return not relayvsr_missing()


def comfyui_installed() -> bool:
    return config.COMFYUI_DIR.is_dir()


def components() -> List[Component]:
    parts = [
        Component("ffmpeg", "FFmpeg", "Reads and writes video", 0.11, 0.4, False, False, ffmpeg_ready, install_ffmpeg),
        Component("engine", "RelayVSR engine", "Python, PyTorch with CUDA and the RelayVSR code",
                  3.4, 9.0, True, False, engine_ready, install_engine),
        Component("weights", "RelayVSR models", "The upscaling model weights", 10.2, 10.3, True, False,
                  weights_ready, install_weights),
    ]
    if comfyui_installed():
        for key, name, keys in (("ltx", "LTX-2.5 restoration models", comfy_models.LTX_PACK),
                                ("sol", "SoL-Refiner models", comfy_models.SOL_PACK)):
            gb = comfy_models.missing_gb(keys) or sum(comfy_models.MODELS[k].size_gb for k in keys)
            parts.append(Component(
                key, name, "For the ComfyUI engines; needs a Hugging Face token", gb, gb, True, True,
                lambda keys=keys: not comfy_models.missing(keys),
                lambda progress, cancel, keys=keys: comfy_models.download_many(keys, progress, cancel)))
    return parts


def needs_setup() -> bool:
    """True while anything the default engine needs is missing."""
    return not all(c.ready for c in components() if not c.optional)


def free_gb() -> float:
    return shutil.disk_usage(config.PROJECT_DIR).free / 1e9


def sync_worker() -> None:
    """Copies the worker script bundled with the app into a standalone project folder when it is
    missing or out of date there, so an updated exe brings its worker along."""
    bundled = config.BUNDLE_DIR / "relayvsr_worker.py"
    target = config.RELAYVSR_WORKER
    if not config.STANDALONE or not bundled.is_file() or bundled.resolve() == target.resolve():
        return
    if not target.is_file() or not filecmp.cmp(bundled, target, shallow=False):
        shutil.copyfile(bundled, target)


# ---------------------------------------------------------------------------
# Installing
# ---------------------------------------------------------------------------
def _downloads_dir() -> Path:
    d = config.TOOLS_DIR / "_downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _add_to_path(directory: Path) -> None:
    if str(directory) not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")


def _extract(zip_path: Path, names: tuple, dest_dir: Path) -> None:
    """Extracts the members whose file names are in `names` (any folder) into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            base = member.rsplit("/", 1)[-1]
            if base in names:
                with zf.open(member) as src, open(dest_dir / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def _run(cmd: list, label: str, progress: Progress, cancel: Optional[threading.Event], env: dict) -> None:
    """Runs a command, showing its latest output line as the step text; stops it on cancel."""
    proc = processes.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                           text=True, encoding="utf-8", errors="replace")
    tail: deque = deque(maxlen=20)

    def pump():
        for line in proc.stdout:
            line = line.strip()
            if line:
                tail.append(line)
                progress(f"{label} · {line[:90]}", 0, 0)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    while proc.poll() is None:
        if cancel is not None and cancel.is_set():
            proc.terminate()
            proc.wait(10)
            raise Cancelled()
        time.sleep(0.2)
    reader.join(5)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {proc.returncode}):\n" + "\n".join(tail))


def install_ffmpeg(progress: Progress, cancel: Optional[threading.Event] = None) -> None:
    z = downloads.download(FFMPEG_URL, _downloads_dir() / "ffmpeg.zip", "Downloading FFmpeg", progress, cancel)
    progress("Unpacking FFmpeg", 0, 0)
    _extract(z, ("ffmpeg.exe", "ffprobe.exe"), config.FFMPEG_BIN_DIR)
    if not (config.FFMPEG_BIN_DIR / "ffprobe.exe").is_file():
        raise RuntimeError("The FFmpeg download didn't contain ffmpeg.exe and ffprobe.exe")
    z.unlink(missing_ok=True)
    _add_to_path(config.FFMPEG_BIN_DIR)


def _uv(progress: Progress, cancel: Optional[threading.Event]) -> tuple:
    """Returns (uv executable, whether it is the setup's own copy). Downloads uv if it isn't installed."""
    found = shutil.which("uv")
    if found and Path(found).resolve() != config.UV_EXE.resolve():
        return found, False
    if not config.UV_EXE.is_file():
        z = downloads.download(UV_URL, _downloads_dir() / "uv.zip", "Downloading uv", progress, cancel)
        _extract(z, ("uv.exe",), config.UV_EXE.parent)
        z.unlink(missing_ok=True)
    return str(config.UV_EXE), True


def install_engine(progress: Progress, cancel: Optional[threading.Event] = None) -> None:
    # RelayVSR code at the tested commit
    if not (config.RELAYVSR_DIR / "refvsr_inference").is_dir():
        z = downloads.download(RELAYVSR_ZIP_URL, _downloads_dir() / "relayvsr.zip", "Downloading RelayVSR code",
                               progress, cancel)
        progress("Unpacking RelayVSR code", 0, 0)
        unpack = _downloads_dir() / "relayvsr"
        shutil.rmtree(unpack, ignore_errors=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(unpack)
        shutil.copytree(next(unpack.iterdir()), config.RELAYVSR_DIR, dirs_exist_ok=True)  # RelayVSR-<commit>/
        shutil.rmtree(unpack, ignore_errors=True)
        z.unlink(missing_ok=True)
    sync_worker()
    check_cancel(cancel)

    # Python environment: uv fetches Python itself, so the PC needs neither Python nor git
    uv, own_uv = _uv(progress, cancel)
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(config.TOOLS_DIR / "python"),
               UV_PYTHON_PREFERENCE="only-managed", UV_LINK_MODE="copy")
    cache = config.TOOLS_DIR / "uv-cache"
    if own_uv:
        env["UV_CACHE_DIR"] = str(cache)  # removed afterwards: it would keep a second copy of torch
    py = str(config.RELAYVSR_PYTHON)
    if not config.RELAYVSR_PYTHON.is_file():
        _run([uv, "venv", "--python", PYTHON_VERSION, str(config.RELAYVSR_DIR / ".venv")],
             "Setting up Python", progress, cancel, env)
    _run([uv, "pip", "install", "--python", py, TORCH_SPEC, "--index-url", TORCH_INDEX],
         "Installing PyTorch with CUDA (about 3 GB)", progress, cancel, env)
    _run([uv, "pip", "install", "--python", py, "-r", str(config.BUNDLE_DIR / REQUIREMENTS_FILE)],
         "Installing RelayVSR packages", progress, cancel, env)
    if own_uv:
        shutil.rmtree(cache, ignore_errors=True)
    if not engine_ready():
        raise RuntimeError("The RelayVSR environment is incomplete after installing; try Install again.")


def install_weights(progress: Progress, cancel: Optional[threading.Event] = None) -> None:
    repo = config.HF_REPOS["relayvsr"]
    for i, (fname, size) in enumerate(WEIGHTS.items(), 1):
        downloads.download(f"https://huggingface.co/{repo}/resolve/main/{fname}", config.RELAYVSR_WEIGHTS_DIR / fname,
                           f"Downloading RelayVSR models ({i} of {len(WEIGHTS)})", progress, cancel,
                           expected_size=size)


def install(progress: Progress, cancel: Optional[threading.Event] = None,
            keys: Optional[List[str]] = None) -> List[str]:
    """Installs what's missing: every required component, plus the optional ones named in keys
    (keys given: only those). Parts that need an NVIDIA GPU are skipped without one.
    Returns the keys it installed."""
    has_gpu = gpu.name() is not None
    done = []
    for c in components():
        wanted = (c.key in keys) if keys is not None else not c.optional
        if not wanted or c.ready or (c.needs_gpu and not has_gpu):
            continue
        check_cancel(cancel)
        c.install(progress, cancel)
        done.append(c.key)
    shutil.rmtree(config.TOOLS_DIR / "_downloads", ignore_errors=True)
    return done


def print_progress(text: str, done: int, total: int) -> None:
    """Progress printer for the command line."""
    pct = f" {done * 100 // total:3d}%" if total else ""
    print(f"\r  {text[:100]:<100}{pct}", end="", flush=True)
