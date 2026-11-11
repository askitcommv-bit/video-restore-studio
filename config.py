"""
Static configuration: folders, model file names and repositories, engine constants.

Importing this module has no side effects (bootstrap.init() creates folders and sets PATH).
Presets live in presets.py, persistent user settings in user_settings.py.
"""

import os
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)


# Where the app's own files live when it is not run from a clone (bundled worker, requirements).
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
# Standalone home for an exe outside any clone: settings, queue, downloaded tools and models.
DATA_HOME = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "VideoRestoreStudio"


def _enclosing_clone(exe_dir: Path):
    """The nearest of the exe's folder and the folders above it that holds a source checkout
    (relayvsr_worker.py), so an exe in a clone or its dist\\ uses that clone; None elsewhere."""
    for d in (exe_dir, *exe_dir.parents[:3]):
        if (d / "relayvsr_worker.py").is_file():
            return d
    return None


# The project folder holds settings, queue, outputs and everything the setup installs.
# STANDALONE: the app owns that folder (exe outside a clone, or VIDEO_RESTORE_HOME for a portable
# install or testing), so it may write its bundled files there; a clone's own files are never touched.
if os.environ.get("VIDEO_RESTORE_HOME"):
    PROJECT_DIR, STANDALONE = Path(os.environ["VIDEO_RESTORE_HOME"]).resolve(), True
elif FROZEN:
    _clone = _enclosing_clone(Path(sys.executable).resolve().parent)
    PROJECT_DIR, STANDALONE = (_clone, False) if _clone else (DATA_HOME, True)
else:
    PROJECT_DIR, STANDALONE = Path(__file__).resolve().parent, False

# Tools the in-app setup downloads (FFmpeg, uv); bootstrap.init() puts them first on PATH
TOOLS_DIR = PROJECT_DIR / "tools"
FFMPEG_BIN_DIR = TOOLS_DIR / "ffmpeg" / "bin"
UV_EXE = TOOLS_DIR / "uv" / "uv.exe"

CONFIG_JSON_PATH = PROJECT_DIR / "config.json"
WORKFLOWS_DIR = PROJECT_DIR / "workflows"
INPUTS_DIR = PROJECT_DIR / "inputs"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
COMPARISONS_DIR = PROJECT_DIR / "comparisons"
RESUME_DIR = PROJECT_DIR / "resume"  # finished segments of stopped jobs, kept until the job completes
DOCS_DIR = PROJECT_DIR / "docs"

# ComfyUI Integration
COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", r"C:\ComfyUI_portable\ComfyUI"))
COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", r"C:\ComfyUI_portable"))
COMFYUI_PYTHON = COMFYUI_ROOT / "python_embeded" / "python.exe"
if not COMFYUI_PYTHON.exists():
    # The exe's own sys.executable is the GUI, not an interpreter
    COMFYUI_PYTHON = Path("python") if FROZEN else Path(sys.executable)

COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFYUI_BASE_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
COMFYUI_WS_URL = f"ws://{COMFYUI_HOST}:{COMFYUI_PORT}/ws"

# Models Directory in ComfyUI
MODELS_DIR = COMFYUI_DIR / "models"
DIFFUSION_MODELS_DIR = MODELS_DIR / "diffusion_models"
TEXT_ENCODERS_DIR = MODELS_DIR / "text_encoders"
VAE_DIR = MODELS_DIR / "vae"
LORAS_DIR = MODELS_DIR / "loras"
LATENT_UPSCALE_DIR = MODELS_DIR / "latent_upscale_models"
UPSCALE_MODELS_DIR = MODELS_DIR / "upscale_models"
INPUT_DIR_COMFY = COMFYUI_DIR / "input"
OUTPUT_DIR_COMFY = COMFYUI_DIR / "output"

# RelayVSR (Streaming 4x Generative Super-Resolution — runs outside ComfyUI in its own venv)
RELAYVSR_DIR = PROJECT_DIR / "third_party" / "RelayVSR"
RELAYVSR_PYTHON = RELAYVSR_DIR / ".venv" / "Scripts" / "python.exe"


def _relayvsr_weights_dir() -> Path:
    """Wherever the weights already are; for a new download, next to ComfyUI's models when ComfyUI is
    installed (shared with its engines), else in the project folder. Installing ComfyUI later doesn't
    move weights the setup already downloaded."""
    comfy, local = MODELS_DIR / "relayvsr", PROJECT_DIR / "models" / "relayvsr"
    for d in (comfy, local):
        if (d / "wan.safetensors").is_file():
            return d
    return comfy if COMFYUI_DIR.is_dir() else local


RELAYVSR_WEIGHTS_DIR = _relayvsr_weights_dir()
RELAYVSR_WORKER = PROJECT_DIR / "relayvsr_worker.py"
RELAYVSR_MODEL_ID = "relayvsr"  # Pseudo superres model id used by the GUI combo box
# Low-res canvas height fed to RelayVSR (output is exactly 4x this before final resample).
# Measured on RTX 5060 Ti 16GB with ~3.5 GB used by the desktop/other apps:
#   240p -> 960p internal: 11.7 GB peak, ~4 FPS
#   270p -> 1080p internal: 12.2 GB peak, spills into shared system memory -> ~0.17 FPS
RELAYVSR_MAX_LR_HEIGHT = 240

# LTX-2.5 Model Filenames
# Int8 models are highly recommended for 16GB VRAM (e.g. RTX 5060 Ti)
MODEL_FILENAMES = {
    # Diffusion Models
    "transformer_int8": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "transformer_bf16": "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    
    # Text Encoders
    "text_encoder_int8": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "text_encoder_bf16": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "prompt_enhancer": "gemma4_e2b_it_bf16.safetensors",
    
    # VAEs
    "video_vae": "ltx-2.5-video-vae-bf16.safetensors",
    "video_vae_conv": "ltx-2.5-video-vae-conv-bf16.safetensors",
    "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors",
    
    # IC-LoRAs & Enhancer LoRAs
    "lora_restore": "ltx-2.5-22b-ic-lora-restore-1.0.safetensors",
    "lora_refine": "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors",
    "lora_cq_enhancer": "ltx2.5-CQ-enhancer-lora-V2.safetensors",
    
    # Latent Upscaler
    "spatial_upscaler": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",

    # NVIDIA SoL-Refiner (1-Step Ultra-Fast Diffusion Refinement)
    "sol_transformer": "sol-refiner-ltx-2.5-h3-transformer-comfy-int8-convrot.safetensors",
    "sol_vae": "sol-refiner-ltx-2.5-h3-video-vae-bf16.safetensors",
}

# Super-Resolution Neural Models (Path A - Instant Deterministic Upscaling)
SUPERRES_MODELS = {
    "RealESRGAN_x2plus.pth": {
        "name": "⚡ Real-ESRGAN x2plus (Natural & Fast — ~15s)",
        "filename": "RealESRGAN_x2plus.pth",
        "scale": 2,
        "description": "Blistering fast edge sharpening and compression cleanup without hallucination or drift.",
    },
    "4x-UltraSharp.pth": {
        "name": "🎯 4x-UltraSharp (Extreme Texture & Crispness)",
        "filename": "4x-UltraSharp.pth",
        "scale": 4,
        "description": "Deep neural edge definition and micro-contrast for film and digital video.",
    }
}

# Hugging Face Repositories
HF_REPOS = {
    "base": "Lightricks/LTX-2.5",
    "restore": "Lightricks/LTX-2.5-22b-IC-LoRA-Restore",
    "refine": "Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details",
    "sol_refiner": "szwagros/SoL-Refiner-LTX-2.5-H3-int8-convrot",
    "cq_enhancer": "CQdesign/LTX-2.5-CQ-Video-and-Image-Enhancer-LoRAs",
    "relayvsr": "kopper/RelayVSR",
}
