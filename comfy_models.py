"""
Model files the ComfyUI engines (LTX-2.5 restore/refine, NVIDIA SoL-Refiner) load: where each one
lives under ComfyUI's models folder, where it comes from on Hugging Face, whether it is installed,
and downloading it. Also the Hugging Face access token the gated Lightricks repos need.

RelayVSR's own files are handled by setup_manager (it doesn't run in ComfyUI).
"""

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import config
import downloads
import user_settings

MIN_SIZE = 1024 * 1024  # smaller files are leftovers of failed downloads, not models


@dataclass(frozen=True)
class ModelFile:
    key: str
    name: str                 # for people
    folder: Path              # under ComfyUI/models
    repo_key: str             # config.HF_REPOS key
    repo_dir: str             # folder inside the repo ("" for the root)
    filename: str             # the recommended (int8) file
    size_gb: float
    bf16_filename: str = ""   # full-precision alternative, also accepted when installed

    @property
    def repo(self) -> str:
        return config.HF_REPOS[self.repo_key]

    def candidates(self) -> Tuple[Path, ...]:
        return tuple(self.folder / f for f in (self.filename, self.bf16_filename) if f)


F = config.MODEL_FILENAMES
MODELS: Dict[str, ModelFile] = {m.key: m for m in (
    ModelFile("transformer", "LTX-2.5 transformer", config.DIFFUSION_MODELS_DIR, "base", "diffusion_models",
              F["transformer_int8"], 20.03, F["transformer_bf16"]),
    ModelFile("text_encoder", "Gemma text encoder", config.TEXT_ENCODERS_DIR, "base", "text_encoders",
              F["text_encoder_int8"], 14.3, F["text_encoder_bf16"]),
    ModelFile("video_vae", "LTX video VAE", config.VAE_DIR, "base", "vae", F["video_vae"], 1.4),
    ModelFile("audio_vae", "LTX audio VAE", config.VAE_DIR, "base", "vae", F["audio_vae"], 0.35),
    ModelFile("lora_restore", "Restore IC-LoRA", config.LORAS_DIR, "restore", "", F["lora_restore"], 1.6),
    ModelFile("lora_refine", "Refine-Details IC-LoRA", config.LORAS_DIR, "refine", "", F["lora_refine"], 1.25),
    ModelFile("spatial_upscaler", "Latent spatial upscaler", config.LATENT_UPSCALE_DIR, "base",
              "latent_upscale_models", F["spatial_upscaler"], 0.93),
    ModelFile("sol_transformer", "SoL-Refiner transformer", config.DIFFUSION_MODELS_DIR, "sol_refiner",
              "diffusion_models", F["sol_transformer"], 21.90),
    ModelFile("sol_vae", "SoL-Refiner VAE", config.VAE_DIR, "sol_refiner", "vae", F["sol_vae"], 1.37),
    ModelFile("lora_cq_enhancer", "CQ Enhancer LoRA", config.LORAS_DIR, "cq_enhancer", "",
              F["lora_cq_enhancer"], 1.28),
    ModelFile("video_vae_conv", "LTX convolution video VAE", config.VAE_DIR, "base", "vae",
              F["video_vae_conv"], 1.45),
)}

# What each ComfyUI engine loads
LTX_BASE = ("transformer", "text_encoder", "video_vae")
LTX_BY_MODE = {
    "restore": LTX_BASE + ("lora_restore",),
    "refine": LTX_BASE + ("lora_refine",),
    "cq_enhancer": LTX_BASE + ("lora_cq_enhancer",),
    "two_stage": LTX_BASE + ("lora_restore", "lora_refine", "spatial_upscaler"),
}
SOL_REFINER = ("sol_transformer", "sol_vae", "spatial_upscaler", "text_encoder")
# Download sets: everything the LTX restore/refine/two-stage presets use, and the SoL-Refiner
LTX_PACK = ("transformer", "text_encoder", "spatial_upscaler", "video_vae", "audio_vae", "lora_restore", "lora_refine")
LORA_PACK = ("lora_restore", "lora_refine")
SOL_PACK = ("sol_vae", "sol_transformer")


def ltx_requirements(mode: str) -> Tuple[str, ...]:
    return LTX_BY_MODE.get(mode, LTX_BY_MODE["two_stage"])


def local_path(key: str) -> Optional[Path]:
    """The installed file for key (int8 preferred), or None."""
    for p in MODELS[key].candidates():
        if p.is_file() and p.stat().st_size > MIN_SIZE:
            return p
    return None


def installed(key: str) -> bool:
    return local_path(key) is not None


def missing(keys: Iterable[str]) -> List[str]:
    return [k for k in keys if not installed(k)]


def missing_gb(keys: Iterable[str]) -> float:
    return sum(MODELS[k].size_gb for k in missing(keys))


# ---------------------------------------------------------------------------
# Hugging Face token (the Lightricks/LTX-2.5 base repo is gated)
# ---------------------------------------------------------------------------
HF_TOKEN_CACHE = Path.home() / ".cache" / "huggingface" / "token"


def get_hf_token() -> Optional[str]:
    """HF_TOKEN, else the token saved in the app's settings, else huggingface-cli's cached token."""
    for tok in (os.environ.get("HF_TOKEN"), user_settings.get_user_setting("hf_token")):
        if tok and str(tok).strip():
            return str(tok).strip()
    try:
        tok = HF_TOKEN_CACHE.read_text(encoding="utf-8").strip()
        return tok or None
    except OSError:
        return None


def set_hf_token(token: str) -> None:
    """Saves the token in the app's settings and huggingface-cli's cache (other tools reuse it)."""
    token = token.strip()
    user_settings.set_user_setting("hf_token", token)
    try:
        HF_TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        HF_TOKEN_CACHE.write_text(token, encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------
def download(key: str, progress: downloads.Progress = downloads._no_progress,
             cancel: Optional[threading.Event] = None, precision: str = "int8") -> Path:
    """Downloads one model into ComfyUI's models folder (skipped when installed). Raises
    errors.DownloadForbidden for a gated repo without an accepted license / valid token."""
    m = MODELS[key]
    have = local_path(key)
    if have is not None:
        return have
    filename = m.bf16_filename if precision == "bf16" and m.bf16_filename else m.filename
    remote = f"{m.repo_dir}/{filename}" if m.repo_dir else filename
    return downloads.download(
        f"https://huggingface.co/{m.repo}/resolve/main/{remote}", m.folder / filename,
        f"Downloading {m.name}", progress, cancel, token=get_hf_token(),
        repo_url=f"https://huggingface.co/{m.repo}")


def download_many(keys: Iterable[str], progress: downloads.Progress = downloads._no_progress,
                  cancel: Optional[threading.Event] = None, precision: str = "int8") -> List[str]:
    """Downloads every missing model of keys; returns the keys it downloaded."""
    done = []
    todo = missing(keys)
    for i, key in enumerate(todo, 1):
        downloads.check_cancel(cancel)
        download(key, lambda text, d, t, i=i: progress(f"{text} ({i} of {len(todo)})", d, t), cancel, precision)
        done.append(key)
    return done
