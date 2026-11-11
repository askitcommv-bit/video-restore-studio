"""
What each engine needs installed: the one place that answers "can this job run here?", for the
desktop app (before it starts a job), the CLI and the engines themselves.
"""

from typing import List

import comfy_models
import config
import setup_manager


def missing(engine_type: str, mode: str = "restore", superres_model: str = "RealESRGAN_x2plus.pth") -> List[str]:
    """Names (for people) of what the engine still needs; empty when it can run."""
    if engine_type == "relayvsr":
        return setup_manager.relayvsr_missing()
    needed = [] if setup_manager.ffmpeg_ready() else ["FFmpeg"]
    if engine_type == "superres":
        if not (config.UPSCALE_MODELS_DIR / superres_model).is_file():
            needed.append(f"{superres_model} (in {config.UPSCALE_MODELS_DIR})")
        return needed
    keys = comfy_models.SOL_REFINER if engine_type == "sol_refiner" else comfy_models.ltx_requirements(mode)
    return needed + [comfy_models.MODELS[k].name for k in comfy_models.missing(keys)]


def missing_model_keys(engine_type: str, mode: str = "restore") -> List[str]:
    """comfy_models keys a ComfyUI diffusion engine (LTX or SoL-Refiner) still needs."""
    keys = comfy_models.SOL_REFINER if engine_type == "sol_refiner" else comfy_models.ltx_requirements(mode)
    return comfy_models.missing(keys)


def engine_label(engine_type: str) -> str:
    return {"relayvsr": "RelayVSR", "superres": "Super-resolution", "sol_refiner": "SoL-Refiner"}.get(
        engine_type, "LTX-2.5")
