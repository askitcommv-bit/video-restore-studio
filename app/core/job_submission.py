"""
What the app needs to decide before a job goes into the queue, without any UI: the suggested output
file, the preset's display name, what the engine still needs installed, and whether the footage is
long enough for an LTX run to deserve a preview first.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import outputs
import presets
import user_settings
import video_preprocessor
from engine import requirements

COMFY_ENGINES = ("superres", "sol_refiner", "diffusion")
# LTX diffusion ranges longer than this get the "render a preview first?" question
LONG_FOOTAGE_FRAMES = 161
PREVIEW_FRAMES = 65


def output_tag(params: Dict[str, Any]) -> str:
    return outputs.output_tag(params)


def suggested_output_path(params: Dict[str, Any]) -> Path:
    return outputs.suggested_output_path(params["input_path"], params, user_settings.get_last_output_folder())


def preset_identity(params: Dict[str, Any], fallback_name: str = "Restoration Job") -> tuple:
    """(preset id, display name) for a job's queue row."""
    preset_id = params.get("preset", presets.DEFAULT_PRESET)
    return preset_id, params.get("preset_name", presets.get_preset(preset_id).get("name", fallback_name))


def missing_requirements(params: Dict[str, Any]) -> List[str]:
    return requirements.missing(params.get("engine_type", "diffusion"), params.get("mode", "restore"),
                                params.get("superres_model", "RealESRGAN_x2plus.pth"))


def needs_comfyui(params: Dict[str, Any]) -> bool:
    return params.get("engine_type", "diffusion") != "relayvsr"


def long_footage(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """For an LTX diffusion job over LONG_FOOTAGE_FRAMES without a frame cap: the range's frames,
    seconds and rough minutes on a 16 GB card; otherwise None."""
    if params.get("engine_type", "diffusion") in ("superres", "relayvsr", "sol_refiner") or params.get("max_frames"):
        return None
    info = video_preprocessor.get_video_info(params["input_path"])
    total = info.get("total_frames", 0)
    in_f, out_f = params.get("start_frame"), params.get("end_frame")
    if in_f is not None or out_f is not None:
        span = max(1, (out_f if out_f is not None else total - 1) - (in_f or 0) + 1)
    else:
        span = total
    if span <= LONG_FOOTAGE_FRAMES:
        return None
    return {"frames": span, "seconds": span / (info.get("fps") or 24.0),
            "minutes": round((span / 65) * (1.5 if params.get("output_size") == "FullHD" else 11), 1)}
