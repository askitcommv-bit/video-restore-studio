"""
Naming of output files, shared by the engine (default names) and the app (suggested names).
"""

from pathlib import Path
from typing import Any, Dict


def output_tag(params: Dict[str, Any]) -> str:
    """Suffix that says how a file was made: <source>_<tag>.mp4."""
    engine_type = params.get("engine_type", "diffusion")
    if engine_type == "superres":
        return f"superres_{params.get('superres_model', 'RealESRGAN_x2plus.pth').split('.')[0].lower()}"
    if engine_type in ("relayvsr", "sol_refiner"):
        return engine_type
    mode = params.get("mode", "restored")
    return "restored" if mode == "restore" else mode


def suggested_output_path(input_path: Path | str, params: Dict[str, Any], output_dir: Path | str) -> Path:
    """An unused <source>_<tag>.mp4 in output_dir."""
    return unique_output_path(output_dir, f"{Path(input_path).stem}_{output_tag(params)}", ".mp4")


def unique_output_path(output_dir: Path | str, base_stem: str, suffix: str = ".mp4") -> Path:
    """
    Returns an unused, unique Path in output_dir.
    If 'base_stem.mp4' already exists, auto-appends ' (1)', ' (2)', etc.
    e.g.:
      clip_restored.mp4       (if doesn't exist)
      clip_restored (1).mp4   (if clip_restored.mp4 exists)
      clip_restored (2).mp4   (if clip_restored (1).mp4 also exists)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not suffix.startswith("."):
        suffix = f".{suffix}"
        
    candidate = out_dir / f"{base_stem}{suffix}"
    if not candidate.exists():
        return candidate
        
    index = 1
    while True:
        candidate = out_dir / f"{base_stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
        index += 1
