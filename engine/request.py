"""
RestoreRequest: everything one restoration job needs, in one object.

Every front end builds one: the desktop queue and the CLI batch from saved job params
(from_params), the CLI and the browser UI from their own options.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import presets


@dataclass
class RestoreRequest:
    input_video_path: Path | str
    output_dir: Optional[Path | str] = None
    output_file: Optional[Path | str] = None
    max_frames: Optional[int] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    engine_type: str = "superres"  # 'superres', 'relayvsr', 'sol_refiner' or 'diffusion' (LTX-2.5)
    superres_model: str = "RealESRGAN_x2plus.pth"
    mode: str = "restore"  # LTX: 'restore', 'refine', 'cq_enhancer' or 'two_stage'
    prompt_preset: str = "archival_general"
    custom_positive_prompt: Optional[str] = None
    custom_negative_prompt: Optional[str] = None
    output_size: str = "FullHD"  # 'HD', 'FullHD', '4K', '8K'
    tile_size: str = "FullHD"  # 'qHD', 'HD', 'FullHD'
    seed: Optional[int] = None
    guide_strength: float = 0.72
    cfg: float = 2.5
    sol_sigma: float = 0.50
    reference_image: Optional[Path | str] = None
    extract_depth: bool = False
    clean_noise: Optional[bool] = None
    stabilize_jitter: Optional[bool] = None
    enable_teacache: Optional[bool] = None
    teacache_thresh: float = 0.06
    auto_compare: bool = True
    output_codec: str = "auto"  # 'auto' (best HEVC encoder, else H.264), 'hevc', 'h264'

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "RestoreRequest":
        """From the job params the desktop app saves with each queued job (settings panel keys)."""
        return cls(
            input_video_path=params["input_path"],
            output_dir=params.get("output_dir"),
            output_file=params.get("output_file"),
            max_frames=params.get("max_frames"),
            start_frame=params.get("start_frame"),
            end_frame=params.get("end_frame"),
            engine_type=params.get("engine_type", "diffusion"),
            superres_model=params.get("superres_model", "RealESRGAN_x2plus.pth"),
            mode=params.get("mode", "restore"),
            prompt_preset=params.get("preset", presets.DEFAULT_PRESET),
            custom_positive_prompt=params.get("positive_prompt"),
            custom_negative_prompt=params.get("negative_prompt"),
            output_size=params.get("output_size", "FullHD"),
            tile_size=params.get("tile_size", "FullHD"),
            seed=params.get("seed"),
            guide_strength=params.get("guide_strength", 0.72),
            cfg=params.get("cfg", 2.5),
            sol_sigma=params.get("sol_sigma", 0.50),
            enable_teacache=params.get("enable_teacache"),
            teacache_thresh=params.get("teacache_thresh", 0.06),
            reference_image=params.get("reference_image"),
            extract_depth=params.get("extract_depth", False),
            clean_noise=params.get("clean_noise"),
            stabilize_jitter=params.get("stabilize_jitter"),
            output_codec=params.get("output_codec", "auto"),
        )
