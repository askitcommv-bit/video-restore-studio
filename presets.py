"""
Restoration presets (engine, mode, resolution, prompts, defect ticks) and the translation from the
UI's defect checkboxes to an LTX-2.5 pipeline.
"""

from config import RELAYVSR_MODEL_ID

# Preset selected on startup and used whenever a job/CLI call doesn't name one
DEFAULT_PRESET = "relayvsr_streaming"

# Comprehensive 1-Click Restoration Presets (Engine, Mode, Resolution, Prompts, Defect Ticks & Est. Speed)
RESTORATION_PRESETS = {
    "master_ltx_8step": {
        "id": "master_ltx_8step",
        "name": "🏆 Master Archival Redraw (LTX 8-Step) [Recommended]",
        "badge": "Highest Quality • Photorealistic Reconstruction",
        "description": "Reconstructs authentic missing micro-details (skin pores, iris, hair, tree bark fissures) with zero boiling. 8 diffusion iterations.",
        "engine_type": "diffusion",
        "mode": "restore",
        "output_size": "FullHD",
        "tile_size": "FullHD",
        "guide_strength": 0.72,
        "cfg": 2.5,
        "sol_sigma": 0.50,
        "superres_model": "RealESRGAN_x2plus.pth",
        "est_speed": "~18s / frame (~5m for 17 frames on RTX 5060 Ti)",
        "positive": (
            "clean sharp archival restoration, pristine image clarity, authentic natural vibrant colors, "
            "photorealistic texture, balanced daylight illumination, fine film detail, modern digital camera capture"
        ),
        "negative": (
            "dirt, scratches, dust specks, emulsion damage, flicker, heavy film grain, blurry, mushy, noisy, "
            "watermark, low resolution, digital artifacts, text, subtitles, logo, oversaturated"
        ),
        "defects": {
            "dirt_scratches": True,
            "jitter_flicker": True,
            "deinterlace": False,
            "noise_compression": True,
            "colorize": False,
            "film_roll": True,
            "micro_detail": False,
        }
    },
    "turbo_ltx_teacache": {
        "id": "turbo_ltx_teacache",
        "name": "⚡ Turbo Archival Redraw (LTX 8-Step + TeaCache)",
        "badge": "2x Accelerated Diffusion • ~8.5s / frame",
        "description": "Full LTX 8-step archival redraw accelerated with DiT Timestep Embedding Cache (TeaCache). Delivers identical photoreal detail in ~2.4 min (nearly 2x faster).",
        "engine_type": "diffusion",
        "mode": "restore",
        "output_size": "FullHD",
        "tile_size": "FullHD",
        "guide_strength": 0.72,
        "cfg": 2.5,
        "sol_sigma": 0.50,
        "enable_teacache": True,
        "teacache_thresh": 0.06,
        "superres_model": "RealESRGAN_x2plus.pth",
        "est_speed": "~8.5s / frame (~2.4m for 17 frames on RTX 5060 Ti)",
        "positive": (
            "clean sharp archival restoration, pristine image clarity, authentic natural vibrant colors, "
            "photorealistic texture, balanced daylight illumination, fine film detail, modern digital camera capture"
        ),
        "negative": (
            "dirt, scratches, dust specks, emulsion damage, flicker, heavy film grain, blurry, mushy, noisy, "
            "watermark, low resolution, digital artifacts, text, subtitles, logo, oversaturated"
        ),
        "defects": {
            "dirt_scratches": True,
            "jitter_flicker": True,
            "deinterlace": False,
            "noise_compression": True,
            "colorize": False,
            "film_roll": True,
            "micro_detail": False,
        }
    },
    "fast_sol_refiner": {
        "id": "fast_sol_refiner",
        "name": "⚡ Fast AI Clean & Denoise (NVIDIA SoL-Refiner)",
        "badge": "Fast Diffusion • ~3.8s / frame",
        "description": "Speed-of-Light 1-step diffusion refinement at sigma 0.50. Cleans compression artifacts, removes flicker, zero boiling (~1 min total).",
        "engine_type": "sol_refiner",
        "mode": "restore",
        "output_size": "FullHD",
        "tile_size": "FullHD",
        "guide_strength": 0.72,
        "cfg": 2.5,
        "sol_sigma": 0.50,
        "superres_model": "RealESRGAN_x2plus.pth",
        "est_speed": "~3.8s / frame (~1m for 17 frames on RTX 5060 Ti)",
        "positive": (
            "clean sharp archival restoration, pristine image clarity, photorealistic texture, authentic natural colors"
        ),
        "negative": (
            "dirt, scratches, dust specks, emulsion damage, flicker, heavy noise, compression artifacts, blur"
        ),
        "defects": {
            "dirt_scratches": True,
            "jitter_flicker": True,
            "deinterlace": False,
            "noise_compression": True,
            "colorize": False,
            "film_roll": False,
            "micro_detail": False,
        }
    },
    "guided_micro_refiner": {
        "id": "guided_micro_refiner",
        "name": "🎯 Guided Detail Refiner (Seamless Chunks • σ=0.35)",
        "badge": "Rich 3D Texture • Seamless Overlap • ~3.5s / frame",
        "description": "NVIDIA SoL-Refiner at the detail-synthesis sweet spot (sigma=0.35) with 8-frame cosine overlap blending. Synthesizes authentic 3D bark fissures, leaf textures, and fabric weaves while locking character identity across all chunks.",
        "engine_type": "sol_refiner",
        "mode": "restore",
        "output_size": "FullHD",
        "tile_size": "FullHD",
        "guide_strength": 0.85,
        "cfg": 1.0,
        "sol_sigma": 0.35,
        "superres_model": "RealESRGAN_x2plus.pth",
        "est_speed": "~3.5s / frame (~55s for 17 frames on RTX 5060 Ti)",
        "positive": "photorealistic clean sharp archival restoration, pristine image clarity, authentic natural vibrant colors, photorealistic texture, fine film detail",
        "negative": "dirt, scratches, dust specks, emulsion damage, flicker, heavy noise, compression artifacts, blur",
        "defects": {
            "dirt_scratches": False,
            "jitter_flicker": False,
            "deinterlace": False,
            "noise_compression": True,
            "colorize": False,
            "film_roll": False,
            "micro_detail": True,
        }
    },
    "cq_enhancer_v2": {
        "id": "cq_enhancer_v2",
        "name": "✨ CQ Enhancer V2 (Face & Detail Reconstruction)",
        "badge": "Generative Detail Synthesis • Faces & Skin • ~18s / frame",
        "description": "Specialized generative enhancement LoRA V2 trained to reconstruct lifelike facial features, eyes, skin pores, and fine 3D textures from degraded low-resolution footage with zero ghosting.",
        "engine_type": "diffusion",
        "mode": "cq_enhancer",
        "output_size": "FullHD",
        "tile_size": "FullHD",
        "guide_strength": 1.0,
        "cfg": 1.0,
        "sol_sigma": 0.35,
        "superres_model": "4x-UltraSharp.pth",
        "est_speed": "~18s / frame (~5m for 17 frames on RTX 5060 Ti)",
        "positive": (
            "clean sharp archival restoration, pristine image clarity, authentic natural vibrant colors, "
            "photorealistic skin texture, fine facial features, authentic natural film detail"
        ),
        "negative": (
            "dirt, scratches, dust specks, emulsion damage, flicker, heavy film grain, blurry, mushy, noisy, "
            "watermark, low resolution, digital artifacts, text, subtitles, logo, oversaturated"
        ),
        "defects": {
            "dirt_scratches": True,
            "jitter_flicker": True,
            "deinterlace": False,
            "noise_compression": True,
            "colorize": False,
            "film_roll": True,
            "micro_detail": True,
        }
    },
    "ultrasharp_master": {
        "id": "ultrasharp_master",
        "name": "💎 4x-UltraSharp Master Upscale",
        "badge": "Instant CUDA • 0% Hallucination • ~0.18s / frame",
        "description": "Deterministic deep convolutional edge refinement (4x-UltraSharp). Sharpens micro-contrasts, fabrics, and textures. Zero hallucinations, 100% identical faces across all chunks.",
        "engine_type": "superres",
        "mode": "restore",
        "output_size": "FullHD",
        "tile_size": "FullHD",
        "guide_strength": 0.72,
        "cfg": 2.5,
        "sol_sigma": 0.50,
        "superres_model": "4x-UltraSharp.pth",
        "est_speed": "~0.18s / frame (~16s for 17 frames on RTX 5060 Ti)",
        "positive": "sharp clean upscale, pristine edges, high contrast, crisp texture",
        "negative": "blurry, noisy, low resolution",
        "defects": {
            "dirt_scratches": False,
            "jitter_flicker": False,
            "deinterlace": False,
            "noise_compression": True,
            "colorize": False,
            "film_roll": False,
            "micro_detail": False,
        }
    },
    "relayvsr_streaming": {
        "id": "relayvsr_streaming",
        "name": "🛰️ RelayVSR Streaming 4x Upscale (Generative, Near Real-Time)",
        "badge": "Wan Keyframes + Lightweight Relay • 720p Output",
        "description": "Sparse Wan generative keyframes (every 15 frames) guide a lightweight video transformer that super-resolves every frame. Realistic detail with temporal consistency, no chunk seams. Upscale only — does not remove scratches or recolor.",
        "engine_type": "relayvsr",
        "mode": "restore",
        "output_size": "HD",
        "tile_size": "FullHD",
        "guide_strength": 0.72,
        "cfg": 2.5,
        "sol_sigma": 0.50,
        "superres_model": RELAYVSR_MODEL_ID,
        "est_speed": "Near real-time after ~1 min model load (RTX 5060 Ti)",
        "positive": "sharp clean upscale, realistic detail",
        "negative": "blurry, noisy, low resolution",
        "defects": {
            "dirt_scratches": False,
            "jitter_flicker": False,
            "deinterlace": False,
            "noise_compression": False,
            "colorize": False,
            "film_roll": False,
            "micro_detail": False,
        }
    },
    "instant_superres": {
        "id": "instant_superres",
        "name": "🚀 Instant Real-Time Upscale (Real-ESRGAN)",
        "badge": "Instant CUDA • ~0.15s / frame",
        "description": "Deterministic neural super-resolution. Ultra-fast edge sharpening (~15s total). Ideal for quick preview checks or clean footage.",
        "engine_type": "superres",
        "mode": "restore",
        "output_size": "FullHD",
        "tile_size": "FullHD",
        "guide_strength": 0.72,
        "cfg": 2.5,
        "sol_sigma": 0.50,
        "superres_model": "RealESRGAN_x2plus.pth",
        "est_speed": "~0.15s / frame (~14s total on RTX 5060 Ti)",
        "positive": "sharp clean upscale, pristine edges, high contrast",
        "negative": "blurry, noisy, low resolution",
        "defects": {
            "dirt_scratches": False,
            "jitter_flicker": False,
            "deinterlace": False,
            "noise_compression": True,
            "colorize": False,
            "film_roll": False,
            "micro_detail": False,
        }
    },
    "vhs_camcorder_tape": {
        "id": "vhs_camcorder_tape",
        "name": "📼 VHS & Camcorder Tape Remaster",
        "badge": "De-interlace + Tape Denoise (LTX 8-Step)",
        "description": "Merges interlaced scanlines, cures chroma bleeding, tape static, and head jitter into progressive 1080p.",
        "engine_type": "diffusion",
        "mode": "restore",
        "output_size": "FullHD",
        "tile_size": "FullHD",
        "guide_strength": 0.70,
        "cfg": 2.5,
        "sol_sigma": 0.50,
        "superres_model": "RealESRGAN_x2plus.pth",
        "est_speed": "~18s / frame (~5m for 17 frames on RTX 5060 Ti)",
        "positive": (
            "remastered video transfer, sharp crisp digital broadcast quality, corrected chroma bleeding, sharp distinct edges, "
            "vibrant natural skin tones, clean solid colors, clear facial expressions"
        ),
        "negative": (
            "VHS static, tracking lines, magnetic tape noise, chroma smear, interlacing lines, blur, VHS timestamp, distortion"
        ),
        "defects": {
            "dirt_scratches": False,
            "jitter_flicker": True,
            "deinterlace": True,
            "noise_compression": True,
            "colorize": False,
            "film_roll": False,
            "micro_detail": False,
        }
    },
    "kodachrome_1950s": {
        "id": "kodachrome_1950s",
        "name": "🎞️ 1950s Kodachrome Film Scan",
        "badge": "Warm 35mm Color & Organic Emulsion",
        "description": "Classic 35mm Kodachrome warmth, rich saturation, authentic film emulsion, and sharp facial features via LTX 8-step.",
        "engine_type": "diffusion",
        "mode": "restore",
        "output_size": "FullHD",
        "tile_size": "FullHD",
        "guide_strength": 0.72,
        "cfg": 2.5,
        "sol_sigma": 0.50,
        "superres_model": "RealESRGAN_x2plus.pth",
        "est_speed": "~18s / frame (~5m for 17 frames on RTX 5060 Ti)",
        "positive": (
            "pristine 1950s Kodachrome 35mm film scan, warm rich saturated colors, authentic period color reproduction, "
            "sharp facial features, crisp clothing fabrics, clean modern telecine transfer"
        ),
        "negative": (
            "faded colors, red-shift dye degradation, vinegar syndrome, stains, scratches, dust, jitter, digital blur"
        ),
        "defects": {
            "dirt_scratches": True,
            "jitter_flicker": False,
            "deinterlace": False,
            "noise_compression": True,
            "colorize": False,
            "film_roll": True,
            "micro_detail": False,
        }
    },
    "silent_1920s_bw": {
        "id": "silent_1920s_bw",
        "name": "📽️ 1920s Silent Film (Monochrome B&W)",
        "badge": "Silver-Gelatin Pure Monochrome",
        "description": "Eliminates tears, dirt, gate jitter, and flicker while preserving deep blacks and luminous historical contrast via LTX 8-step.",
        "engine_type": "diffusion",
        "mode": "restore",
        "output_size": "FullHD",
        "tile_size": "FullHD",
        "guide_strength": 0.72,
        "cfg": 2.5,
        "sol_sigma": 0.50,
        "superres_model": "RealESRGAN_x2plus.pth",
        "est_speed": "~18s / frame (~5m for 17 frames on RTX 5060 Ti)",
        "positive": (
            "restored 1920s historical film, crystal clear sharp focus, smooth motion, high contrast silver-gelatin tones, "
            "deep blacks, luminous highlights, authentic vintage costumes and period textures, razor-sharp edges"
        ),
        "negative": (
            "film flicker, scratches, jitter, gate weave, dust, tears, hair in gate, color, chromatic aberration, modern objects"
        ),
        "defects": {
            "dirt_scratches": True,
            "jitter_flicker": True,
            "deinterlace": False,
            "noise_compression": True,
            "colorize": False,
            "film_roll": True,
            "micro_detail": False,
        }
    },
    "silent_1920s_colorized": {
        "id": "silent_1920s_colorized",
        "name": "🎨 1920s Historical Colorization",
        "badge": "AI Period Color Recovery",
        "description": "Infers period-authentic natural colors, clothing dyes, skin tones, and sunlight from archival scans via LTX 8-step.",
        "engine_type": "diffusion",
        "mode": "restore",
        "output_size": "FullHD",
        "tile_size": "FullHD",
        "guide_strength": 0.72,
        "cfg": 3.0,
        "sol_sigma": 0.50,
        "superres_model": "RealESRGAN_x2plus.pth",
        "est_speed": "~18s / frame (~5m for 17 frames on RTX 5060 Ti)",
        "positive": (
            "restored 1920s archival documentary footage in full natural color, authentic period wool, linen, and brick tones, "
            "realistic natural Caucasian skin tones, cobblestone streets, natural sunny daylight, clean sharp modern film scan"
        ),
        "negative": (
            "black and white, sepia, washed out, unnatural skin tones, garish neon colors, film damage, scratches, dust, flicker"
        ),
        "defects": {
            "dirt_scratches": True,
            "jitter_flicker": True,
            "deinterlace": False,
            "noise_compression": True,
            "colorize": True,
            "film_roll": True,
            "micro_detail": False,
        }
    },
    "two_stage_4k_master": {
        "id": "two_stage_4k_master",
        "name": "💎 4K Ultimate Master (2-Stage Redraw + Refine)",
        "badge": "4K UHD • Maximum Resolution",
        "description": "Stage 1 repairs damage and color at 1080p, Stage 2 latently upscales and refines micro-textures at full 4K.",
        "engine_type": "diffusion",
        "mode": "two_stage",
        "output_size": "4K",
        "tile_size": "HD",
        "guide_strength": 0.72,
        "cfg": 2.5,
        "sol_sigma": 0.50,
        "superres_model": "RealESRGAN_x2plus.pth",
        "est_speed": "~35s / frame (~10m for 17 frames on RTX 5060 Ti)",
        "positive": (
            "master quality archival restoration, authentic vibrant color palette, pristine clean finish, "
            "crisp 4K resolution, ultra-fine texture, authentic cinematography, modern camera capture"
        ),
        "negative": (
            "damage, dirt, scratches, blur, digital compression, plastic smoothing, artifacts, flicker"
        ),
        "defects": {
            "dirt_scratches": True,
            "jitter_flicker": True,
            "deinterlace": False,
            "noise_compression": True,
            "colorize": False,
            "film_roll": True,
            "micro_detail": True,
        }
    },
}

# Backward compatibility map for prompt presets
PROMPT_PRESETS = {
    **RESTORATION_PRESETS,
    "archival_general": RESTORATION_PRESETS["master_ltx_8step"],
    "refine_1080p": {
        "name": "1080p Detail Refinement (Faithful Source)",
        "description": "Micro-texture, hair strands, and fabric weaves at 1080p without altering colors or composition.",
        "mode": "refine",
        "positive": (
            "ultra sharp 1080p fullhd resolution, immaculate micro detail, crisp skin texture, realistic hair strands, "
            "fine fabric weaves, subtle natural cinematic 35mm film grain, razor sharp high-definition rendering"
        ),
        "negative": (
            "blurry, soft, artificial smoothing, plastic skin, ringing artifacts, halation, pixelated, noisy, low-resolution"
        ),
        "tile_size": "FullHD",
        "output_size": "FullHD",
    },
    "refine_upscaled_4k": {
        "name": "4K Detail Refinement (Heavy Master)",
        "description": "Micro-texture, pore details, and 35mm film grain scaled to 4K UHD.",
        "mode": "refine",
        "positive": (
            "ultra sharp 4K resolution, immaculate micro detail, crisp skin texture, realistic hair strands, "
            "fine fabric weaves, subtle natural cinematic 35mm film grain, razor sharp high-definition rendering"
        ),
        "negative": (
            "blurry, soft, artificial smoothing, plastic skin, ringing artifacts, halation, pixelated, noisy, low-resolution"
        ),
        "tile_size": "FullHD",
        "output_size": "4K",
    },
    "two_stage_fullhd": {
        "name": "Two-Stage Ladder (Restore -> 1080p Refine)",
        "description": "Stage 1 repairs damage & colorizes, Stage 2 latently upscales & sharpens micro-details at 1080p.",
        "mode": "two_stage",
        "positive": (
            "master quality archival restoration, authentic vibrant color palette, pristine clean finish, "
            "crisp 1080p resolution, ultra-fine texture, authentic cinematography, modern camera capture"
        ),
        "negative": (
            "damage, dirt, scratches, blur, digital compression, plastic smoothing, artifacts, flicker"
        ),
        "tile_size": "HD",
        "output_size": "FullHD",
    },
    "two_stage_full_restoration": RESTORATION_PRESETS["two_stage_4k_master"]
}


def get_preset(preset_id: str) -> dict:
    """Retrieves preset dictionary by ID with safe fallback to master_ltx_8step."""
    if preset_id in RESTORATION_PRESETS:
        return RESTORATION_PRESETS[preset_id]
    if preset_id in PROMPT_PRESETS:
        item = PROMPT_PRESETS[preset_id]
        if isinstance(item, dict) and "engine_type" in item:
            return item
    # Fallback default
    return RESTORATION_PRESETS["master_ltx_8step"]


def build_pipeline_from_defects(
    defects: dict,
    target_res: str = "FullHD"
) -> dict:
    """
    Translates user-facing defect checkboxes into the optimal LTX-2.5 ComfyUI pipeline:
    - Determines 'mode': 'restore' (1-pass) or 'two_stage' (2-pass ladder)
    - Dynamically generates positive and negative prompt guidance
    - Configures tile sizes and target resolutions
    """
    pos_parts = []
    neg_parts = []

    # 1. Base Restoration Clarity
    pos_parts.append("clean sharp archival restoration, pristine image clarity, photorealistic texture")

    # 2. Dirt & Scratches
    if defects.get("dirt_scratches", True):
        pos_parts.append("undamaged film surface, immaculate clean transfer, clear sharp focus")
        neg_parts.append("dirt, scratches, dust specks, emulsion damage, tears, hair in gate, spots")

    # 3. Jitter & Flicker
    if defects.get("jitter_flicker", True):
        pos_parts.append("smooth cinematic motion, stable exposure, steady camera gate")
        neg_parts.append("film flicker, exposure breathing, gate weave, jitter, frame shake, strobe")

    # 4. De-interlacing
    if defects.get("deinterlace", False):
        neg_parts.append("interlacing lines, comb filter artifacts, jagged scanlines, field tearing")

    # 5. Noise & Compression
    if defects.get("noise_compression", True):
        pos_parts.append("clean noise-free gradients, razor-sharp edges, pristine digital master")
        neg_parts.append("heavy noise, compression artifacts, macroblocking, tape hiss, VHS chroma smear, pixelation, banding")

    # 6. Colorize vs Pure Monochrome
    if defects.get("colorize", False):
        pos_parts.append("vibrant natural period-authentic colors, rich organic skin tones, balanced daylight illumination")
        neg_parts.append("black and white, monochrome, sepia, faded washed out colors, garish oversaturation")
    else:
        pos_parts.append("authentic cinematic lighting and tone, natural contrast")

    # 7. Scanned Film Emulsion
    if defects.get("film_roll", True):
        pos_parts.append("authentic cinematic 35mm film emulsion, subtle organic film texture, natural highlight rolloff")
        neg_parts.append("plastic smoothing, artificial wax figures, synthetic CGI look")
    else:
        pos_parts.append("modern high-definition digital camera capture, clean razor-sharp finish")

    # 8. Micro-Detail Enhancement & Target Resolution
    use_micro_detail = defects.get("micro_detail", False)
    is_upscale_heavy = target_res in ["4K", "8K"]
    
    # If 4K/8K or micro_detail is selected, two-stage ladder gives far superior results
    if use_micro_detail or is_upscale_heavy:
        mode = "two_stage"
        pos_parts.append("immaculate micro detail, sharp high-frequency texture, fine fabric weaves, realistic skin pores")
        neg_parts.append("blurry, soft, artificial smoothing, ringing artifacts, low-resolution")
    else:
        mode = "restore"

    positive_prompt = ", ".join(pos_parts)
    negative_prompt = ", ".join(neg_parts)

    return {
        "mode": mode,
        "output_size": target_res,
        "tile_size": "qHD" if target_res == "FullHD" else "HD",
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "deinterlace": defects.get("deinterlace", False)
    }
