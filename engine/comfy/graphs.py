"""
ComfyUI API prompt graphs for the ComfyUI engines: Real-ESRGAN / UltraSharp super-resolution,
NVIDIA SoL-Refiner, and LTX-2.5 restore / refine / two-stage tiled-fusion restoration.

Pure functions: settings in, graph dict out (sent to ComfyUI by engine.comfy.client).
"""

from typing import Any, Dict, Optional

import config


def build_superres_api_prompt(
    video_filename: str,
    output_prefix: str,
    upscale_model_name: str = "RealESRGAN_x2plus.pth",
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Builds a native ComfyUI API prompt for direct neural super-resolution (Path A).
    Runs Real-ESRGAN or 4x-UltraSharp deterministically on GPU frames without diffusion drift.
    """
    prompt: Dict[str, Any] = {
        "1": {
            "class_type": "LoadVideo",
            "inputs": {"file": video_filename},
            "_meta": {"title": "Load Source Video"}
        },
        "2": {
            "class_type": "GetVideoComponents",
            "inputs": {"video": ["1", 0]},
            "_meta": {"title": "Extract Video Components"}
        },
        "100": {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": upscale_model_name},
            "_meta": {"title": "Load Upscale Model"}
        },
        "101": {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["100", 0],
                "image": ["2", 0]
            },
            "_meta": {"title": "Neural Super-Resolution (GPU)"}
        }
    }

    if target_width and target_height:
        prompt["102"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["101", 0],
                "upscale_method": "lanczos",
                "width": int(target_width),
                "height": int(target_height),
                "crop": "disabled"
            },
            "_meta": {"title": "Scale to Target Canvas"}
        }
        create_video_img = ["102", 0]
    else:
        create_video_img = ["101", 0]

    prompt["103"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": create_video_img,
            "fps": ["2", 2],
            "audio": ["2", 1]
        },
        "_meta": {"title": "Assemble Video"}
    }
    prompt["104"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["103", 0],
            "filename_prefix": output_prefix,
            "format": "auto"
    },
    "_meta": {"title": "Save Restored Video"}
}
    return prompt


def build_sol_refiner_api_prompt(
    video_filename: str,
    output_prefix: str,
    seed: int = 42,
    pos_prompt: str = "",
    neg_prompt: str = "",
    sigma: float = 0.50,
    ref_image_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds a native ComfyUI API prompt for NVIDIA 1-Step SoL-Refiner.
    Pipeline:
    1. LoadVideo -> GetVideoComponents (images, audio, fps)
    2. VAEEncode (CausalDiffusionVAE) -> Latent
    3. LatentUpscaleModelLoader -> LTXVLatentUpsampler (2x spatial latent upscaling)
    4. UNETLoader (SoL-Refiner DiT) + CLIPLoader (Gemma 4) + LTXVConditioning
    5. ManualSigmas ('{sigma}, 0.0') + KSamplerSelect ('euler')
    6. SamplerCustom (1-step forward pass, CFG=1.0, add_noise=True)
    7. VAEDecodeTiled (SoL diffusion decoder)
    8. CreateVideo + SaveVideo
    """
    if not pos_prompt:
        pos_prompt = (
            "photorealistic clean sharp archival restoration, pristine image clarity, authentic natural vibrant colors, "
            "photorealistic texture, balanced daylight illumination, fine film detail, modern digital camera capture"
        )
    if not neg_prompt:
        neg_prompt = (
            "dirt, scratches, dust specks, emulsion damage, flicker, heavy film grain, blurry, mushy, noisy, "
            "watermark, low resolution, digital artifacts, text, subtitles, logo, oversaturated"
        )

    prompt = {
        "1": {
            "class_type": "LoadVideo",
            "inputs": {"file": video_filename},
            "_meta": {"title": "Load Source Video"}
        },
        "2": {
            "class_type": "GetVideoComponents",
            "inputs": {"video": ["1", 0]},
            "_meta": {"title": "Extract Video Components"}
        },
        "3": {
            "class_type": "GetImageSize",
            "inputs": {"image": ["2", 0]},
            "_meta": {"title": "Get Image Dimensions"}
        },
        "10": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": config.MODEL_FILENAMES["sol_transformer"],
                "weight_dtype": "default"
            },
            "_meta": {"title": "Load NVIDIA SoL-Refiner Transformer"}
        },
        "11": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": config.MODEL_FILENAMES["text_encoder_int8"],
                "type": "ltxv",
                "device": "default"
            },
            "_meta": {"title": "Load Gemma Text Encoder"}
        },
        "12": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": config.MODEL_FILENAMES["sol_vae"]
            },
            "_meta": {"title": "Load SoL Video VAE & Diffusion Decoder"}
        },
        "13": {
            "class_type": "LatentUpscaleModelLoader",
            "inputs": {
                "model_name": config.MODEL_FILENAMES["spatial_upscaler"]
            },
            "_meta": {"title": "Load Latent Spatial Upscaler (2x)"}
        },
        "20": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": pos_prompt},
            "_meta": {"title": "Positive Guidance Prompt"}
        },
        "21": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": neg_prompt},
            "_meta": {"title": "Negative Guidance Prompt"}
        },
        "22": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["20", 0],
                "negative": ["21", 0],
                "frame_rate": ["2", 2]
            },
            "_meta": {"title": "LTXV Conditioning"}
        },
        "30": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["2", 0],
                "vae": ["12", 0]
            },
            "_meta": {"title": "Encode Frames to Video Latents"}
        },
        "31": {
            "class_type": "LTXVLatentUpsampler",
            "inputs": {
                "samples": ["30", 0],
                "upscale_model": ["13", 0],
                "vae": ["12", 0]
            },
            "_meta": {"title": "2x Neural Latent Upsampling"}
        },
        "40": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
            "_meta": {"title": "Euler Sampler"}
        },
        "41": {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": f"{sigma}, 0.0"},
            "_meta": {"title": "NVIDIA SoL 1-Step Noise Schedule"}
        },
        "42": {
            "class_type": "SamplerCustom",
            "inputs": {
                "model": ["10", 0],
                "add_noise": True,
                "noise_seed": int(seed),
                "cfg": 1.0,
                "positive": ["22", 0],
                "negative": ["22", 1],
                "sampler": ["40", 0],
                "sigmas": ["41", 0],
                "latent_image": ["31", 0]
            },
            "_meta": {"title": "NVIDIA 1-Step Denoise Refinement"}
        },
        "50": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["42", 0],
                "vae": ["12", 0],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8
            },
            "_meta": {"title": "Decode Latents to High-Res Video"}
        },
        "51": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["50", 0],
                "fps": ["2", 2],
                "audio": ["2", 1]
            },
            "_meta": {"title": "Assemble Video with Audio"}
        },
        "52": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["51", 0],
                "filename_prefix": output_prefix,
                "format": "auto"
            },
            "_meta": {"title": "Save Restored Video"}
        }
    }

    if ref_image_filename:
        prompt["80"] = {
            "class_type": "LoadImage",
            "inputs": {"image": ref_image_filename},
            "_meta": {"title": "Load Character Reference Image"}
        }
        prompt["81"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["80", 0], "vae": ["12", 0]},
            "_meta": {"title": "Encode Reference Image to Latent"}
        }
        prompt["82"] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["22", 0], "latent": ["81", 0]},
            "_meta": {"title": "Inject Reference Latent Conditioning"}
        }
        prompt["42"]["inputs"]["positive"] = ["82", 0]

    return prompt


def build_restoration_api_prompt(
    video_filename: str,
    output_prefix: str,
    mode: str = "restore",  # 'restore', 'refine', or 'two_stage'
    tile_size: str = "FullHD",  # 'qHD', 'HD', 'FullHD'
    output_size: str = "FullHD",  # 'FullHD', '4K', '8K'
    seed: int = 42,
    pos_prompt: str = "restored footage, clean image, pristine detail, sharp focus, natural color, master quality",
    neg_prompt: str = "dirt, scratches, dust specks, emulsion damage, flicker, heavy film grain, blurry, mushy, noisy, watermark, low resolution, digital artifacts",
    guide_strength: float = 0.72,
    cfg: float = 2.5,
    ref_image_filename: Optional[str] = None,
    enable_teacache: bool = False,
    teacache_thresh: float = 0.06,
) -> Dict[str, Any]:
    """
    Builds a native, validated ComfyUI API prompt dictionary for LTX-2.5 video restoration.
    Bypasses complex subgraphs by directly wiring verified execution nodes.
    Supports Single-Stage ('restore', 'refine') and Two-Stage Native Ladder ('two_stage').
    Optional TeaCache (Timestep Embedding Cache) accelerates DiT sampling by ~1.8x-2.2x.
    """
    if mode == "two_stage":
        return _build_two_stage_api_prompt(
            video_filename=video_filename,
            output_prefix=output_prefix,
            tile_size=tile_size,
            output_size=output_size,
            seed=seed,
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            guide_strength=guide_strength,
            cfg=cfg,
            ref_image_filename=ref_image_filename,
            enable_teacache=enable_teacache,
            teacache_thresh=teacache_thresh,
        )
    else:
        return _build_single_stage_api_prompt(
            video_filename=video_filename,
            output_prefix=output_prefix,
            mode=mode,
            tile_size=tile_size,
            output_size=output_size,
            seed=seed,
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            guide_strength=guide_strength,
            cfg=cfg,
            ref_image_filename=ref_image_filename,
            enable_teacache=enable_teacache,
            teacache_thresh=teacache_thresh,
        )


def _build_single_stage_api_prompt(
    video_filename: str,
    output_prefix: str,
    mode: str = "restore",
    tile_size: str = "FullHD",
    output_size: str = "FullHD",
    seed: int = 42,
    pos_prompt: str = "",
    neg_prompt: str = "",
    guide_strength: float = 0.72,
    cfg: float = 2.5,
    ref_image_filename: Optional[str] = None,
    enable_teacache: bool = False,
    teacache_thresh: float = 0.06,
) -> Dict[str, Any]:
    if mode == "cq_enhancer":
        lora_filename = config.MODEL_FILENAMES.get("lora_cq_enhancer", "ltx2.5-CQ-enhancer-lora-V2.safetensors")
        conv_vae = config.VAE_DIR / config.MODEL_FILENAMES.get("video_vae_conv", "ltx-2.5-video-vae-conv-bf16.safetensors")
        vae_filename = config.MODEL_FILENAMES.get("video_vae_conv", "ltx-2.5-video-vae-conv-bf16.safetensors") if conv_vae.exists() else config.MODEL_FILENAMES.get("video_vae", "ltx-2.5-video-vae-bf16.safetensors")
        sampler_name = "euler"
    elif mode == "restore":
        lora_filename = config.MODEL_FILENAMES.get("lora_restore", "ltx-2.5-22b-ic-lora-restore-1.0.safetensors")
        vae_filename = config.MODEL_FILENAMES.get("video_vae", "ltx-2.5-video-vae-bf16.safetensors")
        sampler_name = "euler"
    else:
        lora_filename = config.MODEL_FILENAMES.get("lora_refine", "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors")
        vae_filename = config.MODEL_FILENAMES.get("video_vae", "ltx-2.5-video-vae-bf16.safetensors")
        sampler_name = "euler"

    prompt = {
        # 1. Source Video Ingestion
        "1": {
            "class_type": "LoadVideo",
            "inputs": {"file": video_filename},
            "_meta": {"title": "Load Source Video"}
        },
        "2": {
            "class_type": "GetVideoComponents",
            "inputs": {"video": ["1", 0]},
            "_meta": {"title": "Extract Video Components"}
        },
        "3": {
            "class_type": "GetImageSize",
            "inputs": {"image": ["2", 0]},
            "_meta": {"title": "Get Image Dimensions"}
        },

        # 2. Neural Weights
        "10": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
                "weight_dtype": "default"
            },
            "_meta": {"title": "Load LTX-2.5 Transformer (int8)"}
        },
        "11": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                "type": "ltxv",
                "device": "default"
            },
            "_meta": {"title": "Load Gemma Text Encoder (int8)"}
        },
        "12": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_filename},
            "_meta": {"title": "Load Video VAE"}
        },
        "13": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"},
            "_meta": {"title": "Load Audio VAE"}
        },
        "14": {
            "class_type": "LTXICLoRALoaderModelOnly",
            "inputs": {
                "model": ["10", 0],
                "lora_name": lora_filename,
                "strength_model": 1.0
            },
            "_meta": {"title": f"Load LoRA ({mode})"}
        },

        # 3. Conditioning
        "20": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": pos_prompt},
            "_meta": {"title": "Positive Prompt"}
        },
        "21": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": neg_prompt},
            "_meta": {"title": "Negative Prompt"}
        },
        "22": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["20", 0],
                "negative": ["21", 0],
                "frame_rate": ["2", 2]
            },
            "_meta": {"title": "LTXV Conditioning"}
        },

        # 4. Canvas & Tiled Preprocessing
        "30": {
            "class_type": "LTXVGetTilingSizes",
            "inputs": {
                "image": ["2", 0],
                "width": ["3", 0],
                "height": ["3", 1],
                "frame_count": ["3", 2],
                "tile_size": tile_size,
                "initial_canvas_size": "FullHD",
                "output_size": output_size
            },
            "_meta": {"title": "Get Tiling Sizes"}
        },
        "31": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["2", 0],
                "upscale_method": "lanczos",
                "width": ["30", 4],
                "height": ["30", 5],
                "crop": "disabled"
            },
            "_meta": {"title": "Resize Guide to Target Canvas"}
        },
        "32": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": ["30", 4],
                "height": ["30", 5],
                "length": ["3", 2],
                "batch_size": 1
            },
            "_meta": {"title": "Empty Latent Canvas"}
        },
        "33": {
            "class_type": "VAEEncodeAudio",
            "inputs": {
                "audio": ["2", 1],
                "vae": ["13", 0]
            },
            "_meta": {"title": "Encode Audio to Latents"}
        },
        "34": {
            "class_type": "LTXVSetAudioRefTokens",
            "inputs": {
                "positive": ["22", 0],
                "negative": ["22", 1],
                "audio_latent": ["33", 0]
            },
            "_meta": {"title": "Freeze Source Audio Tokens"}
        },
        "35": {
            "class_type": "LTXAddVideoICLoRAGuide",
            "inputs": {
                "positive": ["34", 0],
                "negative": ["34", 1],
                "vae": ["12", 0],
                "latent": ["32", 0],
                "image": ["31", 0],
                "frame_idx": 0,
                "strength": float(guide_strength),
                "latent_downscale_factor": 1.0,
                "crop": "disabled",
                "use_tiled_encode": False,
                "tile_size": 256,
                "tile_overlap": 64,
                "use_streaming": True,
                "tile_frames": 97
            },
            "_meta": {"title": "Add Video IC-LoRA Guide"}
        },

        # 5. Sampler & Scheduler (8 steps distilled)
        "40": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": sampler_name},
            "_meta": {"title": "Sampler Select"}
        },
        "41": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
            },
            "_meta": {"title": "Distilled 8-Step Sigmas"}
        },
        "42": {
            "class_type": "LTXVTiledFusionSampler",
            "inputs": {
                "model": ["14", 0],
                "positive": ["35", 0],
                "negative": ["35", 1],
                "latents": ["35", 2],
                "sigmas": ["41", 0],
                "sampler": ["40", 0],
                "seed": seed,
                "cfg": float(cfg),
                "tile_width": ["30", 2],
                "tile_height": ["30", 3],
                "overlap_frac": ["30", 6],
                "blend_var": 0.05,
                "grid_cycle": 1,
                "tile_frames": ["30", 7],
                "vae": ["12", 0],
                "canvas_device": "gpu"
            },
            "_meta": {"title": "LTXV Tiled Fusion Sampler"}
        },

        # 6. Decode & Output
        "50": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["42", 0],
                "vae": ["12", 0],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 128,
                "temporal_overlap": 32
            },
            "_meta": {"title": "Tiled Video VAE Decode"}
        },
        "51": {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {
                "samples": ["33", 0],
                "audio_vae": ["13", 0]
            },
            "_meta": {"title": "Audio VAE Decode"}
        },
        "52": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["50", 0],
                "audio": ["51", 0],
                "fps": ["2", 2],
                "bit_depth": 8
            },
            "_meta": {"title": "Assemble Video"}
        },
        "53": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["52", 0],
                "filename_prefix": output_prefix,
                "format": "auto"
            },
            "_meta": {"title": "Save Restored Video"}
        }
    }

    if ref_image_filename:
        prompt["80"] = {
            "class_type": "LoadImage",
            "inputs": {"image": ref_image_filename},
            "_meta": {"title": "Load Character Reference Image"}
        }
        prompt["81"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["80", 0], "vae": ["12", 0]},
            "_meta": {"title": "Encode Reference Image to Latent"}
        }
        prompt["82"] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["35", 0], "latent": ["81", 0]},
            "_meta": {"title": "Inject Reference Latent Conditioning"}
        }
        prompt["42"]["inputs"]["positive"] = ["82", 0]

    if enable_teacache:
        prompt["15"] = {
            "class_type": "TeaCache",
            "inputs": {
                "model": ["14", 0],
                "model_type": "ltxv",
                "rel_l1_thresh": float(teacache_thresh),
                "start_percent": 0.0,
                "end_percent": 1.0,
                "cache_device": "cuda"
            },
            "_meta": {"title": "TeaCache (DiT Acceleration)"}
        }
        prompt["42"]["inputs"]["model"] = ["15", 0]

    return prompt


def _build_two_stage_api_prompt(
    video_filename: str,
    output_prefix: str,
    tile_size: str = "HD",
    output_size: str = "4K",
    seed: int = 42,
    pos_prompt: str = "",
    neg_prompt: str = "",
    guide_strength: float = 0.72,
    cfg: float = 2.5,
    ref_image_filename: Optional[str] = None,
    enable_teacache: bool = False,
    teacache_thresh: float = 0.06,
) -> Dict[str, Any]:
    prompt = {
        # 1. Source Video Ingestion
        "1": {
            "class_type": "LoadVideo",
            "inputs": {"file": video_filename},
            "_meta": {"title": "Load Source Video"}
        },
        "2": {
            "class_type": "GetVideoComponents",
            "inputs": {"video": ["1", 0]},
            "_meta": {"title": "Extract Video Components"}
        },
        "3": {
            "class_type": "GetImageSize",
            "inputs": {"image": ["2", 0]},
            "_meta": {"title": "Get Image Dimensions"}
        },

        # 2. Neural Weights
        "10": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
                "weight_dtype": "default"
            },
            "_meta": {"title": "Load LTX-2.5 Transformer (int8)"}
        },
        "11": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                "type": "ltxv",
                "device": "default"
            },
            "_meta": {"title": "Load Gemma Text Encoder (int8)"}
        },
        "12": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ltx-2.5-video-vae-bf16.safetensors"},
            "_meta": {"title": "Load Video VAE"}
        },
        "13": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"},
            "_meta": {"title": "Load Audio VAE"}
        },
        "14": {
            "class_type": "LTXICLoRALoaderModelOnly",
            "inputs": {
                "model": ["10", 0],
                "lora_name": "ltx-2.5-22b-ic-lora-restore-1.0.safetensors",
                "strength_model": 1.0
            },
            "_meta": {"title": "Load Restore IC-LoRA (Stage 1)"}
        },
        "15": {
            "class_type": "LTXICLoRALoaderModelOnly",
            "inputs": {
                "model": ["10", 0],
                "lora_name": "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors",
                "strength_model": 1.0
            },
            "_meta": {"title": "Load Refine IC-LoRA (Stage 2)"}
        },
        "16": {
            "class_type": "LatentUpscaleModelLoader",
            "inputs": {
                "model_name": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
            },
            "_meta": {"title": "Load 2x Latent Spatial Upscaler"}
        },

        # 3. Conditioning
        "20": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": pos_prompt},
            "_meta": {"title": "Positive Prompt"}
        },
        "21": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": neg_prompt},
            "_meta": {"title": "Negative Prompt"}
        },
        "22": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["20", 0],
                "negative": ["21", 0],
                "frame_rate": ["2", 2]
            },
            "_meta": {"title": "LTXV Conditioning"}
        },

        # 4. Canvas & Tiling Preprocessing
        # For two-stage ladder with 2x spatial latent upscaler:
        # - If target is FullHD (1080p): Stage 1 must be qHD (960px) so 2x upscale = FullHD (1920px)
        # - If target is 4K: Stage 1 is FullHD (1920px) so 2x upscale = 4K (3840px)
        # - If target is 8K: Stage 1 is FullHD or 4K so 2x upscale = 8K
        "30": {
            "class_type": "LTXVGetTilingSizes",
            "inputs": {
                "image": ["2", 0],
                "width": ["3", 0],
                "height": ["3", 1],
                "frame_count": ["3", 2],
                "tile_size": "qHD" if output_size == "FullHD" else tile_size,
                "initial_canvas_size": "qHD" if output_size == "FullHD" else "FullHD",
                "output_size": output_size
            },
            "_meta": {"title": "Get Tiling Sizes"}
        },
        "31": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["2", 0],
                "upscale_method": "lanczos",
                "width": ["30", 0],   # initial_canvas_width
                "height": ["30", 1],  # initial_canvas_height
                "crop": "disabled"
            },
            "_meta": {"title": "Resize Guide to Stage 1 Canvas"}
        },
        "32": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": ["30", 0],
                "height": ["30", 1],
                "length": ["3", 2],
                "batch_size": 1
            },
            "_meta": {"title": "Empty Latent Canvas (Stage 1)"}
        },
        "33": {
            "class_type": "VAEEncodeAudio",
            "inputs": {
                "audio": ["2", 1],
                "vae": ["13", 0]
            },
            "_meta": {"title": "Encode Audio to Latents"}
        },
        "34": {
            "class_type": "LTXVSetAudioRefTokens",
            "inputs": {
                "positive": ["22", 0],
                "negative": ["22", 1],
                "audio_latent": ["33", 0]
            },
            "_meta": {"title": "Freeze Source Audio Tokens"}
        },
        "35": {
            "class_type": "LTXAddVideoICLoRAGuide",
            "inputs": {
                "positive": ["34", 0],
                "negative": ["34", 1],
                "vae": ["12", 0],
                "latent": ["32", 0],
                "image": ["31", 0],
                "frame_idx": 0,
                "strength": 1.0,
                "latent_downscale_factor": 1.0,
                "crop": "disabled",
                "use_tiled_encode": False,
                "tile_size": 256,
                "tile_overlap": 64,
                "use_streaming": True,
                "tile_frames": 97
            },
            "_meta": {"title": "Add Video IC-LoRA Guide (Stage 1)"}
        },

        # 5. Sampler - Stage 1 (8 steps)
        "40": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
            "_meta": {"title": "Sampler Select"}
        },
        "41": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
            },
            "_meta": {"title": "Stage 1 Sigmas (8 steps)"}
        },
        "42": {
            "class_type": "LTXVTiledFusionSampler",
            "inputs": {
                "model": ["14", 0],
                "positive": ["35", 0],
                "negative": ["35", 1],
                "latents": ["35", 2],
                "sigmas": ["41", 0],
                "sampler": ["40", 0],
                "seed": seed,
                "cfg": float(cfg),
                "tile_width": ["30", 2],
                "tile_height": ["30", 3],
                "overlap_frac": ["30", 6],
                "blend_var": 0.05,
                "grid_cycle": 1,
                "tile_frames": ["30", 7],
                "vae": ["12", 0],
                "canvas_device": "gpu"
            },
            "_meta": {"title": "Stage 1 Tiled Fusion Sampler"}
        },

        # 6. Stage 2 - 2x Spatial Latent Upscale & 4K Refine
        "60": {
            "class_type": "LTXVLatentUpsampler",
            "inputs": {
                "samples": ["42", 0],
                "upscale_model": ["16", 0],
                "vae": ["12", 0]
            },
            "_meta": {"title": "2x Spatial Latent Upscaler"}
        },
        "61": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["2", 0],
                "upscale_method": "lanczos",
                "width": ["30", 4],   # output_width (4K)
                "height": ["30", 5],  # output_height (4K)
                "crop": "disabled"
            },
            "_meta": {"title": "Resize Original Guide to Target Canvas"}
        },
        "62": {
            "class_type": "LTXAddVideoICLoRAGuide",
            "inputs": {
                "positive": ["34", 0],
                "negative": ["34", 1],
                "vae": ["12", 0],
                "latent": ["60", 0],  # 2x upscaled latent
                "image": ["61", 0],   # 4K guide frames
                "frame_idx": 0,
                "strength": float(guide_strength),
                "latent_downscale_factor": 1.0,
                "crop": "disabled",
                "use_tiled_encode": False,
                "tile_size": 256,
                "tile_overlap": 64,
                "use_streaming": True,
                "tile_frames": 97
            },
            "_meta": {"title": "Add Video IC-LoRA Guide (Stage 2)"}
        },
        "63": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "0.909375, 0.725, 0.421875, 0.0"
            },
            "_meta": {"title": "Stage 2 Refine Sigmas (3 steps)"}
        },
        "64": {
            "class_type": "LTXVTiledFusionSampler",
            "inputs": {
                "model": ["15", 0],  # Refine LoRA
                "positive": ["62", 0],
                "negative": ["62", 1],
                "latents": ["62", 2],
                "sigmas": ["63", 0],
                "sampler": ["40", 0],
                "seed": seed + 1,
                "cfg": float(cfg),
                "tile_width": ["30", 2],
                "tile_height": ["30", 3],
                "overlap_frac": ["30", 6],
                "blend_var": 0.05,
                "grid_cycle": 1,
                "tile_frames": ["30", 7],
                "vae": ["12", 0],
                "canvas_device": "gpu"
            },
            "_meta": {"title": "Stage 2 Refine Tiled Fusion Sampler"}
        },

        # 7. Decode & Save
        "50": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["64", 0],  # 4K refined latent
                "vae": ["12", 0],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 128,
                "temporal_overlap": 32
            },
            "_meta": {"title": "Tiled Video VAE Decode"}
        },
        "51": {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {
                "samples": ["33", 0],
                "audio_vae": ["13", 0]
            },
            "_meta": {"title": "Audio VAE Decode"}
        },
        "52": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["50", 0],
                "audio": ["51", 0],
                "fps": ["2", 2],
                "bit_depth": 8
            },
            "_meta": {"title": "Assemble Video"}
        },
        "53": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["52", 0],
                "filename_prefix": output_prefix,
                "format": "auto"
            },
            "_meta": {"title": "Save Restored Video"}
        }
    }

    if ref_image_filename:
        prompt["80"] = {
            "class_type": "LoadImage",
            "inputs": {"image": ref_image_filename},
            "_meta": {"title": "Load Character Reference Image"}
        }
        prompt["81"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["80", 0], "vae": ["12", 0]},
            "_meta": {"title": "Encode Reference Image to Latent"}
        }
        prompt["82"] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["35", 0], "latent": ["81", 0]},
            "_meta": {"title": "Inject Reference Latent Conditioning (Stage 1)"}
        }
        prompt["42"]["inputs"]["positive"] = ["82", 0]

        prompt["83"] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["62", 0], "latent": ["81", 0]},
            "_meta": {"title": "Inject Reference Latent Conditioning (Stage 2)"}
        }
        prompt["64"]["inputs"]["positive"] = ["83", 0]

    if enable_teacache:
        prompt["17"] = {
            "class_type": "TeaCache",
            "inputs": {
                "model": ["14", 0],
                "model_type": "ltxv",
                "rel_l1_thresh": float(teacache_thresh),
                "start_percent": 0.0,
                "end_percent": 1.0,
                "cache_device": "cuda"
            },
            "_meta": {"title": "TeaCache Acceleration (Stage 1)"}
        }
        prompt["42"]["inputs"]["model"] = ["17", 0]

        prompt["18"] = {
            "class_type": "TeaCache",
            "inputs": {
                "model": ["15", 0],
                "model_type": "ltxv",
                "rel_l1_thresh": float(teacache_thresh),
                "start_percent": 0.0,
                "end_percent": 1.0,
                "cache_device": "cuda"
            },
            "_meta": {"title": "TeaCache Acceleration (Stage 2)"}
        }
        prompt["64"]["inputs"]["model"] = ["18", 0]

    return prompt


NODE_DESCRIPTIONS = {
    "15": (28, "Applying TeaCache DiT layer acceleration..."),
    "17": (28, "Applying TeaCache DiT acceleration (Stage 1)..."),
    "18": (68, "Applying TeaCache DiT acceleration (Stage 2)..."),
    "1": (32, "Decoding source video frames..."),
    "80": (36, "Loading character reference portrait..."),
    "81": (37, "Encoding reference portrait into VAE latent space..."),
    "82": (38, "Injecting character reference likeness conditioning..."),
    "83": (83, "Injecting character reference likeness conditioning (Stage 2)..."),
    "2": (34, "Extracting video & audio components..."),
    "3": (35, "Inspecting frame dimensions..."),
    "10": (38, "Loading LTX-2.5 22B transformer into VRAM..."),
    "11": (42, "Loading Gemma text encoder..."),
    "12": (45, "Loading Video VAE..."),
    "13": (47, "Loading Audio VAE..."),
    "14": (49, "Attaching Restoration IC-LoRA weights..."),
    "15": (50, "Attaching Refinement IC-LoRA weights..."),
    "16": (52, "Loading 2x Spatial Latent Upscaler..."),
    "20": (53, "Encoding positive prompt..."),
    "21": (54, "Encoding negative prompt..."),
    "22": (55, "Computing LTXV temporal conditioning..."),
    "30": (56, "Calculating adaptive tiled canvas dimensions..."),
    "31": (58, "Scaling guidance frames..."),
    "32": (59, "Allocating empty latent canvas..."),
    "33": (60, "Encoding source audio latents..."),
    "34": (61, "Freezing source audio tokens..."),
    "35": (62, "Injecting video guide latents (Stage 1)..."),
    "40": (63, "Configuring Euler sampler..."),
    "41": (64, "Computing diffusion sigmas..."),
    "42": (65, "Executing Neural Tiled Fusion Sampler (Stage 1)..."),
    "60": (82, "Upscaling latents 2x with spatial diffusion..."),
    "61": (84, "Preparing 4K guidance canvas..."),
    "62": (85, "Injecting 4K guide latents (Stage 2)..."),
    "63": (86, "Configuring refinement sigmas..."),
    "64": (87, "Executing 4K Refine Tiled Sampler (Stage 2)..."),
    "50": (93, "Decoding latents with Tiled VAE to 4K RGB frames..."),
    "51": (96, "Decoding synchronized audio..."),
    "52": (97, "Assembling video and audio tracks..."),
    "53": (98, "Saving rendered video to disk..."),
    # Direct Neural Super-Resolution Nodes (Path A)
    "100": (25, "Loading Neural Super-Resolution model into VRAM..."),
    "101": (65, "Upscaling video frames with CUDA Super-Resolution..."),
    "102": (85, "Scaling to target canvas with Lanczos..."),
    "103": (92, "Assembling video and audio tracks..."),
    "104": (98, "Saving rendered video to disk...")
}
