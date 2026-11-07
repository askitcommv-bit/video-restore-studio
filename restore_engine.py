"""
Core Restoration Engine for LTX-2.5 Video Restoration.
Coordinates:
- Video inspection and preprocessing (deinterlace, 32-multiple canvas, 8n+1 frames)
- Native ComfyUI API Prompt graph generation for Single-Stage and Two-Stage pipelines
- ComfyUI API queueing, WebSocket live tracking, and resilient polling fallback
- Audio preservation and lossless remuxing
- Side-by-side & split-screen comparison rendering
"""

import sys
import os
import json
import time
import shutil
import uuid
import atexit
import subprocess
import threading
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import video_preprocessor
import video_encoders
import comparison_generator
import model_downloader


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


class ComfyUIClient:
    """REST and WebSocket Client for ComfyUI."""

    def __init__(self, host: str = config.COMFYUI_HOST, port: int = config.COMFYUI_PORT):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.client_id = str(uuid.uuid4())

    def is_online(self) -> bool:
        """Checks if ComfyUI server is reachable."""
        return self.try_system_stats() is not None

    def try_system_stats(self, timeout: float = 3) -> Optional[Dict[str, Any]]:
        """One /system_stats round-trip: the stats dict when online, None when unreachable."""
        try:
            return self.get_system_stats(timeout=timeout)
        except Exception:
            return None

    def get_system_stats(self, timeout: float = 5) -> Dict[str, Any]:
        """Fetches ComfyUI system stats including GPU VRAM and torch status."""
        req = urllib.request.Request(f"{self.base_url}/system_stats")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def upload_video(self, file_path: Path | str) -> str:
        """Copies video to ComfyUI input directory and returns filename."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        comfy_input = config.INPUT_DIR_COMFY
        comfy_input.mkdir(parents=True, exist_ok=True)

        target_path = comfy_input / file_path.name
        if target_path.resolve() != file_path.resolve():
            shutil.copyfile(file_path, target_path)

        return file_path.name

    def queue_prompt(self, prompt: Dict[str, Any], extra_data: Optional[Dict[str, Any]] = None) -> str:
        """Submits prompt graph to ComfyUI /prompt endpoint and returns prompt_id."""
        payload = {
            "prompt": prompt,
            "client_id": self.client_id
        }
        if extra_data:
            payload["extra_data"] = extra_data

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            prompt_id = res_json.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"ComfyUI did not return prompt_id: {res_json}")
            return prompt_id

    def interrupt(self) -> bool:
        """Interrupts currently executing prompt in ComfyUI."""
        try:
            req = urllib.request.Request(f"{self.base_url}/interrupt", data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"[COMFYUI] Interrupt notice: {e}")
            return False

    def free_memory(self) -> bool:
        """Asks ComfyUI to unload all models and release VRAM (needed before out-of-process GPU workers)."""
        try:
            body = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/free", data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"[COMFYUI] Free-memory notice: {e}")
            return False

    def track_execution(
        self,
        prompt_id: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Tracks execution progress using WebSocket with fallback to HTTP polling.
        """
        try:
            return self._track_websocket(prompt_id, progress_callback)
        except Exception as e:
            print(f"[COMFYUI] WebSocket tracking notice ({e}); falling back to HTTP polling...")
            return self._track_polling(prompt_id, progress_callback)

    def _track_websocket(
        self,
        prompt_id: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        import websocket

        ws = websocket.WebSocket()
        ws_url_with_id = f"{self.ws_url}?clientId={self.client_id}"
        ws.connect(ws_url_with_id)

        last_node = ""
        current_step = 0
        total_steps = 1
        output_data = {}

        try:
            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    msg_type = message.get("type")
                    data = message.get("data", {})

                    if msg_type == "status":
                        pass
                    elif msg_type == "execution_start":
                        if data.get("prompt_id") == prompt_id:
                            print(f"[COMFYUI] Execution started for prompt: {prompt_id}")
                            if progress_callback:
                                progress_callback(32, 100, "Starting GPU neural execution pipeline...")
                    elif msg_type == "executing":
                        node_id = data.get("node")
                        if node_id is None and data.get("prompt_id") == prompt_id:
                            print(f"[COMFYUI] Execution completed successfully!")
                            if progress_callback:
                                progress_callback(100, 100, "Execution completed!")
                            break
                        last_node = str(node_id) if node_id is not None else ""
                        if progress_callback and last_node in NODE_DESCRIPTIONS:
                            pct, desc = NODE_DESCRIPTIONS[last_node]
                            progress_callback(pct, 100, desc)
                        elif progress_callback and last_node:
                            progress_callback(current_step, total_steps, f"Executing node {last_node}...")
                    elif msg_type == "progress":
                        current_step = data.get("value", 0)
                        total_steps = data.get("max", 1)
                        if progress_callback:
                            pct_ratio = current_step / max(1, total_steps)
                            if last_node == "42":
                                p_val = int(65 + pct_ratio * 16)
                                progress_callback(p_val, 100, f"Neural Tiled Diffusion: Step {current_step}/{total_steps} ({int(pct_ratio*100)}%)")
                            elif last_node == "64":
                                p_val = int(87 + pct_ratio * 5)
                                progress_callback(p_val, 100, f"Stage 2 4K Refinement: Step {current_step}/{total_steps} ({int(pct_ratio*100)}%)")
                            else:
                                p_val = int(60 + pct_ratio * 30)
                                progress_callback(p_val, 100, f"Diffusion step {current_step}/{total_steps} ({int(pct_ratio*100)}%)")
                    elif msg_type == "executed":
                        if data.get("prompt_id") == prompt_id:
                            node_id = data.get("node")
                            if "output" in data:
                                output_data[str(node_id)] = data["output"]
                    elif msg_type == "execution_error":
                        if data.get("prompt_id") == prompt_id:
                            error_msg = data.get("exception_message", "Unknown execution error")
                            raise RuntimeError(f"ComfyUI Execution Error: {error_msg}")
        finally:
            ws.close()

        return output_data

    def _track_polling(
        self,
        prompt_id: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """Polls ComfyUI /history endpoint until prompt_id is present."""
        poll_interval = 2.0
        elapsed = 0

        not_in_queue_count = 0
        while True:
            time.sleep(poll_interval)
            elapsed += int(poll_interval)

            try:
                # 1. Check history first
                req = urllib.request.Request(f"{self.base_url}/history/{prompt_id}")
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if prompt_id in data:
                        job_info = data[prompt_id]
                        status = job_info.get("status", {})
                        if status.get("status_str") == "error":
                            messages = status.get("messages", [])
                            # Same concise message the WebSocket path reports, not the full event log
                            errors = [m[1] for m in messages if isinstance(m, (list, tuple)) and len(m) == 2 and m[0] == "execution_error"]
                            if errors:
                                err = errors[-1]
                                error_msg = f"{err.get('node_type', 'node')} #{err.get('node_id', '?')}: {err.get('exception_message', 'Unknown execution error').strip()}"
                            else:
                                error_msg = "Unknown execution error"
                            raise RuntimeError(f"ComfyUI Execution Error: {error_msg}")

                        outputs = job_info.get("outputs", {})
                        print(f"[COMFYUI] Execution finished according to history.")
                        return outputs

                # 2. Check active queue
                q_req = urllib.request.Request(f"{self.base_url}/queue")
                with urllib.request.urlopen(q_req, timeout=8) as resp:
                    q_data = json.loads(resp.read().decode("utf-8"))
                    running = q_data.get("queue_running", [])
                    pending = q_data.get("queue_pending", [])
                    is_running = any(item[1] == prompt_id for item in running)
                    is_pending = any(item[1] == prompt_id for item in pending)

                    if is_running:
                        not_in_queue_count = 0
                        if progress_callback:
                            progress_callback(50, 100, f"Rendering on GPU ({elapsed}s elapsed)...")
                    elif is_pending:
                        not_in_queue_count = 0
                        if progress_callback:
                            progress_callback(10, 100, f"Queued, waiting to execute ({elapsed}s elapsed)...")
                    else:
                        # Job is no longer in running or pending
                        not_in_queue_count += 1
                        if not_in_queue_count >= 3:
                            print(f"[COMFYUI] Prompt {prompt_id} is no longer in queue. Checking output...")
                            return {}

            except RuntimeError:
                raise
            except Exception as e:
                print(f"[COMFYUI Polling Notice] {e}")


def _format_eta(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{max(minutes, 1)}m"


# Resident out-of-process RelayVSR worker: reused across jobs, killed on cancel, released when idle
_ACTIVE_WORKER: Dict[str, Optional[subprocess.Popen]] = {"proc": None}
_WORKER_LOCK = threading.Lock()
# RelayVSR output is written in 10 s (at 29.97 fps) segments: the most a stopped job can lose
RELAYVSR_SEGMENT_FRAMES = 300
# Set by terminate_active_worker so a job waiting between worker restarts also stops
_CANCEL = threading.Event()
# A starved worker (VRAM spilled into system RAM) is restarted at most this many times per job
RELAYVSR_MAX_RECYCLES = 3
# How long a restart waits for other apps to give back VRAM before loading anyway
RELAYVSR_VRAM_WAIT_S = 60
# Worker footprint (weights + 240p activations + CUDA context) until it reports its own
RELAYVSR_DEFAULT_NEED_GB = 12.5


class VideoRestorationPipeline:
    """End-to-End Orchestrator for LTX-2.5 Video Restoration."""

    def __init__(self, client: Optional[ComfyUIClient] = None):
        self.client = client or ComfyUIClient()

    def restore_footage(
        self,
        input_video_path: Path | str,
        output_dir: Optional[Path | str] = None,
        output_file: Optional[Path | str] = None,
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        engine_type: str = "superres",  # 'superres' (Path A) or 'diffusion' (Path B)
        superres_model: str = "RealESRGAN_x2plus.pth",
        mode: str = "restore",  # 'restore', 'refine', or 'two_stage'
        prompt_preset: str = "archival_general",
        custom_positive_prompt: Optional[str] = None,
        custom_negative_prompt: Optional[str] = None,
        output_size: str = "FullHD",  # 'FullHD', '4K', '8K'
        tile_size: str = "FullHD",    # 'qHD', 'HD', 'FullHD'
        seed: Optional[int] = None,
        guide_strength: float = 0.72,
        cfg: float = 2.5,
        sol_sigma: float = 0.50,
        reference_image: Optional[Path | str] = None,
        extract_depth: bool = False,
        clean_noise: Optional[bool] = None,
        stabilize_jitter: Optional[bool] = None,
        enable_teacache: Optional[bool] = None,
        teacache_thresh: float = 0.06,
        auto_compare: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        output_codec: str = "auto",  # 'auto' (best HEVC encoder, else H.264), 'hevc', 'h264'
    ) -> Dict[str, Any]:
        input_path = Path(input_video_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        if engine_type != "relayvsr":
            self.release_relayvsr_worker()  # hand its ~9.4 GB of VRAM back before ComfyUI work

        # Resolve output directory and unique collision-free filename.
        # (The remembered output folder is only changed by the user's own save-dialog choice —
        # running a job, CLI call or test must not redirect where later jobs are written.)
        if output_file:
            final_output_path = Path(output_file)
            out_folder = final_output_path.parent
        else:
            if output_dir:
                out_folder = Path(output_dir)
            else:
                out_folder = config.get_last_output_folder()

            out_folder.mkdir(parents=True, exist_ok=True)
            if engine_type == "superres":
                mode_tag = "superres"
            elif engine_type == "relayvsr":
                mode_tag = "relayvsr"
            elif engine_type == "sol_refiner":
                mode_tag = "sol_refiner"
            else:
                mode_tag = mode if mode != "restore" else "restored"

            final_output_path = config.get_unique_output_path(
                output_dir=out_folder,
                base_stem=f"{input_path.stem}_{mode_tag}",
                suffix=".mp4"
            )

        # Resolve preset defaults for teacache if not explicitly set
        preset_cfg = config.RESTORATION_PRESETS.get(prompt_preset, {})
        if enable_teacache is None:
            enable_teacache = preset_cfg.get("enable_teacache", False)
        if teacache_thresh is None or teacache_thresh <= 0:
            teacache_thresh = preset_cfg.get("teacache_thresh", 0.06)

        print(f"\n=======================================================")
        print(f" STARTING FOOTAGE RESTORATION: {input_path.name}")
        print(f" Engine: {engine_type.upper()} ({superres_model if engine_type == 'superres' else mode})")
        print(f" Target Output: {final_output_path.name}")
        print(f" Destination: {out_folder}")
        print(f" Output Resolution: {output_size} | Tile: {tile_size}")
        if engine_type == "sol_refiner":
            print(f" SoL Refinement Strength (Sigma): {sol_sigma}")
        elif engine_type == "relayvsr":
            print(f" RelayVSR Streaming 4x (keyframe window 16, low-res canvas <= {config.RELAYVSR_MAX_LR_HEIGHT}p)")
        elif engine_type != "superres":
            print(f" Guide Strength: {guide_strength} | CFG: {cfg}")
            if enable_teacache:
                print(f" TeaCache Acceleration: ENABLED (~2x faster, threshold={teacache_thresh})")
        if start_frame is not None or end_frame is not None:
            print(f" In/Out Range: Frame {start_frame or 0} -> {end_frame or 'End'}")
        print(f"=======================================================")

        # Step 1: Preprocess Video (GPU NVENC accelerated with In/Out frame slicing)
        if progress_callback:
            progress_callback(5, 100, f"Preprocessing canvas with GPU NVENC ({'Direct Canvas' if engine_type == 'superres' else '32-multiple, 8n+1 frames'})...")
        print("\n[STEP 1/5] Inspecting & Preprocessing input footage (GPU NVENC)...")
        raw_info = video_preprocessor.get_video_info(input_path)
        total_src_frames = raw_info["total_frames"]
        print(f"  Input: {raw_info['width']}x{raw_info['height']} @ {raw_info['fps']} fps, {total_src_frames} frames")
        print(f"  Interlaced: {raw_info['is_interlaced']} | Has Audio: {raw_info['has_audio']}")

        # Determine target frame span
        start_f = max(0, int(start_frame)) if start_frame is not None else 0
        end_f = min(total_src_frames - 1, int(end_frame)) if end_frame is not None else total_src_frames - 1
        if max_frames and (end_f - start_f + 1) > max_frames:
            end_f = start_f + max_frames - 1
        span_frames = max(1, end_f - start_f + 1)

        # Smart Auto-Chunking: If footage exceeds the safe single-pass limit,
        # automatically process across consecutive legal chunks and seamlessly stitch with audio.
        # RelayVSR streams frame-by-frame with bounded memory, so it never needs chunking.
        safe_single_limit = 200 if engine_type == "superres" else 97
        if span_frames > safe_single_limit and engine_type != "relayvsr":
            chunk_step = 200 if engine_type == "superres" else 33
            print(f"\n>>> [SMART AUTO-CHUNKER ENGAGED]")
            print(f"   Video span is {span_frames} frames. Automatically chunking into {chunk_step}-frame blocks to prevent memory crash.")
            return self._restore_chunked(
                input_video_path=input_path,
                start_f=start_f,
                end_f=end_f,
                chunk_size=chunk_step,
                output_file=final_output_path,
                engine_type=engine_type,
                superres_model=superres_model,
                mode=mode,
                prompt_preset=prompt_preset,
                custom_positive_prompt=custom_positive_prompt,
                custom_negative_prompt=custom_negative_prompt,
                output_size=output_size,
                tile_size=tile_size,
                seed=seed,
                guide_strength=guide_strength,
                cfg=cfg,
                sol_sigma=sol_sigma,
                reference_image=reference_image,
                extract_depth=extract_depth,
                clean_noise=clean_noise,
                stabilize_jitter=stabilize_jitter,
                enable_teacache=enable_teacache,
                teacache_thresh=teacache_thresh,
                auto_compare=auto_compare,
                progress_callback=progress_callback
            )

        # Create unique prepared target path incorporating In/Out frame tags to avoid file collisions
        range_tag = f"_f{start_frame or 0}_{end_frame}" if (start_frame is not None or end_frame is not None) else ""
        clean_stem = input_path.stem.replace("_prepared", "")
        prep_target = config.INPUTS_DIR / f"{clean_stem}{range_tag}_prepared.mp4"

        # Super-Resolution does not require strict 8n+1 frames
        align_frames = engine_type not in ("superres", "relayvsr")

        # Calculate target scaling dimensions for Super-Resolution
        target_w = None
        target_h = None
        if engine_type == "superres":
            aspect = raw_info["width"] / raw_info["height"] if raw_info["height"] > 0 else 16/9
            if output_size in ["HD", "FullHD"]:
                target_base = 720 if output_size == "HD" else 1080
                if raw_info["width"] >= raw_info["height"]:
                    target_h = target_base
                    target_w = int(round(target_base * aspect))
                else:
                    target_w = target_base
                    target_h = int(round(target_base / aspect))
                target_w = target_w - (target_w % 2)
                target_h = target_h - (target_h % 2)
            elif output_size in ["4K", "8K"]:
                target_base = 2160 if output_size == "4K" else 4320
                if raw_info["width"] >= raw_info["height"]:
                    target_h = target_base
                    target_w = int(round(target_base * aspect))
                else:
                    target_w = target_base
                    target_h = int(round(target_base / aspect))
                target_w = target_w - (target_w % 2)
                target_h = target_h - (target_h % 2)

        should_clean = clean_noise if clean_noise is not None else (engine_type == "superres")
        should_stabilize = stabilize_jitter if stabilize_jitter is not None else (engine_type == "superres")
        is_relay = engine_type == "relayvsr"
        if is_relay:
            # Upscale-only engine trained on real degraded footage: pre-denoising/deflickering
            # (e.g. defect ticks set by auto-detect) would only strip detail it can recover
            should_clean = should_stabilize = False

        # A stopped RelayVSR job resumes from its finished segments, so its prepared input is reused as-is
        relay_done_segments = self._relayvsr_resume_state(
            final_output_path, {"input": str(input_path.resolve()), "start": start_f, "end": end_f,
                                "output_size": output_size, "segment_frames": RELAYVSR_SEGMENT_FRAMES},
        ) if is_relay else 0
        if relay_done_segments and prep_target.exists():
            prepared_video_path, prep_info = prep_target, video_preprocessor.get_video_info(prep_target)
            print(f"  Resuming: reusing prepared input and {relay_done_segments} finished segment(s)")
        else:
            prepared_video_path, prep_info = video_preprocessor.preprocess_video(
                input_path=input_path,
                output_path=prep_target,
                # Deinterlacing progressive footage only softens it; RelayVSR gets it when actually needed
                force_deinterlace=raw_info["is_interlaced"] if is_relay else True,
                clean_noise=should_clean,
                stabilize_jitter=should_stabilize,
                align_frames=align_frames,
                max_frames=max_frames,
                start_frame=start_frame or 0,
                end_frame=end_frame
            )
        print(f"  Prepared: {prep_info['width']}x{prep_info['height']}")
        print(f"  Frames: {prep_info['total_frames']} (range: F {prep_info.get('start_frame', 0)}..{prep_info.get('end_frame', prep_info['total_frames']-1)})")

        # Step 2: Audio Extraction from prepared trimmed video for lossless remux
        audio_extracted_path = None
        if raw_info["has_audio"]:
            if progress_callback:
                progress_callback(10, 100, "Extracting audio track for lossless preservation...")
            print("\n[STEP 2/5] Extracting synchronized audio stream from prepared canvas...")
            audio_extracted_path = video_preprocessor.extract_audio(
                prepared_video_path, config.INPUTS_DIR / f"{final_output_path.stem}_audio.m4a"
            )
            if audio_extracted_path and audio_extracted_path.exists():
                print(f"  Extracted audio: {audio_extracted_path.name} ({audio_extracted_path.stat().st_size / (1024*1024):.2f} MB)")

        if engine_type == "relayvsr":
            return self._restore_relayvsr(
                input_path=input_path,
                prepared_video_path=prepared_video_path,
                prep_info=prep_info,
                audio_extracted_path=audio_extracted_path,
                final_output_path=final_output_path,
                output_size=output_size,
                seed=seed if seed is not None else 42,
                prompt_preset=prompt_preset,
                auto_compare=auto_compare,
                progress_callback=progress_callback,
                done_segments=relay_done_segments,
                output_codec=output_codec,
            )

        # Step 2b: 3D Scene Depth Geometry Extraction (if requested)
        depth_extracted_path = None
        if extract_depth:
            if progress_callback:
                progress_callback(12, 100, "Extracting 3D scene depth geometry (Depth Anything V2)...")
            print("\n[3D GEOMETRY] Extracting frame-by-frame 3D scene depth geometry via Depth Anything V2...")
            depth_target = config.INPUTS_DIR / f"{prepared_video_path.stem}_depth.mp4"
            try:
                depth_extracted_path = video_preprocessor.extract_depth_sequence(
                    prepared_video_path,
                    output_path=depth_target,
                    progress_callback=progress_callback
                )
                print(f"  Extracted 3D metric depth sequence: {depth_extracted_path.name}")
            except Exception as depth_err:
                print(f"  [Depth Estimation Notice] Depth extraction skipped or errored: {depth_err}")

        # Step 2c: Character Likeness Reference Image Alignment & Ingestion (if provided)
        uploaded_ref_filename = None
        if reference_image:
            ref_path = Path(reference_image)
            if not ref_path.exists():
                raise FileNotFoundError(f"Reference image not found: {ref_path}")
            print(f"\n[CHARACTER GUIDANCE] Preparing Character Reference Portrait: {ref_path.name}")
            if progress_callback:
                progress_callback(14, 100, f"Aligning & uploading reference image ({ref_path.name})...")
            prepared_ref_path = video_preprocessor.prepare_reference_image(ref_path)
            uploaded_ref_filename = self.client.upload_video(prepared_ref_path)
            print(f"  Reference image prepared (32-multiple aligned) and uploaded: {uploaded_ref_filename}")

        # Step 3: Check Model Status
        if progress_callback:
            progress_callback(15, 100, "Verifying model weights...")

        output_prefix = final_output_path.stem
        actual_seed = seed if seed is not None else int(time.time() * 1000) % 1000000000

        if engine_type == "superres":
            # Check upscale model file
            model_file = config.UPSCALE_MODELS_DIR / superres_model
            if not model_file.exists():
                raise FileNotFoundError(f"Super-resolution model not found: {model_file}")

            # Upload to ComfyUI & build superres prompt graph
            if progress_callback:
                progress_callback(20, 100, f"Uploading video & configuring {superres_model}...")
            print(f"\n[STEP 3/5] Uploading video to ComfyUI input storage ({superres_model})...")
            uploaded_filename = self.client.upload_video(prepared_video_path)

            prompt_graph = build_superres_api_prompt(
                video_filename=uploaded_filename,
                output_prefix=output_prefix,
                upscale_model_name=superres_model,
                target_width=target_w,
                target_height=target_h,
            )
        elif engine_type == "sol_refiner":
            # NVIDIA SoL-Refiner Model Verification
            status_models = model_downloader.get_model_status()
            required_keys = ["sol_transformer", "sol_vae", "spatial_upscaler", "text_encoder"]
            missing_models = [k for k in required_keys if not status_models[k]["installed"]]

            if missing_models:
                print(f"\n[NOTICE] NVIDIA SoL-Refiner neural weights not yet installed: {missing_models}")
                print(f"  Delivering GPU NVENC preprocessed video to: {final_output_path}")

                if progress_callback:
                    progress_callback(85, 100, "Saving GPU-preprocessed video to destination...")

                shutil.copyfile(prepared_video_path, final_output_path)

                if progress_callback:
                    progress_callback(100, 100, "Done (Weights missing; preprocessed video saved)")

                return {
                    "status": "prepared_only",
                    "models_missing": True,
                    "missing_list": missing_models,
                    "input_file": str(input_path),
                    "prepared_input": str(prepared_video_path),
                    "final_output_path": str(final_output_path),
                    "output_prefix": final_output_path.stem,
                    "mode": "sol_refiner",
                    "preset": prompt_preset,
                    "seed": 42,
                    "dimensions": f"{prep_info['width']}x{prep_info['height']}",
                    "frames": prep_info["total_frames"],
                    "has_audio": raw_info["has_audio"],
                    "audio_file": str(audio_extracted_path) if audio_extracted_path else None
                }

            if progress_callback:
                progress_callback(20, 100, "Uploading video & building NVIDIA 1-Step SoL-Refiner prompt...")
            print("\n[STEP 3/5] Uploading video to ComfyUI input storage (SoL-Refiner)...")
            uploaded_filename = self.client.upload_video(prepared_video_path)

            preset = config.PROMPT_PRESETS.get(prompt_preset, config.PROMPT_PRESETS["archival_general"])
            pos_prompt = custom_positive_prompt or preset["positive"]
            neg_prompt = custom_negative_prompt or preset["negative"]

            prompt_graph = build_sol_refiner_api_prompt(
                video_filename=uploaded_filename,
                output_prefix=output_prefix,
                seed=actual_seed,
                pos_prompt=pos_prompt,
                neg_prompt=neg_prompt,
                sigma=sol_sigma,
                ref_image_filename=uploaded_ref_filename,
            )
        else:
            # LTX-2.5 Diffusion Model Verification
            status_models = model_downloader.get_model_status()
            required_keys = ["transformer", "text_encoder", "video_vae"]
            if mode == "restore":
                required_keys.append("lora_restore")
            elif mode == "refine":
                required_keys.append("lora_refine")
            elif mode == "cq_enhancer":
                required_keys.append("lora_cq_enhancer")
            else:
                required_keys.extend(["lora_restore", "lora_refine", "spatial_upscaler"])

            missing_models = [k for k in required_keys if not status_models[k]["installed"]]

            if missing_models:
                print(f"\n[NOTICE] LTX-2.5 neural model weights not yet installed: {missing_models}")
                print(f"  Delivering GPU NVENC preprocessed video to: {final_output_path}")

                if progress_callback:
                    progress_callback(85, 100, "Saving GPU-preprocessed video to destination...")

                shutil.copyfile(prepared_video_path, final_output_path)

                if progress_callback:
                    progress_callback(100, 100, "Done (Weights missing; preprocessed video saved)")

                return {
                    "status": "prepared_only",
                    "models_missing": True,
                    "missing_list": missing_models,
                    "input_file": str(input_path),
                    "prepared_input": str(prepared_video_path),
                    "final_output_path": str(final_output_path),
                    "output_prefix": final_output_path.stem,
                    "mode": mode,
                    "preset": prompt_preset,
                    "seed": 42,
                    "dimensions": f"{prep_info['width']}x{prep_info['height']}",
                    "frames": prep_info["total_frames"],
                    "has_audio": raw_info["has_audio"],
                    "audio_file": str(audio_extracted_path) if audio_extracted_path else None,
                    "reference_image": str(reference_image) if reference_image else None,
                    "depth_output_path": str(depth_extracted_path) if depth_extracted_path else None,
                }

            # Step 4: Upload to ComfyUI & Build API Prompt
            if progress_callback:
                progress_callback(20, 100, "Uploading video & building ComfyUI prompt...")
            print("\n[STEP 3/5] Uploading video to ComfyUI input storage...")
            uploaded_filename = self.client.upload_video(prepared_video_path)

            preset = config.PROMPT_PRESETS.get(prompt_preset, config.PROMPT_PRESETS["archival_general"])
            pos_prompt = custom_positive_prompt or preset["positive"]
            neg_prompt = custom_negative_prompt or preset["negative"]

            prompt_graph = build_restoration_api_prompt(
                video_filename=uploaded_filename,
                output_prefix=output_prefix,
                mode=mode,
                tile_size=tile_size,
                output_size=output_size,
                seed=actual_seed,
                pos_prompt=pos_prompt,
                neg_prompt=neg_prompt,
                guide_strength=guide_strength,
                cfg=cfg,
                ref_image_filename=uploaded_ref_filename,
                enable_teacache=enable_teacache,
                teacache_thresh=teacache_thresh,
            )

        run_wf_path = config.WORKFLOWS_DIR / f"active_run_{output_prefix}.json"
        with open(run_wf_path, "w", encoding="utf-8") as f:
            json.dump(prompt_graph, f, indent=2)
        print(f"  Saved runnable API prompt: {run_wf_path}")

        # Step 5: Queue & Execute via ComfyUI
        if not self.client.is_online():
            raise ConnectionError(f"ComfyUI is offline at {self.client.base_url}")

        if progress_callback:
            progress_callback(25, 100, "Queueing neural restoration on RTX GPU...")
        print(f"\n[STEP 4/5] Queueing restoration job to ComfyUI (client {self.client.client_id})...")
        prompt_id = self.client.queue_prompt(prompt_graph)
        print(f"  Queued prompt ID: {prompt_id}")

        if progress_callback:
            progress_callback(30, 100, "Initializing neural pipeline on GPU...")

        output_data = self.client.track_execution(
            prompt_id=prompt_id,
            progress_callback=progress_callback
        )

        # Step 6: Retrieve Rendered Output & Mux Audio
        if progress_callback:
            progress_callback(92, 100, "Muxing lossless audio and finalizing output...")
        print("\n[STEP 5/5] Retrieving generated video and finalizing audio...")

        comfy_out_dir = config.OUTPUT_DIR_COMFY
        generated_candidates = list(comfy_out_dir.glob(f"*{output_prefix}*.mp4"))

        if not generated_candidates:
            # Fallback search for recently modified mp4 in comfy_out_dir
            all_recent = sorted(
                list(comfy_out_dir.glob("*.mp4")),
                key=os.path.getmtime,
                reverse=True
            )
            if all_recent:
                latest_vid = all_recent[0]
            else:
                raise FileNotFoundError(f"Could not find rendered video for prefix {output_prefix} in {comfy_out_dir}")
        else:
            latest_vid = sorted(generated_candidates, key=os.path.getmtime, reverse=True)[0]

        print(f"  Retrieved AI rendered video: {latest_vid.name} ({latest_vid.stat().st_size / (1024*1024):.2f} MB)")

        final_info, comparison_res = self._deliver_output(
            latest_vid, audio_extracted_path, final_output_path, prepared_video_path,
            auto_compare, progress_callback, keep_source=True, done_text="Restoration Complete!",
            output_codec=output_codec,
        )

        return {
            "status": "completed",
            "models_missing": False,
            "missing_list": [],
            "workflow_file": str(run_wf_path),
            "input_file": str(input_path),
            "prepared_input": str(prepared_video_path),
            "final_output_path": str(final_output_path),
            "output_prefix": output_prefix,
            "mode": mode,
            "engine_type": engine_type,
            "superres_model": superres_model if engine_type == "superres" else None,
            "reference_image": str(reference_image) if reference_image else None,
            "depth_output_path": str(depth_extracted_path) if depth_extracted_path else None,
            "preset": prompt_preset,
            "enable_teacache": enable_teacache,
            "teacache_thresh": teacache_thresh,
            "seed": actual_seed,
            "dimensions": f"{final_info['width']}x{final_info['height']}",
            "frames": final_info["total_frames"],
            "has_audio": final_info["has_audio"],
            "audio_file": str(audio_extracted_path) if audio_extracted_path else None,
            "comparison": comparison_res,
            "comparison_path": comparison_res.get("side_by_side") if comparison_res else None
        }

    @staticmethod
    def relayvsr_geometry(src_w: int, src_h: int, output_size: str) -> Tuple[int, int]:
        """Returns (lr_height, out_height) for the RelayVSR worker.

        The short side of the source is capped at RELAYVSR_MAX_LR_HEIGHT for the 4x
        pass (VRAM bound), and the 4x result is resampled so its short side matches
        the requested output tier.
        """
        target_short = {"HD": 720, "FullHD": 1080, "4K": 2160, "8K": 4320}.get(output_size, 720)
        short_side = min(src_w, src_h)
        lr_short = min(short_side, config.RELAYVSR_MAX_LR_HEIGHT)
        # Worker heights are expressed in source-height units
        to_height = src_h / short_side
        return int(round(lr_short * to_height)), int(round(target_short * to_height))

    @staticmethod
    def terminate_active_worker() -> bool:
        """Kills the RelayVSR worker mid-job (queue cancellation); the next job starts a fresh one."""
        _CANCEL.set()
        proc = _ACTIVE_WORKER.get("proc")
        if proc is not None and proc.poll() is None:
            proc.kill()
            return True
        return False

    @staticmethod
    def release_relayvsr_worker() -> None:
        """Shuts down the resident RelayVSR worker and frees its ~9.4 GB of VRAM.
        Called when the queue goes idle, before non-RelayVSR jobs, and at interpreter exit."""
        with _WORKER_LOCK:
            proc = _ACTIVE_WORKER.get("proc")
            _ACTIVE_WORKER["proc"] = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.stdin.close()  # serve loop ends on EOF
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
        print("[RelayVSR] Worker released (VRAM freed).")

    @staticmethod
    def _relayvsr_parts_dir(final_output_path: Path) -> Path:
        # Kept apart from inputs/ and outputs/ so clearing those doesn't destroy resumable work
        return config.RESUME_DIR / f"{final_output_path.stem}_relayvsr_parts"

    def _relayvsr_resume_state(self, final_output_path: Path, key: Dict[str, Any]) -> int:
        """Number of finished segments a previous run of this exact job left behind (0 = fresh start).
        A different source/range/size invalidates them; the folder is reset for the new run."""
        parts_dir = self._relayvsr_parts_dir(final_output_path)
        manifest = parts_dir / "manifest.json"
        try:
            if json.loads(manifest.read_text(encoding="utf-8")) == key:
                for leftover in parts_dir.glob(".refvsr-*"):  # half-written segment of a killed run
                    leftover.unlink(missing_ok=True)
                done = 0
                while (parts_dir / f"seg_{done:05d}.mp4").exists():
                    done += 1
                return done
        except (OSError, ValueError):
            pass
        shutil.rmtree(parts_dir, ignore_errors=True)
        parts_dir.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(key), encoding="utf-8")
        return 0

    @staticmethod
    def _concat_segments(segments: List[Path], out_path: Path) -> None:
        """Joins same-codec segments losslessly (stream copy) via ffmpeg's concat demuxer."""
        list_file = out_path.with_suffix(".concat.txt")
        list_file.write_text(
            "".join("file '{}'\n".format(s.resolve().as_posix().replace("'", "'\\''")) for s in segments),
            encoding="utf-8",
        )
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(list_file),
                 "-c", "copy", "-movflags", "+faststart", str(out_path)],
                check=True, capture_output=True, text=True,
            )
        finally:
            list_file.unlink(missing_ok=True)

    def _save_relayvsr_partial(self, parts_dir: Path, audio_path: Optional[Path], final_output_path: Path) -> Optional[Path]:
        """Joins the finished segments of a stopped job into <name>_partial.mp4 (audio trimmed to match).
        The segments stay in place so the job can resume."""
        segments = sorted(parts_dir.glob("seg_*.mp4"))
        if not segments:
            return None
        partial = final_output_path.with_name(f"{final_output_path.stem}_partial.mp4")
        joined = parts_dir / "partial_joined.mp4"
        try:
            self._concat_segments(segments, joined)
            if audio_path and audio_path.exists():
                video_preprocessor.mux_audio_and_video(joined, audio_path, partial)
            else:
                shutil.copyfile(joined, partial)
        except Exception as e:
            print(f"[RelayVSR] Could not assemble partial output: {e}")
            return None
        finally:
            joined.unlink(missing_ok=True)
        print(f"[RelayVSR] Saved finished part ({len(segments) * RELAYVSR_SEGMENT_FRAMES} frames max): {partial}")
        return partial

    @staticmethod
    def _pump_relayvsr(
        proc: subprocess.Popen,
        stop_tokens: Tuple[str, ...],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        speed: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], List[str]]:
        """Relays worker stdout until a line starting with one of stop_tokens.
        Returns (that line, last 30 lines); the line is None if the process exited first.
        SEGMENT lines update `speed` (fps, best, slow, reserved), which feeds the speed/ETA readout."""
        tail: List[str] = []
        speed = {} if speed is None else speed
        for line in iter(proc.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            tail = (tail + [line])[-30:]
            if line.startswith(stop_tokens):
                return line, tail
            if line.startswith("PROGRESS:"):
                try:
                    cur, tot = (int(v) for v in line.split("PROGRESS:")[1].strip().split("/"))
                    if progress_callback:
                        text = f"RelayVSR upscaling frame {cur}/{tot}"
                        if speed.get("fps"):
                            text += f" · {speed['fps']:.2f} fps · ~{_format_eta((tot - cur) / speed['fps'])} left"
                        if speed.get("slow"):
                            text += " · ⚠ slowed: GPU memory full — close other GPU apps"
                        progress_callback(25 + int(65 * cur / max(tot, 1)), 100, text)
                except Exception:
                    pass
                continue
            if line.startswith("SEGMENT:"):
                # SEGMENT: <n> <fps> fps judged=<0|1> slow=<0|1> reserved=<GB>
                try:
                    parts = line.split()
                    fields = dict(p.split("=", 1) for p in parts[4:])
                    speed["fps"] = float(parts[2])
                    speed["slow"] = fields.get("slow") == "1"
                    speed["reserved"] = max(speed.get("reserved", 0.0), float(fields.get("reserved", 0)))
                    if fields.get("judged") == "1":
                        speed["best"] = max(speed.get("best", 0.0), speed["fps"])
                        speed["run_best"] = max(speed.get("run_best", 0.0), speed["fps"])
                except (ValueError, IndexError):
                    pass
            print(f"  [RelayVSR] {line}")
        return None, tail

    @staticmethod
    def _gpu_free_gb() -> Optional[float]:
        """Free VRAM on the first GPU as the driver sees it (all processes), or None if unknown."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            ).stdout.splitlines()[0]
            used, total = (float(v) for v in out.split(","))
            return (total - used) / 1024
        except Exception:
            return None

    def _wait_for_vram(self, need_gb: float, progress_callback: Optional[Callable[[int, int, str], None]]) -> None:
        """After the starved worker is released, gives other apps up to RELAYVSR_VRAM_WAIT_S to leave
        room for a fully-resident restart. Loads anyway afterwards (spilled is slower, not broken)."""
        self.client.free_memory()
        deadline = time.monotonic() + RELAYVSR_VRAM_WAIT_S
        while True:
            if _CANCEL.is_set():
                raise RuntimeError("RelayVSR job canceled while waiting for GPU memory.")
            free = self._gpu_free_gb()
            if free is None or free >= need_gb:
                return
            if time.monotonic() >= deadline:
                print(f"  [RelayVSR] Only {free:.1f} GB VRAM free (needs ~{need_gb:.1f} GB); restarting anyway.")
                return
            if progress_callback:
                progress_callback(25, 100, f"GPU memory is full ({free:.1f} of ~{need_gb:.1f} GB free) — "
                                           "close other GPU apps to restore full speed; waiting...")
            time.sleep(3)

    def _relayvsr_server(self, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> subprocess.Popen:
        """Returns the resident RelayVSR worker, starting it (and loading its weights) only if needed."""
        with _WORKER_LOCK:
            proc = _ACTIVE_WORKER.get("proc")
            if proc is not None and proc.poll() is None:
                print("\n[STEP 3/5] Reusing resident RelayVSR worker (models already loaded)")
                return proc

            # RelayVSR needs ~12 GB; make ComfyUI drop any resident LTX weights first
            # (the ~35 s model load that follows gives ComfyUI ample time to actually release it)
            print("\n[STEP 3/5] Releasing ComfyUI VRAM and loading RelayVSR...")
            self.client.free_memory()
            if progress_callback:
                progress_callback(20, 100, "Loading RelayVSR (Wan keyframes + Flash relay) on GPU...")
            proc = subprocess.Popen(
                [str(config.RELAYVSR_PYTHON), str(config.RELAYVSR_WORKER), "--serve",
                 "--weights-dir", str(config.RELAYVSR_WEIGHTS_DIR)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            _ACTIVE_WORKER["proc"] = proc
        last, tail = self._pump_relayvsr(proc, ("SERVE_READY",))
        if last is None:
            self.release_relayvsr_worker()
            raise RuntimeError("RelayVSR worker failed to start:\n" + "\n".join(tail))
        return proc

    def _deliver_output(
        self,
        rendered: Path,
        audio_path: Optional[Path],
        final_output_path: Path,
        prepared_video_path: Path,
        auto_compare: bool,
        progress_callback: Optional[Callable[[int, int, str], None]],
        keep_source: bool,
        done_text: str,
        output_codec: str = "auto",
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Muxes audio (AAC) onto the rendered video at its final path, re-encoding the video to HEVC
        on the best encoder this machine has unless output_codec is 'h264' (or none works), then
        renders comparisons. keep_source=False consumes `rendered` (a temp file) instead of copying it."""
        codec, encoder = video_encoders.resolve(output_codec)
        if progress_callback:
            progress_callback(92, 100, f"Encoding HEVC on {encoder.label} and adding audio..." if encoder
                              else "Adding audio and finalizing output...")
        print(f"  Output video: {video_encoders.describe(output_codec)}")
        if encoder is not None or (audio_path and audio_path.exists()):
            try:
                video_preprocessor.mux_audio_and_video(rendered, audio_path, final_output_path, encoder=encoder)
            except subprocess.CalledProcessError as e:
                if encoder is None:
                    raise
                # Never lose a finished render to the final re-encode (driver hiccup, odd frame size):
                # deliver the rendered H.264 stream instead
                print(f"  [Output] {encoder.label} HEVC encode failed ({(e.stderr or '').strip()[-200:]}); keeping H.264.")
                video_preprocessor.mux_audio_and_video(rendered, audio_path, final_output_path, encoder=None)
            if not keep_source:
                rendered.unlink(missing_ok=True)
        elif keep_source:
            shutil.copyfile(rendered, final_output_path)
        else:
            shutil.move(str(rendered), str(final_output_path))

        final_info = video_preprocessor.get_video_info(final_output_path)
        print(f"  Final Video: {final_info['width']}x{final_info['height']} @ {final_info['fps']} fps, {final_info['total_frames']} frames")
        print(f"  Delivered to: {final_output_path}")

        comparison_res = None
        if auto_compare:
            try:
                if progress_callback:
                    progress_callback(96, 100, "Rendering side-by-side & split-screen comparisons...")
                comparison_res = comparison_generator.create_comparison(
                    original_video=prepared_video_path,
                    restored_video=final_output_path,
                    mode="both"
                )
            except Exception as comp_err:
                print(f"  [Comparison Notice] Skipping comparison: {comp_err}")

        if progress_callback:
            progress_callback(100, 100, done_text)
        return final_info, comparison_res

    def _restore_relayvsr(
        self,
        input_path: Path,
        prepared_video_path: Path,
        prep_info: Dict[str, Any],
        audio_extracted_path: Optional[Path],
        final_output_path: Path,
        output_size: str,
        seed: int,
        prompt_preset: str,
        auto_compare: bool,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        done_segments: int = 0,
        output_codec: str = "auto",
    ) -> Dict[str, Any]:
        """Runs RelayVSR in its own virtualenv as a subprocess, then muxes audio & renders comparisons.
        Output is rendered as resumable segments: if the job stops, the finished part is saved as
        <name>_partial.mp4 and re-running the same job continues after `done_segments`."""
        missing = [
            str(p) for p in (
                config.RELAYVSR_PYTHON,
                config.RELAYVSR_WORKER,
                config.RELAYVSR_WEIGHTS_DIR / "wan.safetensors",
                config.RELAYVSR_WEIGHTS_DIR / "flash.safetensors",
            ) if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "RelayVSR is not installed. Missing:\n  " + "\n  ".join(missing)
                + "\nRun: python restore_cli.py download --relayvsr"
            )

        lr_h, out_h = self.relayvsr_geometry(prep_info["width"], prep_info["height"], output_size)
        _CANCEL.clear()

        # Best steady speed ever measured for this LR geometry: the yardstick for "starved"
        lr_w = int(round(prep_info["width"] * lr_h / prep_info["height"]))
        speed_key = f"{lr_w - lr_w % 2}x{lr_h - lr_h % 2}"
        known_fps = config.get_user_setting("relayvsr_fps", {}) or {}
        speed: Dict[str, Any] = {"best": float(known_fps.get(speed_key, 0.0))}

        parts_dir = self._relayvsr_parts_dir(final_output_path)
        recycles, allow_recycle, starved_fps = 0, True, 0.0
        while True:
            try:
                proc = self._relayvsr_server(progress_callback)
            except RuntimeError as e:
                if not recycles:
                    raise
                last, tail = None, [str(e)]  # canceled/failed during a restart: keep the finished part
                break
            print(f"\n[STEP 4/5] Running RelayVSR streaming 4x (LR {lr_h}p -> output {out_h}p)...")
            if progress_callback:
                progress_callback(25, 100, "RelayVSR models ready — streaming frames...")

            start_frame = done_segments * RELAYVSR_SEGMENT_FRAMES
            if start_frame:
                print(f"  Resuming at frame {start_frame} ({done_segments} segment(s) already rendered)")
            job = {"video": str(prepared_video_path), "segment_dir": str(parts_dir),
                   "segment_frames": RELAYVSR_SEGMENT_FRAMES, "start_frame": start_frame,
                   "lr_height": lr_h, "out_height": out_h, "seed": seed,
                   "baseline_fps": speed["best"],
                   "recycle": allow_recycle and recycles < RELAYVSR_MAX_RECYCLES}
            speed["run_best"] = 0.0
            try:
                proc.stdin.write(json.dumps(job) + "\n")
                proc.stdin.flush()
            except OSError:
                self.release_relayvsr_worker()
                raise RuntimeError("RelayVSR worker exited unexpectedly before accepting the job.")
            last, tail = self._pump_relayvsr(proc, ("JOB_DONE", "JOB_ERROR", "JOB_RECYCLE"), progress_callback, speed)
            if speed["best"] > known_fps.get(speed_key, 0.0):
                known_fps[speed_key] = round(speed["best"], 3)
                config.set_user_setting("relayvsr_fps", known_fps)
            if last is None or not last.startswith("JOB_RECYCLE"):
                break
            done_segments = sum(1 for _ in parts_dir.glob("seg_*.mp4"))
            # The worker stopped exactly at a segment boundary, so no rendered frame is lost
            if recycles and speed["run_best"] < 1.3 * starved_fps:
                # The last restart didn't help (other apps still hold the VRAM): keep the loaded
                # worker and finish at whatever speed is available rather than reloading again
                print("\n[RelayVSR] Restart did not restore speed; continuing on the loaded worker.")
                allow_recycle = False
                continue
            # Starved worker: restart it with fresh allocations once VRAM has room, then resume
            recycles += 1
            starved_fps = speed.get("fps", 0.0)
            print(f"\n[RelayVSR] Speed collapsed ({last.split(':', 1)[1].strip()}); "
                  f"restarting the worker (#{recycles}/{RELAYVSR_MAX_RECYCLES}) and resuming...")
            self.release_relayvsr_worker()
            # torch's reserved pool plus ~0.8 GB of CUDA context and kernels
            need_gb = speed["reserved"] + 0.8 if speed.get("reserved") else RELAYVSR_DEFAULT_NEED_GB
            try:
                self._wait_for_vram(need_gb, progress_callback)
            except RuntimeError as e:
                last, tail = None, [str(e)]
                break
        if last != "JOB_DONE":
            if last is None:  # process died (crash or cancellation); next job starts a fresh one
                self.release_relayvsr_worker()
            partial = self._save_relayvsr_partial(parts_dir, audio_extracted_path, final_output_path)
            note = f"\nFinished part saved: {partial}. Run the job again to resume." if partial else ""
            raise RuntimeError("RelayVSR job stopped:\n" + "\n".join(tail[-5:]) + note)

        raw_out = config.INPUTS_DIR / f"{final_output_path.stem}_relayvsr_raw.mp4"
        self._concat_segments(sorted(parts_dir.glob("seg_*.mp4")), raw_out)

        print("\n[STEP 5/5] Finalizing audio and video...")
        final_info, comparison_res = self._deliver_output(
            raw_out, audio_extracted_path, final_output_path, prepared_video_path,
            auto_compare, progress_callback, keep_source=False, done_text="RelayVSR Upscale Complete!",
            output_codec=output_codec,
        )
        shutil.rmtree(parts_dir, ignore_errors=True)
        final_output_path.with_name(f"{final_output_path.stem}_partial.mp4").unlink(missing_ok=True)

        return {
            "status": "completed",
            "models_missing": False,
            "missing_list": [],
            "workflow_file": None,
            "input_file": str(input_path),
            "prepared_input": str(prepared_video_path),
            "final_output_path": str(final_output_path),
            "output_prefix": final_output_path.stem,
            "mode": "relayvsr",
            "engine_type": "relayvsr",
            "superres_model": config.RELAYVSR_MODEL_ID,
            "reference_image": None,
            "depth_output_path": None,
            "preset": prompt_preset,
            "enable_teacache": False,
            "teacache_thresh": None,
            "seed": seed,
            "dimensions": f"{final_info['width']}x{final_info['height']}",
            "frames": final_info["total_frames"],
            "has_audio": final_info["has_audio"],
            "audio_file": str(audio_extracted_path) if audio_extracted_path else None,
            "comparison": comparison_res,
            "comparison_path": comparison_res.get("side_by_side") if comparison_res else None
        }

    @staticmethod
    def _calculate_chunks(
        start_f: int,
        end_f: int,
        chunk_size: int = 33,
        overlap: int = 8,
        is_diffusion: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Splits an arbitrary frame span into overlapping legal blocks.
        For diffusion: chunks are legal (8n+1) lengths and overlap by `overlap` frames.
        For superres: chunks are contiguous blocks.
        """
        if not is_diffusion:
            chunks = []
            curr = start_f
            while curr <= end_f:
                c_end = min(end_f, curr + chunk_size - 1)
                chunks.append((curr, c_end))
                curr += chunk_size
            return chunks

        chunks = []
        step = max(8, chunk_size - overlap)
        curr = start_f

        while curr <= end_f:
            c_end = curr + chunk_size - 1
            if c_end >= end_f:
                span = end_f - curr + 1
                legal_n = (span - 1) // 8
                legal_span = max(9, legal_n * 8 + 1)
                c_start = end_f - legal_span + 1
                cand = (max(start_f, c_start), end_f)
                if not chunks or cand != chunks[-1]:
                    chunks.append(cand)
                break
            else:
                chunks.append((curr, c_end))
                curr += step

        return chunks

    @staticmethod
    def _blend_and_stitch_chunks(
        chunk_data: List[Dict[str, Any]],
        output_path: Path,
        fps: float = 29.97,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> None:
        """
        Seamlessly stitches chunked videos together using a smooth cosine S-curve cross-fade
        over overlapping frames, eliminating hard block seams and pops entirely.
        """
        if not chunk_data:
            return

        import cv2
        import numpy as np

        cap0 = cv2.VideoCapture(str(chunk_data[0]["path"]))
        ret, f0 = cap0.read()
        cap0.release()
        if not ret or f0 is None:
            raise RuntimeError(f"Could not read first chunk frame: {chunk_data[0]['path']}")
        h, w = f0.shape[:2]

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "17",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        total_chunks = len(chunk_data)
        for idx, cinfo in enumerate(chunk_data):
            if progress_callback:
                pct = int(90 + (idx / max(1, total_chunks)) * 5)
                progress_callback(pct, 100, f"Blending chunk {idx+1}/{total_chunks}...")

            cap = cv2.VideoCapture(str(cinfo["path"]))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            c_start = cinfo["start"]
            c_len = len(frames)

            if idx == 0:
                if total_chunks > 1:
                    next_start = chunk_data[1]["start"]
                    write_until = max(1, next_start - c_start)
                    for f_idx in range(min(write_until, c_len)):
                        proc.stdin.write(frames[f_idx].tobytes())
                else:
                    for frame in frames:
                        proc.stdin.write(frame.tobytes())
            else:
                prev_info = chunk_data[idx - 1]
                prev_cap = cv2.VideoCapture(str(prev_info["path"]))
                prev_frames = []
                while True:
                    ret, frame = prev_cap.read()
                    if not ret:
                        break
                    prev_frames.append(frame)
                prev_cap.release()

                ov_start = c_start
                ov_end = prev_info["end"]
                ov_len = max(0, ov_end - ov_start + 1)

                # Blend overlapping frames using cosine ramp
                for i in range(ov_len):
                    prev_idx = (ov_start + i) - prev_info["start"]
                    curr_idx = i

                    alpha = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / (ov_len + 1)))

                    f_prev = prev_frames[min(max(0, prev_idx), len(prev_frames) - 1)]
                    f_curr = frames[min(max(0, curr_idx), len(frames) - 1)]

                    blended = cv2.addWeighted(f_curr, float(alpha), f_prev, float(1.0 - alpha), 0)
                    proc.stdin.write(blended.tobytes())

                # Write remainder of current chunk
                if idx < total_chunks - 1:
                    next_start = chunk_data[idx + 1]["start"]
                    write_until = next_start - c_start
                else:
                    write_until = c_len

                for i in range(ov_len, write_until):
                    if i < c_len:
                        proc.stdin.write(frames[i].tobytes())

        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg blending failed with code {proc.returncode}")

    def _restore_chunked(
        self,
        input_video_path: Path | str,
        start_f: int,
        end_f: int,
        chunk_size: int,
        output_file: Path,
        engine_type: str,
        superres_model: str,
        mode: str,
        prompt_preset: str,
        custom_positive_prompt: Optional[str],
        custom_negative_prompt: Optional[str],
        output_size: str,
        tile_size: str,
        seed: Optional[int],
        guide_strength: float,
        cfg: float,
        sol_sigma: float,
        reference_image: Optional[Path | str],
        extract_depth: bool,
        clean_noise: Optional[bool],
        stabilize_jitter: Optional[bool],
        enable_teacache: Optional[bool],
        teacache_thresh: float,
        auto_compare: bool,
        progress_callback: Optional[Callable[[int, int, str], None]]
    ) -> Dict[str, Any]:
        """Automatically restores long footage across contiguous/overlapping chunks and seamlessly stitches."""
        input_path = Path(input_video_path)
        is_diff = (engine_type != "superres")
        chunk_overlap = 8 if is_diff else 0
        chunks = self._calculate_chunks(start_f, end_f, chunk_size, overlap=chunk_overlap, is_diffusion=is_diff)
        total_chunks = len(chunks)
        target_span = end_f - start_f + 1

        # Anchor seed across all chunks to maintain consistent generative trajectories
        anchored_seed = seed if seed is not None else 42

        chunk_dir = config.OUTPUTS_DIR / f"chunks_{output_file.stem}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_files = []
        chunk_data = []

        print(f"\n[AUTO-CHUNKER] Processing {target_span} frames across {total_chunks} chunks ({chunk_size}f/chunk, overlap={chunk_overlap}f)...")

        for idx, (c_start, c_end) in enumerate(chunks):
            chunk_out = chunk_dir / f"chunk_{idx:04d}.mp4"
            chunk_data.append({"path": chunk_out, "start": c_start, "end": c_end})

            if chunk_out.exists() and chunk_out.stat().st_size > 1000:
                print(f"[AUTO-CHUNKER] Chunk {idx+1}/{total_chunks} (F {c_start}..{c_end}) already rendered, skipping.")
                chunk_files.append(chunk_out)
                continue

            def make_chunk_cb(c_idx, c_s, c_e):
                def cb(pct, total, desc):
                    if progress_callback:
                        overall = int(((c_idx + (pct / 100.0)) / total_chunks) * 90)
                        progress_callback(overall, 100, f"Chunk {c_idx+1}/{total_chunks} (F {c_s}..{c_e}) | {desc}")
                return cb

            print(f"\n--- [AUTO-CHUNKER] Chunk {idx+1}/{total_chunks}: Frames {c_start} -> {c_end} ---")
            self.restore_footage(
                input_video_path=input_path,
                output_file=chunk_out,
                start_frame=c_start,
                end_frame=c_end,
                engine_type=engine_type,
                superres_model=superres_model,
                mode=mode,
                prompt_preset=prompt_preset,
                custom_positive_prompt=custom_positive_prompt,
                custom_negative_prompt=custom_negative_prompt,
                output_size=output_size,
                tile_size=tile_size,
                seed=anchored_seed,
                guide_strength=guide_strength,
                cfg=cfg,
                sol_sigma=sol_sigma,
                reference_image=reference_image,
                extract_depth=extract_depth,
                clean_noise=clean_noise,
                stabilize_jitter=False,
                enable_teacache=enable_teacache,
                teacache_thresh=teacache_thresh,
                auto_compare=False,
                progress_callback=make_chunk_cb(idx, c_start, c_end)
            )
            chunk_files.append(chunk_out)

        # Stitch all rendered chunks
        temp_concat_video = chunk_dir / "stitched_temp.mp4"
        raw_meta = video_preprocessor.get_video_info(input_path)
        fps_val = raw_meta.get("fps", 29.97)

        if is_diff and len(chunk_data) > 1:
            if progress_callback:
                progress_callback(92, 100, "Seamlessly cross-fading overlapping chunks (Zero Seam Pop)...")
            self._blend_and_stitch_chunks(
                chunk_data=chunk_data,
                output_path=temp_concat_video,
                fps=fps_val,
                progress_callback=progress_callback
            )
        else:
            if progress_callback:
                progress_callback(92, 100, "Concatenating rendered chunks...")
            concat_list_file = chunk_dir / "concat_list.txt"
            with open(concat_list_file, "w", encoding="utf-8") as f:
                for cf in chunk_files:
                    f.write(f"file '{cf.resolve().as_posix()}'\n")

            cmd_concat = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list_file),
                "-c", "copy",
                str(temp_concat_video)
            ]
            subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Mux audio from original source corresponding to start_f -> end_f
        if progress_callback:
            progress_callback(95, 100, "Extracting and preserving synchronized audio track...")

        raw_meta = video_preprocessor.get_video_info(input_path)
        fps_val = raw_meta.get("fps", 29.97)
        start_time_sec = start_f / fps_val
        duration_sec = target_span / fps_val

        output_file.parent.mkdir(parents=True, exist_ok=True)
        if raw_meta.get("has_audio"):
            cmd_mux = [
                "ffmpeg", "-y",
                "-i", str(temp_concat_video),
                "-ss", str(start_time_sec),
                "-t", str(duration_sec),
                "-i", str(input_path),
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(output_file)
            ]
        else:
            cmd_mux = [
                "ffmpeg", "-y",
                "-i", str(temp_concat_video),
                "-c", "copy",
                str(output_file)
            ]
        subprocess.run(cmd_mux, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Before/After comparison
        comp_path = None
        if auto_compare:
            if progress_callback:
                progress_callback(97, 100, "Generating side-by-side Before/After comparison video...")
            try:
                comp_orig = input_path
                temp_slice = None
                if start_f > 0 or target_span < raw_meta.get("total_frames", target_span):
                    temp_slice = chunk_dir / "orig_slice_for_comp.mp4"
                    cmd_slice = [
                        "ffmpeg", "-y",
                        "-ss", str(start_time_sec),
                        "-t", str(duration_sec),
                        "-i", str(input_path),
                        "-c:v", "copy",
                        "-c:a", "copy",
                        str(temp_slice)
                    ]
                    subprocess.run(cmd_slice, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    comp_orig = temp_slice

                comp_res = comparison_generator.create_comparison(
                    original_video=comp_orig,
                    restored_video=output_file,
                    mode="side_by_side"
                )
                comp_path = comp_res.get("side_by_side")
            except Exception as e:
                print(f"[AUTO-CHUNKER] Comparison notice: {e}")

        # Clean up chunk directory
        try:
            shutil.rmtree(chunk_dir)
        except Exception:
            pass

        if progress_callback:
            progress_callback(100, 100, "Full-length video restoration complete!")

        return {
            "status": "completed",
            "final_output_path": str(output_file),
            "comparison_path": str(comp_path) if comp_path else None,
            "total_chunks": total_chunks,
            "frames": target_span,
            "mode": mode,
            "preset": prompt_preset
        }


# A resident RelayVSR worker must never outlive the app/CLI that started it
atexit.register(VideoRestorationPipeline.release_relayvsr_worker)
