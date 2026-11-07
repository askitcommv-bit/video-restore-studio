import json
import urllib.request

def build_two_stage_prompt(
    video_filename: str = "test_restore_input.mp4",
    output_prefix: str = "test_two_stage_output",
    tile_size: str = "HD",
    output_size: str = "4K",
    seed: int = 42,
    pos_prompt: str = "restored footage, clean image, pristine detail, sharp focus, natural color, master quality",
    neg_prompt: str = "dirt, scratches, dust specks, emulsion damage, flicker, heavy film grain, blurry, mushy, noisy, watermark, low resolution, digital artifacts",
) -> dict:
    prompt = {
        # 1. Source Video
        "1": {
            "class_type": "LoadVideo",
            "inputs": {
                "file": video_filename
            },
            "_meta": {"title": "Load Source Video"}
        },
        "2": {
            "class_type": "GetVideoComponents",
            "inputs": {
                "video": ["1", 0]
            },
            "_meta": {"title": "Extract Video Components"}
        },
        "3": {
            "class_type": "GetImageSize",
            "inputs": {
                "image": ["2", 0]
            },
            "_meta": {"title": "Get Image Dimensions"}
        },

        # 2. Models
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
            "inputs": {
                "vae_name": "ltx-2.5-video-vae-bf16.safetensors"
            },
            "_meta": {"title": "Load Video VAE"}
        },
        "13": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "ltx-2.5-audio-vae-bf16.safetensors"
            },
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
            "inputs": {
                "clip": ["11", 0],
                "text": pos_prompt
            },
            "_meta": {"title": "Positive Prompt"}
        },
        "21": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["11", 0],
                "text": neg_prompt
            },
            "_meta": {"title": "Negative Prompt"}
        },
        "22": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["20", 0],
                "negative": ["21", 0],
                "frame_rate": ["2", 2]  # fps
            },
            "_meta": {"title": "LTXV Conditioning"}
        },

        # 4. Canvas & Tiling Preprocessing
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
                "width": ["30", 0],   # initial_canvas_width (Stage 1)
                "height": ["30", 1],  # initial_canvas_height (Stage 1)
                "crop": "disabled"
            },
            "_meta": {"title": "Resize Guide to Stage 1 Canvas"}
        },
        "32": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": ["30", 0],   # initial_canvas_width
                "height": ["30", 1],  # initial_canvas_height
                "length": ["3", 2],   # frame_count
                "batch_size": 1
            },
            "_meta": {"title": "Empty Latent Canvas (Stage 1)"}
        },
        "33": {
            "class_type": "VAEEncodeAudio",
            "inputs": {
                "audio": ["2", 1],   # audio
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

        # 5. Sampler & Scheduler - Stage 1 (8 steps)
        "40": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler"
            },
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
                "model": ["14", 0],         # Restore IC-LoRA
                "positive": ["35", 0],
                "negative": ["35", 1],
                "latents": ["35", 2],
                "sigmas": ["41", 0],
                "sampler": ["40", 0],
                "seed": seed,
                "cfg": 1.0,
                "tile_width": ["30", 2],    # tile_width
                "tile_height": ["30", 3],   # tile_height
                "overlap_frac": ["30", 6],  # overlap_frac
                "blend_var": 0.05,
                "grid_cycle": 1,
                "tile_frames": ["30", 7],   # tile_frames
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
            "_meta": {"title": "Resize Original Guide to 4K"}
        },
        "62": {
            "class_type": "LTXAddVideoICLoRAGuide",
            "inputs": {
                "positive": ["34", 0],
                "negative": ["34", 1],
                "vae": ["12", 0],
                "latent": ["60", 0],  # upscaled latent!
                "image": ["61", 0],   # 4K guide!
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
                "model": ["15", 0],         # Refine IC-LoRA
                "positive": ["62", 0],
                "negative": ["62", 1],
                "latents": ["62", 2],
                "sigmas": ["63", 0],
                "sampler": ["40", 0],
                "seed": seed + 1,
                "cfg": 1.0,
                "tile_width": ["30", 2],    # tile_width
                "tile_height": ["30", 3],   # tile_height
                "overlap_frac": ["30", 6],  # overlap_frac
                "blend_var": 0.05,
                "grid_cycle": 1,
                "tile_frames": ["30", 7],   # tile_frames
                "vae": ["12", 0],
                "canvas_device": "gpu"
            },
            "_meta": {"title": "Stage 2 Refine Tiled Fusion Sampler"}
        },

        # 7. Decode & Output
        "50": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["64", 0],
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
                "fps": ["2", 2],  # fps
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

    return prompt

if __name__ == "__main__":
    prompt = build_two_stage_prompt()
    print(f"Generated 2-stage graph with {len(prompt)} nodes.")

    # Validate against ComfyUI /prompt endpoint
    # Note: don't queue right now since another prompt is currently executing, just check structure
    print("Graph built successfully.")
