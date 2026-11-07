"""
Video Preprocessing Utilities for LTX-2.5 Restoration Pipeline.
Handles:
- ffprobe metadata inspection (resolution, frame count, fps, audio, interlacing)
- Deinterlacing (essential for tape / broadcast / archival footage)
- Resolution rounding to exact multiples of 32 (model constraint)
- Aspect ratio preservation (4:3 vs 16:9 vs custom)
- Frame count alignment to model-legal (8n + 1)
- Audio extraction and lossless remuxing
"""

import subprocess
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
import config

_NVENC_AVAILABLE: Optional[bool] = None

def check_nvenc_available() -> bool:
    """Checks if NVIDIA NVENC hardware H.264 encoder is available on system."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=0.1:size=64x64:rate=10",
            "-c:v", "h264_nvenc", "-f", "null", "-"
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _NVENC_AVAILABLE = (res.returncode == 0)
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE

def get_video_encoder_args(cq: int = 16, preset: str = "p4") -> list:
    """
    Returns GPU NVENC encoding arguments if NVIDIA NVENC is available,
    otherwise falls back to CPU libx264.
    """
    if check_nvenc_available():
        return ["-c:v", "h264_nvenc", "-preset", preset, "-cq", str(cq), "-pix_fmt", "yuv420p"]
    else:
        return ["-c:v", "libx264", "-preset", "medium", "-crf", str(cq), "-pix_fmt", "yuv420p"]


def get_video_info(video_path: Path | str) -> Dict[str, Any]:
    """Inspects video file using ffprobe and returns structured metadata."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,field_order,pix_fmt",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json",
        str(video_path)
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    probe = json.loads(result.stdout)
    
    stream = probe["streams"][0] if probe.get("streams") else {}
    fmt = probe.get("format", {})
    
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    
    # Calculate FPS
    r_fps = stream.get("r_frame_rate", "24/1")
    if "/" in r_fps:
        num, den = map(float, r_fps.split("/"))
        fps = num / den if den != 0 else 24.0
    else:
        fps = float(r_fps)
        
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    
    # Count frames
    nb_frames = stream.get("nb_frames")
    if nb_frames and nb_frames.isdigit():
        total_frames = int(nb_frames)
    elif duration > 0:
        total_frames = int(round(duration * fps))
    else:
        total_frames = 0
        
    field_order = stream.get("field_order", "progressive")
    is_interlaced = field_order not in ["progressive", "unknown"] and field_order != ""
    
    # Check audio
    audio_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,channels,sample_rate",
        "-of", "json",
        str(video_path)
    ]
    audio_res = subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    audio_probe = json.loads(audio_res.stdout) if audio_res.returncode == 0 else {}
    has_audio = len(audio_probe.get("streams", [])) > 0
    
    aspect_ratio = width / height if height > 0 else 16/9
    
    return {
        "path": str(video_path),
        "filename": video_path.name,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "fps_rational": r_fps,  # exact rate (e.g. "30000/1001") for frame<->time math
        "total_frames": total_frames,
        "duration": round(duration, 3),
        "aspect_ratio": round(aspect_ratio, 4),
        "aspect_ratio_str": "4:3" if abs(aspect_ratio - 4/3) < 0.05 else ("16:9" if abs(aspect_ratio - 16/9) < 0.05 else f"{width}:{height}"),
        "is_interlaced": is_interlaced,
        "field_order": field_order,
        "pix_fmt": stream.get("pix_fmt", ""),
        "has_audio": has_audio,
        "size_mb": round(int(fmt.get("size", 0)) / (1024 * 1024), 2)
    }


def calculate_model_dimensions(width: int, height: int, target_base: int = 1080) -> Tuple[int, int]:
    """
    Calculates target dimensions that preserve aspect ratio while ensuring
    both width and height are strictly divisible by 32 (LTX requirement).
    """
    aspect = width / height
    
    if width >= height:
        # Landscape
        new_width = target_base
        new_height = int(round(target_base / aspect))
    else:
        # Portrait
        new_height = target_base
        new_width = int(round(target_base * aspect))
        
    # Round to nearest multiple of 32
    legal_width = int(round(new_width / 32.0)) * 32
    legal_height = int(round(new_height / 32.0)) * 32
    
    legal_width = max(32, legal_width)
    legal_height = max(32, legal_height)
    
    return legal_width, legal_height


def calculate_legal_frame_count(total_frames: int, mode: str = "trim") -> int:
    """
    LTX-2.5 requires frame count to be (8n + 1), e.g. 17, 25, 33, 49, 65, 81, 97, 129, 257.
    mode: 'trim' (round down to nearest 8n+1) or 'pad' (round up).
    """
    if total_frames <= 1:
        return 9  # minimum useful
    
    n = (total_frames - 1) // 8
    if mode == "trim" or n == 0:
        return max(9, n * 8 + 1)
    else:
        return (n + 1) * 8 + 1


def preprocess_video(
    input_path: Path | str,
    output_path: Optional[Path | str] = None,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    force_deinterlace: bool = True,
    clean_noise: bool = False,
    stabilize_jitter: bool = False,
    align_frames: bool = True,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    target_fps: Optional[float] = None
) -> Tuple[Path, Dict[str, Any]]:
    """
    Sanitizes video to guarantee 100% compliance with LTX-2.5 inputs:
    - Progressive scan (deinterlaces using bwdif filter if needed)
    - Dimensions strictly divisible by 32
    - Frame count matching (8n + 1)
    - Precise In/Out point trimming with synchronized audio
    - Fast start MOOV atom for instant playback
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    info = get_video_info(input_path)
    
    if output_path is None:
        output_dir = input_path.parent
        output_path = output_dir / f"{input_path.stem}_prepared.mp4"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
    # Calculate target dimensions
    if target_width is None or target_height is None:
        target_w, target_h = calculate_model_dimensions(info["width"], info["height"], target_base=min(info["width"], 1920))
    else:
        target_w = int(round(target_width / 32.0)) * 32
        target_h = int(round(target_height / 32.0)) * 32
        
    # Frame count and In/Out slicing
    total_src_frames = info["total_frames"]
    start_f = max(0, int(start_frame)) if start_frame is not None else 0
    if end_frame is not None:
        end_f = min(total_src_frames - 1, max(start_f, int(end_frame)))
        slice_frames = end_f - start_f + 1
    else:
        slice_frames = max(1, total_src_frames - start_f)
        
    if max_frames and slice_frames > max_frames:
        slice_frames = max_frames
        
    if align_frames:
        legal_frames = calculate_legal_frame_count(slice_frames, mode="trim")
    else:
        legal_frames = slice_frames

    effective_end_f = start_f + legal_frames - 1
        
    # Video filters
    vf_filters = []
    
    # 1. Deinterlace if requested or detected
    if force_deinterlace or info["is_interlaced"]:
        vf_filters.append("bwdif=mode=send_frame:parity=auto:deint=all")
        
    # 2. Constant frame rate: sources with timestamp gaps (dropped frames in the rip) are
    # filled by repeating frames, so frame n sits exactly at n/fps. Every later stage
    # (frame-index trim below, the time-based audio trim, model writers) assumes that;
    # without it each gap shifted the video ahead of the audio (7.4 s on S01E01).
    vf_filters.append(f"fps={info['fps_rational']}")

    # 3. Trim to legal frames from start_frame
    vf_filters.append(f"select=between(n\\,{start_f}\\,{effective_end_f})")
    vf_filters.append("setpts=PTS-STARTPTS")

    # 4. Exposure flicker stabilization. Keeps every frame: the previous mpdecimate +
    # setpts=N/FRAME_RATE/TB dropped every near-duplicate frame (~31% of a sitcom episode)
    # and re-timed the rest, so outputs played fast and drifted out of sync with the audio.
    if stabilize_jitter:
        vf_filters.append("deflicker=size=5:mode=pm")

    # 5. Temporal Anti-Boiling / Noise Reduction
    if clean_noise:
        vf_filters.append("hqdn3d=2:2:4:4")

    # 6. Scale to legal dimensions
    vf_filters.append(f"scale={target_w}:{target_h}:flags=lanczos")
    
    filter_chain = ",".join(vf_filters)
    fps_arg = ["-r", str(target_fps)] if target_fps else []
    
    # Audio slicing to match exact video In/Out slice
    audio_args = []
    if info["has_audio"]:
        # Exact rational rate: the rounded 29.97 drifts ~10 ms per 5 min against 30000/1001
        fps_val = float(Fraction(info["fps_rational"])) if info["fps"] > 0 else 24.0
        start_sec = start_f / fps_val
        end_sec = (start_f + legal_frames) / fps_val
        af_chain = f"atrim=start={start_sec:.6f}:end={end_sec:.6f},asetpts=PTS-STARTPTS"
        audio_args = ["-af", af_chain, "-c:a", "aac", "-b:a", "192k"]
    else:
        audio_args = ["-an"]
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", filter_chain,
        "-vframes", str(legal_frames),
        *fps_arg,
        *audio_args,
        *get_video_encoder_args(cq=16, preset="p4"),
        "-movflags", "+faststart",
        str(output_path)
    ]
    
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    
    new_info = get_video_info(output_path)
    new_info["start_frame"] = start_f
    new_info["end_frame"] = effective_end_f
    return output_path, new_info


def extract_audio(video_path: Path | str, audio_output_path: Optional[Path | str] = None) -> Optional[Path]:
    """
    Extracts pristine audio track from video into high-compatibility AAC (.m4a)
    or uncompressed WAV, cleanly handling all source audio codecs (Opus, AAC, AC3, PCM, etc.).
    """
    video_path = Path(video_path)
    info = get_video_info(video_path)
    if not info["has_audio"]:
        return None
        
    if audio_output_path is None:
        audio_output_path = video_path.parent / f"{video_path.stem}_audio.m4a"
    else:
        audio_output_path = Path(audio_output_path)
        
    audio_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Target high-bitrate AAC (.m4a) for universal MP4 container compatibility
    target_m4a = audio_output_path.with_suffix(".m4a")
    cmd_aac = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-c:a", "aac",
        "-b:a", "256k",
        str(target_m4a)
    ]
    
    res = subprocess.run(cmd_aac, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0 and target_m4a.exists() and target_m4a.stat().st_size > 0:
        return target_m4a
        
    # Clean up failed 0-byte file if any
    if target_m4a.exists() and target_m4a.stat().st_size == 0:
        try:
            target_m4a.unlink()
        except Exception:
            pass

    # Fallback to uncompressed 16-bit PCM WAV
    target_wav = audio_output_path.with_suffix(".wav")
    cmd_wav = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-c:a", "pcm_s16le",
        str(target_wav)
    ]
    res_wav = subprocess.run(cmd_wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res_wav.returncode == 0 and target_wav.exists() and target_wav.stat().st_size > 0:
        return target_wav

    return None


def mux_audio_and_video(
    video_path: Path | str,
    audio_path: Path | str,
    output_path: Path | str,
    shortest: bool = True,
    encoder=None,
) -> Path:
    """Muxes audio (as AAC) with the video. The video stream is copied, or re-encoded with
    `encoder` (a video_encoders.Encoder, e.g. HEVC on NVENC) when one is given."""
    import video_encoders
    video_path = Path(video_path)
    audio_path = Path(audio_path) if audio_path else None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_audio = audio_path is not None and audio_path.exists() and audio_path.stat().st_size > 0

    if not has_audio and encoder is None:
        import shutil
        shutil.copyfile(video_path, output_path)
        return output_path

    # -shortest cuts at an AAC packet boundary and dropped up to ~4 trailing video frames;
    # trimming the output to the video's exact duration keeps every frame.
    video_duration = get_video_info(video_path)["duration"] if shortest and has_audio else 0
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        *(["-i", str(audio_path)] if has_audio else []),
        "-map", "0:v:0", *(["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"] if has_audio else []),
        *video_encoders.video_args(encoder),
        "-movflags", "+faststart",
        *(["-t", f"{video_duration:.3f}"] if video_duration > 0 else []),
        str(output_path)
    ]
    
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return output_path


def analyze_video_defects(video_path: Path | str, num_sample_frames: int = 15) -> Dict[str, Any]:
    """
    Performs fast computer vision and signal analysis on video frames to detect:
    - Interlaced scanlines (via ffmpeg idet)
    - Monochrome / B&W / Sepia vs full color (via HSV saturation analysis)
    - Exposure flicker / gate jitter (luminance standard deviation across frames)
    - Heavy film grain, noise, or scratch variance (Laplacian sharpness/noise estimation)
    - SD vs HD resolution and aspect ratio (4:3 vs 16:9)

    Returns recommended defect check values and human-readable diagnostics.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    import cv2
    import numpy as np
    import re

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        return {
            "is_interlaced": False,
            "is_monochrome": False,
            "is_sd": False,
            "has_grain_or_noise": False,
            "has_flicker": False,
            "aspect_ratio_str": "16:9",
            "detected_tags": ["Standard Video"],
            "summary_text": "Unable to analyze video frames.",
            "suggested_defects": {
                "dirt_scratches": True,
                "jitter_flicker": False,
                "deinterlace": False,
                "noise_compression": True,
                "colorize": False,
                "film_roll": False,
                "micro_detail": False,
            }
        }

    # Sample evenly across the clip
    sample_indices = np.linspace(0, max(0, total_frames - 1), min(num_sample_frames, total_frames), dtype=int)
    saturations = []
    brightnesses = []
    laplacians = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturations.append(float(np.mean(hsv[:, :, 1])))
        brightnesses.append(float(np.mean(hsv[:, :, 2])))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacians.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    cap.release()

    mean_sat = float(np.mean(saturations)) if saturations else 50.0
    is_monochrome = (mean_sat < 18.0)
    flicker_score = float(np.std(brightnesses)) if len(brightnesses) > 1 else 0.0
    mean_lap = float(np.mean(laplacians)) if laplacians else 500.0
    is_sd = (width < 1280 or height < 720)
    aspect_ratio = width / height if height > 0 else 16/9
    aspect_ratio_str = "4:3" if abs(aspect_ratio - 4/3) < 0.1 else ("16:9" if abs(aspect_ratio - 16/9) < 0.1 else f"{width}:{height}")

    # Interlacing detection via ffmpeg idet
    is_interlaced = False
    try:
        cmd = ['ffmpeg', '-i', str(video_path), '-filter:v', 'idet', '-frames:v', '30', '-an', '-f', 'null', '-']
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        matches = re.findall(r'Multi frame detection:\s+TFF:\s+(\d+)\s+BFF:\s+(\d+)\s+Progressive:\s+(\d+)', res.stderr)
        if matches:
            tff, bff, prog = [int(x) for x in matches[-1]]
            if (tff + bff) > prog:
                is_interlaced = True
    except Exception:
        pass

    has_grain_or_noise = (mean_lap > 400.0 or is_sd)
    has_flicker = (flicker_score > 3.0)

    tags = []
    if is_sd:
        tags.append(f"SD {height}p ({aspect_ratio_str})")
    else:
        tags.append(f"HD {width}×{height}")

    if is_interlaced:
        tags.append("Interlaced Scanlines")
    if is_monochrome:
        tags.append("Monochrome / B&W")
    else:
        tags.append("Color")

    if has_grain_or_noise:
        tags.append("Heavy Grain & Specks")
    if has_flicker:
        tags.append("Exposure Flicker")

    summary_text = " • ".join(tags)

    suggested = {
        "dirt_scratches": True,
        "jitter_flicker": True if (has_flicker or is_sd) else False,
        "deinterlace": is_interlaced,
        "noise_compression": True if is_sd else (mean_lap > 300),
        "colorize": is_monochrome,
        "film_roll": (aspect_ratio_str == "4:3" or is_sd or has_grain_or_noise),
        "micro_detail": False,
    }

    return {
        "is_interlaced": is_interlaced,
        "is_monochrome": is_monochrome,
        "is_sd": is_sd,
        "has_grain_or_noise": has_grain_or_noise,
        "has_flicker": has_flicker,
        "aspect_ratio_str": aspect_ratio_str,
        "mean_saturation": round(mean_sat, 1),
        "flicker_score": round(flicker_score, 2),
        "noise_laplacian": round(mean_lap, 1),
        "detected_tags": tags,
        "summary_text": summary_text,
        "suggested_defects": suggested
    }


def prepare_reference_image(image_path: Path | str, target_dir: Optional[Path] = None) -> Path:
    """
    Prepares a character or style reference image for Video DiT & VAE ingestion.
    Ensures dimensions are strictly multiples of 32 to avoid tensor rearrange/patchify errors.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {image_path}")

    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load reference image: {image_path}")

    h, w = img.shape[:2]
    new_w = max(64, (w // 32) * 32)
    new_h = max(64, (h // 32) * 32)

    if (new_w != w) or (new_h != h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    out_dir = Path(target_dir) if target_dir else config.INPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_aligned32.png"
    cv2.imwrite(str(out_path), img)
    return out_path


def extract_depth_sequence(
    video_path: Path | str,
    output_path: Optional[Path | str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Path:
    """
    Extracts frame-by-frame 3D metric scene depth using Depth Anything V2 Small on CUDA.
    Produces a synchronized grayscale depth video for geometric anchoring.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    out_file = Path(output_path) if output_path else config.INPUTS_DIR / f"{video_path.stem}_depth.mp4"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    worker_script = config.PROJECT_DIR / "extract_depth_worker.py"
    embedded_py = Path("C:/ComfyUI_portable/python_embeded/python.exe")

    # If ComfyUI embedded Python with CUDA is available, execute via worker subprocess for guaranteed RTX acceleration
    if embedded_py.exists() and worker_script.exists():
        if progress_callback:
            progress_callback(0, 100, "Starting Depth Anything V2 on NVIDIA RTX GPU...")

        cmd = [
            str(embedded_py),
            str(worker_script),
            "--video", str(video_path),
            "--output", str(out_file),
            "--batch-size", "4"
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        )

        total_frames = 1
        for line in iter(proc.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            if line.startswith("INIT: Total frames="):
                try:
                    parts = line.split("Total frames=")[1].split(",")[0]
                    total_frames = max(1, int(parts))
                except Exception:
                    pass
            elif line.startswith("PROGRESS:"):
                try:
                    parts = line.split("PROGRESS:")[1].strip().split("/")
                    cur = int(parts[0])
                    tot = int(parts[1])
                    if progress_callback:
                        progress_callback(cur, tot, f"Estimating 3D depth geometry ({cur}/{tot})...")
                except Exception:
                    pass

        proc.stdout.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Depth extraction process failed with code {ret}")

        return out_file

    # Fallback to direct in-process Python
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from transformers import pipeline

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if progress_callback:
        progress_callback(0, total_frames, "Loading Depth Anything V2 neural model...")

    device = 0 if torch.cuda.is_available() else -1
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))

    batch_frames = []
    batch_size = 4
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_frames.append(Image.fromarray(rgb))

        if len(batch_frames) >= batch_size:
            depth_results = depth_pipe(batch_frames, batch_size=batch_size)
            for d_res in depth_results:
                depth_np = np.array(d_res["depth"].convert("RGB"))
                depth_np = cv2.resize(depth_np, (w, h))
                writer.write(cv2.cvtColor(depth_np, cv2.COLOR_RGB2BGR))
                processed += 1
                if progress_callback:
                    progress_callback(processed, total_frames, f"Estimating 3D depth geometry ({processed}/{total_frames})...")
            batch_frames = []

    if batch_frames:
        depth_results = depth_pipe(batch_frames, batch_size=len(batch_frames))
        for d_res in depth_results:
            depth_np = np.array(d_res["depth"].convert("RGB"))
            depth_np = cv2.resize(depth_np, (w, h))
            writer.write(cv2.cvtColor(depth_np, cv2.COLOR_RGB2BGR))
            processed += 1
            if progress_callback:
                progress_callback(processed, total_frames, f"Estimating 3D depth geometry ({processed}/{total_frames})...")

    cap.release()
    writer.release()
    return out_file

