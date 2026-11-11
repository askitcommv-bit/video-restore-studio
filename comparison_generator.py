"""
Comparison Generator for LTX-2.5 Video Restoration.
Creates:
- Side-by-Side (Before vs After) split videos with labeled overlays
- 50/50 Vertical Wipe split-screen comparisons with divider line
- High-resolution frame-by-frame comparison stills
"""

import subprocess
from pathlib import Path
from typing import Optional

import processes
from video_preprocessor import get_video_info, get_video_encoder_args


def _get_font_arg() -> str:
    """Returns safe font parameter for ffmpeg drawtext filter on Windows."""
    for font in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]:
        if Path(font).exists():
            clean = font.replace(":", "\\:")
            return f"fontfile='{clean}':"
    return ""


def generate_side_by_side(
    original_path: Path | str,
    restored_path: Path | str,
    output_path: Path | str,
    label_left: str = "ORIGINAL FOOTAGE",
    label_right: str = "LTX-2.5 RESTORED",
    output_height: int = 1080
) -> Path:
    """
    Creates a synchronized side-by-side comparison video (Original Left, Restored Right)
    with clean text badges.
    """
    original_path = Path(original_path)
    restored_path = Path(restored_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    font_opt = _get_font_arg()

    # Filter graph (scale=-2 forces even width for H.264 compliance):
    filter_complex = (
        f"[0:v]scale=-2:{output_height}:flags=lanczos,"
        f"drawbox=x=20:y=20:w=320:h=45:color=black@0.75:t=fill,"
        f"drawtext={font_opt}text='{label_left}':x=35:y=32:fontsize=22:fontcolor=white:box=0[left];"
        f"[1:v]scale=-2:{output_height}:flags=lanczos,"
        f"drawbox=x=20:y=20:w=320:h=45:color=black@0.75:t=fill,"
        f"drawtext={font_opt}text='{label_right}':x=35:y=32:fontsize=22:fontcolor=#00e676:box=0[right];"
        f"[left][right]hstack=inputs=2[v]"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original_path),
        "-i", str(restored_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        *get_video_encoder_args(cq=18, preset="p4"),
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path)
    ]
    
    processes.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return output_path


def generate_split_screen(
    original_path: Path | str,
    restored_path: Path | str,
    output_path: Path | str,
    split_ratio: float = 0.5,
    label_left: str = "ORIGINAL",
    label_right: str = "RESTORED"
) -> Path:
    """
    Creates a 50/50 vertical split-screen with a white divider line:
    Left side shows original, right side shows restored.
    """
    original_path = Path(original_path)
    restored_path = Path(restored_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    info_restored = get_video_info(restored_path)
    w = (info_restored["width"] // 2) * 2
    h = (info_restored["height"] // 2) * 2
    split_x = ((int(round(w * split_ratio))) // 2) * 2
    font_opt = _get_font_arg()
    
    # Crop original to left half, crop restored to right half, overlay with divider line
    filter_complex = (
        f"[0:v]scale={w}:{h}:flags=lanczos,crop={split_x}:{h}:0:0,"
        f"drawbox=x=20:y=20:w=200:h=40:color=black@0.75:t=fill,"
        f"drawtext={font_opt}text='{label_left}':x=30:y=30:fontsize=20:fontcolor=white:box=0[left];"
        f"[1:v]scale={w}:{h}:flags=lanczos,"
        f"drawbox=x={w-220}:y=20:w=200:h=40:color=black@0.75:t=fill,"
        f"drawtext={font_opt}text='{label_right}':x={w-210}:y=30:fontsize=20:fontcolor=#00e676:box=0[right_full];"
        f"[right_full][left]overlay=0:0[combined];"
        f"[combined]drawbox=x={split_x-2}:y=0:w=4:h={h}:color=white@0.9:t=fill[v]"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original_path),
        "-i", str(restored_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        *get_video_encoder_args(cq=18, preset="p4"),
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path)
    ]
    
    processes.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return output_path


def generate_frame_comparison(
    original_path: Path | str,
    restored_path: Path | str,
    output_image_path: Path | str,
    timestamp_sec: float = 1.0
) -> Path:
    """Extracts a still frame from both videos and generates a side-by-side comparison image."""
    original_path = Path(original_path)
    restored_path = Path(restored_path)
    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    
    info_restored = get_video_info(restored_path)
    dur = info_restored.get("duration", 0)
    actual_ts = min(timestamp_sec, max(0.0, dur / 2.0)) if dur > 0 else 0.0

    font_opt = _get_font_arg()
    filter_complex = (
        f"[0:v]scale=-2:1080:flags=lanczos,drawbox=x=20:y=20:w=280:h=45:color=black@0.75:t=fill,"
        f"drawtext={font_opt}text='BEFORE':x=40:y=32:fontsize=24:fontcolor=white:box=0[left];"
        f"[1:v]scale=-2:1080:flags=lanczos,drawbox=x=20:y=20:w=280:h=45:color=black@0.75:t=fill,"
        f"drawtext={font_opt}text='AFTER':x=40:y=32:fontsize=24:fontcolor=#00e676:box=0[right];"
        f"[left][right]hstack=inputs=2[v]"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{actual_ts:.3f}",
        "-i", str(original_path),
        "-ss", f"{actual_ts:.3f}",
        "-i", str(restored_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-vframes", "1",
        str(output_image_path)
    ]
    
    processes.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return output_image_path


def create_comparison(
    original_video: Path | str,
    restored_video: Path | str,
    output_dir: Optional[Path | str] = None,
    mode: str = "both"
) -> dict:
    """Convenience helper to generate comparison artifacts."""
    import config
    original = Path(original_video)
    restored = Path(restored_video)
    out_dir = Path(output_dir) if output_dir else config.COMPARISONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    if mode in ["side_by_side", "both"]:
        sbs_path = out_dir / f"{restored.stem}_vs_{original.stem}_sbs.mp4"
        generate_side_by_side(original, restored, sbs_path)
        results["side_by_side"] = str(sbs_path)
        
    if mode in ["split", "both"]:
        split_path = out_dir / f"{restored.stem}_vs_{original.stem}_split.mp4"
        generate_split_screen(original, restored, split_path)
        results["split"] = str(split_path)

    try:
        still_path = out_dir / f"{restored.stem}_comparison.jpg"
        generate_frame_comparison(original, restored, still_path, timestamp_sec=1.0)
        results["still"] = str(still_path)
    except Exception as e:
        print(f"Could not generate comparison still: {e}")

    return results

