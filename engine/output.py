"""
Final delivery of a rendered video: audio muxed in, HEVC re-encode on the best encoder the PC has
(unless H.264 was chosen or none works), and the optional before/after comparisons.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import comparison_generator
import video_encoders
import video_preprocessor
from engine.progress import ProgressCallback, report


def deliver(
    rendered: Path,
    audio_path: Optional[Path],
    final_output_path: Path,
    prepared_video_path: Path,
    auto_compare: bool,
    progress: ProgressCallback,
    keep_source: bool,
    done_text: str,
    output_codec: str = "auto",
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Writes final_output_path from `rendered`, returns (its video info, comparison result).
    keep_source=False consumes `rendered` (a temp file) instead of copying it."""
    codec, encoder = video_encoders.resolve(output_codec)
    report(progress, 92, 100, f"Encoding HEVC on {encoder.label} and adding audio..." if encoder
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
            report(progress, 96, 100, "Rendering side-by-side & split-screen comparisons...")
            comparison_res = comparison_generator.create_comparison(
                original_video=prepared_video_path,
                restored_video=final_output_path,
                mode="both"
            )
        except Exception as comp_err:
            print(f"  [Comparison Notice] Skipping comparison: {comp_err}")

    report(progress, 100, 100, done_text)
    return final_info, comparison_res
