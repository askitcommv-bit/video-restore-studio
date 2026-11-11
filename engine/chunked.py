"""
Long footage on the ComfyUI engines: split the range into chunks a single pass can hold, render each
(resuming already-rendered chunks), then join them (cross-faded overlaps for diffusion) and mux
the source's audio for that range.
"""

import dataclasses
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import comparison_generator
import config
import processes
import video_preprocessor
from engine.progress import ProgressCallback, report
from engine.request import RestoreRequest


def calculate_chunks(start_f: int, end_f: int, chunk_size: int = 33, overlap: int = 8,
                     is_diffusion: bool = True) -> List[Tuple[int, int]]:
    """Splits a frame span into blocks.
    Diffusion: legal (8n+1) lengths overlapping by `overlap` frames. Super-resolution: contiguous."""
    if not is_diffusion:
        chunks = []
        curr = start_f
        while curr <= end_f:
            chunks.append((curr, min(end_f, curr + chunk_size - 1)))
            curr += chunk_size
        return chunks

    chunks = []
    step = max(8, chunk_size - overlap)
    curr = start_f
    while curr <= end_f:
        c_end = curr + chunk_size - 1
        if c_end >= end_f:
            span = end_f - curr + 1
            legal_span = max(9, (span - 1) // 8 * 8 + 1)
            cand = (max(start_f, end_f - legal_span + 1), end_f)
            if not chunks or cand != chunks[-1]:
                chunks.append(cand)
            break
        chunks.append((curr, c_end))
        curr += step
    return chunks


def blend_and_stitch(chunk_data: List[Dict[str, Any]], output_path: Path, fps: float = 29.97,
                     progress: ProgressCallback = None) -> None:
    """Joins overlapping chunks with a cosine S-curve cross-fade over the overlap, so there are
    no hard seams between them."""
    if not chunk_data:
        return

    import cv2
    import numpy as np

    def read_all(path) -> list:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    cap0 = cv2.VideoCapture(str(chunk_data[0]["path"]))
    ret, f0 = cap0.read()
    cap0.release()
    if not ret or f0 is None:
        raise RuntimeError(f"Could not read first chunk frame: {chunk_data[0]['path']}")
    h, w = f0.shape[:2]

    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
           str(output_path)]
    proc = processes.popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    total_chunks = len(chunk_data)
    for idx, cinfo in enumerate(chunk_data):
        report(progress, int(90 + (idx / max(1, total_chunks)) * 5), 100, f"Blending chunk {idx+1}/{total_chunks}...")
        frames = read_all(cinfo["path"])
        c_start = cinfo["start"]
        c_len = len(frames)

        if idx == 0:
            write_until = max(1, chunk_data[1]["start"] - c_start) if total_chunks > 1 else c_len
            for frame in frames[:write_until]:
                proc.stdin.write(frame.tobytes())
            continue

        prev_info = chunk_data[idx - 1]
        prev_frames = read_all(prev_info["path"])
        ov_start = c_start
        ov_len = max(0, prev_info["end"] - ov_start + 1)

        # Blend overlapping frames using a cosine ramp
        for i in range(ov_len):
            prev_idx = (ov_start + i) - prev_info["start"]
            alpha = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / (ov_len + 1)))
            f_prev = prev_frames[min(max(0, prev_idx), len(prev_frames) - 1)]
            f_curr = frames[min(max(0, i), len(frames) - 1)]
            proc.stdin.write(cv2.addWeighted(f_curr, float(alpha), f_prev, float(1.0 - alpha), 0).tobytes())

        # Remainder of the current chunk, up to where the next one starts
        write_until = chunk_data[idx + 1]["start"] - c_start if idx < total_chunks - 1 else c_len
        for i in range(ov_len, write_until):
            if i < c_len:
                proc.stdin.write(frames[i].tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg blending failed with code {proc.returncode}")


def run(request: RestoreRequest, start_f: int, end_f: int, chunk_size: int, output_file: Path,
        restore_one: Callable[[RestoreRequest, ProgressCallback], Dict[str, Any]],
        progress: ProgressCallback) -> Dict[str, Any]:
    """Renders request's range [start_f, end_f] chunk by chunk through restore_one, then joins them."""
    input_path = Path(request.input_video_path)
    is_diff = request.engine_type != "superres"
    chunk_overlap = 8 if is_diff else 0
    chunks = calculate_chunks(start_f, end_f, chunk_size, overlap=chunk_overlap, is_diffusion=is_diff)
    total_chunks = len(chunks)
    target_span = end_f - start_f + 1
    # One seed for all chunks keeps the generative trajectories consistent
    anchored_seed = request.seed if request.seed is not None else 42

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

        def chunk_progress(pct, total, desc, c_idx=idx, c_s=c_start, c_e=c_end):
            overall = int(((c_idx + (pct / 100.0)) / total_chunks) * 90)
            report(progress, overall, 100, f"Chunk {c_idx+1}/{total_chunks} (F {c_s}..{c_e}) | {desc}")

        print(f"\n--- [AUTO-CHUNKER] Chunk {idx+1}/{total_chunks}: Frames {c_start} -> {c_end} ---")
        restore_one(dataclasses.replace(
            request, input_video_path=input_path, output_dir=None, output_file=chunk_out, max_frames=None,
            start_frame=c_start, end_frame=c_end, seed=anchored_seed, stabilize_jitter=False, auto_compare=False,
        ), chunk_progress)
        chunk_files.append(chunk_out)

    # Join the rendered chunks
    temp_concat_video = chunk_dir / "stitched_temp.mp4"
    raw_meta = video_preprocessor.get_video_info(input_path)
    fps_val = raw_meta.get("fps", 29.97)

    if is_diff and len(chunk_data) > 1:
        report(progress, 92, 100, "Seamlessly cross-fading overlapping chunks (Zero Seam Pop)...")
        blend_and_stitch(chunk_data, temp_concat_video, fps=fps_val, progress=progress)
    else:
        report(progress, 92, 100, "Concatenating rendered chunks...")
        concat_list_file = chunk_dir / "concat_list.txt"
        with open(concat_list_file, "w", encoding="utf-8") as f:
            for cf in chunk_files:
                f.write(f"file '{cf.resolve().as_posix()}'\n")
        processes.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list_file), "-c", "copy",
                       str(temp_concat_video)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Audio of the source for start_f..end_f
    report(progress, 95, 100, "Extracting and preserving synchronized audio track...")
    start_time_sec = start_f / fps_val
    duration_sec = target_span / fps_val

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if raw_meta.get("has_audio"):
        cmd_mux = ["ffmpeg", "-y", "-i", str(temp_concat_video), "-ss", str(start_time_sec), "-t", str(duration_sec),
                   "-i", str(input_path), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
                   "-b:a", "192k", "-shortest", str(output_file)]
    else:
        cmd_mux = ["ffmpeg", "-y", "-i", str(temp_concat_video), "-c", "copy", str(output_file)]
    processes.run(cmd_mux, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    comp_path = None
    if request.auto_compare:
        report(progress, 97, 100, "Generating side-by-side Before/After comparison video...")
        try:
            comp_orig = input_path
            if start_f > 0 or target_span < raw_meta.get("total_frames", target_span):
                comp_orig = chunk_dir / "orig_slice_for_comp.mp4"
                processes.run(["ffmpeg", "-y", "-ss", str(start_time_sec), "-t", str(duration_sec),
                               "-i", str(input_path), "-c:v", "copy", "-c:a", "copy", str(comp_orig)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            comp_path = comparison_generator.create_comparison(
                original_video=comp_orig, restored_video=output_file, mode="side_by_side").get("side_by_side")
        except Exception as e:
            print(f"[AUTO-CHUNKER] Comparison notice: {e}")

    shutil.rmtree(chunk_dir, ignore_errors=True)
    report(progress, 100, 100, "Full-length video restoration complete!")

    return {
        "status": "completed",
        "final_output_path": str(output_file),
        "comparison_path": str(comp_path) if comp_path else None,
        "total_chunks": total_chunks,
        "frames": target_span,
        "mode": request.mode,
        "preset": request.prompt_preset,
    }
