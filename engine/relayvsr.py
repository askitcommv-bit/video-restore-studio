"""
RelayVSR engine: streaming 4x super-resolution in its own Python environment.

The worker (relayvsr_worker.py) runs as a resident subprocess, reused across jobs, killed on cancel
and released when idle. Output is written in segments so a stopped job resumes where it stopped,
and a worker starved of VRAM by other apps is restarted at a segment boundary (no frame lost).
"""

import atexit
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
import gpu
import processes
import user_settings
import video_preprocessor
from engine import output
from engine.comfy.client import ComfyUIClient
from engine.progress import ProgressCallback, format_eta, report

# Output is written in 10 s (at 29.97 fps) segments: the most a stopped job can lose
SEGMENT_FRAMES = 300
# A starved worker (VRAM spilled into system RAM) is restarted at most this many times per job
MAX_RECYCLES = 3
# How long a restart waits for other apps to give back VRAM before loading anyway
VRAM_WAIT_S = 60
# Worker footprint (weights + 240p activations + CUDA context) until it reports its own
DEFAULT_NEED_GB = 12.5

_worker: Dict[str, Optional[subprocess.Popen]] = {"proc": None}
_worker_lock = threading.Lock()
# Set by terminate_worker so a job waiting between worker restarts also stops
_cancel = threading.Event()


def geometry(src_w: int, src_h: int, output_size: str) -> Tuple[int, int]:
    """Returns (lr_height, out_height) for the worker.

    The short side of the source is capped at RELAYVSR_MAX_LR_HEIGHT for the 4x pass (VRAM bound),
    and the 4x result is resampled so its short side matches the requested output tier."""
    target_short = {"HD": 720, "FullHD": 1080, "4K": 2160, "8K": 4320}.get(output_size, 720)
    short_side = min(src_w, src_h)
    lr_short = min(short_side, config.RELAYVSR_MAX_LR_HEIGHT)
    # Worker heights are expressed in source-height units
    to_height = src_h / short_side
    return int(round(lr_short * to_height)), int(round(target_short * to_height))


def terminate_worker() -> bool:
    """Kills the worker mid-job (queue cancellation); the next job starts a fresh one."""
    _cancel.set()
    proc = _worker.get("proc")
    if proc is not None and proc.poll() is None:
        proc.kill()
        return True
    return False


def release_worker() -> None:
    """Shuts the resident worker down and frees its ~9.4 GB of VRAM.
    Called when a queue goes idle, before non-RelayVSR jobs, and at interpreter exit."""
    with _worker_lock:
        proc = _worker.get("proc")
        _worker["proc"] = None
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.stdin.close()  # serve loop ends on EOF
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    print("[RelayVSR] Worker released (VRAM freed).")


# A resident worker must never outlive the app/CLI that started it
atexit.register(release_worker)


# ---------------------------------------------------------------------------
# Segments and resume
# ---------------------------------------------------------------------------
def parts_dir(final_output_path: Path) -> Path:
    # Kept apart from inputs/ and outputs/ so clearing those doesn't destroy resumable work
    return config.RESUME_DIR / f"{final_output_path.stem}_relayvsr_parts"


def resume_state(final_output_path: Path, key: Dict[str, Any]) -> int:
    """Number of finished segments a previous run of this exact job left behind (0 = fresh start).
    A different source/range/size invalidates them; the folder is reset for the new run."""
    folder = parts_dir(final_output_path)
    manifest = folder / "manifest.json"
    try:
        if json.loads(manifest.read_text(encoding="utf-8")) == key:
            for leftover in folder.glob(".refvsr-*"):  # half-written segment of a killed run
                leftover.unlink(missing_ok=True)
            done = 0
            while (folder / f"seg_{done:05d}.mp4").exists():
                done += 1
            return done
    except (OSError, ValueError):
        pass
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(key), encoding="utf-8")
    return 0


def concat_segments(segments: List[Path], out_path: Path) -> None:
    """Joins same-codec segments losslessly (stream copy) via ffmpeg's concat demuxer."""
    list_file = out_path.with_suffix(".concat.txt")
    list_file.write_text(
        "".join("file '{}'\n".format(s.resolve().as_posix().replace("'", "'\\''")) for s in segments),
        encoding="utf-8",
    )
    try:
        processes.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(list_file),
             "-c", "copy", "-movflags", "+faststart", str(out_path)],
            check=True, capture_output=True, text=True,
        )
    finally:
        list_file.unlink(missing_ok=True)


def save_partial(folder: Path, audio_path: Optional[Path], final_output_path: Path) -> Optional[Path]:
    """Joins the finished segments of a stopped job into <name>_partial.mp4 (audio trimmed to match).
    The segments stay in place so the job can resume."""
    segments = sorted(folder.glob("seg_*.mp4"))
    if not segments:
        return None
    partial = final_output_path.with_name(f"{final_output_path.stem}_partial.mp4")
    joined = folder / "partial_joined.mp4"
    try:
        concat_segments(segments, joined)
        if audio_path and audio_path.exists():
            video_preprocessor.mux_audio_and_video(joined, audio_path, partial)
        else:
            shutil.copyfile(joined, partial)
    except Exception as e:
        print(f"[RelayVSR] Could not assemble partial output: {e}")
        return None
    finally:
        joined.unlink(missing_ok=True)
    print(f"[RelayVSR] Saved finished part ({len(segments) * SEGMENT_FRAMES} frames max): {partial}")
    return partial


# ---------------------------------------------------------------------------
# The worker process
# ---------------------------------------------------------------------------
def _pump(proc: subprocess.Popen, stop_tokens: Tuple[str, ...], progress: ProgressCallback = None,
          speed: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], List[str]]:
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
                text = f"RelayVSR upscaling frame {cur}/{tot}"
                if speed.get("fps"):
                    text += f" · {speed['fps']:.2f} fps · ~{format_eta((tot - cur) / speed['fps'])} left"
                if speed.get("slow"):
                    text += " · ⚠ slowed: GPU memory full — close other GPU apps"
                report(progress, 25 + int(65 * cur / max(tot, 1)), 100, text)
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


def _wait_for_vram(client: ComfyUIClient, need_gb: float, progress: ProgressCallback) -> None:
    """After the starved worker is released, gives other apps up to VRAM_WAIT_S to leave room for a
    fully-resident restart. Loads anyway afterwards (spilled is slower, not broken)."""
    client.free_memory()
    deadline = time.monotonic() + VRAM_WAIT_S
    while True:
        if _cancel.is_set():
            raise RuntimeError("RelayVSR job canceled while waiting for GPU memory.")
        free = gpu.free_gb()
        if free is None or free >= need_gb:
            return
        if time.monotonic() >= deadline:
            print(f"  [RelayVSR] Only {free:.1f} GB VRAM free (needs ~{need_gb:.1f} GB); restarting anyway.")
            return
        report(progress, 25, 100, f"GPU memory is full ({free:.1f} of ~{need_gb:.1f} GB free) — "
                                  "close other GPU apps to restore full speed; waiting...")
        time.sleep(3)


def _server(client: ComfyUIClient, progress: ProgressCallback = None) -> subprocess.Popen:
    """Returns the resident worker, starting it (and loading its weights) only if needed."""
    with _worker_lock:
        proc = _worker.get("proc")
        if proc is not None and proc.poll() is None:
            print("\n[STEP 3/5] Reusing resident RelayVSR worker (models already loaded)")
            return proc

        # RelayVSR needs ~12 GB; make ComfyUI drop any resident LTX weights first
        # (the ~35 s model load that follows gives ComfyUI ample time to actually release it)
        print("\n[STEP 3/5] Releasing ComfyUI VRAM and loading RelayVSR...")
        client.free_memory()
        report(progress, 20, 100, "Loading RelayVSR (Wan keyframes + Flash relay) on GPU...")
        proc = processes.popen(
            [str(config.RELAYVSR_PYTHON), str(config.RELAYVSR_WORKER), "--serve",
             "--weights-dir", str(config.RELAYVSR_WEIGHTS_DIR)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        _worker["proc"] = proc
    last, tail = _pump(proc, ("SERVE_READY",))
    if last is None:
        release_worker()
        raise RuntimeError("RelayVSR worker failed to start:\n" + "\n".join(tail))
    return proc


# ---------------------------------------------------------------------------
# A job
# ---------------------------------------------------------------------------
def run(
    client: ComfyUIClient,
    prepared_video_path: Path,
    prep_info: Dict[str, Any],
    audio_path: Optional[Path],
    final_output_path: Path,
    output_size: str,
    seed: int,
    auto_compare: bool,
    progress: ProgressCallback = None,
    done_segments: int = 0,
    output_codec: str = "auto",
) -> Dict[str, Any]:
    """Upscales prepared_video_path into final_output_path (audio muxed, comparisons rendered).
    If the job stops, the finished part is saved as <name>_partial.mp4 and re-running the same job
    continues after `done_segments`. Returns the final video info and comparison result."""
    lr_h, out_h = geometry(prep_info["width"], prep_info["height"], output_size)
    _cancel.clear()

    # Best steady speed ever measured for this LR geometry: the yardstick for "starved"
    lr_w = int(round(prep_info["width"] * lr_h / prep_info["height"]))
    speed_key = f"{lr_w - lr_w % 2}x{lr_h - lr_h % 2}"
    known_fps = user_settings.get_user_setting("relayvsr_fps", {}) or {}
    speed: Dict[str, Any] = {"best": float(known_fps.get(speed_key, 0.0))}

    folder = parts_dir(final_output_path)
    recycles, allow_recycle, starved_fps = 0, True, 0.0
    while True:
        try:
            proc = _server(client, progress)
        except RuntimeError as e:
            if not recycles:
                raise
            last, tail = None, [str(e)]  # canceled/failed during a restart: keep the finished part
            break
        print(f"\n[STEP 4/5] Running RelayVSR streaming 4x (LR {lr_h}p -> output {out_h}p)...")
        report(progress, 25, 100, "RelayVSR models ready — streaming frames...")

        start_frame = done_segments * SEGMENT_FRAMES
        if start_frame:
            print(f"  Resuming at frame {start_frame} ({done_segments} segment(s) already rendered)")
        job = {"video": str(prepared_video_path), "segment_dir": str(folder),
               "segment_frames": SEGMENT_FRAMES, "start_frame": start_frame,
               "lr_height": lr_h, "out_height": out_h, "seed": seed,
               "baseline_fps": speed["best"],
               "recycle": allow_recycle and recycles < MAX_RECYCLES}
        speed["run_best"] = 0.0
        try:
            proc.stdin.write(json.dumps(job) + "\n")
            proc.stdin.flush()
        except OSError:
            release_worker()
            raise RuntimeError("RelayVSR worker exited unexpectedly before accepting the job.")
        last, tail = _pump(proc, ("JOB_DONE", "JOB_ERROR", "JOB_RECYCLE"), progress, speed)
        if speed["best"] > known_fps.get(speed_key, 0.0):
            known_fps[speed_key] = round(speed["best"], 3)
            user_settings.set_user_setting("relayvsr_fps", known_fps)
        if last is None or not last.startswith("JOB_RECYCLE"):
            break
        done_segments = sum(1 for _ in folder.glob("seg_*.mp4"))
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
              f"restarting the worker (#{recycles}/{MAX_RECYCLES}) and resuming...")
        release_worker()
        # torch's reserved pool plus ~0.8 GB of CUDA context and kernels
        need_gb = speed["reserved"] + 0.8 if speed.get("reserved") else DEFAULT_NEED_GB
        try:
            _wait_for_vram(client, need_gb, progress)
        except RuntimeError as e:
            last, tail = None, [str(e)]
            break
    if last != "JOB_DONE":
        if last is None:  # process died (crash or cancellation); next job starts a fresh one
            release_worker()
        partial = save_partial(folder, audio_path, final_output_path)
        note = f"\nFinished part saved: {partial}. Run the job again to resume." if partial else ""
        raise RuntimeError("RelayVSR job stopped:\n" + "\n".join(tail[-5:]) + note)

    raw_out = config.INPUTS_DIR / f"{final_output_path.stem}_relayvsr_raw.mp4"
    concat_segments(sorted(folder.glob("seg_*.mp4")), raw_out)

    print("\n[STEP 5/5] Finalizing audio and video...")
    final_info, comparison_res = output.deliver(
        raw_out, audio_path, final_output_path, prepared_video_path,
        auto_compare, progress, keep_source=False, done_text="RelayVSR Upscale Complete!",
        output_codec=output_codec,
    )
    shutil.rmtree(folder, ignore_errors=True)
    final_output_path.with_name(f"{final_output_path.stem}_partial.mp4").unlink(missing_ok=True)
    return {"final_info": final_info, "comparison": comparison_res}
