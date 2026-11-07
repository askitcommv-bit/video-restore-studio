"""
Standalone CUDA worker for RelayVSR streaming 4x video super-resolution.
Invoked via the isolated RelayVSR virtualenv (third_party/RelayVSR/.venv).

RelayVSR always upscales exactly 4x, so the input is first resized to --lr-height
(the "low-res canvas") and the 4x result is resampled to --out-height if given.

--serve keeps the ~10 GB of weights resident and takes one JSON job per stdin line,
so a queue of clips pays the model load once instead of per clip.

On a 16 GB card the full model (~12.4 GB with activations) plus the desktop's own GPU use does
not fit, and the driver then spills arbitrary allocations into system RAM, roughly halving speed.
So the worker measures free VRAM first and keeps just enough of Wan's 30 transformer blocks in
pinned system RAM, copying each onto the GPU right before it runs. Wan only runs on keyframes
(every window_size-1 frames), so that costs ~15 ms per offloaded block per keyframe.

RelayVSR does the same work for every frame, so a drop in per-segment speed means the GPU is
starved (usually VRAM overflowing into system RAM because other apps grabbed memory). After
SLOW_SEGMENTS such segments the job ends at the segment boundary with JOB_RECYCLE so the parent
can restart the worker with fresh, fully-resident allocations and resume.
"""

import sys
import json
import math
import time
import queue
import argparse
import threading
import subprocess
from pathlib import Path

RELAYVSR_DIR = Path(__file__).resolve().parent / "third_party" / "RelayVSR"
sys.path.insert(0, str(RELAYVSR_DIR))

import torch
import torch.nn.functional as F

from refvsr_inference.checkpoints import load_checkpoints
from refvsr_inference.media import VideoWriter, input_fps, iter_frames
from refvsr_inference.models.wan.sampling import WanVSRPipeline
from refvsr_inference.runtime.streaming import CollaborativeFrameStream, InputFrame

# A segment slower than SLOW_RATIO x the best known speed counts as starved
SLOW_RATIO = 0.7
SLOW_SEGMENTS = 2


# Measured footprint with every block resident (weights + 240p activations + CUDA context), and
# the free VRAM to leave for the desktop/other apps so nothing is pushed into system RAM
FULL_FOOTPRINT_GB = 12.4
HEADROOM_GB = 2.0


class Recycle(Exception):
    """Raised at a segment boundary when the GPU is starved; all finished segments are kept."""


def gpu_free_gb():
    """Free VRAM across all processes (nvidia-smi), or None if it can't be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.splitlines()[0]
        used, total = (float(v) for v in out.split(","))
        return (total - used) / 1024
    except Exception:
        return None


def auto_offload_blocks(block_gb, num_blocks):
    """How many Wan blocks must live in system RAM for the worker to fit with headroom."""
    free = gpu_free_gb()
    if free is None:
        return 0
    deficit = FULL_FOOTPRINT_GB + HEADROOM_GB - free
    return max(0, min(num_blocks, math.ceil(deficit / block_gb))) if deficit > 0 else 0


def offload_blocks(blocks, device):
    """Moves `blocks` to pinned CPU memory; each is copied to `device` just before its forward
    and released right after (the caching allocator reuses that memory for the next block)."""
    for block in blocks:
        tensors = list(block.parameters()) + list(block.buffers())
        host = [t.data.to("cpu").pin_memory() for t in tensors]
        for t, h in zip(tensors, host):
            t.data = h

        def onload(_module, _args, _kwargs=None, tensors=tensors, host=host):
            for t, h in zip(tensors, host):
                t.data = h.to(device, non_blocking=True)

        def release(_module, _args, _output, tensors=tensors, host=host):
            for t, h in zip(tensors, host):
                t.data = h  # kernels already queued keep the GPU copy alive until they finish

        block.register_forward_pre_hook(onload)
        block.register_forward_hook(release)


def even_size(height, width, target_h):
    """Aspect-preserving (h, w) for target_h, both rounded to even numbers."""
    target_w = int(round(width * target_h / height))
    return target_h - (target_h % 2), target_w - (target_w % 2)


def resize_uint8(frame, size):
    """Antialiased bicubic resize of a uint8 CHW tensor."""
    if tuple(frame.shape[-2:]) == tuple(size):
        return frame
    x = frame.unsqueeze(0).float()
    x = F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)
    return x.round_().clamp_(0, 255).to(torch.uint8)[0]


def load_relay(weights_dir, window_size, seed, offload=-1):
    """Loads Wan + Flash once and returns the streaming relay (reused by every job in --serve mode).
    offload = number of Wan blocks kept in system RAM (-1 = as many as free VRAM requires)."""
    weights = Path(weights_dir)
    wan_ckpt, flash_ckpt = weights / "wan.safetensors", weights / "flash.safetensors"
    for ckpt in (wan_ckpt, flash_ckpt):
        if not ckpt.is_file():
            raise FileNotFoundError(f"RelayVSR checkpoint not found: {ckpt}")
    if not torch.cuda.is_available():
        raise RuntimeError("RelayVSR requires a CUDA GPU.")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    print("MODEL_LOAD: Loading RelayVSR Wan + Flash checkpoints...", flush=True)
    # Loaded on the CPU first so offloaded blocks never touch VRAM (an allocation that lands in
    # system RAM while VRAM is momentarily full would stay there and slow every frame)
    wan, flash = load_checkpoints(wan_ckpt, flash_ckpt, device="cpu", dtype=dtype)
    blocks = list(wan.blocks)
    block_gb = sum(p.numel() * p.element_size() for p in blocks[0].parameters()) / 2**30
    count = auto_offload_blocks(block_gb, len(blocks)) if offload < 0 else min(offload, len(blocks))
    offloaded = blocks[len(blocks) - count:] if count else []
    offload_blocks(offloaded, device)
    offloaded_ids = set(map(id, offloaded))
    for module in [m for m in wan.children() if m is not wan.blocks] + [b for b in blocks if id(b) not in offloaded_ids]:
        module.to(device)
    for param in wan.parameters(recurse=False):  # top-level tensors such as scale_shift_table
        param.data = param.data.to(device)
    for name, buf in wan.named_buffers(recurse=False):  # prompt_embeds
        setattr(wan, name, buf.to(device))
    flash = flash.to(device)
    print(f"OFFLOAD: {count}/{len(blocks)} Wan blocks in system RAM ({count * block_gb:.1f} GB)", flush=True)
    sampler = WanVSRPipeline(
        scheduler_config={"num_train_timesteps": 1000, "shift": 5.0},
        sampling={
            "execution_mode": "streaming",
            "num_inference_steps": 1,
            "timesteps": [1000],
            "guidance_scale": 1.0,
            "seed": seed,
            "kv_cache_window_size": wan.temporal_context_frames,
        },
    )
    sampler.prepare_streaming(model=wan, prompt_embeds=wan.prompt_embeds, device=device, compute_dtype=dtype)
    relay = CollaborativeFrameStream(
        wan_model=wan,
        flash_model=flash,
        wan_pipeline=sampler,
        prompt_embeds=wan.prompt_embeds,
        device=device,
        compute_dtype=dtype,
        window_size=window_size,
        seed=seed,
    )
    print(f"MODEL_READY: {torch.cuda.memory_allocated() / 2**30:.2f} GB allocated", flush=True)
    return relay


class SegmentWriter:
    """Writes frames as consecutive seg_NNNNN.mp4 files of `segment_frames` each. Every segment is
    published atomically when full (VideoWriter writes to a temp file and renames on close), so a
    killed job leaves only complete, playable segments behind; numbering follows the global frame
    index so a resumed job continues the same sequence."""

    def __init__(self, segment_dir, fps, segment_frames, start_frame):
        self.dir, self.fps, self.size = Path(segment_dir), fps, segment_frames
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = start_frame  # global frame index of the next frame
        self.writer = None

    def write(self, frame):
        if self.writer is None:
            self.writer = VideoWriter(self.dir / f"seg_{self.index // self.size:05d}.mp4", self.fps).__enter__()
        self.writer.write(frame)
        self.index += 1
        if self.index % self.size == 0:
            self.close()

    def close(self, exc_type=None):
        if self.writer is not None:
            self.writer.__exit__(exc_type, None, None)
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(exc_type)  # a trailing partial segment is only published on success


@torch.no_grad()
def run_job(relay, video, output=None, lr_height=0, out_height=0, max_frames=None, seed=42, profile=False,
            segment_dir=None, segment_frames=300, start_frame=0, baseline_fps=0.0, recycle=False):
    """Upscales one video with an already-loaded relay. Prints INIT/GEOMETRY/PROGRESS/SEGMENT/STATS/DONE
    lines. With segment_dir, output goes to resumable segments instead of `output`; start_frame (a multiple
    of segment_frames) skips frames already rendered by an earlier, interrupted run. With recycle=True,
    SLOW_SEGMENTS consecutive segments below SLOW_RATIO x max(baseline_fps, best seen) raise Recycle."""
    video_path = Path(video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if segment_dir is None and output is None:
        raise ValueError("run_job needs `output` or `segment_dir`.")
    if start_frame % segment_frames:
        raise ValueError("start_frame must be a multiple of segment_frames.")
    relay.seed = seed  # the per-stream generator is seeded from this at stream start

    import av
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        total_frames = stream.frames or 0
        src_w, src_h = stream.codec_context.width, stream.codec_context.height
    if max_frames:
        total_frames = min(total_frames, max_frames) if total_frames else max_frames
    fps = input_fps(video_path)

    lr_size = even_size(src_h, src_w, lr_height) if lr_height else (src_h, src_w)
    out_size = even_size(src_h, src_w, out_height) if out_height else None
    final_size = out_size or (lr_size[0] * 4, lr_size[1] * 4)
    print(f"INIT: Total frames={total_frames}, FPS={float(fps):.3f}, Dimensions={src_w}x{src_h}", flush=True)
    print(f"GEOMETRY: LR {lr_size[1]}x{lr_size[0]} -> 4x {lr_size[1]*4}x{lr_size[0]*4} -> output {final_size[1]}x{final_size[0]}", flush=True)

    if start_frame and total_frames and start_frame >= total_frames:
        print(f"DONE: all {total_frames} frames already rendered", flush=True)
        return
    remaining = (max_frames - start_frame) if max_frames else None
    frames = iter_frames(video_path, max_frames=remaining, start_frame=start_frame)
    packets = (  # a (resumed) stream always starts at local index 0
        InputFrame(i, resize_uint8(rgb, lr_size), source_time=i / float(fps))
        for i, rgb in enumerate(frames)
    )
    sink = (SegmentWriter(segment_dir, fps, segment_frames, start_frame) if segment_dir
            else VideoWriter(Path(output), fps))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processed = 0
    seg_started, best_fps, slow_run = started, float(baseline_fps or 0.0), 0
    stage_ms = {}
    # x264 encoding runs on its own thread so the GPU never waits on the encoder
    encode_q: queue.Queue = queue.Queue(maxsize=8)
    encode_err: list = []

    def encode_loop(writer):
        try:
            while (frame := encode_q.get()) is not None:
                writer.write(frame)
        except Exception as exc:
            encode_err.append(exc)
            while encode_q.get() is not None:  # drain so the producer never blocks
                pass

    try:
        with sink as writer:
            encoder = threading.Thread(target=encode_loop, args=(writer,), daemon=True)
            encoder.start()
            try:
                for result in relay.frames(packets):
                    if encode_err:
                        break
                    if profile:
                        for stage, span in result.timings:
                            stage_ms[stage] = stage_ms.get(stage, 0.0) + span.milliseconds()
                    x = result.rgb.float().add_(1).mul_(127.5)
                    if out_size:
                        x = F.interpolate(x, size=out_size, mode="bicubic", align_corners=False, antialias=True)
                    rgb = x[0].round_().clamp_(0, 255).to(torch.uint8)
                    encode_q.put(rgb.permute(1, 2, 0).contiguous().cpu())
                    processed += 1
                    if processed == relay.window_size:
                        # First window includes Triton JIT compilation; time steady state separately
                        warm_t = time.perf_counter()
                    done = start_frame + processed
                    print(f"PROGRESS: {done}/{max(total_frames, done)}", flush=True)
                    if done % segment_frames == 0:
                        now = time.perf_counter()
                        seg_fps = segment_frames / max(now - seg_started, 1e-6)
                        seg_started = now
                        # Return freed blocks to the driver so later allocations can land back in
                        # VRAM if it has room again (instead of staying in spilled system memory)
                        torch.cuda.empty_cache()
                        reserved = torch.cuda.memory_reserved() / 2**30
                        # The first segment of a run includes model warm-up/JIT, so it is never judged
                        judged = processed > segment_frames
                        slow = judged and best_fps > 0 and seg_fps < SLOW_RATIO * best_fps
                        print(f"SEGMENT: {done // segment_frames} {seg_fps:.3f} fps judged={int(judged)} "
                              f"slow={int(slow)} reserved={reserved:.2f}", flush=True)
                        if judged:
                            best_fps = max(best_fps, seg_fps)
                        slow_run = slow_run + 1 if slow else 0
                        if recycle and slow_run >= SLOW_SEGMENTS and segment_dir:
                            raise Recycle(f"{seg_fps:.2f} fps vs {best_fps:.2f} fps best")
            finally:
                encode_q.put(None)
                encoder.join()
            if encode_err:
                raise encode_err[0]
    finally:
        packets.close()
        frames.close()

    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"STATS: {processed} frames in {elapsed:.1f}s ({processed / max(elapsed, 1e-6):.2f} FPS), peak VRAM {peak:.2f} GB", flush=True)
    if processed > relay.window_size:
        steady = processed - relay.window_size
        steady_s = time.perf_counter() - warm_t
        print(f"STATS: steady-state {steady} frames in {steady_s:.1f}s ({steady / max(steady_s, 1e-6):.2f} FPS)", flush=True)
    for stage, ms in sorted(stage_ms.items(), key=lambda kv: -kv[1]):
        print(f"PROFILE: {stage:<16} {ms / 1000:8.1f}s total", flush=True)
    print(f"DONE: Output saved to {segment_dir or output}", flush=True)


def serve(relay):
    """Keeps the models resident: one JSON job per stdin line, answered by JOB_DONE / JOB_ERROR.
    Exits when stdin closes (the parent released the worker or exited)."""
    print("SERVE_READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            run_job(relay, **json.loads(line))
            print("JOB_DONE", flush=True)
        except Recycle as exc:
            print(f"JOB_RECYCLE: {exc}", flush=True)
        except Exception as exc:
            torch.cuda.empty_cache()
            print(f"JOB_ERROR: {type(exc).__name__}: {exc}".replace("\n", " "), flush=True)


def main():
    parser = argparse.ArgumentParser(description="RelayVSR 4x streaming video super-resolution worker.")
    parser.add_argument("--serve", action="store_true", help="Load once, then take JSON jobs from stdin (one per line)")
    parser.add_argument("--video", "-v", help="Input video path")
    parser.add_argument("--output", "-o", help="Output MP4 path (video only, no audio)")
    parser.add_argument("--weights-dir", required=True, help="Folder holding wan.safetensors + flash.safetensors")
    parser.add_argument("--lr-height", type=int, default=0, help="Resize input to this height before 4x (0 = native)")
    parser.add_argument("--out-height", type=int, default=0, help="Resample 4x result to this height (0 = keep 4x)")
    parser.add_argument("--window-size", type=int, default=16, help="Keyframe interval + 1")
    parser.add_argument("--offload-blocks", type=int, default=-1,
                        help="Wan blocks kept in system RAM (-1 = auto from free VRAM, 0 = none)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true", help="Print per-stage GPU time totals (Wan / Flash)")
    args = parser.parse_args()
    if not args.serve and not (args.video and args.output):
        parser.error("--video and --output are required unless --serve is given")

    try:
        relay = load_relay(args.weights_dir, args.window_size, args.seed, args.offload_blocks)
        if args.serve:
            serve(relay)
        else:
            run_job(relay, args.video, args.output, args.lr_height, args.out_height,
                    args.max_frames, args.seed, args.profile)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
