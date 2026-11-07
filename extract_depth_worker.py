"""
Standalone CUDA worker for frame-by-frame 3D metric scene depth estimation using Depth Anything V2 Small.
Invoked either directly or via ComfyUI's embedded CUDA Python runtime.
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description="Extract 3D metric depth sequence from video.")
    parser.add_argument("--video", "-v", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output depth video path")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Inference batch size")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"INIT: Total frames={total_frames}, FPS={fps}, Dimensions={w}x{h}", flush=True)

    device = 0 if torch.cuda.is_available() else -1
    print(f"MODEL_LOAD: Loading Depth Anything V2 on device={device}...", flush=True)
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
    print("MODEL_READY", flush=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))

    batch_frames = []
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_frames.append(Image.fromarray(rgb))

        if len(batch_frames) >= args.batch_size:
            depth_results = depth_pipe(batch_frames, batch_size=args.batch_size)
            for d_res in depth_results:
                depth_np = np.array(d_res["depth"].convert("RGB"))
                depth_np = cv2.resize(depth_np, (w, h))
                writer.write(cv2.cvtColor(depth_np, cv2.COLOR_RGB2BGR))
                processed += 1
                print(f"PROGRESS: {processed}/{total_frames}", flush=True)
            batch_frames = []

    if batch_frames:
        depth_results = depth_pipe(batch_frames, batch_size=len(batch_frames))
        for d_res in depth_results:
            depth_np = np.array(d_res["depth"].convert("RGB"))
            depth_np = cv2.resize(depth_np, (w, h))
            writer.write(cv2.cvtColor(depth_np, cv2.COLOR_RGB2BGR))
            processed += 1
            print(f"PROGRESS: {processed}/{total_frames}", flush=True)

    cap.release()
    writer.release()
    print(f"DONE: Output saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()
