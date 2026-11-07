"""
Command Line Interface for LTX-2.5 Video Restoration Suite.
Usage:
  python restore_cli.py status
  python restore_cli.py download [--precision int8|bf16] [--loras-only] [--relayvsr]
  python restore_cli.py preprocess --input "clip.mp4"
  python restore_cli.py restore --input "clip.mp4" [--preset archival_general] [--resolution 4K]
  python restore_cli.py refine --input "clip.mp4" [--resolution 4K]
  python restore_cli.py compare --original "clip.mp4" --restored "restored.mp4"
  python restore_cli.py presets
"""

import sys
import time
import glob
import argparse
from pathlib import Path

# Configure UTF-8 console output for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import model_downloader
import video_preprocessor
import comparison_generator
from restore_engine import VideoRestorationPipeline, ComfyUIClient


def cmd_status(args):
    """Checks server, GPU VRAM, and model weights."""
    print("\n[SYSTEM DIAGNOSTICS]")
    client = ComfyUIClient()
    if client.is_online():
        print(f" ComfyUI Server: [ONLINE] at {client.base_url}")
        try:
            stats = client.get_system_stats()
            devices = stats.get("devices", [])
            for d in devices:
                vram_gb = d.get("vram_total", 0) / (1024**3)
                vram_free = d.get("vram_free", 0) / (1024**3)
                print(f" Device: {d.get('name')} | VRAM: {vram_free:.1f} GB free / {vram_gb:.1f} GB total")
        except Exception as e:
            print(f" Could not read stats: {e}")
    else:
        print(f" ComfyUI Server: [OFFLINE] at {client.base_url}")
        print(f" -> Launch ComfyUI via: {config.COMFYUI_ROOT / 'run_nvidia_gpu.bat'}")

    print()
    model_downloader.print_status_table()


def cmd_download(args):
    """Downloads required model weights."""
    if args.relayvsr:
        print("\nInstalling RelayVSR (code + isolated CUDA venv + ~10.2 GB weights)...")
        model_downloader.install_relayvsr()
        return
    prec = "int8" if not args.bf16 else "bf16"
    print(f"\nStarting model download (Precision: {prec.upper()})...")
    model_downloader.download_all_models(precision=prec, only_loras=args.loras_only)


def cmd_preprocess(args):
    """Inspects and sanitizes a video file."""
    input_path = Path(args.input)
    print(f"\n[INSPECTING VIDEO]: {input_path}")
    raw = video_preprocessor.get_video_info(input_path)
    for k, v in raw.items():
        print(f"  {k}: {v}")

    print("\n[PREPROCESSING FOR LTX-2.5]...")
    out, info = video_preprocessor.preprocess_video(
        input_path=input_path,
        output_path=args.output,
        force_deinterlace=args.deinterlace,
        max_frames=args.max_frames,
        start_frame=getattr(args, "start_frame", None),
        end_frame=getattr(args, "end_frame", None)
    )
    print(f"\n[DONE] Prepared video created at: {out}")
    print(f"  Canvas: {info['width']}x{info['height']} (divisible by 32)")
    print(f"  Frames: {info['total_frames']} (legal 8n+1, range: F {info.get('start_frame', 0)}..{info.get('end_frame', info['total_frames']-1)})")


def make_cli_progress():
    def cb(curr, tot, msg):
        print(f"[{curr:3d}%] {msg}", flush=True)
    return cb


def cmd_restore(args):
    """Executes archival footage restoration pipeline with 1-click preset auto-configuration."""
    preset_key = getattr(args, "preset", config.DEFAULT_PRESET) or config.DEFAULT_PRESET
    p = config.get_preset(preset_key)

    # Auto-resolve engine, mode, resolution, strengths unless explicitly overridden by CLI flags
    engine_type = getattr(args, "engine", None) or p.get("engine_type", "diffusion")
    mode = getattr(args, "mode", None) or p.get("mode", "restore")
    res_val = args.resolution if (getattr(args, "resolution", None) and args.resolution != "FullHD") else p.get("output_size", "FullHD")
    tile_val = args.tile if (getattr(args, "tile", None) and args.tile != "FullHD") else p.get("tile_size", "FullHD")
    guide_val = getattr(args, "guide_strength", None) or p.get("guide_strength", 0.72)
    cfg_val = getattr(args, "cfg", None) or p.get("cfg", 2.5)
    sol_sigma_val = getattr(args, "sigma", None) or p.get("sol_sigma", 0.50)
    superres_model_val = getattr(args, "superres_model", None) or p.get("superres_model", "RealESRGAN_x2plus.pth")
    pos_prompt = getattr(args, "prompt", None) or p.get("positive")
    neg_prompt = getattr(args, "negative", None) or p.get("negative")

    print(f"\n[1-CLICK PRESET]: {p.get('name', preset_key)}")
    print(f"  • Engine: {engine_type.upper()} ({p.get('badge', '')})")
    print(f"  • Mode: {mode} | Resolution: {res_val} | Tile: {tile_val}")
    print(f"  • Speed Estimate: {p.get('est_speed', 'N/A')}")
    print(f"  • Features: {p.get('description', '')}")

    teacache_flag = getattr(args, "teacache", None)
    teacache_val = teacache_flag if teacache_flag is not None else p.get("enable_teacache", False)
    teacache_thresh_val = getattr(args, "teacache_thresh", None) or p.get("teacache_thresh", 0.06)

    pipeline = VideoRestorationPipeline()
    res = pipeline.restore_footage(
        input_video_path=args.input,
        output_dir=getattr(args, "output", None),
        output_file=getattr(args, "output_file", None),
        max_frames=getattr(args, "max_frames", None),
        start_frame=getattr(args, "start_frame", None),
        end_frame=getattr(args, "end_frame", None),
        mode=mode,
        engine_type=engine_type,
        superres_model=superres_model_val,
        prompt_preset=preset_key,
        custom_positive_prompt=pos_prompt,
        custom_negative_prompt=neg_prompt,
        output_size=res_val,
        tile_size=tile_val,
        guide_strength=guide_val,
        cfg=cfg_val,
        sol_sigma=sol_sigma_val,
        seed=args.seed,
        enable_teacache=teacache_val,
        teacache_thresh=teacache_thresh_val,
        reference_image=getattr(args, "reference", None),
        extract_depth=getattr(args, "depth", False),
        progress_callback=make_cli_progress(),
        output_codec=getattr(args, "codec", "auto"),
    )
    print("\n[SUCCESS] Restoration job processed!")
    print(f"  Status: {res['status']}")
    print(f"  Output Video: {res.get('final_output_path')}")
    print(f"  Configured Workflow: {res.get('workflow_file')}")
    print(f"  Prepared Video: {res.get('prepared_input')}")
    if res.get("reference_image"):
        print(f"  Reference Portrait: {res['reference_image']}")
    if res.get("depth_output_path"):
        print(f"  3D Depth Sequence: {res['depth_output_path']}")
    print(f"  Output Prefix: {res.get('output_prefix')}")
    print("\nYou can run this in ComfyUI or open the Web UI dashboard!")


def cmd_ladder(args):
    """Executes 2-Stage Native Ladder restoration pipeline (Clean + 2x Spatial Upscale + 4K Refine)."""
    pipeline = VideoRestorationPipeline()
    res = pipeline.restore_footage(
        input_video_path=args.input,
        output_dir=getattr(args, "output", None),
        output_file=getattr(args, "output_file", None),
        max_frames=getattr(args, "max_frames", None),
        start_frame=getattr(args, "start_frame", None),
        end_frame=getattr(args, "end_frame", None),
        mode="two_stage",
        engine_type="diffusion",
        prompt_preset=args.preset,
        custom_positive_prompt=args.prompt,
        custom_negative_prompt=args.negative,
        output_size=args.resolution,
        tile_size=args.tile,
        seed=args.seed,
        reference_image=getattr(args, "reference", None),
        extract_depth=getattr(args, "depth", False),
        progress_callback=make_cli_progress(),
        output_codec=getattr(args, "codec", "auto"),
    )
    print("\n[SUCCESS] 2-Stage Ladder Restoration job processed!")
    print(f"  Status: {res['status']}")
    print(f"  Output Video: {res.get('final_output_path')}")
    print(f"  Configured Workflow: {res['workflow_file']}")
    print(f"  Prepared Video: {res['prepared_input']}")
    if res.get("reference_image"):
        print(f"  Reference Portrait: {res['reference_image']}")
    if res.get("depth_output_path"):
        print(f"  3D Depth Sequence: {res['depth_output_path']}")


def cmd_refine(args):
    """Executes detail refinement pipeline."""
    pipeline = VideoRestorationPipeline()
    res = pipeline.restore_footage(
        input_video_path=args.input,
        output_dir=getattr(args, "output", None),
        output_file=getattr(args, "output_file", None),
        max_frames=getattr(args, "max_frames", None),
        start_frame=getattr(args, "start_frame", None),
        end_frame=getattr(args, "end_frame", None),
        mode="refine",
        engine_type="diffusion",
        prompt_preset=args.preset or "refine_upscaled_4k",
        custom_positive_prompt=args.prompt,
        custom_negative_prompt=args.negative,
        output_size=args.resolution,
        tile_size=args.tile,
        seed=args.seed,
        reference_image=getattr(args, "reference", None),
        extract_depth=getattr(args, "depth", False),
        progress_callback=make_cli_progress(),
        output_codec=getattr(args, "codec", "auto"),
    )
    print("\n[SUCCESS] Detail Refinement job configured!")
    print(f"  Configured Workflow: {res['workflow_file']}")
    if res.get("reference_image"):
        print(f"  Reference Portrait: {res['reference_image']}")
    if res.get("depth_output_path"):
        print(f"  3D Depth Sequence: {res['depth_output_path']}")


def cmd_sol(args):
    """Executes NVIDIA 1-Step SoL-Refiner restoration pipeline (~10s)."""
    pipeline = VideoRestorationPipeline()
    res = pipeline.restore_footage(
        input_video_path=args.input,
        output_dir=getattr(args, "output", None),
        output_file=getattr(args, "output_file", None),
        max_frames=getattr(args, "max_frames", None),
        start_frame=getattr(args, "start_frame", None),
        end_frame=getattr(args, "end_frame", None),
        engine_type="sol_refiner",
        sol_sigma=getattr(args, "sigma", 0.50),
        prompt_preset=args.preset,
        custom_positive_prompt=args.prompt,
        custom_negative_prompt=args.negative,
        seed=args.seed,
        reference_image=getattr(args, "reference", None),
        extract_depth=getattr(args, "depth", False),
        progress_callback=make_cli_progress(),
        output_codec=getattr(args, "codec", "auto"),
    )
    print("\n[SUCCESS] NVIDIA 1-Step SoL-Refiner job processed!")
    print(f"  Status: {res['status']}")
    print(f"  Output Video: {res.get('final_output_path')}")
    print(f"  Configured Workflow: {res.get('workflow_file')}")
    print(f"  Prepared Video: {res.get('prepared_input')}")
    if res.get("reference_image"):
        print(f"  Reference Portrait: {res['reference_image']}")
    if res.get("depth_output_path"):
        print(f"  3D Depth Sequence: {res['depth_output_path']}")


def cmd_compare(args):
    """Generates comparison videos."""
    orig = Path(args.original)
    rest = Path(args.restored)
    out_dir = config.COMPARISONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[GENERATING COMPARISONS]...")
    if args.type in ["side_by_side", "both"]:
        sbs_path = out_dir / f"{orig.stem}_vs_{rest.stem}_sidebyside.mp4"
        print(f"  Rendering Side-by-Side: {sbs_path.name}...")
        comparison_generator.generate_side_by_side(orig, rest, sbs_path)
        print(f"  [OK] Saved: {sbs_path}")

    if args.type in ["split", "both"]:
        split_path = out_dir / f"{orig.stem}_vs_{rest.stem}_split.mp4"
        print(f"  Rendering 50/50 Split Wipe: {split_path.name}...")
        comparison_generator.generate_split_screen(orig, rest, split_path)
        print(f"  [OK] Saved: {split_path}")


def cmd_presets(args):
    """Lists all available 1-click restoration presets."""
    print("=" * 85)
    print(" LTX-2.5 SIMPLIFIED 1-CLICK RESTORATION PRESETS")
    print("=" * 85)
    for key, p in config.RESTORATION_PRESETS.items():
        print(f"\nPreset ID: [{key}]")
        print(f"  Name:        {p['name']}")
        print(f"  Engine:      {p['engine_type'].upper()} ({p.get('mode', 'restore')})")
        print(f"  Speed:       {p.get('est_speed', 'N/A')}")
        print(f"  Badge:       {p.get('badge', '')}")
        print(f"  Output Res:  {p.get('output_size', 'FullHD')} (Tile: {p.get('tile_size', 'FullHD')})")
        print(f"  Description: {p['description']}")
    print("\n" + "=" * 85)


def cmd_batch(args):
    """Processes multiple video files sequentially or resumes pending jobs from queue_state.json."""
    from app.core.queue_manager import QueueManager

    qm = QueueManager()

    # Case 1: Run pending jobs directly from saved queue state
    if getattr(args, "from_queue", False):
        pending_jobs = [j for j in qm.jobs if j.status == "pending"]
        if not pending_jobs:
            print("\n[BATCH QUEUE] No pending jobs found in queue_state.json.")
            return

        print(f"\n=======================================================")
        print(f" PROCESSING RESUMED BATCH QUEUE ({len(pending_jobs)} JOBS)")
        print(f"=======================================================")

        pipeline = VideoRestorationPipeline()
        completed = 0
        failed = 0

        for i, job in enumerate(pending_jobs, 1):
            print(f"\n-------------------------------------------------------")
            print(f" [QUEUE JOB {i}/{len(pending_jobs)}] {job.filename}")
            print(f" Preset: {job.preset_name} | Engine: {job.engine_type.upper()}")
            print(f" Target: {job.output_filename}")
            print(f"-------------------------------------------------------")

            job.status = "processing"
            job.started_at = time.time()
            qm._save_queue_state()

            try:
                res = pipeline.restore_footage(
                    input_video_path=job.input_path,
                    output_file=job.output_path,
                    output_dir=job.params.get("output_dir"),
                    max_frames=job.params.get("max_frames"),
                    start_frame=job.params.get("start_frame"),
                    end_frame=job.params.get("end_frame"),
                    mode=job.mode,
                    engine_type=job.engine_type,
                    superres_model=job.params.get("superres_model", "RealESRGAN_x2plus.pth"),
                    prompt_preset=job.preset_id,
                    custom_positive_prompt=job.params.get("positive_prompt"),
                    custom_negative_prompt=job.params.get("negative_prompt"),
                    output_size=job.params.get("output_size", "FullHD"),
                    tile_size=job.params.get("tile_size", "FullHD"),
                    guide_strength=job.params.get("guide_strength", 0.72),
                    cfg=job.params.get("cfg", 2.5),
                    sol_sigma=job.params.get("sol_sigma", 0.50),
                    seed=job.params.get("seed"),
                    enable_teacache=job.params.get("enable_teacache"),
                    teacache_thresh=job.params.get("teacache_thresh", 0.06),
                    reference_image=job.params.get("reference_image"),
                    extract_depth=job.params.get("extract_depth", False),
                    clean_noise=job.params.get("clean_noise"),
                    stabilize_jitter=job.params.get("stabilize_jitter"),
                    progress_callback=make_cli_progress()
                )
                job.status = "completed"
                job.progress = 100
                job.completed_at = time.time()
                job.result = res
                completed += 1
                print(f"[OK] Completed: {job.output_filename}")
            except Exception as e:
                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = time.time()
                failed += 1
                print(f"[ERROR] Failed {job.filename}: {e}")
            finally:
                qm._save_queue_state()

        print(f"\n=======================================================")
        print(f" BATCH QUEUE SUMMARY: {completed} Completed, {failed} Failed out of {len(pending_jobs)}")
        print(f"=======================================================\n")
        return

    # Case 2: Process files specified via --inputs
    raw_inputs = getattr(args, "inputs", []) or []
    expanded_files = []
    for item in raw_inputs:
        matched = glob.glob(item)
        if matched:
            expanded_files.extend(matched)
        else:
            p = Path(item)
            if p.exists():
                expanded_files.append(str(p))

    if not expanded_files:
        print("\n[ERROR] No valid input videos specified. Use --inputs <file1> <file2> or --from-queue.")
        return

    # Remove duplicates preserving order
    unique_files = list(dict.fromkeys(expanded_files))

    preset_key = getattr(args, "preset", config.DEFAULT_PRESET) or config.DEFAULT_PRESET
    p = config.get_preset(preset_key)
    engine_type = getattr(args, "engine", None) or p.get("engine_type", "diffusion")
    mode = getattr(args, "mode", None) or p.get("mode", "restore")
    res_val = args.resolution if (getattr(args, "resolution", None) and args.resolution != "FullHD") else p.get("output_size", "FullHD")
    tile_val = args.tile if (getattr(args, "tile", None) and args.tile != "FullHD") else p.get("tile_size", "FullHD")
    guide_val = getattr(args, "guide_strength", None) or p.get("guide_strength", 0.72)
    cfg_val = getattr(args, "cfg", None) or p.get("cfg", 2.5)
    sol_sigma_val = getattr(args, "sigma", None) or p.get("sol_sigma", 0.50)
    superres_model_val = getattr(args, "superres_model", None) or p.get("superres_model", "RealESRGAN_x2plus.pth")
    pos_prompt = getattr(args, "prompt", None) or p.get("positive")
    neg_prompt = getattr(args, "negative", None) or p.get("negative")
    teacache_flag = getattr(args, "teacache", None)
    teacache_val = teacache_flag if teacache_flag is not None else p.get("enable_teacache", False)
    teacache_thresh_val = getattr(args, "teacache_thresh", None) or p.get("teacache_thresh", 0.06)

    base_params = {
        "engine_type": engine_type,
        "superres_model": superres_model_val,
        "guide_strength": guide_val,
        "cfg": cfg_val,
        "sol_sigma": sol_sigma_val,
        "enable_teacache": teacache_val,
        "teacache_thresh": teacache_thresh_val,
        "mode": mode,
        "preset": preset_key,
        "positive_prompt": pos_prompt,
        "negative_prompt": neg_prompt,
        "output_size": res_val,
        "tile_size": tile_val,
        "seed": args.seed,
        "clean_noise": True,
        "stabilize_jitter": True,
        "max_frames": getattr(args, "max_frames", None)
    }

    if getattr(args, "add_only", False):
        added = qm.add_batch_files(
            file_paths=unique_files,
            base_params=base_params,
            preset_id=preset_key,
            preset_name=p.get("name", preset_key)
        )
        print(f"\n[OK] Added {len(added)} jobs to queue_state.json.")
        print(f"Run them later with: python restore_cli.py batch --from-queue")
        return

    print(f"\n=======================================================")
    print(f" STARTING BATCH RESTORATION ({len(unique_files)} VIDEOS)")
    print(f" Preset: {p.get('name', preset_key)} | Engine: {engine_type.upper()}")
    print(f" Resolution: {res_val} | Tile: {tile_val}")
    print(f"=======================================================")

    pipeline = VideoRestorationPipeline()
    out_dir = Path(args.output) if getattr(args, "output", None) else config.get_last_output_folder()
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = f"superres_{superres_model_val.split('.')[0].lower()}" if engine_type == "superres" else ("relayvsr" if engine_type == "relayvsr" else (mode if mode != "restore" else "restored"))

    results = []
    for idx, fpath in enumerate(unique_files, 1):
        fp = Path(fpath)
        out_file = config.get_unique_output_path(
            output_dir=out_dir,
            base_stem=f"{fp.stem}_{mode_tag}",
            suffix=".mp4"
        )

        print(f"\n-------------------------------------------------------")
        print(f" [BATCH {idx}/{len(unique_files)}] {fp.name}")
        print(f" Target Output: {out_file.name}")
        print(f"-------------------------------------------------------")

        try:
            res = pipeline.restore_footage(
                input_video_path=fp,
                output_file=out_file,
                output_dir=out_dir,
                max_frames=getattr(args, "max_frames", None),
                mode=mode,
                engine_type=engine_type,
                superres_model=superres_model_val,
                prompt_preset=preset_key,
                custom_positive_prompt=pos_prompt,
                custom_negative_prompt=neg_prompt,
                output_size=res_val,
                tile_size=tile_val,
                guide_strength=guide_val,
                cfg=cfg_val,
                sol_sigma=sol_sigma_val,
                seed=args.seed,
                enable_teacache=teacache_val,
                teacache_thresh=teacache_thresh_val,
                reference_image=getattr(args, "reference", None),
                extract_depth=getattr(args, "depth", False),
                progress_callback=make_cli_progress()
            )
            results.append({"file": fp.name, "status": "SUCCESS", "output": str(out_file)})
            print(f"[OK] Completed: {out_file.name}")
        except Exception as e:
            results.append({"file": fp.name, "status": "FAILED", "error": str(e)})
            print(f"[ERROR] Failed {fp.name}: {e}")

    print(f"\n=======================================================")
    print(f" BATCH PROCESSING COMPLETED: {sum(1 for r in results if r['status'] == 'SUCCESS')}/{len(unique_files)} Succeeded")
    print(f"=======================================================")
    for r in results:
        status_symbol = "[OK]" if r["status"] == "SUCCESS" else "[FAIL]"
        detail = r.get("output") or r.get("error")
        print(f" {status_symbol} {r['file']} -> {detail}")
    print()


def main():
    parser = argparse.ArgumentParser(description="LTX-2.5 Video Restoration CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = subparsers.add_parser("status", help="Check server and model status")
    p_status.set_defaults(func=cmd_status)

    # download
    p_dl = subparsers.add_parser("download", help="Download model weights")
    p_dl.add_argument("--bf16", action="store_true", help="Download full BF16 instead of INT8")
    p_dl.add_argument("--loras-only", action="store_true", help="Download only Restore & Refine IC-LoRAs")
    p_dl.add_argument("--relayvsr", action="store_true", help="Install RelayVSR streaming 4x upscaler (separate venv + weights)")
    p_dl.set_defaults(func=cmd_download)

    # preprocess
    p_prep = subparsers.add_parser("preprocess", help="Preprocess video for LTX-2.5")
    p_prep.add_argument("--input", "-i", required=True, help="Input video path")
    p_prep.add_argument("--output", "-o", default=None, help="Output video path")
    p_prep.add_argument("--deinterlace", action="store_true", default=True, help="Force deinterlacing")
    p_prep.add_argument("--max-frames", type=int, default=None, help="Cap maximum frames")
    p_prep.add_argument("--start-frame", "--in", type=int, default=None, dest="start_frame", help="In Point / Start frame index (0-based)")
    p_prep.add_argument("--end-frame", "--out", type=int, default=None, dest="end_frame", help="Out Point / End frame index (0-based)")
    p_prep.set_defaults(func=cmd_preprocess)

    # restore
    p_res = subparsers.add_parser("restore", help="Restore archival footage with 1-click presets")
    p_res.add_argument("--input", "-i", required=True, help="Input video path")
    p_res.add_argument("--output", "-o", default=None, help="Target output folder")
    p_res.add_argument("--output-file", default=None, help="Specific target output file path")
    p_res.add_argument("--max-frames", type=int, default=None, help="Cap maximum frames to process (e.g. 65 for preview)")
    p_res.add_argument("--start-frame", "--in", type=int, default=None, dest="start_frame", help="In Point / Start frame index (0-based)")
    p_res.add_argument("--end-frame", "--out", type=int, default=None, dest="end_frame", help="Out Point / End frame index (0-based)")
    p_res.add_argument("--preset", default=config.DEFAULT_PRESET, help=f"Restoration preset ID (default: {config.DEFAULT_PRESET})")
    p_res.add_argument("--engine", default=None, choices=["diffusion", "sol_refiner", "superres", "relayvsr"], help="Override engine type")
    p_res.add_argument("--prompt", default=None, help="Custom positive prompt override")
    p_res.add_argument("--negative", default=None, help="Custom negative prompt override")
    p_res.add_argument("--resolution", default="FullHD", choices=["HD", "FullHD", "4K", "8K"], help="Target output resolution (default: FullHD)")
    p_res.add_argument("--tile", default="FullHD", choices=["qHD", "HD", "FullHD"], help="Sampling tile size (default: FullHD)")
    p_res.add_argument("--codec", default="auto", choices=["auto", "hevc", "h264"],
                       help="Final video codec: auto = HEVC on the best encoder found (NVENC/QSV/AMF/x265), else H.264")
    p_res.add_argument("--seed", type=int, default=None, help="RNG seed")
    p_res.add_argument("--guide-strength", type=float, default=None, dest="guide_strength", help="Override LTX guide strength")
    p_res.add_argument("--cfg", type=float, default=None, help="Override CFG scale")
    p_res.add_argument("--sigma", type=float, default=None, help="Override SoL-Refiner sigma strength")
    p_res.add_argument("--ref", "--reference", dest="reference", default=None, help="Path to character reference image to lock facial likeness and identity")
    p_res.add_argument("--depth", action="store_true", default=False, help="Extract 3D metric scene depth sequence with Depth Anything V2")
    p_res.add_argument("--teacache", action="store_true", default=None, help="Enable TeaCache DiT layer acceleration (~2x speedup)")
    p_res.add_argument("--no-teacache", action="store_false", dest="teacache", help="Disable TeaCache DiT layer acceleration")
    p_res.add_argument("--teacache-thresh", type=float, default=None, dest="teacache_thresh", help="TeaCache relative L1 threshold (default: 0.06)")
    p_res.set_defaults(func=cmd_restore)

    # ladder / two_stage
    p_lad = subparsers.add_parser("ladder", help="2-Stage Native Ladder (Restore ➔ 2x Upscale ➔ 4K Refine)")
    p_lad.add_argument("--input", "-i", required=True, help="Input video path")
    p_lad.add_argument("--output", "-o", default=None, help="Target output folder")
    p_lad.add_argument("--output-file", default=None, help="Specific target output file path")
    p_lad.add_argument("--max-frames", type=int, default=None, help="Cap maximum frames to process (e.g. 65 for preview)")
    p_lad.add_argument("--start-frame", "--in", type=int, default=None, dest="start_frame", help="In Point / Start frame index (0-based)")
    p_lad.add_argument("--end-frame", "--out", type=int, default=None, dest="end_frame", help="Out Point / End frame index (0-based)")
    p_lad.add_argument("--preset", default="archival_general", help="Restoration preset ID")
    p_lad.add_argument("--prompt", default=None, help="Custom positive prompt")
    p_lad.add_argument("--negative", default=None, help="Custom negative prompt")
    p_lad.add_argument("--resolution", default="4K", choices=["FullHD", "4K", "8K"], help="Target output resolution")
    p_lad.add_argument("--tile", default="HD", choices=["qHD", "HD", "FullHD"], help="Sampling tile size")
    p_lad.add_argument("--seed", type=int, default=None, help="RNG seed")
    p_lad.add_argument("--ref", "--reference", dest="reference", default=None, help="Path to character reference image to lock facial likeness and identity")
    p_lad.add_argument("--depth", action="store_true", default=False, help="Extract 3D metric scene depth sequence with Depth Anything V2")
    p_lad.set_defaults(func=cmd_ladder)

    # refine
    p_ref = subparsers.add_parser("refine", help="Refine details of upscaled footage")
    p_ref.add_argument("--input", "-i", required=True, help="Input video path")
    p_ref.add_argument("--output", "-o", default=None, help="Target output folder")
    p_ref.add_argument("--output-file", default=None, help="Specific target output file path")
    p_ref.add_argument("--max-frames", type=int, default=None, help="Cap maximum frames to process (e.g. 65 for preview)")
    p_ref.add_argument("--start-frame", "--in", type=int, default=None, dest="start_frame", help="In Point / Start frame index (0-based)")
    p_ref.add_argument("--end-frame", "--out", type=int, default=None, dest="end_frame", help="Out Point / End frame index (0-based)")
    p_ref.add_argument("--preset", default="refine_upscaled_4k", help="Preset ID")
    p_ref.add_argument("--prompt", default=None, help="Custom positive prompt")
    p_ref.add_argument("--negative", default=None, help="Custom negative prompt")
    p_ref.add_argument("--resolution", default="4K", choices=["FullHD", "4K", "8K"], help="Target output resolution")
    p_ref.add_argument("--tile", default="HD", choices=["qHD", "HD", "FullHD"], help="Sampling tile size")
    p_ref.add_argument("--seed", type=int, default=None, help="RNG seed")
    p_ref.add_argument("--ref", "--reference", dest="reference", default=None, help="Path to character reference image to lock facial likeness and identity")
    p_ref.add_argument("--depth", action="store_true", default=False, help="Extract 3D metric scene depth sequence with Depth Anything V2")
    p_ref.set_defaults(func=cmd_refine)

    # sol (NVIDIA 1-Step SoL-Refiner)
    p_sol = subparsers.add_parser("sol", help="NVIDIA 1-Step SoL-Refiner (~10s)")
    p_sol.add_argument("--input", "-i", required=True, help="Input video path")
    p_sol.add_argument("--output", "-o", default=None, help="Target output folder")
    p_sol.add_argument("--output-file", default=None, help="Specific target output file path")
    p_sol.add_argument("--max-frames", type=int, default=None, help="Cap maximum frames to process (e.g. 65 for preview)")
    p_sol.add_argument("--start-frame", "--in", type=int, default=None, dest="start_frame", help="In Point / Start frame index (0-based)")
    p_sol.add_argument("--end-frame", "--out", type=int, default=None, dest="end_frame", help="Out Point / End frame index (0-based)")
    p_sol.add_argument("--preset", default="archival_general", help="Restoration preset ID")
    p_sol.add_argument("--prompt", default=None, help="Custom positive prompt")
    p_sol.add_argument("--negative", default=None, help="Custom negative prompt")
    p_sol.add_argument("--seed", type=int, default=None, help="RNG seed")
    p_sol.add_argument("--sigma", type=float, default=0.50, help="Detail refinement sigma strength: 0.35 (subtle) to 0.65 (rich detail), default: 0.50")
    p_sol.add_argument("--ref", "--reference", dest="reference", default=None, help="Path to character reference image to lock facial likeness and identity")
    p_sol.add_argument("--depth", action="store_true", default=False, help="Extract 3D metric scene depth sequence with Depth Anything V2")
    p_sol.set_defaults(func=cmd_sol)

    # compare
    p_comp = subparsers.add_parser("compare", help="Create comparison videos")
    p_comp.add_argument("--original", "-o", required=True, help="Original video path")
    p_comp.add_argument("--restored", "-r", required=True, help="Restored video path")
    p_comp.add_argument("--type", choices=["side_by_side", "split", "both"], default="both", help="Comparison layout")
    p_comp.set_defaults(func=cmd_compare)

    # presets
    p_presets = subparsers.add_parser("presets", help="List restoration presets")
    p_presets.set_defaults(func=cmd_presets)

    # batch
    p_batch = subparsers.add_parser("batch", help="Batch process multiple videos or resume queue")
    p_batch.add_argument("--inputs", "-i", nargs="+", default=[], help="Input video paths or wildcard patterns (e.g. inputs/*.mp4)")
    p_batch.add_argument("--from-queue", action="store_true", help="Process pending jobs saved in queue_state.json")
    p_batch.add_argument("--add-only", action="store_true", help="Add jobs to queue_state.json without immediate execution")
    p_batch.add_argument("--output", "-o", default=None, help="Target output folder")
    p_batch.add_argument("--max-frames", type=int, default=None, help="Cap maximum frames per clip")
    p_batch.add_argument("--preset", default=config.DEFAULT_PRESET, help=f"Restoration preset ID (default: {config.DEFAULT_PRESET})")
    p_batch.add_argument("--engine", default=None, choices=["diffusion", "sol_refiner", "superres", "relayvsr"], help="Override engine type")
    p_batch.add_argument("--superres-model", default=None, help="Super-resolution model filename")
    p_batch.add_argument("--prompt", default=None, help="Custom positive prompt override")
    p_batch.add_argument("--negative", default=None, help="Custom negative prompt override")
    p_batch.add_argument("--resolution", default="FullHD", choices=["HD", "FullHD", "4K", "8K"], help="Target output resolution")
    p_batch.add_argument("--tile", default="FullHD", choices=["qHD", "HD", "FullHD"], help="Sampling tile size")
    p_batch.add_argument("--seed", type=int, default=None, help="RNG seed")
    p_batch.add_argument("--guide-strength", type=float, default=None, dest="guide_strength", help="Override LTX guide strength")
    p_batch.add_argument("--cfg", type=float, default=None, help="Override CFG scale")
    p_batch.add_argument("--sigma", type=float, default=None, help="Override SoL-Refiner sigma strength")
    p_batch.add_argument("--ref", "--reference", dest="reference", default=None, help="Path to character reference image")
    p_batch.add_argument("--depth", action="store_true", default=False, help="Extract 3D metric scene depth")
    p_batch.add_argument("--teacache", action="store_true", default=None, help="Enable TeaCache DiT layer acceleration")
    p_batch.add_argument("--no-teacache", action="store_false", dest="teacache", help="Disable TeaCache DiT acceleration")
    p_batch.add_argument("--teacache-thresh", type=float, default=None, dest="teacache_thresh", help="TeaCache threshold")
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
