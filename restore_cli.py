"""
Command Line Interface for LTX-2.5 Video Restoration Suite.
Usage:
  python restore_cli.py status
  python restore_cli.py download [--bf16] [--loras-only] [--sol] [--relayvsr] [--token hf_...]
  python restore_cli.py token [hf_...]
  python restore_cli.py preprocess --input "clip.mp4"
  python restore_cli.py restore --input "clip.mp4" [--preset archival_general] [--resolution 4K]
  python restore_cli.py refine --input "clip.mp4" [--resolution 4K]
  python restore_cli.py compare --original "clip.mp4" --restored "restored.mp4"
  python restore_cli.py presets
"""

import argparse
import dataclasses
import glob
import time
from pathlib import Path

import bootstrap
import comfy_models
import comparison_generator
import config
import outputs
import presets
import setup_manager
import user_settings
import video_preprocessor
from engine import ComfyUIClient, RestoreRequest, VideoRestorationPipeline
from errors import DownloadForbidden


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
    print_model_status()


def print_model_status():
    """Installed vs missing: what RelayVSR needs, and the ComfyUI engines' models."""
    print("=" * 80)
    print(" MODEL STATUS")
    print("=" * 80)
    for c in setup_manager.components():
        if not c.optional:
            print(f"{c.name:<28} | {'[INSTALLED]' if c.ready else '[MISSING]':<12} | {c.download_gb:6.2f} GB")
    print("-" * 80)
    missing_gb = 0.0
    for key, m in comfy_models.MODELS.items():
        path = comfy_models.local_path(key)
        if path is None:
            missing_gb += m.size_gb
        print(f"{key:<28} | {'[INSTALLED]' if path else '[MISSING]':<12} | {m.size_gb:6.2f} GB | {(path or m.folder / m.filename).name}")
    print("-" * 80)
    print(" ComfyUI models: all installed." if not missing_gb else f" ComfyUI models still to download: ~{missing_gb:.2f} GB")
    tok = comfy_models.get_hf_token()
    print(f" Hugging Face token: {'set' if tok else 'not set (needed for the gated Lightricks/LTX-2.5 models)'}")
    print("=" * 80)


def cmd_download(args):
    """Downloads what an engine needs: RelayVSR (the in-app setup's steps) or the ComfyUI models."""
    if args.token:
        comfy_models.set_hf_token(args.token)
    if args.relayvsr:
        print("\nInstalling RelayVSR (FFmpeg if missing, code, isolated CUDA venv, ~10.2 GB weights)...")
        done = setup_manager.install(setup_manager.print_progress)
        print(f"\n[RelayVSR] Installed: {', '.join(done) or 'nothing (already set up)'}. "
              f"Weights: {config.RELAYVSR_WEIGHTS_DIR}")
        return
    keys = comfy_models.SOL_PACK if args.sol else comfy_models.LORA_PACK if args.loras_only else comfy_models.LTX_PACK
    prec = "bf16" if args.bf16 else "int8"
    print(f"\nDownloading {len(comfy_models.missing(keys))} missing model(s) (precision: {prec.upper()})...")
    try:
        done = comfy_models.download_many(keys, setup_manager.print_progress, precision=prec)
    except DownloadForbidden as e:
        print(f"\n[!] {e}")
        return
    print(f"\n[DONE] Downloaded: {', '.join(done) or 'nothing (already installed)'}")


def cmd_token(args):
    """Saves the Hugging Face access token the gated LTX-2.5 models need."""
    token = args.token or input("Enter your Hugging Face token (hf_...): ").strip()
    if token:
        comfy_models.set_hf_token(token)
        print("[OK] Hugging Face token saved.")
    else:
        print("[!] Cancelled.")


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
    preset_key = getattr(args, "preset", presets.DEFAULT_PRESET) or presets.DEFAULT_PRESET
    p = presets.get_preset(preset_key)

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
    res = pipeline.run(RestoreRequest(
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
        output_codec=getattr(args, "codec", "auto"),
    ), make_cli_progress())
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
    res = pipeline.run(RestoreRequest(
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
        output_codec=getattr(args, "codec", "auto"),
    ), make_cli_progress())
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
    res = pipeline.run(RestoreRequest(
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
        output_codec=getattr(args, "codec", "auto"),
    ), make_cli_progress())
    print("\n[SUCCESS] Detail Refinement job configured!")
    print(f"  Configured Workflow: {res['workflow_file']}")
    if res.get("reference_image"):
        print(f"  Reference Portrait: {res['reference_image']}")
    if res.get("depth_output_path"):
        print(f"  3D Depth Sequence: {res['depth_output_path']}")


def cmd_sol(args):
    """Executes NVIDIA 1-Step SoL-Refiner restoration pipeline (~10s)."""
    pipeline = VideoRestorationPipeline()
    res = pipeline.run(RestoreRequest(
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
        output_codec=getattr(args, "codec", "auto"),
    ), make_cli_progress())
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
    for key, p in presets.RESTORATION_PRESETS.items():
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
                request = dataclasses.replace(RestoreRequest.from_params(job.params), output_file=job.output_path,
                                              prompt_preset=job.preset_id, mode=job.mode, engine_type=job.engine_type)
                res = pipeline.run(request, make_cli_progress())
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

    preset_key = getattr(args, "preset", presets.DEFAULT_PRESET) or presets.DEFAULT_PRESET
    p = presets.get_preset(preset_key)
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
    out_dir = Path(args.output) if getattr(args, "output", None) else user_settings.get_last_output_folder()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, fpath in enumerate(unique_files, 1):
        fp = Path(fpath)
        out_file = outputs.suggested_output_path(fp, base_params, out_dir)

        print(f"\n-------------------------------------------------------")
        print(f" [BATCH {idx}/{len(unique_files)}] {fp.name}")
        print(f" Target Output: {out_file.name}")
        print(f"-------------------------------------------------------")

        try:
            res = pipeline.run(RestoreRequest(
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
            ), make_cli_progress())
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
    p_dl.add_argument("--sol", action="store_true", help="Download the NVIDIA SoL-Refiner models")
    p_dl.add_argument("--token", default=None, help="Save this Hugging Face token first (gated LTX-2.5 repo)")
    p_dl.set_defaults(func=cmd_download)

    # token
    p_tok = subparsers.add_parser("token", help="Save the Hugging Face access token")
    p_tok.add_argument("token", nargs="?", default=None, help="The token (prompted for when omitted)")
    p_tok.set_defaults(func=cmd_token)

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
    p_res.add_argument("--preset", default=presets.DEFAULT_PRESET, help=f"Restoration preset ID (default: {presets.DEFAULT_PRESET})")
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
    p_batch.add_argument("--preset", default=presets.DEFAULT_PRESET, help=f"Restoration preset ID (default: {presets.DEFAULT_PRESET})")
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
    bootstrap.init()
    args.func(args)


if __name__ == "__main__":
    main()
