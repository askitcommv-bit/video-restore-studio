"""
One job on a ComfyUI engine (super-resolution, SoL-Refiner, LTX-2.5): build the prompt graph for the
prepared video, run it in ComfyUI, and deliver the result with audio.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import presets
import video_preprocessor
from engine import output, requirements
from engine.comfy import graphs
from engine.comfy.client import ComfyUIClient, find_rendered_output
from engine.progress import ProgressCallback, report
from engine.request import RestoreRequest
from errors import ComfyUIOffline


def superres_target_size(raw_info: Dict[str, Any], output_size: str) -> tuple:
    """(width, height) for super-resolution's resize to the output tier, even-sized; (None, None) for none."""
    base = {"HD": 720, "FullHD": 1080, "4K": 2160, "8K": 4320}.get(output_size)
    if base is None:
        return None, None
    aspect = raw_info["width"] / raw_info["height"] if raw_info["height"] > 0 else 16 / 9
    if raw_info["width"] >= raw_info["height"]:
        w, h = int(round(base * aspect)), base
    else:
        w, h = base, int(round(base / aspect))
    return w - (w % 2), h - (h % 2)


def _prepared_only(request: RestoreRequest, input_path: Path, prepared_video_path: Path, prep_info: Dict[str, Any],
                   raw_info: Dict[str, Any], audio_path: Optional[Path], final_output_path: Path,
                   missing_models: List[str], mode: str, progress: ProgressCallback,
                   depth_path: Optional[Path] = None) -> Dict[str, Any]:
    """Models aren't installed: deliver the preprocessed canvas instead, saying so."""
    print(f"\n[NOTICE] Neural model weights not yet installed: {missing_models}")
    print(f"  Delivering GPU NVENC preprocessed video to: {final_output_path}")
    report(progress, 85, 100, "Saving GPU-preprocessed video to destination...")
    shutil.copyfile(prepared_video_path, final_output_path)
    report(progress, 100, 100, "Done (Weights missing; preprocessed video saved)")
    res = {
        "status": "prepared_only",
        "models_missing": True,
        "missing_list": missing_models,
        "input_file": str(input_path),
        "prepared_input": str(prepared_video_path),
        "final_output_path": str(final_output_path),
        "output_prefix": final_output_path.stem,
        "mode": mode,
        "preset": request.prompt_preset,
        "seed": 42,
        "dimensions": f"{prep_info['width']}x{prep_info['height']}",
        "frames": prep_info["total_frames"],
        "has_audio": raw_info["has_audio"],
        "audio_file": str(audio_path) if audio_path else None,
    }
    if request.engine_type != "sol_refiner":
        res["reference_image"] = str(request.reference_image) if request.reference_image else None
        res["depth_output_path"] = str(depth_path) if depth_path else None
    return res


def run(client: ComfyUIClient, request: RestoreRequest, input_path: Path, prepared_video_path: Path,
        prep_info: Dict[str, Any], raw_info: Dict[str, Any], audio_path: Optional[Path], final_output_path: Path,
        enable_teacache: bool, teacache_thresh: float, progress: ProgressCallback) -> Dict[str, Any]:
    engine_type, mode = request.engine_type, request.mode

    # 3D scene depth (Depth Anything V2), if requested
    depth_path = None
    if request.extract_depth:
        report(progress, 12, 100, "Extracting 3D scene depth geometry (Depth Anything V2)...")
        print("\n[3D GEOMETRY] Extracting frame-by-frame 3D scene depth geometry via Depth Anything V2...")
        try:
            depth_path = video_preprocessor.extract_depth_sequence(
                prepared_video_path, output_path=config.INPUTS_DIR / f"{prepared_video_path.stem}_depth.mp4",
                progress_callback=progress)
            print(f"  Extracted 3D metric depth sequence: {depth_path.name}")
        except Exception as depth_err:
            print(f"  [Depth Estimation Notice] Depth extraction skipped or errored: {depth_err}")

    # Character reference portrait, if given
    uploaded_ref = None
    if request.reference_image:
        ref_path = Path(request.reference_image)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        print(f"\n[CHARACTER GUIDANCE] Preparing Character Reference Portrait: {ref_path.name}")
        report(progress, 14, 100, f"Aligning & uploading reference image ({ref_path.name})...")
        uploaded_ref = client.upload_video(video_preprocessor.prepare_reference_image(ref_path))
        print(f"  Reference image prepared (32-multiple aligned) and uploaded: {uploaded_ref}")

    report(progress, 15, 100, "Verifying model weights...")
    output_prefix = final_output_path.stem
    actual_seed = request.seed if request.seed is not None else int(time.time() * 1000) % 1000000000
    preset = presets.PROMPT_PRESETS.get(request.prompt_preset, presets.PROMPT_PRESETS["archival_general"])
    pos_prompt = request.custom_positive_prompt or preset["positive"]
    neg_prompt = request.custom_negative_prompt or preset["negative"]

    if engine_type == "superres":
        model_file = config.UPSCALE_MODELS_DIR / request.superres_model
        if not model_file.exists():
            raise FileNotFoundError(f"Super-resolution model not found: {model_file}")
        report(progress, 20, 100, f"Uploading video & configuring {request.superres_model}...")
        print(f"\n[STEP 3/5] Uploading video to ComfyUI input storage ({request.superres_model})...")
        target_w, target_h = superres_target_size(raw_info, request.output_size)
        prompt_graph = graphs.build_superres_api_prompt(
            video_filename=client.upload_video(prepared_video_path), output_prefix=output_prefix,
            upscale_model_name=request.superres_model, target_width=target_w, target_height=target_h)
    else:
        missing_models = requirements.missing_model_keys(engine_type, mode)
        if missing_models:
            return _prepared_only(request, input_path, prepared_video_path, prep_info, raw_info, audio_path,
                                  final_output_path, missing_models,
                                  "sol_refiner" if engine_type == "sol_refiner" else mode, progress, depth_path)
        if engine_type == "sol_refiner":
            report(progress, 20, 100, "Uploading video & building NVIDIA 1-Step SoL-Refiner prompt...")
            print("\n[STEP 3/5] Uploading video to ComfyUI input storage (SoL-Refiner)...")
            prompt_graph = graphs.build_sol_refiner_api_prompt(
                video_filename=client.upload_video(prepared_video_path), output_prefix=output_prefix,
                seed=actual_seed, pos_prompt=pos_prompt, neg_prompt=neg_prompt, sigma=request.sol_sigma,
                ref_image_filename=uploaded_ref)
        else:
            report(progress, 20, 100, "Uploading video & building ComfyUI prompt...")
            print("\n[STEP 3/5] Uploading video to ComfyUI input storage...")
            prompt_graph = graphs.build_restoration_api_prompt(
                video_filename=client.upload_video(prepared_video_path), output_prefix=output_prefix, mode=mode,
                tile_size=request.tile_size, output_size=request.output_size, seed=actual_seed,
                pos_prompt=pos_prompt, neg_prompt=neg_prompt, guide_strength=request.guide_strength,
                cfg=request.cfg, ref_image_filename=uploaded_ref, enable_teacache=enable_teacache,
                teacache_thresh=teacache_thresh)

    config.WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
    run_wf_path = config.WORKFLOWS_DIR / f"active_run_{output_prefix}.json"
    run_wf_path.write_text(json.dumps(prompt_graph, indent=2), encoding="utf-8")
    print(f"  Saved runnable API prompt: {run_wf_path}")

    if not client.is_online():
        raise ComfyUIOffline(client.base_url)
    report(progress, 25, 100, "Queueing neural restoration on RTX GPU...")
    print(f"\n[STEP 4/5] Queueing restoration job to ComfyUI (client {client.client_id})...")
    prompt_id = client.queue_prompt(prompt_graph)
    print(f"  Queued prompt ID: {prompt_id}")
    report(progress, 30, 100, "Initializing neural pipeline on GPU...")
    client.track_execution(prompt_id=prompt_id, progress_callback=progress)

    report(progress, 92, 100, "Muxing lossless audio and finalizing output...")
    print("\n[STEP 5/5] Retrieving generated video and finalizing audio...")
    rendered = find_rendered_output(output_prefix)
    print(f"  Retrieved AI rendered video: {rendered.name} ({rendered.stat().st_size / (1024*1024):.2f} MB)")
    final_info, comparison_res = output.deliver(
        rendered, audio_path, final_output_path, prepared_video_path, request.auto_compare, progress,
        keep_source=True, done_text="Restoration Complete!", output_codec=request.output_codec)

    return {
        "status": "completed",
        "models_missing": False,
        "missing_list": [],
        "workflow_file": str(run_wf_path),
        "input_file": str(input_path),
        "prepared_input": str(prepared_video_path),
        "final_output_path": str(final_output_path),
        "output_prefix": output_prefix,
        "mode": mode,
        "engine_type": engine_type,
        "superres_model": request.superres_model if engine_type == "superres" else None,
        "reference_image": str(request.reference_image) if request.reference_image else None,
        "depth_output_path": str(depth_path) if depth_path else None,
        "preset": request.prompt_preset,
        "enable_teacache": enable_teacache,
        "teacache_thresh": teacache_thresh,
        "seed": actual_seed,
        "dimensions": f"{final_info['width']}x{final_info['height']}",
        "frames": final_info["total_frames"],
        "has_audio": final_info["has_audio"],
        "audio_file": str(audio_path) if audio_path else None,
        "comparison": comparison_res,
        "comparison_path": comparison_res.get("side_by_side") if comparison_res else None,
    }
