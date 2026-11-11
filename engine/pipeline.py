"""
The restoration pipeline: one RestoreRequest in, one finished video out.

It resolves the output file, splits long ranges for the ComfyUI engines (engine.chunked), prepares
the source (deinterlace, canvas, frame range) and its audio, then hands over to the engine:
RelayVSR (engine.relayvsr) or a ComfyUI engine (engine.comfy.job).
"""

import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

import config
import outputs
import presets
import user_settings
import video_preprocessor
from engine import chunked, relayvsr, requirements
from engine.comfy import job as comfy_job
from engine.comfy.client import ComfyUIClient
from engine.progress import ProgressCallback, report
from engine.request import RestoreRequest
from errors import SetupRequired


class VideoRestorationPipeline:
    def __init__(self, client: Optional[ComfyUIClient] = None):
        self.client = client or ComfyUIClient()

    def run(self, request: RestoreRequest, progress: ProgressCallback = None) -> Dict[str, Any]:
        input_path = Path(request.input_video_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        engine_type = request.engine_type
        if engine_type == "relayvsr":
            missing = requirements.missing("relayvsr")
            if missing:
                raise SetupRequired("RelayVSR", missing)
        else:
            relayvsr.release_worker()  # hand its ~9.4 GB of VRAM back before ComfyUI work

        final_output_path = self._output_path(request, input_path)

        # Preset defaults for TeaCache when not set explicitly
        preset_cfg = presets.RESTORATION_PRESETS.get(request.prompt_preset, {})
        enable_teacache = request.enable_teacache
        if enable_teacache is None:
            enable_teacache = preset_cfg.get("enable_teacache", False)
        teacache_thresh = request.teacache_thresh
        if teacache_thresh is None or teacache_thresh <= 0:
            teacache_thresh = preset_cfg.get("teacache_thresh", 0.06)

        self._print_banner(request, input_path, final_output_path, enable_teacache, teacache_thresh)

        # Step 1: inspect and prepare the source
        report(progress, 5, 100, "Preprocessing canvas with GPU NVENC "
               f"({'Direct Canvas' if engine_type == 'superres' else '32-multiple, 8n+1 frames'})...")
        print("\n[STEP 1/5] Inspecting & Preprocessing input footage (GPU NVENC)...")
        raw_info = video_preprocessor.get_video_info(input_path)
        total_src_frames = raw_info["total_frames"]
        print(f"  Input: {raw_info['width']}x{raw_info['height']} @ {raw_info['fps']} fps, {total_src_frames} frames")
        print(f"  Interlaced: {raw_info['is_interlaced']} | Has Audio: {raw_info['has_audio']}")

        start_f = max(0, int(request.start_frame)) if request.start_frame is not None else 0
        end_f = min(total_src_frames - 1, int(request.end_frame)) if request.end_frame is not None else total_src_frames - 1
        if request.max_frames and (end_f - start_f + 1) > request.max_frames:
            end_f = start_f + request.max_frames - 1
        span_frames = max(1, end_f - start_f + 1)

        # Longer than one pass can hold on a ComfyUI engine: render in chunks and join them.
        # RelayVSR streams frame by frame with bounded memory and never needs this.
        safe_single_limit = 200 if engine_type == "superres" else 97
        if span_frames > safe_single_limit and engine_type != "relayvsr":
            chunk_step = 200 if engine_type == "superres" else 33
            print("\n>>> [SMART AUTO-CHUNKER ENGAGED]")
            print(f"   Video span is {span_frames} frames. Automatically chunking into {chunk_step}-frame blocks to prevent memory crash.")
            request = dataclasses.replace(request, enable_teacache=enable_teacache, teacache_thresh=teacache_thresh)
            return chunked.run(request, start_f, end_f, chunk_step, final_output_path, self.run, progress)

        prepared_video_path, prep_info, done_segments = self._prepare(request, input_path, raw_info,
                                                                      final_output_path, start_f, end_f)

        # Step 2: the prepared range's audio, for a lossless remux at the end
        audio_path = None
        if raw_info["has_audio"]:
            report(progress, 10, 100, "Extracting audio track for lossless preservation...")
            print("\n[STEP 2/5] Extracting synchronized audio stream from prepared canvas...")
            audio_path = video_preprocessor.extract_audio(
                prepared_video_path, config.INPUTS_DIR / f"{final_output_path.stem}_audio.m4a")
            if audio_path and audio_path.exists():
                print(f"  Extracted audio: {audio_path.name} ({audio_path.stat().st_size / (1024*1024):.2f} MB)")

        if engine_type == "relayvsr":
            seed = request.seed if request.seed is not None else 42
            res = relayvsr.run(self.client, prepared_video_path, prep_info, audio_path, final_output_path,
                               request.output_size, seed, request.auto_compare, progress, done_segments,
                               request.output_codec)
            final_info, comparison_res = res["final_info"], res["comparison"]
            return {
                "status": "completed",
                "models_missing": False,
                "missing_list": [],
                "workflow_file": None,
                "input_file": str(input_path),
                "prepared_input": str(prepared_video_path),
                "final_output_path": str(final_output_path),
                "output_prefix": final_output_path.stem,
                "mode": "relayvsr",
                "engine_type": "relayvsr",
                "superres_model": config.RELAYVSR_MODEL_ID,
                "reference_image": None,
                "depth_output_path": None,
                "preset": request.prompt_preset,
                "enable_teacache": False,
                "teacache_thresh": None,
                "seed": seed,
                "dimensions": f"{final_info['width']}x{final_info['height']}",
                "frames": final_info["total_frames"],
                "has_audio": final_info["has_audio"],
                "audio_file": str(audio_path) if audio_path else None,
                "comparison": comparison_res,
                "comparison_path": comparison_res.get("side_by_side") if comparison_res else None,
            }

        return comfy_job.run(self.client, request, input_path, prepared_video_path, prep_info, raw_info,
                             audio_path, final_output_path, enable_teacache, teacache_thresh, progress)

    # -- steps ----------------------------------------------------------------
    @staticmethod
    def _output_path(request: RestoreRequest, input_path: Path) -> Path:
        """The file to write: request.output_file, else an unused <source>_<tag>.mp4 in output_dir
        (or the remembered output folder). Running a job never changes the remembered folder; only
        the user's own save-dialog choice does."""
        if request.output_file:
            return Path(request.output_file)
        out_folder = Path(request.output_dir) if request.output_dir else user_settings.get_last_output_folder()
        return outputs.suggested_output_path(input_path, {"engine_type": request.engine_type, "mode": request.mode,
                                                          "superres_model": request.superres_model}, out_folder)

    @staticmethod
    def _prepare(request: RestoreRequest, input_path: Path, raw_info: Dict[str, Any], final_output_path: Path,
                 start_f: int, end_f: int):
        """Returns (prepared video, its info, RelayVSR segments already done)."""
        engine_type = request.engine_type
        is_relay = engine_type == "relayvsr"
        # In/Out frames in the name avoid collisions between ranges of the same source
        range_tag = (f"_f{request.start_frame or 0}_{request.end_frame}"
                     if (request.start_frame is not None or request.end_frame is not None) else "")
        prep_target = config.INPUTS_DIR / f"{input_path.stem.replace('_prepared', '')}{range_tag}_prepared.mp4"

        should_clean = request.clean_noise if request.clean_noise is not None else (engine_type == "superres")
        should_stabilize = (request.stabilize_jitter if request.stabilize_jitter is not None
                            else (engine_type == "superres"))
        if is_relay:
            # Upscale-only engine trained on real degraded footage: pre-denoising/deflickering
            # (e.g. defect ticks set by auto-detect) would only strip detail it can recover
            should_clean = should_stabilize = False

        # A stopped RelayVSR job resumes from its finished segments, so its prepared input is reused as-is
        done_segments = relayvsr.resume_state(
            final_output_path, {"input": str(input_path.resolve()), "start": start_f, "end": end_f,
                                "output_size": request.output_size, "segment_frames": relayvsr.SEGMENT_FRAMES},
        ) if is_relay else 0
        if done_segments and prep_target.exists():
            prepared, prep_info = prep_target, video_preprocessor.get_video_info(prep_target)
            print(f"  Resuming: reusing prepared input and {done_segments} finished segment(s)")
        else:
            prepared, prep_info = video_preprocessor.preprocess_video(
                input_path=input_path,
                output_path=prep_target,
                # Deinterlacing progressive footage only softens it; RelayVSR gets it when actually needed
                force_deinterlace=raw_info["is_interlaced"] if is_relay else True,
                clean_noise=should_clean,
                stabilize_jitter=should_stabilize,
                align_frames=engine_type not in ("superres", "relayvsr"),  # diffusion needs 8n+1 frames
                max_frames=request.max_frames,
                start_frame=request.start_frame or 0,
                end_frame=request.end_frame,
            )
        print(f"  Prepared: {prep_info['width']}x{prep_info['height']}")
        print(f"  Frames: {prep_info['total_frames']} (range: F {prep_info.get('start_frame', 0)}..{prep_info.get('end_frame', prep_info['total_frames']-1)})")
        return prepared, prep_info, done_segments

    @staticmethod
    def _print_banner(request: RestoreRequest, input_path: Path, final_output_path: Path,
                      enable_teacache: bool, teacache_thresh: float) -> None:
        engine_type = request.engine_type
        print("\n=======================================================")
        print(f" STARTING FOOTAGE RESTORATION: {input_path.name}")
        print(f" Engine: {engine_type.upper()} ({request.superres_model if engine_type == 'superres' else request.mode})")
        print(f" Target Output: {final_output_path.name}")
        print(f" Destination: {final_output_path.parent}")
        print(f" Output Resolution: {request.output_size} | Tile: {request.tile_size}")
        if engine_type == "sol_refiner":
            print(f" SoL Refinement Strength (Sigma): {request.sol_sigma}")
        elif engine_type == "relayvsr":
            print(f" RelayVSR Streaming 4x (keyframe window 16, low-res canvas <= {config.RELAYVSR_MAX_LR_HEIGHT}p)")
        elif engine_type != "superres":
            print(f" Guide Strength: {request.guide_strength} | CFG: {request.cfg}")
            if enable_teacache:
                print(f" TeaCache Acceleration: ENABLED (~2x faster, threshold={teacache_thresh})")
        if request.start_frame is not None or request.end_frame is not None:
            print(f" In/Out Range: Frame {request.start_frame or 0} -> {request.end_frame or 'End'}")
        print("=======================================================")

