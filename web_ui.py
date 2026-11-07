"""
Web UI Dashboard for LTX-2.5 Video Restoration Suite.
Provides:
- 1-Click Simplified Presets (Master LTX 8-Step, Turbo TeaCache, NVIDIA SoL-Refiner, Real-ESRGAN, UltraSharp)
- Interactive In / Out Range Marks Selector with live video scrubbing & legal 8n+1 frame snapping
- Thread-safe Asynchronous Job Queue System with real-time progress, live logs & cancellation
- Direct Video Library browser for existing footage in inputs/
- Synchronized Before/After Comparison Video Player
"""

import sys
import os
import json
import time
import shutil
import uuid
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import model_downloader
import video_preprocessor
import comparison_generator
from restore_engine import VideoRestorationPipeline, ComfyUIClient

PORT = int(os.environ.get("PORT", 7860))


class JobQueueManager:
    """Thread-safe background queue manager for video restoration jobs."""

    def __init__(self):
        self.queue: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.active_job: Optional[Dict[str, Any]] = None
        self.lock = threading.Lock()
        self.stop_requested = False
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "active": self.active_job,
                "queue": list(self.queue),
                "history": list(self.history)
            }

    def add_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            input_path = payload.get("input_path")
            if not input_path or not Path(input_path).exists():
                raise FileNotFoundError(f"Input video file not found: {input_path}")

            job_id = f"job_{int(time.time()*1000)}_{str(uuid.uuid4())[:6]}"
            preset_key = payload.get("preset", config.DEFAULT_PRESET)
            preset_info = config.RESTORATION_PRESETS.get(preset_key, {})

            # In/Out Frame Resolution & Safety Clamping
            raw_meta = video_preprocessor.get_video_info(Path(input_path))
            total_src_frames = raw_meta.get("total_frames", 0)

            engine_type = payload.get("engine_type") or preset_info.get("engine_type", "diffusion")
            start_frame = payload.get("start_frame")
            end_frame = payload.get("end_frame")

            if start_frame is not None:
                start_frame = max(0, int(start_frame))
            else:
                start_frame = 0

            if end_frame is not None:
                end_frame = min(total_src_frames - 1, int(end_frame))
            else:
                end_frame = total_src_frames - 1

            frame_count = max(1, end_frame - start_frame + 1)
            is_chunked = engine_type != "relayvsr" and frame_count > (200 if engine_type == "superres" else 97)

            job = {
                "id": job_id,
                "status": "queued",
                "progress": 0,
                "step": "Queued in studio backlog",
                "input_path": str(input_path),
                "filename": Path(input_path).name,
                "preset": preset_key,
                "preset_name": preset_info.get("name", preset_key),
                "preset_badge": preset_info.get("badge", ""),
                "engine_type": engine_type,
                "mode": payload.get("mode") or preset_info.get("mode", "restore"),
                "output_size": payload.get("resolution") or preset_info.get("output_size", "FullHD"),
                "tile_size": payload.get("tile") or preset_info.get("tile_size", "FullHD"),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "total_src_frames": total_src_frames,
                "auto_chunked": is_chunked,
                "custom_prompt": payload.get("prompt"),
                "custom_negative": payload.get("negative"),
                "seed": payload.get("seed"),
                "sol_sigma": float(payload.get("sol_sigma", preset_info.get("sol_sigma", 0.50))),
                "superres_model": payload.get("superres_model") or preset_info.get("superres_model", "RealESRGAN_x2plus.pth"),
                "reference_image": payload.get("reference_image"),
                "extract_depth": bool(payload.get("extract_depth", False)),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "completed_at": None,
                "output_video": None,
                "comparison_video": None,
                "error": None
            }

            self.queue.append(job)
            return job

    def cancel_job(self, job_id: str) -> bool:
        with self.lock:
            # Check if active
            if self.active_job and self.active_job.get("id") == job_id:
                self.active_job["status"] = "cancelled"
                self.active_job["step"] = "Cancelled by user"
                # Signal ComfyUI to interrupt
                try:
                    ComfyUIClient().interrupt()
                    VideoRestorationPipeline.terminate_active_worker()
                except Exception:
                    pass
                return True

            # Check if pending in queue
            for idx, j in enumerate(self.queue):
                if j.get("id") == job_id:
                    removed = self.queue.pop(idx)
                    removed["status"] = "cancelled"
                    removed["step"] = "Cancelled before execution"
                    self.history.insert(0, removed)
                    return True
        return False

    def clear_history(self):
        with self.lock:
            self.history.clear()

    def _worker_loop(self):
        while not self.stop_requested:
            job_to_run = None
            with self.lock:
                if not self.active_job and len(self.queue) > 0:
                    job_to_run = self.queue.pop(0)
                    job_to_run["status"] = "running"
                    job_to_run["step"] = "Initializing restoration pipeline..."
                    job_to_run["progress"] = 2
                    self.active_job = job_to_run

            if job_to_run:
                self._execute_job(job_to_run)
            else:
                # Queue drained: free the resident RelayVSR worker's VRAM (no-op when none is loaded)
                VideoRestorationPipeline.release_relayvsr_worker()
                time.sleep(1.0)

    def _execute_job(self, job: Dict[str, Any]):
        pipeline = VideoRestorationPipeline()

        def on_progress(pct: int, total: int, desc: str):
            with self.lock:
                if self.active_job and self.active_job.get("id") == job.get("id"):
                    self.active_job["progress"] = pct
                    self.active_job["step"] = desc

        try:
            res = pipeline.restore_footage(
                input_video_path=job["input_path"],
                engine_type=job["engine_type"],
                mode=job["mode"],
                prompt_preset=job["preset"],
                custom_positive_prompt=job.get("custom_prompt"),
                custom_negative_prompt=job.get("custom_negative"),
                output_size=job["output_size"],
                tile_size=job["tile_size"],
                start_frame=job["start_frame"],
                end_frame=job["end_frame"],
                seed=job.get("seed"),
                sol_sigma=job.get("sol_sigma", 0.50),
                superres_model=job.get("superres_model", "RealESRGAN_x2plus.pth"),
                reference_image=job.get("reference_image"),
                extract_depth=job.get("extract_depth", False),
                progress_callback=on_progress
            )

            with self.lock:
                job["status"] = "completed"
                job["progress"] = 100
                job["step"] = "Restoration completed successfully!"
                job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                job["output_video"] = str(res.get("final_output_path")) if res.get("final_output_path") else None
                job["comparison_video"] = str(res.get("comparison_path")) if res.get("comparison_path") else None
                self.history.insert(0, job)
                self.active_job = None

        except Exception as e:
            err_msg = str(e)
            with self.lock:
                job["status"] = "failed"
                job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

                # Parse common failure reasons into actionable human advice
                if "ArrayMemoryError" in err_msg or "Unable to allocate" in err_msg:
                    friendly = (
                        "System RAM Out of Memory: The video had too many frames for full diffusion redraw. "
                        "Fix: Use the In/Out marks selector to pick a shot (17 to 97 frames), or switch preset to 'Instant Super-Res (Real-ESRGAN)' to upscale the full episode."
                    )
                elif "OutOfMemoryError" in err_msg or "CUDA out of memory" in err_msg:
                    friendly = (
                        "GPU VRAM Out of Memory: Try lowering Tiling Stencil to 'HD' or 'qHD' in pipeline settings."
                    )
                elif "offline" in err_msg.lower() or "connection" in err_msg.lower():
                    friendly = "ComfyUI Server is offline. Ensure ComfyUI background process is active on Port 8188."
                else:
                    friendly = err_msg

                job["error"] = friendly
                job["step"] = f"Failed: {friendly}"
                self.history.insert(0, job)
                self.active_job = None


# Global Queue Manager Instance
queue_manager = JobQueueManager()


class RestorationHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the dashboard and REST API."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.replace("__DEFAULT_PRESET__", config.DEFAULT_PRESET).encode("utf-8"))
            return

        elif path == "/api/status":
            self.send_json_response(self.get_system_status())
            return

        elif path == "/api/presets":
            self.send_json_response(config.RESTORATION_PRESETS)
            return

        elif path == "/api/inputs":
            inputs_list = []
            for f in sorted(config.INPUTS_DIR.glob("*.mp4")):
                try:
                    info = video_preprocessor.get_video_info(f)
                    inputs_list.append({
                        "filename": f.name,
                        "path": str(f),
                        "width": info.get("width", 0),
                        "height": info.get("height", 0),
                        "fps": info.get("fps", 0),
                        "total_frames": info.get("total_frames", 0),
                        "duration": info.get("duration", 0),
                        "has_audio": info.get("has_audio", False),
                        "size_mb": info.get("size_mb", 0),
                        "aspect_ratio_str": info.get("aspect_ratio_str", "16:9"),
                        "url": f"/files/inputs/{f.name}"
                    })
                except Exception:
                    pass
            self.send_json_response(inputs_list)
            return

        elif path == "/api/queue":
            self.send_json_response(queue_manager.get_state())
            return

        elif path == "/api/outputs":
            outputs = []
            for f in sorted(config.OUTPUTS_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True):
                outputs.append({
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "url": f"/files/outputs/{f.name}"
                })
            comparisons = []
            for f in sorted(config.COMPARISONS_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True):
                comparisons.append({
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "url": f"/files/comparisons/{f.name}"
                })
            self.send_json_response({"outputs": outputs, "comparisons": comparisons})
            return

        elif path.startswith("/files/outputs/"):
            fname = urllib.parse.unquote(path.replace("/files/outputs/", ""))
            file_path = config.OUTPUTS_DIR / fname
            self.serve_file(file_path, "video/mp4")
            return

        elif path.startswith("/files/comparisons/"):
            fname = urllib.parse.unquote(path.replace("/files/comparisons/", ""))
            file_path = config.COMPARISONS_DIR / fname
            self.serve_file(file_path, "video/mp4")
            return

        elif path.startswith("/files/inputs/"):
            fname = urllib.parse.unquote(path.replace("/files/inputs/", ""))
            file_path = config.INPUTS_DIR / fname
            self.serve_file(file_path, "video/mp4")
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/upload":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type:
                boundary = content_type.split("boundary=")[1].encode("utf-8")
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)

                parts = body.split(boundary)
                for part in parts:
                    if b'filename="' in part:
                        headers_part, file_data = part.split(b"\r\n\r\n", 1)
                        file_data = file_data.rstrip(b"\r\n--")

                        fname = "uploaded_clip.mp4"
                        for line in headers_part.decode("utf-8", errors="ignore").split("\r\n"):
                            if "filename=" in line:
                                fname = line.split('filename="')[1].split('"')[0]

                        save_path = config.INPUTS_DIR / fname
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(save_path, "wb") as f:
                            f.write(file_data)

                        meta = video_preprocessor.get_video_info(save_path)
                        self.send_json_response({
                            "success": True,
                            "filename": fname,
                            "path": str(save_path),
                            "url": f"/files/inputs/{fname}",
                            "metadata": meta
                        })
                        return

            self.send_json_response({"success": False, "error": "Invalid upload"}, status=400)
            return

        elif path == "/api/queue/add":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            try:
                job = queue_manager.add_job(payload)
                self.send_json_response({"success": True, "job": job})
            except Exception as e:
                self.send_json_response({"success": False, "error": str(e)}, status=400)
            return

        elif path == "/api/queue/cancel":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            job_id = payload.get("job_id")
            success = queue_manager.cancel_job(job_id)
            self.send_json_response({"success": success})
            return

        elif path == "/api/queue/clear":
            queue_manager.clear_history()
            self.send_json_response({"success": True})
            return

        elif path == "/api/restore":
            # Direct compatibility endpoint that wraps through queue
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            try:
                job = queue_manager.add_job(payload)
                self.send_json_response({"success": True, "job": job})
            except Exception as e:
                self.send_json_response({"success": False, "error": str(e)}, status=500)
            return

        super().do_POST()

    def send_json_response(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode("utf-8"))

    def serve_file(self, file_path: Path, mime_type: str):
        if not file_path.exists():
            self.send_error(404, "File not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(file_path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def get_system_status(self) -> Dict[str, Any]:
        client = ComfyUIClient()
        online = client.is_online()
        stats = {}
        if online:
            try:
                stats = client.get_system_stats()
            except Exception:
                pass
        models = model_downloader.get_model_status()
        q_state = queue_manager.get_state()
        return {
            "server_online": online,
            "server_url": client.base_url,
            "stats": stats,
            "models": models,
            "active_job": q_state.get("active"),
            "queue_len": len(q_state.get("queue", []))
        }


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LTX-2.5 Video Restoration & Detail Studio</title>
    <style>
        :root {
            --bg-primary: #090d16;
            --bg-secondary: #121824;
            --bg-card: #1b2234;
            --bg-hover: #222b40;
            --text-main: #f0f6fc;
            --text-muted: #8b949e;
            --accent: #38bdf8;
            --accent-green: #22c55e;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --accent-purple: #a855f7;
            --border: #283347;
            --radius-md: 8px;
            --radius-lg: 12px;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg-primary); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }

        /* Left Navigation / Sidebar */
        .sidebar { width: 320px; background: var(--bg-secondary); border-right: 1px solid var(--border); display: flex; flex-direction: column; }
        .sidebar-header { padding: 18px 20px; border-bottom: 1px solid var(--border); }
        .sidebar-header h1 { font-size: 17px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
        .badge { font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.5px; }
        .badge-purple { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
        .badge-green { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
        .badge-amber { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
        .badge-blue { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }

        .status-badge { display: flex; align-items: center; gap: 7px; font-size: 12px; margin-top: 8px; color: var(--text-muted); }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-red); }
        .dot.online { background: var(--accent-green); box-shadow: 0 0 10px var(--accent-green); }

        .sidebar-scroll { flex: 1; overflow-y: auto; padding: 16px 20px; }
        .section-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted); margin-bottom: 10px; font-weight: 700; }
        
        .lib-item { background: var(--bg-card); padding: 10px 12px; border-radius: var(--radius-md); font-size: 12px; border: 1px solid var(--border); margin-bottom: 8px; cursor: pointer; transition: 0.15s; }
        .lib-item:hover { background: var(--bg-hover); border-color: var(--accent); }
        .lib-item.active { border-color: var(--accent); background: rgba(56, 189, 248, 0.1); }
        .lib-name { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .lib-meta { font-size: 11px; color: var(--text-muted); margin-top: 4px; display: flex; justify-content: space-between; }

        /* Main Workspace */
        .workspace { flex: 1; display: flex; flex-direction: column; overflow-y: auto; }
        .topbar { height: 56px; border-bottom: 1px solid var(--border); background: var(--bg-secondary); display: flex; align-items: center; justify-content: space-between; padding: 0 24px; position: sticky; top: 0; z-index: 10; }
        .topbar h2 { font-size: 15px; font-weight: 600; }
        
        .content { padding: 20px 24px; max-width: 1400px; margin: 0 auto; width: 100%; display: grid; grid-template-columns: 1fr 420px; gap: 20px; }
        .card { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 20px; margin-bottom: 20px; }
        .card h3 { font-size: 14px; font-weight: 700; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; }

        /* Video Player & In/Out Scrubber */
        .video-player-container { background: #000; border-radius: var(--radius-md); overflow: hidden; border: 1px solid var(--border); margin-bottom: 14px; position: relative; }
        .main-video { width: 100%; max-height: 340px; display: block; object-fit: contain; }
        
        .in-out-panel { background: var(--bg-card); border-radius: var(--radius-md); padding: 14px; border: 1px solid var(--border); }
        .scrub-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
        .scrub-slider { flex: 1; accent-color: var(--accent); cursor: pointer; }
        .frame-counter { font-family: monospace; font-size: 13px; font-weight: 600; color: var(--accent); min-width: 130px; text-align: right; }
        
        .btn-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
        .btn-step { background: var(--bg-secondary); color: var(--text-main); border: 1px solid var(--border); padding: 6px 10px; border-radius: 6px; font-size: 12px; cursor: pointer; transition: 0.15s; font-weight: 500; }
        .btn-step:hover { background: var(--bg-hover); border-color: var(--accent); }
        .btn-marker { background: rgba(56, 189, 248, 0.15); color: var(--accent); border: 1px solid var(--accent); font-weight: 600; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; }
        .btn-marker:hover { background: rgba(56, 189, 248, 0.25); }

        .range-pill { font-size: 12px; padding: 6px 12px; background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 6px; color: #4ade80; display: inline-flex; align-items: center; gap: 8px; font-weight: 500; }
        .alert-box { padding: 12px 14px; border-radius: var(--radius-md); font-size: 12px; line-height: 1.5; margin-top: 10px; display: flex; gap: 10px; align-items: flex-start; }
        .alert-amber { background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); color: #fde68a; }

        /* Presets Grid */
        .preset-grid { display: grid; grid-template-columns: 1fr; gap: 10px; }
        .preset-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 12px 14px; cursor: pointer; transition: 0.15s; position: relative; }
        .preset-card:hover { border-color: var(--accent); background: var(--bg-hover); }
        .preset-card.active { border-color: var(--accent); background: rgba(56, 189, 248, 0.08); box-shadow: 0 0 0 1px var(--accent); }
        .preset-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
        .preset-title { font-size: 13px; font-weight: 600; color: #fff; }
        .preset-desc { font-size: 11px; color: var(--text-muted); line-height: 1.4; }
        .preset-footer { display: flex; gap: 8px; font-size: 11px; color: var(--accent); margin-top: 6px; }

        /* Queue & Status Column */
        .btn-add-queue { width: 100%; padding: 14px; background: var(--accent-green); color: #000; border: none; border-radius: var(--radius-md); font-size: 14px; font-weight: 700; cursor: pointer; transition: 0.2s; display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 16px; }
        .btn-add-queue:hover { background: #16a34a; color: #fff; }
        .btn-add-queue:disabled { opacity: 0.5; cursor: not-allowed; }

        .active-job-card { background: var(--bg-card); border: 1px solid var(--accent); border-radius: var(--radius-md); padding: 16px; margin-bottom: 16px; position: relative; }
        .progress-bar-wrap { height: 8px; background: var(--bg-secondary); border-radius: 4px; overflow: hidden; margin: 12px 0 8px 0; }
        .progress-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-green)); width: 0%; transition: width 0.3s; }
        .job-step-text { font-size: 12px; color: var(--accent); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

        .btn-cancel { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 6px 12px; border-radius: 6px; font-size: 11px; font-weight: 600; cursor: pointer; }
        .btn-cancel:hover { background: rgba(239, 68, 68, 0.3); }

        .queue-table { width: 100%; border-collapse: collapse; font-size: 12px; }
        .queue-table th { text-align: left; padding: 8px; color: var(--text-muted); font-size: 11px; border-bottom: 1px solid var(--border); }
        .queue-table td { padding: 10px 8px; border-bottom: 1px solid var(--border); vertical-align: middle; }

        /* Output & History */
        .output-box { background: var(--bg-card); border-radius: var(--radius-md); padding: 12px; margin-bottom: 12px; border: 1px solid var(--border); }
        .output-box video { width: 100%; max-height: 220px; background: #000; border-radius: 6px; margin-top: 8px; }
    </style>
</head>
<body>
    <!-- Sidebar: Video Library -->
    <div class="sidebar">
        <div class="sidebar-header">
            <h1><span>🎬</span> Video Restore Studio</h1>
            <div class="status-badge">
                <span class="dot" id="server-dot"></span>
                <span id="server-status">Checking ComfyUI...</span>
            </div>
            <div style="font-size: 11px; margin-top: 4px; color: var(--text-muted);" id="vram-info"></div>
        </div>

        <div class="sidebar-scroll">
            <div class="section-title">Footage In Studio (inputs/)</div>
            <div id="video-library-list">
                <!-- Dynamically populated -->
            </div>

            <div style="margin-top: 16px;">
                <button class="btn-step" style="width: 100%; padding: 10px;" onclick="document.getElementById('file-upload').click()">
                    ➕ Upload New Video Clip
                </button>
                <input type="file" id="file-upload" style="display: none;" accept="video/*" onchange="uploadNewVideo(this.files[0])">
            </div>
        </div>
    </div>

    <!-- Main Workspace -->
    <div class="workspace">
        <div class="topbar">
            <h2>Archival Detail Restoration & Queue Manager</h2>
            <div style="font-size: 12px; display: flex; gap: 12px; align-items: center;">
                <span id="queue-indicator" class="badge badge-blue">Queue Idle</span>
                <span style="color: var(--text-muted);">Port 7860 • Local Studio</span>
            </div>
        </div>

        <div class="content">
            <!-- Left Main Column: Inspector, In/Out Trimmer & Presets -->
            <div class="main-column">
                
                <!-- 1. Video Inspector & In/Out Marks -->
                <div class="card">
                    <h3>
                        <span>1. Footage Inspector & In/Out Marks</span>
                        <span id="current-file-badge" class="badge badge-purple">No Clip Selected</span>
                    </h3>

                    <div class="video-player-container">
                        <video id="player" class="main-video" controls></video>
                    </div>

                    <div class="in-out-panel">
                        <div class="scrub-row">
                            <input type="range" id="scrubber" class="scrub-slider" min="0" max="100" value="0" oninput="onScrub(this.value)">
                            <div class="frame-counter" id="frame-display">F 0 / 0</div>
                        </div>

                        <div class="btn-row">
                            <button class="btn-step" onclick="stepFrames(-10)">⏮ -10</button>
                            <button class="btn-step" onclick="stepFrames(-1)">◀ -1</button>
                            <button class="btn-step" onclick="stepFrames(1)">▶ +1</button>
                            <button class="btn-step" onclick="stepFrames(10)">⏭ +10</button>
                            <div style="width: 1px; height: 20px; background: var(--border); margin: 0 4px;"></div>
                            <button class="btn-marker" onclick="setInPoint()">[ Set IN Point</button>
                            <button class="btn-marker" onclick="setOutPoint()">Set OUT Point ]</button>
                        </div>

                        <div class="btn-row" style="margin-top: 8px;">
                            <span style="font-size: 11px; color: var(--text-muted); margin-right: 4px;">Quick Range:</span>
                            <button class="btn-step" onclick="applyQuickShot(17)">⚡ 17 Frames (0.5s Fast)</button>
                            <button class="btn-step" onclick="applyQuickShot(33)">🎬 33 Frames (1.1s Shot)</button>
                            <button class="btn-step" onclick="applyQuickShot(65)">📽️ 65 Frames (2.2s Shot)</button>
                            <button class="btn-step" onclick="applyQuickShot(97)">🏆 97 Frames (3.2s Scene)</button>
                            <button class="btn-step" onclick="applyFullVideo()">🎞️ Entire Video</button>
                        </div>

                        <div style="margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
                            <div class="range-pill" id="range-pill">
                                📍 In: Frame 0 | Out: Frame 16 (17 frames, 0.57s) • Legal 8n+1 ✓
                            </div>
                        </div>

                        <!-- Smart Auto-Chunking Active Indicator -->
                        <div id="chunking-info" class="alert-box alert-amber" style="display: none;">
                            <span>⚡</span>
                            <div>
                                <strong>Smart Auto-Chunking Active:</strong> Selected <span id="chunk-frame-count">0</span> frames.
                                The studio will automatically process this footage across <span id="chunk-block-count">0</span> safe blocks and seamlessly stitch with audio.
                                <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;" id="chunk-eta-hint"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 2. Simplified 1-Click Restoration Presets -->
                <div class="card">
                    <h3><span>2. 1-Click Restoration Preset</span></h3>
                    <div class="preset-grid" id="preset-container">
                        <!-- Populated dynamically -->
                    </div>
                </div>
            </div>

            <!-- Right Column: Queue Manager & Job Progress -->
            <div class="queue-column">
                
                <!-- Add to Queue Action -->
                <button class="btn-add-queue" id="btn-enqueue" onclick="enqueueJob()">
                    <span>➕</span> Add to Restoration Queue
                </button>

                <!-- Currently Processing Job -->
                <div class="card" id="active-job-section">
                    <h3>
                        <span>⚡ Active Restoration Job</span>
                        <button class="btn-cancel" id="btn-cancel-active" onclick="cancelActiveJob()" style="display: none;">⏹️ Cancel</button>
                    </h3>
                    <div id="active-job-body" style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 16px 0;">
                        No jobs currently running. Add a job above to start!
                    </div>
                </div>

                <!-- Pending Queue List -->
                <div class="card">
                    <h3><span>📋 Queued Jobs</span></h3>
                    <div id="pending-queue-body">
                        <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 12px 0;">
                            Queue is empty.
                        </div>
                    </div>
                </div>

                <!-- Completed Restorations & Comparisons -->
                <div class="card">
                    <h3>
                        <span>🔍 Restored Footage & History</span>
                        <button class="btn-step" style="font-size: 10px; padding: 3px 6px;" onclick="clearHistory()">Clear</button>
                    </h3>
                    <div id="history-container">
                        <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 14px 0;">
                            Completed jobs will appear here.
                        </div>
                    </div>
                </div>

            </div>
        </div>
    </div>

    <script>
        let currentClip = null;
        let inputsData = [];
        let presetsData = {};
        let selectedPresetKey = "__DEFAULT_PRESET__";
        let inFrame = 0;
        let outFrame = 16;
        let fpsVal = 29.97;
        let totalFramesVal = 17;

        const player = document.getElementById("player");
        const scrubber = document.getElementById("scrubber");
        const frameDisplay = document.getElementById("frame-display");
        const rangePill = document.getElementById("range-pill");
        const chunkingInfo = document.getElementById("chunking-info");

        // Player synchronization
        player.addEventListener("loadedmetadata", () => {
            scrubber.max = totalFramesVal > 0 ? totalFramesVal - 1 : 100;
            scrubber.value = inFrame;
            updateScrubDisplay();
        });

        player.addEventListener("timeupdate", () => {
            const currentF = Math.floor(player.currentTime * fpsVal);
            scrubber.value = currentF;
            updateScrubDisplay();
        });

        function updateScrubDisplay() {
            const currentF = parseInt(scrubber.value);
            const timeSec = (currentF / (fpsVal || 30)).toFixed(2);
            frameDisplay.textContent = `F ${currentF} / ${totalFramesVal} (${timeSec}s)`;
        }

        function onScrub(val) {
            const f = parseInt(val);
            player.currentTime = f / (fpsVal || 30);
            updateScrubDisplay();
        }

        function stepFrames(delta) {
            let nextF = Math.max(0, Math.min(totalFramesVal - 1, parseInt(scrubber.value) + delta));
            scrubber.value = nextF;
            onScrub(nextF);
        }

        function setInPoint() {
            inFrame = parseInt(scrubber.value);
            if (outFrame < inFrame) {
                outFrame = Math.min(totalFramesVal - 1, inFrame + 16);
            }
            updateRangeDisplay();
        }

        function setOutPoint() {
            outFrame = parseInt(scrubber.value);
            if (outFrame < inFrame) {
                inFrame = Math.max(0, outFrame - 16);
            }
            updateRangeDisplay();
        }

        function applyQuickShot(nFrames) {
            const currentF = parseInt(scrubber.value);
            inFrame = currentF;
            outFrame = Math.min(totalFramesVal - 1, currentF + nFrames - 1);
            updateRangeDisplay();
        }

        function applyFullVideo() {
            inFrame = 0;
            outFrame = totalFramesVal - 1;
            updateRangeDisplay();
        }

        function snapLegal8n1(frames) {
            if (frames <= 1) return 9;
            const n = Math.floor((frames - 1) / 8);
            return Math.max(9, n * 8 + 1);
        }

        function updateRangeDisplay() {
            const rawSlice = Math.max(1, outFrame - inFrame + 1);
            const preset = presetsData[selectedPresetKey] || {};
            const isDiffusion = !["superres", "relayvsr"].includes(preset.engine_type);

            let effectiveSlice = rawSlice;
            if (isDiffusion) {
                effectiveSlice = snapLegal8n1(rawSlice);
            }

            const sec = (effectiveSlice / (fpsVal || 30)).toFixed(2);
            const safeLimit = isDiffusion ? 97 : 200;
            const isChunked = effectiveSlice > safeLimit;

            if (isChunked) {
                const blockSize = isDiffusion ? 33 : 200;
                const totalBlocks = Math.ceil(effectiveSlice / (isDiffusion ? 32 : 200));
                rangePill.innerHTML = `📍 In: Frame ${inFrame} | Out: Frame ${inFrame + effectiveSlice - 1} (${effectiveSlice} frames, ${sec}s) • Auto-Chunking (${totalBlocks} blocks) ⚡`;
                
                chunkingInfo.style.display = "flex";
                document.getElementById("chunk-frame-count").textContent = effectiveSlice;
                document.getElementById("chunk-block-count").textContent = totalBlocks;
                
                if (isDiffusion) {
                    const estMinutes = Math.round(totalBlocks * 2.8);
                    const hours = Math.floor(estMinutes / 60);
                    const mins = estMinutes % 60;
                    document.getElementById("chunk-eta-hint").innerHTML = `Diffusion Redraw: Estimated render time ~${hours > 0 ? hours + 'h ' : ''}${mins}m. (Tip: For fast ~15-min upscale, choose <em>Instant Real-Time Upscale</em>).`;
                } else {
                    const estMinutes = Math.round(totalBlocks * 0.5);
                    document.getElementById("chunk-eta-hint").innerHTML = `Super-Resolution: Estimated render time ~${estMinutes} minutes.`;
                }
            } else {
                rangePill.innerHTML = `📍 In: Frame ${inFrame} | Out: Frame ${inFrame + effectiveSlice - 1} (${effectiveSlice} frames, ${sec}s) ${isDiffusion ? '• Single Shot (8n+1) ✓' : '• Super-Res ✓'}`;
                chunkingInfo.style.display = "none";
            }
        }

        async function fetchInputs() {
            try {
                const res = await fetch("/api/inputs");
                inputsData = await res.json();
                renderLibrary();

                // Auto select first clip if none active
                if (!currentClip && inputsData.length > 0) {
                    selectClip(inputsData[0]);
                }
            } catch (e) {
                console.error("Fetch inputs error:", e);
            }
        }

        function renderLibrary() {
            const container = document.getElementById("video-library-list");
            container.innerHTML = "";
            inputsData.forEach(clip => {
                const item = document.createElement("div");
                item.className = "lib-item" + (currentClip && currentClip.filename === clip.filename ? " active" : "");
                item.onclick = () => selectClip(clip);
                item.innerHTML = `
                    <div class="lib-name">${clip.filename}</div>
                    <div class="lib-meta">
                        <span>${clip.width}×${clip.height} • ${clip.total_frames} frames</span>
                        <span>${clip.duration}s</span>
                    </div>
                `;
                container.appendChild(item);
            });
        }

        function selectClip(clip) {
            currentClip = clip;
            fpsVal = clip.fps || 29.97;
            totalFramesVal = clip.total_frames || 17;
            player.src = clip.url;
            document.getElementById("current-file-badge").textContent = `${clip.filename} (${clip.total_frames} frames)`;

            // Set default range: if long, default to first 17 or 33 frames
            inFrame = 0;
            outFrame = Math.min(totalFramesVal - 1, 16);
            scrubber.max = totalFramesVal - 1;
            scrubber.value = 0;

            renderLibrary();
            updateRangeDisplay();
            updateScrubDisplay();
        }

        async function fetchPresets() {
            try {
                const res = await fetch("/api/presets");
                presetsData = await res.json();
                renderPresets();
            } catch (e) {
                console.error("Fetch presets error:", e);
            }
        }

        function renderPresets() {
            const container = document.getElementById("preset-container");
            container.innerHTML = "";

            for (const [key, p] of Object.entries(presetsData)) {
                const card = document.createElement("div");
                card.className = "preset-card" + (key === selectedPresetKey ? " active" : "");
                card.onclick = () => selectPreset(key);

                const engineBadge = (p.engine_type === "superres" || p.engine_type === "relayvsr") ? 
                    `<span class="badge badge-green">Full Length Safe</span>` :
                    (p.engine_type === "sol_refiner" ? 
                        `<span class="badge badge-blue">Zero Morphing</span>` :
                        `<span class="badge badge-purple">Shot Redraw</span>`);

                card.innerHTML = `
                    <div class="preset-top">
                        <span class="preset-title">${p.name}</span>
                        ${engineBadge}
                    </div>
                    <div class="preset-desc">${p.description}</div>
                    <div class="preset-footer">
                        <span>Speed: ${p.est_speed || 'N/A'}</span>
                    </div>
                `;
                container.appendChild(card);
            }
        }

        function selectPreset(key) {
            selectedPresetKey = key;
            renderPresets();
            updateRangeDisplay();
        }

        async function enqueueJob() {
            if (!currentClip) {
                alert("Please select a video clip from the library first.");
                return;
            }

            const preset = presetsData[selectedPresetKey] || {};
            const rawSlice = Math.max(1, outFrame - inFrame + 1);
            let effectiveSlice = rawSlice;
            if (!["superres", "relayvsr"].includes(preset.engine_type)) {
                effectiveSlice = snapLegal8n1(rawSlice);
            }

            const payload = {
                input_path: currentClip.path,
                preset: selectedPresetKey,
                start_frame: inFrame,
                end_frame: inFrame + effectiveSlice - 1,
                engine_type: preset.engine_type,
                mode: preset.mode,
                resolution: preset.output_size,
                tile: preset.tile_size,
                sol_sigma: preset.sol_sigma,
                superres_model: preset.superres_model
            };

            const btn = document.getElementById("btn-enqueue");
            btn.disabled = true;
            btn.innerHTML = `<span>⏳</span> Adding to Queue...`;

            try {
                const res = await fetch("/api/queue/add", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.success) {
                    fetchQueue();
                } else {
                    alert("Queue error: " + data.error);
                }
            } catch (e) {
                alert("Request error: " + e);
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<span>➕</span> Add to Restoration Queue`;
            }
        }

        async function cancelActiveJob() {
            const res = await fetch("/api/queue");
            const data = await res.json();
            if (data.active) {
                await fetch("/api/queue/cancel", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ job_id: data.active.id })
                });
                fetchQueue();
            }
        }

        async function removeQueuedJob(jobId) {
            await fetch("/api/queue/cancel", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ job_id: jobId })
            });
            fetchQueue();
        }

        async function clearHistory() {
            await fetch("/api/queue/clear", { method: "POST" });
            fetchQueue();
        }

        async function fetchQueue() {
            try {
                const res = await fetch("/api/queue");
                const data = await res.json();
                renderQueue(data);
            } catch (e) {
                console.error("Queue fetch error:", e);
            }
        }

        function renderQueue(data) {
            const activeBody = document.getElementById("active-job-body");
            const cancelBtn = document.getElementById("btn-cancel-active");
            const indicator = document.getElementById("queue-indicator");

            // Active Job
            if (data.active) {
                const j = data.active;
                cancelBtn.style.display = "block";
                indicator.className = "badge badge-green";
                indicator.textContent = "Processing Job";

                activeBody.innerHTML = `
                    <div style="font-weight: 700; color: #fff; margin-bottom: 4px;">${j.preset_name}</div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">
                        ${j.filename} • Frames ${j.start_frame} - ${j.end_frame} (${j.frame_count} frames)
                    </div>
                    <div class="progress-bar-wrap">
                        <div class="progress-bar-fill" style="width: ${j.progress}%"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 11px;">
                        <span class="job-step-text">${j.step}</span>
                        <strong style="color: #fff;">${j.progress}%</strong>
                    </div>
                `;
            } else {
                cancelBtn.style.display = "none";
                indicator.className = "badge badge-blue";
                indicator.textContent = `Queue (${data.queue.length} pending)`;
                activeBody.innerHTML = `
                    <div style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 12px 0;">
                        ${data.queue.length > 0 ? 'Next job preparing to run...' : 'No jobs currently running.'}
                    </div>
                `;
            }

            // Pending Queue
            const pendingBody = document.getElementById("pending-queue-body");
            if (data.queue.length === 0) {
                pendingBody.innerHTML = `<div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 8px 0;">Queue is empty.</div>`;
            } else {
                let html = `<table class="queue-table"><thead><tr><th>Footage</th><th>Preset</th><th>Frames</th><th></th></tr></thead><tbody>`;
                data.queue.forEach(q => {
                    html += `
                        <tr>
                            <td><strong>${q.filename}</strong></td>
                            <td><span class="badge badge-purple">${q.preset}</span></td>
                            <td>${q.frame_count}f</td>
                            <td><button class="btn-step" style="padding: 2px 6px; font-size: 10px;" onclick="removeQueuedJob('${q.id}')">✕</button></td>
                        </tr>
                    `;
                });
                html += `</tbody></table>`;
                pendingBody.innerHTML = html;
            }

            // History & Completed
            const historyBody = document.getElementById("history-container");
            if (data.history.length === 0) {
                historyBody.innerHTML = `<div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 8px 0;">Completed jobs will show here.</div>`;
            } else {
                let html = "";
                data.history.forEach(h => {
                    if (h.status === "completed" && h.output_video) {
                        const outUrl = `/files/outputs/${h.output_video.split('\\\\').pop().split('/').pop()}`;
                        html += `
                            <div class="output-box">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong style="font-size: 12px; color: #4ade80;">✓ ${h.preset_name}</strong>
                                    <span style="font-size: 10px; color: var(--text-muted);">${h.completed_at || ''}</span>
                                </div>
                                <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">${h.filename} (Frames ${h.start_frame}-${h.end_frame})</div>
                                <video src="${outUrl}" controls></video>
                            </div>
                        `;
                    } else if (h.status === "failed") {
                        html += `
                            <div class="output-box" style="border-color: rgba(239, 68, 68, 0.4); background: rgba(239, 68, 68, 0.05);">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong style="font-size: 12px; color: #f87171;">✕ Job Failed</strong>
                                    <span style="font-size: 10px; color: var(--text-muted);">${h.completed_at || ''}</span>
                                </div>
                                <div style="font-size: 11px; color: var(--text-muted); margin: 4px 0;">${h.filename} (${h.preset})</div>
                                <div style="font-size: 11px; color: #fca5a5; line-height: 1.4;">${h.error || 'Unknown execution error'}</div>
                            </div>
                        `;
                    }
                });
                historyBody.innerHTML = html || `<div style="color: var(--text-muted); font-size: 12px; text-align: center;">No completed videos yet.</div>`;
            }
        }

        async function fetchSystemStatus() {
            try {
                const res = await fetch("/api/status");
                const data = await res.json();
                const dot = document.getElementById("server-dot");
                const txt = document.getElementById("server-status");
                if (data.server_online) {
                    dot.className = "dot online";
                    txt.textContent = "ComfyUI Online (Port 8188)";
                    const dev = data.stats.devices?.[0];
                    if (dev) {
                        const freeGb = (dev.vram_free / (1024**3)).toFixed(1);
                        const totGb = (dev.vram_total / (1024**3)).toFixed(1);
                        document.getElementById("vram-info").textContent = `GPU: RTX 5060 Ti (${freeGb} GB / ${totGb} GB free)`;
                    }
                } else {
                    dot.className = "dot";
                    txt.textContent = "ComfyUI Offline";
                }
            } catch (e) {}
        }

        async function uploadNewVideo(file) {
            if (!file) return;
            const formData = new FormData();
            formData.append("file", file, file.name);
            document.getElementById("current-file-badge").textContent = `Uploading ${file.name}...`;

            try {
                const res = await fetch("/api/upload", { method: "POST", body: formData });
                const data = await res.json();
                if (data.success) {
                    await fetchInputs();
                    selectClip(data.metadata);
                } else {
                    alert("Upload error: " + data.error);
                }
            } catch (e) {
                alert("Upload error: " + e);
            }
        }

        // Initialize Studio
        fetchSystemStatus();
        fetchInputs();
        fetchPresets();
        fetchQueue();
        setInterval(fetchSystemStatus, 4000);
        setInterval(fetchQueue, 2000);
    </script>
</body>
</html>
"""

def run_server():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, RestorationHandler)
    print(f"\n=======================================================")
    print(f" LTX-2.5 RESTORATION STUDIO WEB UI")
    print(f" Server running at: http://localhost:{PORT}")
    print(f" Connected to ComfyUI at: {config.COMFYUI_BASE_URL}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Web UI server...")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
