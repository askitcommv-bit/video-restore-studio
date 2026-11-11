"""
ComfyUI REST / WebSocket client: queue a prompt graph, follow its progress, fetch the result.
"""

import json
import os
import shutil
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import config
from engine.comfy.graphs import NODE_DESCRIPTIONS
from errors import ComfyExecutionError


class ComfyUIClient:
    """REST and WebSocket Client for ComfyUI."""

    def __init__(self, host: str = config.COMFYUI_HOST, port: int = config.COMFYUI_PORT):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.client_id = str(uuid.uuid4())

    def is_online(self) -> bool:
        """Checks if ComfyUI server is reachable."""
        return self.try_system_stats() is not None

    def try_system_stats(self, timeout: float = 3) -> Optional[Dict[str, Any]]:
        """One /system_stats round-trip: the stats dict when online, None when unreachable."""
        try:
            return self.get_system_stats(timeout=timeout)
        except Exception:
            return None

    def get_system_stats(self, timeout: float = 5) -> Dict[str, Any]:
        """Fetches ComfyUI system stats including GPU VRAM and torch status."""
        req = urllib.request.Request(f"{self.base_url}/system_stats")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def upload_video(self, file_path: Path | str) -> str:
        """Copies video to ComfyUI input directory and returns filename."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        comfy_input = config.INPUT_DIR_COMFY
        comfy_input.mkdir(parents=True, exist_ok=True)

        target_path = comfy_input / file_path.name
        if target_path.resolve() != file_path.resolve():
            shutil.copyfile(file_path, target_path)

        return file_path.name

    def queue_prompt(self, prompt: Dict[str, Any], extra_data: Optional[Dict[str, Any]] = None) -> str:
        """Submits prompt graph to ComfyUI /prompt endpoint and returns prompt_id."""
        payload = {
            "prompt": prompt,
            "client_id": self.client_id
        }
        if extra_data:
            payload["extra_data"] = extra_data

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            prompt_id = res_json.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"ComfyUI did not return prompt_id: {res_json}")
            return prompt_id

    def interrupt(self) -> bool:
        """Interrupts currently executing prompt in ComfyUI."""
        try:
            req = urllib.request.Request(f"{self.base_url}/interrupt", data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"[COMFYUI] Interrupt notice: {e}")
            return False

    def free_memory(self) -> bool:
        """Asks ComfyUI to unload all models and release VRAM (needed before out-of-process GPU workers)."""
        try:
            body = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/free", data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"[COMFYUI] Free-memory notice: {e}")
            return False

    def track_execution(
        self,
        prompt_id: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Tracks execution progress using WebSocket with fallback to HTTP polling.
        """
        try:
            return self._track_websocket(prompt_id, progress_callback)
        except Exception as e:
            print(f"[COMFYUI] WebSocket tracking notice ({e}); falling back to HTTP polling...")
            return self._track_polling(prompt_id, progress_callback)

    def _track_websocket(
        self,
        prompt_id: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        import websocket

        ws = websocket.WebSocket()
        ws_url_with_id = f"{self.ws_url}?clientId={self.client_id}"
        ws.connect(ws_url_with_id)

        last_node = ""
        current_step = 0
        total_steps = 1
        output_data = {}

        try:
            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    msg_type = message.get("type")
                    data = message.get("data", {})

                    if msg_type == "status":
                        pass
                    elif msg_type == "execution_start":
                        if data.get("prompt_id") == prompt_id:
                            print(f"[COMFYUI] Execution started for prompt: {prompt_id}")
                            if progress_callback:
                                progress_callback(32, 100, "Starting GPU neural execution pipeline...")
                    elif msg_type == "executing":
                        node_id = data.get("node")
                        if node_id is None and data.get("prompt_id") == prompt_id:
                            print(f"[COMFYUI] Execution completed successfully!")
                            if progress_callback:
                                progress_callback(100, 100, "Execution completed!")
                            break
                        last_node = str(node_id) if node_id is not None else ""
                        if progress_callback and last_node in NODE_DESCRIPTIONS:
                            pct, desc = NODE_DESCRIPTIONS[last_node]
                            progress_callback(pct, 100, desc)
                        elif progress_callback and last_node:
                            progress_callback(current_step, total_steps, f"Executing node {last_node}...")
                    elif msg_type == "progress":
                        current_step = data.get("value", 0)
                        total_steps = data.get("max", 1)
                        if progress_callback:
                            pct_ratio = current_step / max(1, total_steps)
                            if last_node == "42":
                                p_val = int(65 + pct_ratio * 16)
                                progress_callback(p_val, 100, f"Neural Tiled Diffusion: Step {current_step}/{total_steps} ({int(pct_ratio*100)}%)")
                            elif last_node == "64":
                                p_val = int(87 + pct_ratio * 5)
                                progress_callback(p_val, 100, f"Stage 2 4K Refinement: Step {current_step}/{total_steps} ({int(pct_ratio*100)}%)")
                            else:
                                p_val = int(60 + pct_ratio * 30)
                                progress_callback(p_val, 100, f"Diffusion step {current_step}/{total_steps} ({int(pct_ratio*100)}%)")
                    elif msg_type == "executed":
                        if data.get("prompt_id") == prompt_id:
                            node_id = data.get("node")
                            if "output" in data:
                                output_data[str(node_id)] = data["output"]
                    elif msg_type == "execution_error":
                        if data.get("prompt_id") == prompt_id:
                            raise ComfyExecutionError(data.get("exception_message", "Unknown execution error"),
                                                      data.get("exception_type", ""))
        finally:
            ws.close()

        return output_data

    def _track_polling(
        self,
        prompt_id: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """Polls ComfyUI /history endpoint until prompt_id is present."""
        poll_interval = 2.0
        elapsed = 0

        not_in_queue_count = 0
        while True:
            time.sleep(poll_interval)
            elapsed += int(poll_interval)

            try:
                # 1. Check history first
                req = urllib.request.Request(f"{self.base_url}/history/{prompt_id}")
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if prompt_id in data:
                        job_info = data[prompt_id]
                        status = job_info.get("status", {})
                        if status.get("status_str") == "error":
                            messages = status.get("messages", [])
                            # Same concise message the WebSocket path reports, not the full event log
                            errors = [m[1] for m in messages if isinstance(m, (list, tuple)) and len(m) == 2 and m[0] == "execution_error"]
                            err = errors[-1] if errors else {}
                            if errors:
                                error_msg = f"{err.get('node_type', 'node')} #{err.get('node_id', '?')}: {err.get('exception_message', 'Unknown execution error').strip()}"
                            else:
                                error_msg = "Unknown execution error"
                            raise ComfyExecutionError(error_msg, err.get("exception_type", ""))

                        outputs = job_info.get("outputs", {})
                        print(f"[COMFYUI] Execution finished according to history.")
                        return outputs

                # 2. Check active queue
                q_req = urllib.request.Request(f"{self.base_url}/queue")
                with urllib.request.urlopen(q_req, timeout=8) as resp:
                    q_data = json.loads(resp.read().decode("utf-8"))
                    running = q_data.get("queue_running", [])
                    pending = q_data.get("queue_pending", [])
                    is_running = any(item[1] == prompt_id for item in running)
                    is_pending = any(item[1] == prompt_id for item in pending)

                    if is_running:
                        not_in_queue_count = 0
                        if progress_callback:
                            progress_callback(50, 100, f"Rendering on GPU ({elapsed}s elapsed)...")
                    elif is_pending:
                        not_in_queue_count = 0
                        if progress_callback:
                            progress_callback(10, 100, f"Queued, waiting to execute ({elapsed}s elapsed)...")
                    else:
                        # Job is no longer in running or pending
                        not_in_queue_count += 1
                        if not_in_queue_count >= 3:
                            print(f"[COMFYUI] Prompt {prompt_id} is no longer in queue. Checking output...")
                            return {}

            except (RuntimeError, ComfyExecutionError):
                raise
            except Exception as e:
                print(f"[COMFYUI Polling Notice] {e}")


def find_rendered_output(output_prefix: str) -> Path:
    """The video ComfyUI wrote for output_prefix (newest match), else its newest .mp4."""
    out_dir = config.OUTPUT_DIR_COMFY
    candidates = list(out_dir.glob(f"*{output_prefix}*.mp4")) or list(out_dir.glob("*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"Could not find rendered video for prefix {output_prefix} in {out_dir}")
    return max(candidates, key=os.path.getmtime)
