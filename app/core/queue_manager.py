"""
Queue Manager for LTX-2.5 Video Restoration Suite.
Manages sequential execution of restoration jobs, batch file additions,
live progress dispatching, error resilience, and queue persistence.
"""

import time
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict

from PySide6.QtCore import QObject, Signal, QThread

import config
import video_preprocessor
from restore_engine import VideoRestorationPipeline


@dataclass
class QueueJob:
    """Represents a single restoration task in the batch queue."""
    job_id: str
    input_path: str
    output_path: str
    preset_id: str
    preset_name: str
    engine_type: str
    mode: str
    params: Dict[str, Any]
    status: str = "pending"  # pending, processing, completed, failed, canceled
    progress: int = 0
    current_step_text: str = "In queue"
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

    @property
    def filename(self) -> str:
        return Path(self.input_path).name

    @property
    def output_filename(self) -> str:
        return Path(self.output_path).name

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueJob":
        return cls(**data)


class QueueWorker(QThread):
    """Executes a single QueueJob in a dedicated thread."""
    progress = Signal(int, int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, pipeline: VideoRestorationPipeline, params: Dict[str, Any]):
        super().__init__()
        self.pipeline = pipeline
        self.params = params
        self._is_cancelled = False

    def run(self):
        try:
            def on_progress(curr, tot, msg):
                if not self._is_cancelled:
                    self.progress.emit(curr, tot, msg)

            res = self.pipeline.restore_footage(
                input_video_path=self.params["input_path"],
                output_dir=self.params.get("output_dir"),
                output_file=self.params.get("output_file"),
                max_frames=self.params.get("max_frames"),
                start_frame=self.params.get("start_frame"),
                end_frame=self.params.get("end_frame"),
                engine_type=self.params.get("engine_type", "diffusion"),
                superres_model=self.params.get("superres_model", "RealESRGAN_x2plus.pth"),
                mode=self.params.get("mode", "restore"),
                prompt_preset=self.params.get("preset", config.DEFAULT_PRESET),
                custom_positive_prompt=self.params.get("positive_prompt"),
                custom_negative_prompt=self.params.get("negative_prompt"),
                output_size=self.params.get("output_size", "FullHD"),
                tile_size=self.params.get("tile_size", "FullHD"),
                seed=self.params.get("seed"),
                guide_strength=self.params.get("guide_strength", 0.72),
                cfg=self.params.get("cfg", 2.5),
                sol_sigma=self.params.get("sol_sigma", 0.50),
                enable_teacache=self.params.get("enable_teacache"),
                teacache_thresh=self.params.get("teacache_thresh", 0.06),
                reference_image=self.params.get("reference_image"),
                extract_depth=self.params.get("extract_depth", False),
                clean_noise=self.params.get("clean_noise"),
                stabilize_jitter=self.params.get("stabilize_jitter"),
                progress_callback=on_progress
            )
            if not self._is_cancelled:
                self.finished.emit(res)
        except Exception as e:
            if not self._is_cancelled:
                self.error.emit(str(e))

    def cancel(self):
        self._is_cancelled = True
        # Out-of-process GPU workers (RelayVSR) are killed directly
        self.pipeline.terminate_active_worker()


class QueueManager(QObject):
    """
    Manages queue lifecycle, sequential execution, and notifications.
    """
    job_added = Signal(object)          # QueueJob
    job_updated = Signal(object)        # QueueJob
    job_removed = Signal(str)           # job_id
    job_started = Signal(object)        # QueueJob
    job_completed = Signal(object, dict)# QueueJob, result
    job_failed = Signal(object, str)    # QueueJob, error_msg
    queue_status_changed = Signal(bool, int, int) # is_running, pending_count, completed_count
    queue_cleared = Signal()

    QUEUE_SAVE_FILE = config.PROJECT_DIR / "queue_state.json"

    def __init__(self, pipeline: Optional[VideoRestorationPipeline] = None):
        super().__init__()
        self.pipeline = pipeline or VideoRestorationPipeline()
        self.jobs: List[QueueJob] = []
        self.is_running: bool = False
        self.is_paused: bool = False
        self.current_job: Optional[QueueJob] = None
        self._worker: Optional[QueueWorker] = None

        self._load_queue_state()

    def add_job(
        self,
        input_path: str,
        output_path: str,
        params: Dict[str, Any],
        preset_id: str = config.DEFAULT_PRESET,
        preset_name: str = "Master Archival Redraw"
    ) -> QueueJob:
        """Adds a single restoration job to the queue."""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job_params = dict(params)
        job_params["output_file"] = output_path
        job_params["output_dir"] = str(Path(output_path).parent)

        engine_type = job_params.get("engine_type", "diffusion")
        mode = job_params.get("mode", "restore")

        job = QueueJob(
            job_id=job_id,
            input_path=str(input_path),
            output_path=str(output_path),
            preset_id=preset_id,
            preset_name=preset_name,
            engine_type=engine_type,
            mode=mode,
            params=job_params,
            status="pending",
            progress=0,
            current_step_text="In Queue"
        )
        self.jobs.append(job)
        self.job_added.emit(job)
        self._emit_status()
        self._save_queue_state()
        return job

    def add_batch_files(
        self,
        file_paths: List[str],
        base_params: Dict[str, Any],
        preset_id: str = config.DEFAULT_PRESET,
        preset_name: str = "Master Archival Redraw",
        output_dir: Optional[Path] = None,
    ) -> List[QueueJob]:
        """Adds multiple files to the queue using base settings with automatic unique filenames
        in `output_dir` (default: the remembered output folder)."""
        added = []
        out_folder = Path(output_dir) if output_dir else config.get_last_output_folder()
        out_folder.mkdir(parents=True, exist_ok=True)
        engine_type = base_params.get("engine_type", "diffusion")
        mode = base_params.get("mode", "restore")

        if engine_type == "superres":
            sr_name = base_params.get("superres_model", "RealESRGAN_x2plus.pth").split(".")[0].lower()
            mode_tag = f"superres_{sr_name}"
        elif engine_type == "relayvsr":
            mode_tag = "relayvsr"
        else:
            mode_tag = mode if mode != "restore" else "restored"

        for fpath in file_paths:
            fp = Path(fpath)
            if not fp.exists():
                continue

            unique_out = config.get_unique_output_path(
                output_dir=out_folder,
                base_stem=f"{fp.stem}_{mode_tag}",
                suffix=".mp4"
            )

            file_params = dict(base_params)
            file_params["input_path"] = str(fp)
            file_params["output_file"] = str(unique_out)
            file_params["output_dir"] = str(out_folder)
            # Reset specific in/out marks for batch additions unless full clip
            file_params["start_frame"] = None
            file_params["end_frame"] = None

            job = self.add_job(
                input_path=str(fp),
                output_path=str(unique_out),
                params=file_params,
                preset_id=preset_id,
                preset_name=preset_name
            )
            added.append(job)

        return added

    def remove_job(self, job_id: str) -> bool:
        """Removes a job by ID from the queue."""
        if self.current_job and self.current_job.job_id == job_id:
            # Active job cannot be simply removed while processing, cancel first
            self.stop_current_job()

        for i, j in enumerate(self.jobs):
            if j.job_id == job_id:
                self.jobs.pop(i)
                self.job_removed.emit(job_id)
                self._emit_status()
                self._save_queue_state()
                return True
        return False

    def clear_completed(self):
        """Clears all completed or failed jobs."""
        initial_len = len(self.jobs)
        self.jobs = [j for j in self.jobs if j.status in ["pending", "processing"]]
        if len(self.jobs) != initial_len:
            self.queue_cleared.emit()
            for j in self.jobs:
                self.job_added.emit(j)
            self._emit_status()
            self._save_queue_state()

    def clear_all(self):
        """Stops runner and removes all jobs."""
        self.stop_queue()
        self.jobs.clear()
        self.queue_cleared.emit()
        self._emit_status()
        self._save_queue_state()

    def move_job(self, from_idx: int, to_idx: int):
        """Moves a job within the queue order."""
        if 0 <= from_idx < len(self.jobs) and 0 <= to_idx < len(self.jobs):
            job = self.jobs.pop(from_idx)
            self.jobs.insert(to_idx, job)
            self.queue_cleared.emit()
            for j in self.jobs:
                self.job_added.emit(j)
            self._save_queue_state()

    def start_queue(self):
        """Starts or resumes sequential execution of the queue."""
        if self.is_running and not self.is_paused:
            return

        self.is_running = True
        self.is_paused = False
        self._emit_status()
        if self.current_job is None:  # un-pausing while a job still runs must not start a second one
            self._process_next()

    def pause_queue(self):
        """Pauses after the current job completes."""
        self.is_paused = True
        self._emit_status()

    def stop_queue(self):
        """Stops the queue runner and halts active job."""
        self.is_running = False
        self.is_paused = False
        self.stop_current_job()
        self.pipeline.release_relayvsr_worker()
        self._emit_status()

    def stop_current_job(self):
        """Stops the running job. RelayVSR jobs save their finished part and stay resumable."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()  # kills the GPU worker; the engine then saves <name>_partial.mp4
            if not self._worker.wait(30000):  # hard-stop only if it couldn't wind down
                self._worker.terminate()
                self._worker.wait(1500)
            self._worker = None

        if self.current_job:
            self.current_job.status = "canceled"
            self.current_job.current_step_text = "Stopped — finished part saved. Resume continues from there."
            self.job_updated.emit(self.current_job)
            self._save_queue_state()
            self.current_job = None

    def resume_job(self, job_id: str) -> bool:
        """Re-queues a stopped/failed job; RelayVSR picks up after its last finished segment."""
        for job in self.jobs:
            if job.job_id == job_id and job.status in ("canceled", "failed"):
                job.status = "pending"
                job.progress = 0
                job.current_step_text = "Queued to resume"
                self.job_updated.emit(job)
                self._save_queue_state()
                self.start_queue()
                return True
        return False

    def _process_next(self):
        """Finds next pending job and executes it."""
        if not self.is_running or self.is_paused:
            # Idle (stopped/paused): a resident RelayVSR worker would just sit on ~9.4 GB of VRAM
            self.pipeline.release_relayvsr_worker()
            return

        next_job = None
        for j in self.jobs:
            if j.status == "pending":
                next_job = j
                break

        if not next_job:
            # Queue complete!
            self.is_running = False
            self.current_job = None
            self.pipeline.release_relayvsr_worker()
            self._emit_status()
            self._save_queue_state()
            return

        self.current_job = next_job
        self.current_job.status = "processing"
        self.current_job.started_at = time.time()
        self.current_job.progress = 0
        self.current_job.current_step_text = "Starting..."
        self.job_started.emit(self.current_job)
        self.job_updated.emit(self.current_job)
        self._emit_status()
        self._save_queue_state()

        self._worker = QueueWorker(self.pipeline, self.current_job.params)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_progress(self, curr: int, tot: int, msg: str):
        if not self.current_job:
            return
        pct = int(round((curr / tot) * 100)) if tot > 0 else curr
        self.current_job.progress = pct
        self.current_job.current_step_text = msg
        self.job_updated.emit(self.current_job)

    def _on_worker_finished(self, result: Dict[str, Any]):
        if not self.current_job:
            return
        job = self.current_job
        job.status = "completed"
        job.progress = 100
        job.current_step_text = "Completed"
        job.completed_at = time.time()
        job.result = result

        final_out = result.get("final_output_path")
        if final_out:
            job.output_path = str(final_out)

        self.job_completed.emit(job, result)
        self.job_updated.emit(job)
        self.current_job = None
        self._worker = None

        self._save_queue_state()
        self._emit_status()

        # Trigger next job
        self._process_next()

    def _on_worker_error(self, err_msg: str):
        if not self.current_job:
            return
        job = self.current_job
        job.status = "failed"
        job.error_message = err_msg
        job.current_step_text = f"Error: {err_msg}"
        job.completed_at = time.time()

        self.job_failed.emit(job, err_msg)
        self.job_updated.emit(job)
        self.current_job = None
        self._worker = None

        self._save_queue_state()
        self._emit_status()

        # Continue to next job in queue even if this one failed
        self._process_next()

    def _emit_status(self):
        pending = sum(1 for j in self.jobs if j.status == "pending")
        completed = sum(1 for j in self.jobs if j.status == "completed")
        self.queue_status_changed.emit(self.is_running, pending, completed)

    def _save_queue_state(self):
        try:
            data = [j.to_dict() for j in self.jobs]
            with open(self.QUEUE_SAVE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            pass

    def _load_queue_state(self):
        if not self.QUEUE_SAVE_FILE.exists():
            return
        try:
            with open(self.QUEUE_SAVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.jobs = [QueueJob.from_dict(d) for d in data if isinstance(d, dict)]
        except Exception as e:
            self.jobs = []
        for job in self.jobs:  # the app closed mid-job: offer to resume instead of a stuck "processing"
            if job.status == "processing":
                job.status = "canceled"
                job.current_step_text = "Interrupted when the app closed — Resume continues from the last finished part."
