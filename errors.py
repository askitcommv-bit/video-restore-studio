"""
Typed errors shared by the engine, the setup and every front end (desktop app, CLI, browser UI).

Front ends decide what to do from the type (open the setup, show a hint), never from message text.
friendly_message() turns any exception into one short, actionable line for people.
"""

from typing import Iterable, Optional


class AppError(Exception):
    """Base for errors the app expects and can explain."""


class FFmpegMissing(AppError):
    def __init__(self):
        super().__init__("FFmpeg isn't installed. Video Restore Studio uses it to read and write video.")


class SetupRequired(AppError):
    """Something an engine needs isn't installed; `missing` names it for people."""

    def __init__(self, engine: str, missing: Iterable[str]):
        self.engine = engine
        self.missing = list(missing)
        super().__init__(f"{engine} isn't fully installed. Missing: {', '.join(self.missing)}. "
                         "Use Set up downloads… in the ☰ menu, or: python restore_cli.py download")


class DownloadForbidden(AppError):
    """The server refused a download (401/403): usually a gated Hugging Face repo whose license
    hasn't been accepted, or a missing/invalid access token."""

    def __init__(self, url: str, repo_url: Optional[str] = None, detail: str = ""):
        self.url = url
        self.repo_url = repo_url
        msg = f"Access denied for {url}"
        if repo_url:
            msg += (f". Accept the license at {repo_url} with your Hugging Face account and set an access "
                    "token (Set up downloads… → Hugging Face token)")
        super().__init__(msg + (f" ({detail})" if detail else ""))


class ComfyUIOffline(AppError):
    def __init__(self, url: str):
        super().__init__(f"ComfyUI is offline at {url}. Start ComfyUI (☰ → Launch ComfyUI) and try again.")


class ComfyExecutionError(AppError):
    """A ComfyUI node failed; `exception_type` is ComfyUI's own (e.g. torch.OutOfMemoryError)."""

    def __init__(self, message: str, exception_type: str = ""):
        self.exception_type = exception_type
        super().__init__(f"ComfyUI Execution Error: {message}")


def friendly_message(exc: BaseException) -> str:
    """One actionable line for people; known failure kinds get a concrete fix."""
    if isinstance(exc, ComfyExecutionError):
        kind = exc.exception_type
        # Older torch reports CUDA OOM as a plain RuntimeError; only its text says so
        if "OutOfMemory" in kind or "CUDA out of memory" in str(exc):
            return "GPU memory ran out. Lower the tiling (HD or qHD) under Customize, or pick a shorter range."
        if kind in ("numpy.core._exceptions._ArrayMemoryError", "MemoryError") or "ArrayMemoryError" in kind:
            return ("System memory ran out: too many frames for one diffusion pass. Mark a shorter range "
                    "(17 to 97 frames) with I/O, or use an upscaling preset for full episodes.")
    if isinstance(exc, MemoryError):
        return "System memory ran out. Mark a shorter range with I/O, or use an upscaling preset."
    return str(exc) or exc.__class__.__name__
