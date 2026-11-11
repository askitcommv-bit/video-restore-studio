"""
Final-output video codec: picks the best HEVC encoder this machine can actually run.

An encoder being listed by ffmpeg doesn't mean the hardware is there (an ffmpeg build lists NVENC,
QSV and AMF everywhere), so each candidate is tried on a few frames with the exact settings used
for real, in order of preference: GPU encoders first (fast, and off the CUDA cores RelayVSR uses),
then x265 on the CPU. The result is cached for the process.

Output codec choices:
  auto  HEVC on the best working encoder, or H.264 as rendered when none works
  hevc  HEVC (same encoder choice as auto)
  h264  keep the rendered H.264 stream as is (no re-encode)
"""

import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import processes

OUTPUT_CODECS = ("auto", "hevc", "h264")


@dataclass(frozen=True)
class Encoder:
    name: str      # ffmpeg encoder
    label: str     # shown to the user
    args: tuple    # quality settings (visually transparent for 720p-1080p restorations)


# Measured on a real 720p RelayVSR episode (2.4 Mbps H.264 CRF 18 intermediate): NVENC at cq 27
# halves the video bitrate (1.2 Mbps) at 44.4 dB PSNR / 0.992 SSIM, indistinguishable at 2x zoom,
# in 14 s for 3 minutes of video; x265 crf 24 is similar in size but took 110 s. The other GPU
# encoders get settings of the same strength. -tag:v hvc1 lets Apple players open the MP4.
HEVC_ENCODERS = (
    Encoder("hevc_nvenc", "NVIDIA NVENC",
            ("-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "27", "-b:v", "0",
             "-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "32", "-profile:v", "main")),
    Encoder("hevc_qsv", "Intel Quick Sync", ("-preset", "slow", "-global_quality", "26")),
    Encoder("hevc_amf", "AMD AMF", ("-quality", "quality", "-rc", "cqp", "-qp_i", "26", "-qp_p", "28")),
    Encoder("libx265", "CPU (x265)", ("-preset", "medium", "-crf", "24", "-x265-params", "log-level=error")),
)
COMMON_ARGS = ("-pix_fmt", "yuv420p", "-tag:v", "hvc1")

_lock = threading.Lock()
_best: dict = {}


def _works(enc: Encoder) -> bool:
    """Encodes 10 frames of a test pattern with the real settings; True if the encoder runs here."""
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30",
           "-frames:v", "10", "-c:v", enc.name, *enc.args, *COMMON_ARGS, "-f", "null", "-"]
    try:
        r = processes.run(cmd, capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def best_hevc() -> Optional[Encoder]:
    """The preferred HEVC encoder that works on this machine (None if none does). Cached."""
    with _lock:
        if "hevc" not in _best:
            _best["hevc"] = next((e for e in HEVC_ENCODERS if _works(e)), None)
        return _best["hevc"]


def probed() -> bool:
    """True once best_hevc() has run (the UI shows a placeholder until then)."""
    return "hevc" in _best


def resolve(choice: Optional[str]) -> Tuple[str, Optional[Encoder]]:
    """Maps a user choice to (codec, encoder). encoder None means keep the rendered H.264 stream."""
    choice = choice if choice in OUTPUT_CODECS else "auto"
    if choice == "h264":
        return "h264", None
    enc = best_hevc()
    return ("hevc", enc) if enc else ("h264", None)


def describe(choice: Optional[str]) -> str:
    """One line for the UI, e.g. 'HEVC on NVIDIA NVENC'."""
    codec, enc = resolve(choice)
    if codec == "h264":
        return "H.264 (no HEVC encoder works here)" if choice in (None, "auto", "hevc") else "H.264, as rendered"
    return f"HEVC on {enc.label}"


def video_args(encoder: Optional[Encoder]) -> list:
    """ffmpeg video-stream arguments for the final file."""
    return ["-c:v", "copy"] if encoder is None else ["-c:v", encoder.name, *encoder.args, *COMMON_ARGS]
