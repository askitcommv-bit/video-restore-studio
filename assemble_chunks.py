"""Joins the chunk_*.mp4 files of an interrupted chunked LTX run. Usage: assemble_chunks.py <chunk_dir>"""
import sys
import subprocess
from pathlib import Path

chunk_dir = Path(sys.argv[1])
chunks = sorted(chunk_dir.glob("chunk_*.mp4"))
print("Found chunks:", len(chunks))

concat_list = chunk_dir / "concat_overnight.txt"
with open(concat_list, "w", encoding="utf-8") as f:
    for c in chunks:
        posix_path = c.resolve().as_posix()
        f.write(f"file '{posix_path}'\n")

out_video = Path(r"C:\AIdev\video-restore\outputs\overnight_106_chunks_restored.mp4")
cmd = [
    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
    "-i", str(concat_list),
    "-c", "copy",
    str(out_video)
]
subprocess.run(cmd, check=True)
print("Successfully assembled:", out_video.name, "size MB:", round(out_video.stat().st_size / (1024*1024), 2))
