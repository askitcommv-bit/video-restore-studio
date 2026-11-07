"""
High-Performance Model Downloader and Health Checker for LTX-2.5 Video Restoration.
Features:
- Direct streaming download directly into ComfyUI models folders
- ZERO duplicate disk cache / ZERO slow copying
- Automatic HTTP Range Resumption (pick up right where you left off if interrupted)
- Real-time download speed in MB/s, transferred size (GB/MB), percentage, and ETA
- Clear Hugging Face authentication guidance for gated Lightricks/LTX-2.5 models
"""

import sys
import os
import time
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional, Callable

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import requests


def get_model_status() -> Dict[str, Dict]:
    """Inspects ComfyUI models folder and checks existence of all required models."""
    status = {}
    
    # 1. Base Transformer
    int8_trans = config.DIFFUSION_MODELS_DIR / config.MODEL_FILENAMES["transformer_int8"]
    bf16_trans = config.DIFFUSION_MODELS_DIR / config.MODEL_FILENAMES["transformer_bf16"]
    trans_exists = (int8_trans.exists() and int8_trans.stat().st_size > 1024*1024) or \
                   (bf16_trans.exists() and bf16_trans.stat().st_size > 1024*1024)
    status["transformer"] = {
        "installed": trans_exists,
        "path": str(int8_trans if int8_trans.exists() else (bf16_trans if bf16_trans.exists() else int8_trans)),
        "variant": "int8" if (int8_trans.exists() and int8_trans.stat().st_size > 1024*1024) else \
                   ("bf16" if (bf16_trans.exists() and bf16_trans.stat().st_size > 1024*1024) else "missing"),
        "recommended_file": config.MODEL_FILENAMES["transformer_int8"],
        "target_dir": config.DIFFUSION_MODELS_DIR,
        "repo": config.HF_REPOS["base"],
        "repo_subpath": f"diffusion_models/{config.MODEL_FILENAMES['transformer_int8']}",
        "size_gb": 20.03
    }
    
    # 2. Text Encoder
    int8_te = config.TEXT_ENCODERS_DIR / config.MODEL_FILENAMES["text_encoder_int8"]
    bf16_te = config.TEXT_ENCODERS_DIR / config.MODEL_FILENAMES["text_encoder_bf16"]
    te_exists = (int8_te.exists() and int8_te.stat().st_size > 1024*1024) or \
                (bf16_te.exists() and bf16_te.stat().st_size > 1024*1024)
    status["text_encoder"] = {
        "installed": te_exists,
        "path": str(int8_te if int8_te.exists() else (bf16_te if bf16_te.exists() else int8_te)),
        "variant": "int8" if (int8_te.exists() and int8_te.stat().st_size > 1024*1024) else \
                   ("bf16" if (bf16_te.exists() and bf16_te.stat().st_size > 1024*1024) else "missing"),
        "recommended_file": config.MODEL_FILENAMES["text_encoder_int8"],
        "target_dir": config.TEXT_ENCODERS_DIR,
        "repo": config.HF_REPOS["base"],
        "repo_subpath": f"text_encoders/{config.MODEL_FILENAMES['text_encoder_int8']}",
        "size_gb": 14.3
    }
    
    # 3. Video VAE
    v_vae = config.VAE_DIR / config.MODEL_FILENAMES["video_vae"]
    status["video_vae"] = {
        "installed": v_vae.exists() and v_vae.stat().st_size > 1024*1024,
        "path": str(v_vae),
        "target_dir": config.VAE_DIR,
        "repo": config.HF_REPOS["base"],
        "repo_subpath": f"vae/{config.MODEL_FILENAMES['video_vae']}",
        "size_gb": 1.4
    }
    
    # 4. Audio VAE
    a_vae = config.VAE_DIR / config.MODEL_FILENAMES["audio_vae"]
    status["audio_vae"] = {
        "installed": a_vae.exists() and a_vae.stat().st_size > 1024*1024,
        "path": str(a_vae),
        "target_dir": config.VAE_DIR,
        "repo": config.HF_REPOS["base"],
        "repo_subpath": f"vae/{config.MODEL_FILENAMES['audio_vae']}",
        "size_gb": 0.35
    }
    
    # 5. Restore IC-LoRA
    lora_restore = config.LORAS_DIR / config.MODEL_FILENAMES["lora_restore"]
    status["lora_restore"] = {
        "installed": lora_restore.exists() and lora_restore.stat().st_size > 1024*1024,
        "path": str(lora_restore),
        "target_dir": config.LORAS_DIR,
        "repo": config.HF_REPOS["restore"],
        "repo_subpath": config.MODEL_FILENAMES["lora_restore"],
        "size_gb": 1.6
    }
    
    # 6. Refine Details IC-LoRA
    lora_refine = config.LORAS_DIR / config.MODEL_FILENAMES["lora_refine"]
    status["lora_refine"] = {
        "installed": lora_refine.exists() and lora_refine.stat().st_size > 1024*1024,
        "path": str(lora_refine),
        "target_dir": config.LORAS_DIR,
        "repo": config.HF_REPOS["refine"],
        "repo_subpath": config.MODEL_FILENAMES["lora_refine"],
        "size_gb": 1.25
    }
    
    # 7. Spatial Upscaler
    spatial = config.LATENT_UPSCALE_DIR / config.MODEL_FILENAMES["spatial_upscaler"]
    status["spatial_upscaler"] = {
        "installed": spatial.exists() and spatial.stat().st_size > 1024*1024,
        "path": str(spatial),
        "target_dir": config.LATENT_UPSCALE_DIR,
        "repo": config.HF_REPOS["base"],
        "repo_subpath": f"latent_upscale_models/{config.MODEL_FILENAMES['spatial_upscaler']}",
        "size_gb": 0.93
    }

    # 8. NVIDIA SoL-Refiner Transformer (1-Step Fast Denoise)
    sol_trans = config.DIFFUSION_MODELS_DIR / config.MODEL_FILENAMES["sol_transformer"]
    status["sol_transformer"] = {
        "installed": sol_trans.exists() and sol_trans.stat().st_size > 1024*1024,
        "path": str(sol_trans),
        "target_dir": config.DIFFUSION_MODELS_DIR,
        "repo": config.HF_REPOS["sol_refiner"],
        "repo_subpath": f"diffusion_models/{config.MODEL_FILENAMES['sol_transformer']}",
        "size_gb": 21.90
    }

    # 9. NVIDIA SoL-Refiner Diffusion Decoder VAE
    sol_vae = config.VAE_DIR / config.MODEL_FILENAMES["sol_vae"]
    status["sol_vae"] = {
        "installed": sol_vae.exists() and sol_vae.stat().st_size > 1024*1024,
        "path": str(sol_vae),
        "target_dir": config.VAE_DIR,
        "repo": config.HF_REPOS["sol_refiner"],
        "repo_subpath": f"vae/{config.MODEL_FILENAMES['sol_vae']}",
        "size_gb": 1.37
    }

    # 10. CQ Enhancer LoRA V2
    cq_lora = config.LORAS_DIR / config.MODEL_FILENAMES["lora_cq_enhancer"]
    status["lora_cq_enhancer"] = {
        "installed": cq_lora.exists() and cq_lora.stat().st_size > 1024*1024,
        "path": str(cq_lora),
        "target_dir": config.LORAS_DIR,
        "repo": config.HF_REPOS["cq_enhancer"],
        "repo_subpath": config.MODEL_FILENAMES["lora_cq_enhancer"],
        "size_gb": 1.28
    }

    # 11. Convolution Video VAE
    v_vae_conv = config.VAE_DIR / config.MODEL_FILENAMES["video_vae_conv"]
    status["video_vae_conv"] = {
        "installed": v_vae_conv.exists() and v_vae_conv.stat().st_size > 1024*1024,
        "path": str(v_vae_conv),
        "target_dir": config.VAE_DIR,
        "repo": config.HF_REPOS["base"],
        "repo_subpath": f"vae/{config.MODEL_FILENAMES['video_vae_conv']}",
        "size_gb": 1.45
    }
    
    return status


def print_status_table():
    """Prints a terminal overview of installed vs missing models."""
    status = get_model_status()
    print("=" * 80)
    print(" LTX-2.5 RESTORATION PIPELINE - MODEL STATUS")
    print("=" * 80)
    print(f"{'Component':<20} | {'Status':<12} | {'Size':<9} | {'Filename'}")
    print("-" * 80)
    
    total_missing_gb = 0.0
    for key, item in status.items():
        state = "[INSTALLED]" if item["installed"] else "[MISSING]"
        size_str = f"{item['size_gb']:.2f} GB"
        filename = Path(item["path"]).name
        if not item["installed"]:
            total_missing_gb += item["size_gb"]
        print(f"{key:<20} | {state:<12} | {size_str:<9} | {filename}")
        
    print("-" * 80)
    if total_missing_gb == 0.0:
        print("[SUCCESS] All required models are installed and ready!")
    else:
        print(f"[NOTICE] Total download required: ~{total_missing_gb:.2f} GB")
    print("=" * 80)


RELAYVSR_GIT_URL = "https://github.com/kopperx/RelayVSR.git"
RELAYVSR_WEIGHT_FILES = ["flash.safetensors", "wan.safetensors"]
RELAYVSR_PIP_DEPS = [
    "diffusers==0.37.1", "accelerate>=1.0,<2", "triton-windows>=3.6,<3.7", "einops>=0.8",
    "av>=17,<19", "safetensors>=0.6", "numpy>=1.26", "pillow>=10",
]


def install_relayvsr() -> None:
    """Clones RelayVSR, builds its isolated CUDA 12.8 venv (via uv) and downloads its weights.

    RelayVSR pins torch 2.10, which conflicts with ComfyUI's embedded Python, so it
    gets its own venv under third_party/RelayVSR/.venv and runs as a subprocess.
    """
    import subprocess
    import shutil as _shutil

    if not config.RELAYVSR_DIR.exists():
        config.RELAYVSR_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"[RelayVSR] Cloning {RELAYVSR_GIT_URL}...")
        subprocess.run(["git", "clone", "--depth", "1", RELAYVSR_GIT_URL, str(config.RELAYVSR_DIR)], check=True)

    if not config.RELAYVSR_PYTHON.exists():
        uv = _shutil.which("uv")
        if not uv:
            raise RuntimeError("uv is required to build the RelayVSR venv: pip install uv")
        venv_dir = config.RELAYVSR_DIR / ".venv"
        print("[RelayVSR] Creating Python 3.13 venv...")
        subprocess.run([uv, "venv", "--python", "3.13", str(venv_dir)], check=True)
        py = str(config.RELAYVSR_PYTHON)
        print("[RelayVSR] Installing torch 2.10 (CUDA 12.8)...")
        subprocess.run([uv, "pip", "install", "--python", py, "torch==2.10.0",
                        "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
        print("[RelayVSR] Installing inference dependencies...")
        subprocess.run([uv, "pip", "install", "--python", py, *RELAYVSR_PIP_DEPS], check=True)

    for fname in RELAYVSR_WEIGHT_FILES:
        url = f"https://huggingface.co/{config.HF_REPOS['relayvsr']}/resolve/main/{fname}"
        download_file_with_resume(url, config.RELAYVSR_WEIGHTS_DIR / fname, repo_id=config.HF_REPOS["relayvsr"])
    print(f"[RelayVSR] Ready. Weights: {config.RELAYVSR_WEIGHTS_DIR}")


def get_hf_token() -> Optional[str]:
    """Retrieves Hugging Face token from environment, config.json, or cached credentials."""
    token = os.environ.get("HF_TOKEN")
    if token and token.strip():
        return token.strip()

    cfg_token = config.get_user_setting("hf_token")
    if cfg_token and str(cfg_token).strip():
        return str(cfg_token).strip()

    hf_cache = Path.home() / ".cache" / "huggingface" / "token"
    if hf_cache.exists():
        try:
            tok = hf_cache.read_text(encoding="utf-8").strip()
            if tok:
                return tok
        except Exception:
            pass

    return None


def set_hf_token(token: str) -> None:
    """Saves Hugging Face token to user config and ~/.cache/huggingface/token."""
    token = token.strip()
    config.set_user_setting("hf_token", token)
    try:
        cache_dir = Path.home() / ".cache" / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "token").write_text(token, encoding="utf-8")
    except Exception:
        pass


def prompt_for_hf_token_if_needed(repo_id: str) -> Optional[str]:
    """If downloading from gated repository and no token exists, prompts user."""
    token = get_hf_token()
    if token:
        return token

    if "Lightricks/LTX-2.5" in repo_id:
        print("\n" + "=" * 80)
        print(" [!] HUGGING FACE AUTHENTICATION REQUIRED (Gated Model)")
        print("=" * 80)
        print(" The base foundation model repository 'Lightricks/LTX-2.5' is GATED.")
        print()
        print(" To download the model weights:")
        print("   1. Log in to Hugging Face and visit:")
        print("      https://huggingface.co/Lightricks/LTX-2.5")
        print("      Click 'Agree and Access' (accepts the free LTX community license).")
        print()
        print("   2. Generate a free Read access token at:")
        print("      https://huggingface.co/settings/tokens")
        print("=" * 80)

        try:
            tok_input = input("\nEnter your Hugging Face Token (starts with hf_...): ").strip()
            if tok_input:
                set_hf_token(tok_input)
                print("[OK] Token saved successfully!\n")
                return tok_input
        except Exception:
            pass

    return None


def download_file_with_resume(
    url: str,
    dest_path: Path,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    expected_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, float, str], None]] = None
) -> Path:
    """
    Downloads a large file directly to dest_path with:
    - HTTP Range resumption
    - Real-time rolling speed calculation (MB/s)
    - Progress bar, percentage, and ETA
    - Direct write without duplicate cache
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(dest_path.name + ".downloading")

    # If completed file already exists and has valid size
    if dest_path.exists():
        cur_sz = dest_path.stat().st_size
        if expected_size and cur_sz == expected_size:
            print(f"[SKIP] Already installed: {dest_path.name} ({cur_sz / (1024**3):.2f} GB)")
            return dest_path
        elif not expected_size and cur_sz > 1024 * 1024:
            print(f"[SKIP] Already installed: {dest_path.name}")
            return dest_path

    # Check for existing partial download to resume
    initial_bytes = 0
    if part_path.exists():
        initial_bytes = part_path.stat().st_size

    headers = {
        "User-Agent": "LTX-2.5-Studio-Downloader/1.0"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if initial_bytes > 0:
        headers["Range"] = f"bytes={initial_bytes}-"
        print(f"[RESUME] Found partial download: {initial_bytes / (1024**3):.2f} GB. Resuming from byte {initial_bytes:,}...")

    # Open stream
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=30)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to {url}: {e}")

    # Check response status
    if r.status_code in [401, 403]:
        err_msg = r.headers.get("X-Error-Message") or r.text.strip()
        repo_url = f"https://huggingface.co/{repo_id}" if repo_id else "https://huggingface.co"
        print("\n" + "!" * 80)
        print(" [!] GATED MODEL ACCESS REQUIRED")
        print(f" File: {dest_path.name}")
        if repo_id:
            print(f" Repository: {repo_id}")
        if err_msg:
            print(f" Hugging Face Notice: {err_msg}")
        print()
        print(" Solution (Quick One-Time Acceptance):")
        print(f"   1. Open in your browser: {repo_url}")
        print("   2. Click 'Agree and Access' (accept terms with your HF account).")
        print("   3. Your existing HF_TOKEN will then authorize the download automatically!")
        print("!" * 80 + "\n")
        raise PermissionError(f"License agreement required for {repo_url}")

    if r.status_code == 206:
        # Resumed partial content
        content_range = r.headers.get("content-range", "")
        if "/" in content_range:
            total_bytes = int(content_range.split("/")[-1])
        else:
            total_bytes = initial_bytes + int(r.headers.get("content-length", 0))
        open_mode = "ab"
    elif r.status_code == 200:
        # Full content starting from scratch
        total_bytes = int(r.headers.get("content-length", 0))
        initial_bytes = 0
        open_mode = "wb"
    elif r.status_code == 416:
        # Range not satisfiable (file may already be fully downloaded)
        if part_path.exists() and part_path.stat().st_size > 0:
            part_path.rename(dest_path)
            print(f"[DONE] File already complete: {dest_path.name}")
            return dest_path
        open_mode = "wb"
        initial_bytes = 0
        headers.pop("Range", None)
        r = requests.get(url, headers=headers, stream=True, timeout=30)
        total_bytes = int(r.headers.get("content-length", 0))
    else:
        raise RuntimeError(f"Unexpected HTTP status {r.status_code} from {url}")

    downloaded = initial_bytes
    chunk_size = 1024 * 1024  # 1 MB buffer for high streaming throughput
    rolling_window = deque()
    last_ui_update = time.time()
    start_time = time.time()

    total_display_str = f"{total_bytes / (1024**3):.2f} GB" if total_bytes >= 1024**3 else f"{total_bytes / (1024**2):.1f} MB"

    print(f"Target: {dest_path.name} ({total_display_str})")
    print(f"Folder: {dest_path.parent}")

    with open(part_path, open_mode) as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue

            f.write(chunk)
            chunk_len = len(chunk)
            downloaded += chunk_len
            now = time.time()

            # Record chunk for rolling speed calculation (1.5-second moving window)
            rolling_window.append((now, chunk_len))
            while rolling_window and now - rolling_window[0][0] > 1.5:
                rolling_window.popleft()

            # Update terminal UI every 0.15s
            if now - last_ui_update >= 0.15 or downloaded >= total_bytes:
                last_ui_update = now
                window_bytes = sum(b for t, b in rolling_window)
                window_time = max(0.01, now - rolling_window[0][0]) if len(rolling_window) > 1 else 0.1
                speed_bps = window_bytes / window_time
                speed_mb = speed_bps / (1024 * 1024)

                # ETA
                remaining_bytes = max(0, total_bytes - downloaded)
                eta_sec = remaining_bytes / max(1.0, speed_bps)
                if eta_sec < 3600:
                    eta_str = f"{int(eta_sec // 60):02d}m {int(eta_sec % 60):02d}s"
                else:
                    eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60):02d}m"

                # Progress Bar
                pct = (downloaded / max(1, total_bytes)) * 100.0
                bar_len = 28
                filled = min(bar_len, int(bar_len * downloaded / max(1, total_bytes)))
                bar_str = "=" * filled + (">" if filled < bar_len else "") + " " * (bar_len - filled - (1 if filled < bar_len else 0))

                # Size string
                if total_bytes >= 1024**3:
                    down_str = f"{downloaded / (1024**3):.2f}"
                else:
                    down_str = f"{downloaded / (1024**2):.1f}"

                # Real-time console update
                line = f"\r [{pct:5.1f}%] [{bar_str}] {down_str} / {total_display_str} | {speed_mb:5.1f} MB/s | ETA: {eta_str} "
                sys.stdout.write(line)
                sys.stdout.flush()

                if progress_callback:
                    progress_callback(downloaded, total_bytes, speed_mb, eta_str)

    # Verification and atomic rename
    final_size = part_path.stat().st_size
    if total_bytes > 0 and final_size != total_bytes:
        raise IOError(f"File size mismatch: Expected {total_bytes} bytes, got {final_size} bytes.")

    part_path.rename(dest_path)
    total_time = time.time() - start_time
    avg_speed = (downloaded - initial_bytes) / max(1.0, total_time) / (1024 * 1024)
    print(f"\n[DONE] Installed: {dest_path.name} (Avg: {avg_speed:.1f} MB/s in {int(total_time)}s)\n")
    return dest_path


def download_single_model(component_key: str, precision: str = "int8", hf_token: Optional[str] = None) -> Path:
    """Downloads an individual model file directly from Hugging Face to ComfyUI models."""
    status = get_model_status()
    if component_key not in status:
        raise ValueError(f"Unknown component: {component_key}")

    item = status[component_key]
    target_dir = item["target_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    repo_id = item["repo"]
    filename = item["repo_subpath"]

    # Choose int8 or bf16
    if component_key == "transformer":
        chosen_file = config.MODEL_FILENAMES["transformer_int8"] if precision == "int8" else config.MODEL_FILENAMES["transformer_bf16"]
        filename = f"diffusion_models/{chosen_file}"
        dest_filename = chosen_file
    elif component_key == "text_encoder":
        chosen_file = config.MODEL_FILENAMES["text_encoder_int8"] if precision == "int8" else config.MODEL_FILENAMES["text_encoder_bf16"]
        filename = f"text_encoders/{chosen_file}"
        dest_filename = chosen_file
    else:
        dest_filename = Path(filename).name

    dest_path = target_dir / dest_filename
    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
        print(f"[SKIP] {dest_filename} already installed at {dest_path}")
        return dest_path

    token = hf_token or prompt_for_hf_token_if_needed(repo_id)

    download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    print(f"\n--> Fetching {dest_filename} from {repo_id} (~{item['size_gb']:.1f} GB)...")

    return download_file_with_resume(
        url=download_url,
        dest_path=dest_path,
        repo_id=repo_id,
        token=token
    )


def download_all_models(precision: str = "int8", only_loras: bool = False, only_sol: bool = False, hf_token: Optional[str] = None):
    """Downloads all missing models for the restoration pipeline."""
    status = get_model_status()

    # Prioritize base models (user already accepted license), then LoRAs, or SoL
    if only_sol:
        keys = ["sol_vae", "sol_transformer"]
    elif only_loras:
        keys = ["lora_restore", "lora_refine"]
    else:
        keys = ["transformer", "text_encoder", "spatial_upscaler", "video_vae", "audio_vae", "lora_restore", "lora_refine"]

    pending_gated = []

    for key in keys:
        if not status[key]["installed"]:
            try:
                download_single_model(key, precision=precision, hf_token=hf_token)
            except PermissionError as pe:
                repo_url = f"https://huggingface.co/{status[key]['repo']}"
                pending_gated.append((key, repo_url))
                print(f"[!] Skipped {key} - Gated license acceptance required at {repo_url}")
            except Exception as e:
                print(f"[ERROR] Failed downloading {key}: {e}")
        else:
            print(f"[OK] {key} is already installed.")

    print("\n" + "=" * 80)
    if pending_gated:
        print(" [!] NOTICE: GATED LICENSES PENDING ACCEPTANCE")
        print(" The following models could not be downloaded because one-time license")
        print(" acceptance is required on Hugging Face:")
        for comp, r_url in pending_gated:
            print(f"   • {comp:<16} : {r_url}")
        print()
        print(" To enable them:")
        print("   1. Click each link above in your browser and click 'Agree and Access'.")
        print("   2. Re-run this downloader to fetch them immediately with your HF_TOKEN!")
    else:
        print(" [COMPLETED] Model installation check finished.")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        prec = "int8"
        if "--bf16" in sys.argv:
            prec = "bf16"
        only_loras = "--loras-only" in sys.argv
        only_sol = "--sol" in sys.argv
        
        token_arg = None
        for i, a in enumerate(sys.argv):
            if a == "--token" and i + 1 < len(sys.argv):
                token_arg = sys.argv[i + 1]
                set_hf_token(token_arg)
                
        download_all_models(precision=prec, only_loras=only_loras, only_sol=only_sol, hf_token=token_arg)
    elif len(sys.argv) > 1 and sys.argv[1] == "token":
        if len(sys.argv) > 2:
            set_hf_token(sys.argv[2])
            print("[OK] Hugging Face token saved successfully!")
        else:
            tok = input("Enter your Hugging Face Token (hf_...): ").strip()
            if tok:
                set_hf_token(tok)
                print("[OK] Hugging Face token saved successfully!")
            else:
                print("[!] Cancelled.")
    else:
        print_status_table()
        cur_tok = get_hf_token()
        if cur_tok:
            print(f"Hugging Face Token: [CONFIGURED] ({cur_tok[:7]}...{cur_tok[-4:]})")
        else:
            print("Hugging Face Token: [NOT SET] (Required for gated Lightricks/LTX-2.5 base models)")
            print("Set token via: python model_downloader.py token")
