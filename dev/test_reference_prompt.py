import os
import json
import time
from pathlib import Path
import config
from engine.comfy.client import ComfyUIClient
from engine.comfy.graphs import build_sol_refiner_api_prompt

def test_reference_sol():
    client = ComfyUIClient()
    # A prepared clip already in ComfyUI's input folder
    video_filename = os.environ.get("TEST_VIDEO", "sample_archive_540p_prepared.mp4")
    ref_image_filename = "character_reference_man.png"
    output_prefix = "sol_test_reference_man"
    
    # Build standard prompt
    prompt = build_sol_refiner_api_prompt(
        video_filename=video_filename,
        output_prefix=output_prefix,
        seed=42,
        pos_prompt="photorealistic clean sharp restoration of the smiling man in the tree, identical facial likeness, natural daylight, crisp 1080p detail",
        neg_prompt="blurry, mushy, noisy, distortion, artificial smoothing, oversaturated",
        sigma=0.50
    )
    
    # Inject Reference Image nodes:
    # 60: LoadImage
    # 61: VAEEncode
    # 62: ReferenceLatent
    prompt["60"] = {
        "class_type": "LoadImage",
        "inputs": {"image": ref_image_filename},
        "_meta": {"title": "Load Character Reference Image"}
    }
    prompt["61"] = {
        "class_type": "VAEEncode",
        "inputs": {"pixels": ["60", 0], "vae": ["12", 0]},
        "_meta": {"title": "Encode Reference Image to Latent"}
    }
    prompt["62"] = {
        "class_type": "ReferenceLatent",
        "inputs": {"conditioning": ["22", 0], "latent": ["61", 0]},
        "_meta": {"title": "Inject Reference Latent Conditioning"}
    }
    
    # Rewire SamplerCustom positive input to use ReferenceLatent output
    prompt["42"]["inputs"]["positive"] = ["62", 0]
    
    print("\n--- Submitting Reference-Guided SoL-Refiner ---")
    t0 = time.time()
    prompt_id = client.queue_prompt(prompt)
    print(f"Queued prompt ID: {prompt_id}")
    
    def on_prog(step, total, msg):
        print(f"[{step}/{total}] {msg}")
        
    outputs = client.track_execution(prompt_id, on_prog)
    t1 = time.time()
    print(f"Completed in {t1 - t0:.1f}s!")
    print(f"Outputs: {outputs}")

if __name__ == "__main__":
    test_reference_sol()
