import os
import json
import time
from pathlib import Path
import config
from restore_engine import build_sol_refiner_api_prompt, ComfyUIClient

def test_sol_sigma(sigma: float, output_prefix: str):
    client = ComfyUIClient()
    # A prepared clip already in ComfyUI's input folder
    video_filename = os.environ.get("TEST_VIDEO", "sample_archive_540p_prepared.mp4")
    preset = config.PROMPT_PRESETS["archival_general"]
    
    prompt_graph = build_sol_refiner_api_prompt(
        video_filename=video_filename,
        output_prefix=output_prefix,
        seed=42,
        pos_prompt=preset["positive"],
        neg_prompt=preset["negative"],
        sigma=sigma
    )
    
    print(f"\n=======================================================")
    print(f"--- Submitting SoL-Refiner with sigma={sigma} ({output_prefix}) ---")
    print(f"=======================================================")
    t0 = time.time()
    prompt_id = client.queue_prompt(prompt_graph)
    print(f"Queued prompt ID: {prompt_id}")
    
    def on_prog(step, total, msg):
        print(f"[{step}/{total}] {msg}")
        
    outputs = client.track_execution(prompt_id, on_prog)
    t1 = time.time()
    print(f"Completed in {t1 - t0:.1f}s!")
    print(f"Outputs: {outputs}")

if __name__ == "__main__":
    test_sol_sigma(0.40, "sol_test_sigma_040")
    test_sol_sigma(0.55, "sol_test_sigma_055")
