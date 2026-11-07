# VRAM Optimization & Performance Guide for RTX 5060 Ti (16 GB) 🚀

This guide explains how to get maximum performance and prevent Out-Of-Memory (OOM) errors on a single 16GB GPU when restoring footage to 4K and 8K.

---

## 1. INT8 Quantization Stack (Recommended)

Running 22-Billion parameter video diffusion models at full BF16 precision requires over 39 GB of VRAM. Lightricks provides official **Comfy INT8 with Convolutional Rotation** (`comfy-int8-convrot`) weights that retain virtually identical visual fidelity while reducing weights footprint by 50%:

| Model Component | BF16 Size | INT8 Size | Recommendation (16GB GPU) |
|---|---|---|---|
| **Distilled Transformer** | 39.1 GB | **20.0 GB** | **Use INT8** |
| **Text Encoder (Gemma 4)** | 24.5 GB | **14.3 GB** | **Use INT8** |
| **Video VAE** | 1.4 GB | 1.4 GB | BF16 |
| **Audio VAE** | 0.35 GB | 0.35 GB | BF16 |
| **Restore / Refine LoRAs** | 1.6 GB | 1.6 GB | BF16 |

ComfyUI dynamically offloads the text encoder from VRAM after conditioning generation, keeping active diffusion VRAM well within 16 GB.

---

## 2. Choosing Tile Sizes in Tiled Fusion

In the `Preprocess → Get Tiling Sizes` node or Web UI:

- **`tile_size: HD` (960×544)**:
  - **Ideal for 16GB VRAM.**
  - Steps through the 4K canvas in overlapping HD chunks.
  - Matches the trained spatial window of the `Restore` IC-LoRA.
- **`tile_size: FullHD` (1024×576)**:
  - Best for `Refine Details` on 4K footage.
  - Highest micro-detail density.
- **`tile_size: qHD` (640×360)**:
  - Ultra-low memory fallback for very long video sequences.

---

## 3. Frame Count Calculation (8n + 1)

LTX-2.5 3D spatio-temporal VAE compresses time by a factor of 8. The input frame count must strictly satisfy:

$$\text{Total Frames} = 8n + 1$$

Common legal frame counts:
- **1.0 second @ 24fps:** $25 \text{ frames } (8 \times 3 + 1)$
- **2.0 seconds @ 24fps:** $49 \text{ frames } (8 \times 6 + 1)$
- **3.0 seconds @ 24fps:** $73 \text{ frames } (8 \times 9 + 1)$
- **4.0 seconds @ 24fps:** $97 \text{ frames } (8 \times 12 + 1)$ (Default in LTX workflows)
- **5.3 seconds @ 24fps:** $129 \text{ frames } (8 \times 16 + 1)$
- **10.6 seconds @ 24fps:** $257 \text{ frames } (8 \times 32 + 1)$

`video_preprocessor.py` automatically trims or aligns any clip to the nearest legal count so you never have to calculate this manually.

---

## 4. Tiled VAE Decoding

Converting the shared latent canvas back to RGB pixels at 4K resolution can spike VRAM.
The workflow incorporates `LTXVTiledVAEDecode`, which tiles the decoding pass independently of the sampling pass to keep peak VRAM under 14 GB.
