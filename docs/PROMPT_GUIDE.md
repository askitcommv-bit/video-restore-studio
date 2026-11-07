# LTX-2.5 Prompting Guide for Video Restoration & Refinement 🎨

Guidance derived from the official Lightricks VFX engineering team for `Restore` and `Refine Details` IC-LoRAs.

---

## 1. Golden Rules of Tiled Prompting

Because **Tiled Fusion** processes overlapping spatial tiles on a large canvas (4K / 8K), **every individual tile receives the full prompt**:

> [!WARNING]
> **Do NOT name specific objects, single people, or distinct props in your prompt!**
> If you write *"a man wearing a red hat"*, multiple individual tiles may generate a man wearing a red hat, leading to duplicate figures across the canvas.

### ✅ What to Include in Positive Prompts:
- **Rendering & Style:** Sharpness, clarity, modern digital camera capture, clean film scan.
- **Lighting & Atmosphere:** Balanced daylight illumination, golden hour backlight, soft diffused indoor lighting.
- **Micro-textures:** Skin pore texture, realistic hair strands, wool/linen weave, fine 35mm silver-halide film grain, crisp architectural brick.
- **Color & Palette:** Authentic vintage Technicolor warmth, saturated natural skin tones, muted historical dyes, accurate earth tones.

### ❌ What to Exclude / Put in Negative Prompts:
- **Negative Prompt:** `dirt, scratches, dust specks, emulsion tears, flicker, gate weave, heavy noise, blurry, mushy, watermark, subtitles, low resolution, digital artifacts, oversaturated, modern logos, plastic skin`.

---

## 2. Choosing Between Restore and Refine

| Characteristic | `Restore` IC-LoRA | `Refine Details` IC-LoRA |
|---|---|---|
| **Primary Goal** | Archival cleanup, damage repair, inferring plausible color | Generating fine micro-textures without drifting |
| **Input Quality** | Low-res (540p / 480p), damaged, B&W, analog tape | Clean or upscaled video with already usable color |
| **Look Shift** | **Intentional change** (modern digital finish) | **Faithful to source** (only sharpens micro-details) |
| **Recommended Tile** | `HD` (960×544) | `HD` or `FullHD` (1024×576) |
| **Aspect Ratios** | Preserves 4:3 (e.g. 1440×1056) or 16:9 (1920×1088) | Native 16:9 / 4K / 8K |

---

## 3. Two-Stage Progressive Native Ladder

For severely degraded archival footage that needs 4K delivery:
1. **Stage 1 (Restore):**
   - Canvas: Native source resolution or 1080p (`1920×1088` for 16:9, `1440×1056` for 4:3).
   - Adapter: `ltx-2.5-22b-ic-lora-restore-1.0.safetensors`.
   - Purpose: Clean dirt, eliminate flicker, colorize, and establish composition.
2. **Stage 2 (Refine):**
   - Canvas: 4K (`3840×2176`).
   - Latent Spatial Upscaler: `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`.
   - Adapter: `ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors`.
   - Purpose: Rebuild fine hair, cloth, stone, and skin texture at 4K spec.
