# minimax-h3-image-server

[日本語](README.md) | English

A standalone image server + browser UI that runs
[MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) (33B; text-to-image plus editing
with up to 16 reference images) in **about 21GB of VRAM, at 1.8 s for 1024² and 4.1 s for 2048²**. With W4A8
quantization and text-encoder layer streaming it **also runs on a single 16GB GPU, and with transformer block
streaming on an 8GB-class GPU** (while keeping 1GB of VRAM free: 1.7–2.6 s @1024², 3.4–4.2 s @2048², 6.2–7.0 s for a
2048²-class edit; a setting for PCs with 32GB of RAM is included; outputs are pixel-identical to the fully resident
configuration). **It also runs on a PC with a 6GB GPU and 16GB of RAM (about 11GB free)**: with the transformer and
the text encoder streamed straight from the quantization cache's page cache, only ~2GB of RAM is required; each
generation re-reads 13–20GB from disk, so speed depends on the SSD (**measured on Windows with an RTX 3050 6GB, 16GB
of RAM and a Gen3 NVMe: about 23 s @1024²**; Linux, Gen4 NVMe, RTX 4000 SFF Ada: 10.5 s @1024², 14.4 s @2048²;
[details](#a-pc-with-16gb-of-ram-and-a-6gb-gpu-streaming-the-transformer-from-the-file--page-cache-too)).
Built on diffusers (main). Code is MIT-licensed.

- The original bf16 weights are 66GB (transformer) + 66.7GB (text encoder), which do not fit together even on an 80GB
  GPU. This server keeps the transformer at ~19GB and the text encoder at ~17GB with **a precomputed AdaLN table
  (bit-exact, no approximation) + int8 weight-only quantization + an nf4 text encoder**.
- The default is **3 steps with the pDMD 2-step LoRA merged**: 1.7× faster than 6 steps at nearly the same quality.
- **The configuration is chosen automatically from a VRAM budget** (`H3IMG_VRAM_GB`; by default the smaller of the
  GPU's size and the actual free memory minus a 1GB reserve; on OOM it automatically steps down one level).
- Low-VRAM profiles (`32gb` and below): the transformer becomes W4A8 via comfy-kitchen (~11GB, with faster GEMMs than
  int8); the 51 text-encoder layers are streamed one at a time from host RAM, the unused lm_head is dropped and the
  embedding table stays on the CPU, leaving ~0.3GB resident (bit-identical output). The VAE is placed on the GPU only
  during encode/decode. **Peak ~29GB on a single 32GB GPU, ~13GB on a single 16GB GPU.**
- With `H3IMG_TRANSFORMER_STREAM=1` (automatic in the `8gb` profile) the 50 transformer blocks are also streamed from
  host RAM with prefetching, which brings it down to **the 8GB class (4.2GB actually used)**. Thanks to prefetching,
  the speed is about the same as the 16GB configuration.
- An asynchronous job API and a UI for generation, reference editing, history, download and deletion.

## Contents

1. [VRAM requirements](#vram-requirements)
2. [Measured speed](#measured-speed)
3. [Installation](#installation)
4. [Running and usage](#running-and-usage) (with [launch examples](#launch-examples-gpu-vram-budget-quantization))
5. [Environment variables](#environment-variables)
6. [How the VRAM was reduced and the speed increased](#how-the-vram-was-reduced-and-the-speed-increased)
7. [Writing prompts (the official guidance)](#writing-prompts-the-official-guidance)
8. [Investigation notes](#investigation-notes)
9. [License notes](#license-notes)

## VRAM requirements

| Component | Resident | Notes |
|---|---|---|
| transformer (int8wo + AdaLN table) + VAE | **~19GB** | Peak during generation 20.8GB (1024²) to 21.4GB (2048² edit). 4096² not measured |
| text_encoder (Qwen3-VL-32B, nf4, 51 layers) | **~17GB** | Placed on a second GPU automatically if one exists (`H3IMG_TE_GPU`) |

**The configuration is chosen automatically from a VRAM budget.** The budget is the smaller of `H3IMG_VRAM_GB`
(default `auto` = the GPU's total memory) and "the actual free memory at load time minus a 1GB reserve"
(`H3IMG_VRAM_RESERVE_GB`, kept free for the display and other applications). Setting e.g. `H3IMG_VRAM_GB=16` at
startup runs the 16GB-class configuration even on a large GPU.

| Profile | Configuration | Selected when (budget) | Measured peak (torch) | Required host RAM |
|---|---|---|---|---|
| `48gb` | int8wo, TE resident (fastest) | ≥ 38.7GB (≥ 21.7GB if the TE is on another GPU) | 37–38.4GB | 2GB |
| `32gb` | W4A8, TE resident | ≥ 30.5GB (≥ 13.6GB with a separate TE GPU) | 28.7–30.2GB | 2.5GB |
| `16gb` | W4A8 + TE layer streaming + VAE swap | ≥ 13.3GB | 12.5–13.0GB | 18.3GB (3.5GB with `file`) |
| `8gb` | the above + transformer block streaming | otherwise | 2.4–2.9GB (+1.4GB if the VAE fits on the GPU) | 27.2GB (12.4GB with `file`; 2.3GB if the transformer is `file` too) |

`file` reads the text encoder layer by layer straight from the file instead of pinning it in RAM; it is selected by default on PCs
with less than 48GB of RAM ([details below](#vram-reserve-and-a-setting-for-pcs-with-little-ram)).

- With two GPUs, the TE goes to the second one by default (`H3IMG_TE_GPU`). Choose the GPU by index, e.g.
  `H3IMG_GPU=1`.
- If a CUDA OOM happens during generation (for example another process took VRAM after loading), the server
  **automatically steps down one profile and retries** (`H3IMG_VRAM_FALLBACK=0` disables this).
- Explicit `H3IMG_TRANSFORMER_QUANT` / `H3IMG_TE_STREAM` / `H3IMG_TRANSFORMER_STREAM` override the budget.
- The selected profile is shown as `engine.profile` in `/api/status` and as `profile` in job results.

The W4A8 configurations were verified on an RTX PRO 6000 (sm_120) with its free memory capped at 23.5GB / 15.5GB /
7.3GB (on real cards, the `8gb` profile has been verified on an RTX 4000 SFF Ada 20GB (sm_89, free memory capped at
6GB) and on a **Windows RTX 3050 6GB (sm_86)**; real 12GB / 24GB cards are untested).

Host RAM for building the cache: bf16 is handled one tensor at a time, so no large amount of RAM is needed
(peak +a few GB). For disk space see [Installation](#installation).

## Measured speed

RTX PRO 6000 Blackwell (96GB, transformer) + RTX PRO 5000 Blackwell (48GB, text_encoder), fixed seed, generation
time only (model already loaded).

| Case | **3 steps (default)** | 6 steps | 2 steps |
|---|---|---|---|
| T2I 1024² (generated at the requested size) | **1.5–1.8 s** | 2.4 s | 0.9–1.5 s |
| T2I 2048² | **4.1 s** | 7.0 s | 3.0 s |
| Reference edit 1024² (1 reference) | **4.5 s** | 7.4 s | 3.4 s |
| Reference edit 2048² (1 reference) | **6.7 s** | 11.3 s | 5.0 s |
| Peak VRAM (GPU0) | 20.8–21.4GB | same | same |

W4A8 configurations (`ck-w4a8`, 3 steps, TE on the same GPU, TE weights pinned):

| Case | 32GB (TE resident) | 16GB (TE streaming + VAE swap) | 8GB (+ block streaming) |
|---|---|---|---|
| T2I 1024² | 0.7 s | 1.2 s | 1.8 s |
| T2I 2048² | 2.8 s | 3.4 s | 3.5 s |
| Reference edit 1024×1536 (1 reference) | — | 5.0 s | — |
| Reference edit 1664×2496 (1 reference) | 5.6 s | 6.0–6.3 s | 6.1–6.5 s |
| T2I 2048², 6 steps | — | 6.1 s (measured at a 24GB-equivalent cap) | — |
| Peak VRAM (torch) | 28.7–30.2GB | 12.5–13.1GB | 2.4–2.9GB (process max 4.2GB) |

W4A8 has **faster GEMMs than int8wo** (2048²: 4.1 s → 2.8 s), so the cost of TE streaming is absorbed. In side-by-side
visual comparisons with int8wo at the same seed, quality looked equivalent (fine details shift as if the seed changed;
see method 6 [below](#how-the-vram-was-reduced-and-the-speed-increased)).

- Load (on the first generation): text_encoder 3–8 s + transformer cache 6–14 s + VAE 1 s ≈ 20 s. Only the very
  first run without a cache takes about 40 s more (quantizing from the bf16 shards).
- Quality: 3 steps look nearly identical to 6 steps at 100% zoom. **2 steps show a fine woven pattern**, so use them
  for previews. However, **at 1024² with a short prompt such as "mountain and river", 3 steps can also show a grid-like
  grain on mountainsides and water** (independent of the quantization/streaming configuration: the 8gb and 32gb
  profiles are pixel-identical and int8wo on 48gb shows the same; 6 steps removes it, also on a Windows RTX 3050, where
  1024² goes from 23 s to 42.5 s). For landscapes with large smooth areas, use 6 steps or a detailed prompt written the
  official way.
- With `generation_area: "native"` (generate on the officially recommended minimum 4MP canvas, then downscale), 1024²
  takes as long as 2048².
- For comparison: loading the same weights plainly as bnb int8 (`load_in_8bit` in `from_pretrained`) gives
  3.5 s @1024² / 9.7 s @2048², peak 33–34GB.

## Installation

### Requirements

- NVIDIA GPU (see above). Verified on sm_120 (Blackwell). torch 2.11.0+cu130 requires a CUDA 13 driver.
- Python 3.12, [uv](https://github.com/astral-sh/uv) (pip also works).
- Disk (resident): the small parts of the original model (VAE etc.) 2.5GB + LoRA 1.4GB + quantized text encoder 17GB +
  transformer cache 19GiB (int8wo) or 10.7GiB (ck-w4a8) per step count you use = **about 32GB with ck-w4a8**.
  The bf16 sources (text encoder 66.7GB, transformer 62GB) are **downloaded one 5GB shard at a time, quantized, and
  deleted**, so the peak during preparation is the resident size + about 5GB (`--keep-source` keeps everything,
  which avoids re-downloading when you later build a cache for another step count).
- Model preparation (`prepare_models.py`) also runs with little VRAM: the text encoder is quantized to nf4 layer by
  layer (0.3GB of VRAM, 1GB of RAM) and the transformer quantization cache is built with 2GB of VRAM and 3GB of RAM
  even on the 6GB-class profile (measured under a 12GB RAM cap). **A PC with a 6GB GPU and 16GB of RAM can go from
  download to generation on its own** (disk space permitting).

### Steps

```bash
git clone <this repo> minimax-h3-image-server && cd minimax-h3-image-server
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# without uv: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Prepare the models (first time only: downloads the 64GB model + LoRA and quantizes the text encoder; needs a GPU, <1GB of VRAM)
.venv/bin/python scripts/prepare_models.py

# Optional: build the transformer quantization cache in advance (otherwise built on the first generation)
.venv/bin/python scripts/prepare_models.py --skip-download --build-cache 3
```

If another installation (e.g. diffusers-movie-server) already has the models, reference them with symlinks instead of
downloading:

```bash
.venv/bin/python scripts/prepare_models.py --skip-download \
  --link-te /path/to/te_bnb-4bit_prune1 --link-cache /path/to/cache --link-lora /path/to/lora
```

The contents of `models/` (`te/`, `lora/`, `cache/`) may be symlinks or real directories.

### Windows

Windows wheels exist for all dependencies (torch 2.11+cu130, comfy-kitchen, bitsandbytes, torchao; use `triton-windows`
for Triton). The only OS-specific code is reading RAM figures and page-cache hints (`madvise` / `fadvise`); on Windows the
former uses Win32 APIs and the latter is replaced by `FILE_FLAG_SEQUENTIAL_SCAN`.
**Verified on a real machine**: RTX 3050 6GB, 16GB of RAM (~11GB free), Gen3 DRAM-less NVMe (Samsung PM991a). The `8gb`
profile (W4A8, TE and transformer both `file`) is selected automatically; 1024² takes about 23 s, of which reading the
13GB text encoder from disk is ~10 s (1.3GB/s). Output matches Linux in content (not pixel-identical, different GPU
arithmetic). Write prompts in English (Japanese prompts may be ignored; the UI's "enhance" button converts them).

With uv (install it with `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` and
reopen PowerShell afterwards):

```powershell
git clone <this repo> minimax-h3-image-server; cd minimax-h3-image-server
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-windows.txt
.venv\Scripts\python.exe scripts\prepare_models.py     # prepare the models (needs a GPU, <1GB of VRAM)
.\run.ps1                                                                 # http://127.0.0.1:8633/
```

Without uv (the standard venv + pip work the same, as long as Python 3.12 is installed):

```powershell
git clone <this repo> minimax-h3-image-server; cd minimax-h3-image-server
py -3.12 -m venv .venv                      # or python -m venv .venv if `py` is unavailable (must be 3.12)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
.venv\Scripts\python.exe scripts\prepare_models.py
.\run.ps1
```

- An NVIDIA driver supporting CUDA 13 is required.
- On PCs with little RAM the weights are streamed with the `file` method on both Linux and Windows (read directly from
  the file, no page cache). The `mmap` method (page cache + `madvise`) is Linux-only.
- If PowerShell refuses to run `.\run.ps1` because of the execution policy, start it with
  `powershell -ExecutionPolicy Bypass -File .\run.ps1`.
- If your torch build does not support `expandable_segments`, it only warns and continues.

### Hugging Face access

The official repository is public, but some environments need `HF_TOKEN`. `scripts/prepare_models.py` uses the usual
`huggingface_hub` cache (`~/.cache/huggingface/hub`).
If a download stalls with no network traffic (this happened with the xet transfer mode of recent `huggingface_hub`),
rerun with `HF_HUB_DISABLE_XET=1` to use plain HTTP (resumable).

## Running and usage

```bash
./run.sh                        # http://127.0.0.1:8633/ (listens on localhost only by default)
H3IMG_HOST=0.0.0.0 ./run.sh     # expose on the LAN (no authentication, so only on a trusted network)
H3IMG_PORT=8634 ./run.sh        # change the port
H3IMG_STEPS=6 ./run.sh          # make 6 steps the default
```

### Launch examples (GPU, VRAM budget, quantization)

Without any options, the configuration (profile) is chosen from the GPU's size and its actual free memory. If another
process takes VRAM after loading and generation hits an OOM, the server steps down one profile and retries.

```bash
# Automatic (default): budget = min(GPU0 total, free - 1GB reserve); picks 48gb / 32gb / 16gb / 8gb
./run.sh

# Choose the GPU by index (transformer on GPU1, TE on the same GPU)
H3IMG_GPU=1 H3IMG_TE_GPU=1 ./run.sh

# Two GPUs: transformer on GPU0, TE on GPU1 (the default when two GPUs exist)
H3IMG_GPU=0 H3IMG_TE_GPU=1 ./run.sh

# Set the budget: run the 16GB-class configuration (W4A8 + TE streaming) even on a 96GB GPU
H3IMG_VRAM_GB=16 ./run.sh

# Force the 8GB-class configuration (required host RAM ~27GB; on PCs with <48GB RAM it switches to file, ~2GB + cache)
H3IMG_VRAM_GB=8 ./run.sh

# Set only the quantization (overrides the budget): W4A8 with the TE resident (the 32GB-class configuration)
H3IMG_TRANSFORMER_QUANT=ck-w4a8 H3IMG_TE_STREAM=0 ./run.sh

# Disable the automatic step-down on OOM (return the OOM as an error)
H3IMG_VRAM_FALLBACK=0 ./run.sh
```

| Option | Values | Meaning |
|---|---|---|
| `H3IMG_GPU` / `H3IMG_TE_GPU` | GPU index | GPU for the transformer / text encoder. Without `H3IMG_TE_GPU`, a second GPU is used if present |
| `H3IMG_VRAM_GB` | `auto` / number | The budget. `auto` = the GPU's total memory; the smaller of this and (free − reserve) selects the profile below |
| `H3IMG_TRANSFORMER_QUANT` | `auto` / `int8wo` / `ck-w4a8` | Transformer quantization (19GB / 11GB) |
| `H3IMG_TE_STREAM` | `auto` / `0` / `1` | Text-encoder layer streaming (resident 17GB → 0.3GB) |
| `H3IMG_TRANSFORMER_STREAM` | `auto` / `0` / `1` | Transformer block streaming (resident 11GB → 1GB) |
| `H3IMG_VRAM_FALLBACK` | `1` / `0` | Step down one profile automatically on OOM during generation |
| `H3IMG_VRAM_RESERVE_GB` | number (default 1.0) | VRAM that is always left free (for the display etc.) |
| `H3IMG_TE_HOST` | `auto` / `pinned` / `file` / `mmap` | Where the text encoder's weights live (`file` is for PCs with ≤32GB of RAM) |
| `H3IMG_TRANSFORMER_HOST` | `auto` / `pinned` / `file` / `mmap` | Where the transformer blocks live (`file` is for PCs with 16GB of RAM; speed then depends on the SSD) |

| Profile | Quantization | TE | transformer | Budget needed | Speed (1024² / 2048², TE pinned) |
|---|---|---|---|---|---|
| `48gb` | int8wo | resident | resident | 38.7GB (21.7GB with a separate TE GPU) | 1.8 s / 4.1 s |
| `32gb` | W4A8 | resident | resident | 30.5GB (13.6GB with a separate TE GPU) | 0.7 s / 2.8 s |
| `16gb` | W4A8 | streamed | resident | 13.3GB | 1.2 s / 3.4 s |
| `8gb` | W4A8 | streamed | streamed | otherwise | 1.8 s / 3.5 s |

The selected profile appears in the startup log (`VRAM profile: ...`), in `/api/status` as `engine.profile`, and in
job results as `profile`. Each `auto` setting follows the budget; explicitly set values take precedence.

Browser UI: mode (t2i / reference edit), prompt, size, steps, canvas, seed, reference images (up to 16), history,
download, delete. It follows the OS light/dark setting.

API (asynchronous jobs):

| Method | Path | Description |
|---|---|---|
| POST | `/api/assets` | Reference image (PNG/JPG/WebP/BMP, multipart `file`) → `{id, kind, filename, size}` |
| POST | `/api/jobs` | Generation job (202). Body below |
| GET | `/api/jobs?limit=N` / `/api/jobs/{id}` | History / status (`queued/running/completed/failed/interrupted`, `image_url`) |
| DELETE | `/api/jobs/{id}` / `/api/outputs/{file}` | Delete history (interrupts a running job) / delete an output |
| POST | `/api/prompts/enhance` | Prompt enhancement (see below) |
| POST | `/api/interrupt` | `{job_id?}` interrupt |
| POST | `/api/admin/unload` / `/api/admin/load` | Free VRAM (keep the process) / explicit load (`{steps}`) |
| GET | `/api/health` / `/api/status` | Status |

```json
{"mode": "t2i" | "edit", "prompt": "...", "width": 2048, "height": 2048, "seed": 42,
 "steps": 2 | 3 | 6, "generation_area": "native" | "requested",
 "conditions": [{"asset_id": "..."}]}
```

- In `edit`, refer to the images in the prompt as `<Picture 1>`, `<Picture 2>`, ... (the official convention; long,
  specific instructions work well).
- `width` / `height` are rounded to multiples of 32. The upper limit is `H3IMG_MAX_PIXELS` (default 4096²).
- Changing `steps` in a request swaps the transformer (cache load ~12 s).

## Writing prompts (the official guidance)

The model expects **a detailed English description of the whole image**. On 2026-11-11 the official GitHub published a
[prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3-Image/tree/main/skills/h3-image-prompt-writing)
(`SKILL.md` + `references/t2i-en.txt` / `edit-en.txt`, intended for agents such as Claude Code) that collects the rules
and complete examples. A self-hosted deployment like this one, unlike the official API, does not insert an
enhancement step, so a prompt written with these rules is exactly what the model receives. A summary:

**General rules**
- **Task before style**: what the image is and what it is for → the main subjects and their state or action → style
  last.
- **Layout before details**: give the number of subjects, their positions, the viewpoint and the empty space first. For
  dense content, name the regions first and then fill each one ("a title across the top; below it, a photo on the
  left and three short paragraphs on the right").
- **Style through visible details**: instead of "premium" or "cinematic", write color, material, light and texture
  ("soft side light, brushed-metal reflections, generous white space").
- **Quote every word that appears**: keep its language and spacing, and give its position, size hierarchy and line
  breaks. When you want the model to have copy, write the actual words.
- **Say when there is no text**: in text-to-image, add "The image contains no text, letters or numbers." (prevents
  invented titles and decorative lettering). In an edit, say that no other lettering is added, which keeps the source
  image's text.
- **Get data right**: for charts, give values, units, relationships and time ranges. Mark example data as
  illustrative and never present invented numbers as real.
- **Grids by rows and columns**: "2 rows and 3 columns" / "3×3 (rows × columns)". Describe irregular layouts directly
  ("3 panels on top and 2 below"); address a cell as "row 2, column 1".

**Reference-image editing (edit)**
- Refer to the inputs as `<Picture 1>`, `<Picture 2>`, ... in upload order.
- One paragraph in the order **definitions → the target and each source's role → changed details**. Describe the
  finished state rather than giving commands ("The former location is filled by continuous tabletop").
- Definitions: `The red cup on the left in <Picture 1> is <Subject 1>.` — one `<Subject N>` per scope that needs its
  own relationship. Define a color-only donor as a property and say that its shape and material are not borrowed.
- Roles: start the target with `The target image …` and give each Subject one relationship. Full preservation uses
  the fixed sentence `<Subject N> is fully preserved.`; partial preservation says what remains and what changes;
  property transfer reads `<Subject 2> supplies the exact body color transferred to <Subject 1>.`; a weak reference
  names the borrowed quality and what may vary. Moving, rotating, duplicating and relighting all count as changes.
- Changed details: for a removal, say what fills the space. Distinguish moving from copying. For text changes, give
  the complete new line. Do not re-describe what is preserved.
- Multi-round editing: one goal per round. Instead of "the one from before", name the image used this round, what
  changes and what stays.
- Official example (a local color change):
  > The red cup on the left in <Picture 1> is <Subject 1>. The lemon on the right in <Picture 1> is <Subject 2>.
  > The target image shows <Subject 1> with a blue finish. <Subject 1> is partially preserved; its shape, material,
  > position, and scale remain while its surface color changes. <Subject 2> is fully preserved. …

**Final check (from the skill)**
Every word that appears is quoted and spelled as given / counts, positions, left-right, foreground-background and
contact match the request / values, units and labels are complete / each reference has a role and what is preserved
is named / the aspect ratio fits the layout / no vague quality word remains where a visible detail could say it.

Aspect ratios range from 1:3 to 3:1. The official API recommends ~16MP; this server defaults to 4MP (2048²-equivalent).

**Prompt enhancement (a feature of this server)**
The UI's "enhance" button / `POST /api/prompts/enhance` sends the official rules bundled with the prompt-enhancement
model [MiniMaxAI/MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE)
(`prompt/t2i/task_prompt.txt` / `prompt/edit/task_prompt.txt`, used by the official API's enhancement step) as the
system prompt, plus the skill's guide text (`references/t2i-en.txt` / `edit-en.txt`, disable with
`H3IMG_ENHANCE_SKILL_GUIDE=0`), to an OpenAI-compatible LLM (e.g. a Gemma model on llama.cpp server). A short
instruction (Japanese works too) becomes an English prompt that follows the official rules, plus a recommended aspect
ratio. This avoids loading the PE model itself (~19GB in FP4), at the cost of possibly lower accuracy than the official
PE. In edit mode the reference images are sent to the LLM too (a multimodal LLM is required). Configure with
`H3IMG_LLM_BASE_URL` / `H3IMG_LLM_MODEL` / `H3IMG_LLM_API_KEY` (environment variables or `.env`). The rule documents
are fetched into `models/prompts/` from the PE repository and GitHub on first use.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `H3IMG_HOST` / `H3IMG_PORT` | 127.0.0.1 / 8633 | Listen address and port |
| `H3IMG_STEPS` | 3 | Default step count (2 / 3 / 6) |
| `H3IMG_GPU` / `H3IMG_TE_GPU` | 0 / (1 if two GPUs) | GPU index for the transformer / text encoder (`H3IMG_DEVICE=cuda:N` / `H3IMG_TE_DEVICE` also work) |
| `H3IMG_VRAM_GB` | auto | VRAM budget (GB). auto = the GPU's total memory; the smaller of this and (free − reserve) selects the profile |
| `H3IMG_VRAM_RESERVE_GB` | 1.0 | VRAM always left free (for the display and other applications) |
| `H3IMG_VRAM_FALLBACK` | 1 | Step down one profile and retry on a CUDA OOM during generation |
| `H3IMG_TRANSFORMER_QUANT` | auto | Transformer quantization (`int8wo` 19GiB / `ck-w4a8` ~11GB). auto follows the profile |
| `H3IMG_TE_STREAM` | auto | Text-encoder layer streaming (with lm_head removal, CPU embedding and VAE swap). Set 0 / 1 explicitly |
| `H3IMG_TRANSFORMER_STREAM` | auto | Streaming of the 50 transformer blocks (8GB class). Set 0 / 1 explicitly |
| `H3IMG_TE_HOST` | auto | Text-encoder weights: `pinned` (fast, ~13GB of RAM) / `file` (each layer read straight from the file; for PCs with little RAM, any OS) / `mmap` (page cache + madvise, Linux only). auto = pinned with ≥48GB of RAM, otherwise file |
| `H3IMG_TRANSFORMER_HOST` | auto | Transformer blocks (8gb profile): `pinned` (~10GB of RAM) / `file` or `mmap` (streamed from the quantization cache's page cache, overflow re-read from disk; for PCs with 16GB of RAM). auto follows the TE rule |
| `H3IMG_TRANSFORMER_DROP_BLOCKS` | auto | With `file` / `mmap`, how many trailing blocks to re-read (file) or drop (mmap) after use. auto = only the overflow that does not fit in RAM; 0 = never |
| `H3IMG_KEEP_SOURCE` | 0 | Set to 1 to keep the transformer bf16 shards downloaded one by one during the cache build (files that were already present are never deleted either way) |
| `H3IMG_GENERATION_AREA` | requested | Default canvas (`native` = at least 4MP) |
| `H3IMG_PRELOAD` | 0 | 1 = load at startup |
| `H3IMG_MODELS_DIR` | `./models` | Location of te / lora / cache |
| `H3IMG_TE_DIR` / `H3IMG_LORA_PATH` / `H3IMG_CACHE_DIR` / `H3IMG_MODEL_DIR` | under the above / HF cache | Individual overrides |
| `H3IMG_MAX_PIXELS` | 4096² | Maximum pixels per request |
| `H3IMG_MAX_QUEUE` / `H3IMG_MAX_UPLOAD_MB` | 4 / 200 | Queue length / upload limit |
| `H3IMG_LLM_BASE_URL` / `H3IMG_LLM_MODEL` / `H3IMG_LLM_API_KEY` | (unset = enhancement disabled) | OpenAI-compatible LLM for prompt enhancement (may also go in `.env`) |
| `H3IMG_LLM_TIMEOUT_S` | 180 | LLM timeout |
| `H3IMG_ENHANCE_SKILL_GUIDE` | 1 | Also send the official prompt-writing skill's guide text when enhancing |

## How the VRAM was reduced and the speed increased

The original weights are 66GB (transformer, bf16) + 66.7GB (text encoder, bf16), which do not fit together even on an
80GB GPU. Methods 1–5 bring this to ~19GB (transformer side) + ~17GB (text encoder) and make 6 steps 1.5× faster and
3 steps more than 2× faster. Methods 6–8 add single-GPU 32GB / 16GB / 8GB configurations. All numbers are
measurements (RTX PRO 6000, 1024² / 2048²).

### The overall idea (from 133GB to 4GB: what goes where)

There are three ways to reduce VRAM: **make it smaller**, **don't keep it on the GPU**, and **put it on the GPU only
at the moment it is used**. This server applies them per component. The key point of view is that **the GPU is a
place for computation, not for storage**.

| Component | Original | Method | Result |
|---|---|---|---|
| AdaLN projections (39% of the transformer) | 26GB | The input is fixed (the sampling timesteps), so **precompute the outputs** and drop the weights (1) | 68MB, zero error |
| The remaining 312 Linear layers | 40GB | **Make smaller**: int8 (2) → W4A8 (6) | 19GB → 11GB |
| transformer blocks | 11GB | **Only when used**: prefetch the 50 blocks from host RAM (8) | 1GB resident on the GPU |
| text_encoder (Qwen3-VL-32B) | 66.7GB | **Make smaller**: nf4 + cut the 13 unused layers (3) → **Don't keep**: drop the unused lm_head, look up the embedding table on the CPU (7) | 17GB → 0.3GB |
| The text encoder's 51 layers | 11.6GB | **Only when used**: stream one layer at a time (7) | 0 resident |
| VAE | 1.4GB | **Only when used**: on the GPU only during encode / decode (7) | 0 resident |

"Only when used" does not cost speed for two reasons. The text encoder runs once per prompt, so streaming its layers
adds about a second per generation. The transformer runs all 50 blocks every step, but if block i+1 is copied on a
separate CUDA stream while block i computes (prefetching), the PCIe transfer hides behind the computation. As a
result the 8GB configuration runs at the same speed as the 16GB one, with pixel-identical output.

How far each method is used is decided automatically from the VRAM budget
([launch examples](#launch-examples-gpu-vram-budget-quantization)). What is not kept on the GPU stays in host RAM, so
the 8GB configuration uses about 29GB of host RAM (about 12GB required with the `file` setting, which reads the text
encoder in the page cache).

### 1. Replace the AdaLN projections with a precomputed table (VRAM −26GB, zero error)

Each transformer block has a linear layer `adaln_proj` that turns the timestep embedding (2688 dims) into modulation
parameters for 6 kinds × 3 modalities (96768 dims). At 520MB × 50 layers = **26GB, it is 39% of all weights**, but its
input is **a function of the timestep only** and does not depend on the tokens.

MiniMax-H3-Image has fixed sampling timesteps (a 6-step distilled model; the σ schedule is stored in the checkpoint and
each step has its own output head). The timestep used for reference-image rows is a single value too. So each block's
`adaln_proj` only ever sees **7 different inputs**: compute those 7 outputs in advance, replace the layer with a
`[7, 96768]` table (68MB in total), and just look up a row during generation.

- The outputs are computed with the same operation as the original forward (`F.linear(silu(temb).to(bf16), W, b)`) on
  the same device and dtype, so **the results are bit-identical** (not an approximation). This is checked on block 0
  with `torch.equal`.
- The compute barely changes (it was a small 7-row × 50-layer matmul), so **the benefit is VRAM and load time**.
- Each cache has a fixed set of timesteps. Steps 2 / 3 (other σ schedules) use separate tables and caches; an
  unknown timestep raises an exception instead of silently giving wrong results.
- Community versions apply a similar idea as a rank-8 low-rank approximation ("AdaLN pruned"), but with fixed
  timesteps there is no need to approximate: an exact table suffices. The idea came from handling the same structure
  in the video version of H3.

### 2. int8 weight-only quantization for the remaining Linear layers (VRAM 40GB → 19GB, 1.5× faster)

The 312 attention and FF Linear layers (50 blocks × 6 + token refiner 2 × 6) are quantized with torchao's
`Int8WeightOnlyConfig(PerRow)`. Layers that must stay in fp32 (input projection, timestep embedding, output heads,
RoPE) are left alone, following the model's own settings.

- The bf16 shards (66GB) are moved to the GPU **one tensor at a time** and quantized there, so bf16 is never fully
  materialized on the CPU or GPU (peak during load = resident + 1.5GB; a few GB of host RAM).
- The quantized weights are saved in `models/cache/` and load in about 12 s afterwards.
- In single-layer benchmarks it matched bnb's LLM.int8, but for the whole model **int8wo is 1.6× faster**
  (1024²: bnb 3.5 s → 2.4 s); bnb's outlier decomposition overhead adds up.
- Pitfall: without a dtype, `accelerate.set_module_tensor_to_device` casts to the meta model's default dtype (fp32).
  Norms and others became fp32, all activations flowed in fp32, and it was initially 2× slower.

### 3. Prune the text encoder to nf4 + 51 layers (66.7GB → 17GB)

The Qwen3-VL-32B used for conditioning has the same weights as MiniMax-H3 (video), and the model reads only
`hidden_states[50]` (the output of the 51st layer). So it is loaded with `num_hidden_layers` set to 51 in the config
(layers 51–63 and lm_head are discarded), quantized to bitsandbytes nf4, and saved.
The quantization does not go through `from_pretrained(quantization_config=...)`: the bf16 shards are read from the
mmap one tensor at a time, quantized on the GPU with `Params4bit`, and written as one safetensors file per layer
(`app/te_quant.py`). The former loads the whole model onto the GPU and needs 17GB of VRAM; the latter needs 0.3GB of
VRAM and 1GB of RAM, and the tensor values are bit-identical (verified).

- With 50 layers the value would be taken after the final norm and the conditioning would change, so **51 layers is
  correct**.
- Encoding takes only 0.1–0.5 s per call, so speed is essentially unaffected.
- With a second GPU, only the text encoder goes there and only `hidden_states[50]` is passed to the transformer side
  through the host (in one environment a direct GPU-to-GPU P2P copy silently returned all zeros, so direct copies are
  not used).

### 4. pDMD 2-step LoRA: 6 steps → 3 steps (1.7× faster, nearly the same quality)

[akhaliq/minimax-h3-image-pdmd-2step](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step) (rank 128,
Apache-2.0) is merged in fp32 into the bf16 weights before quantization and run with the shift-12 σ schedule
(3 steps: `[1.0, 0.96, 0.857, 0.0]`).

- 70–90% of the time is denoising, so cutting steps helps the most (2048²: 7.0 s → 4.1 s).
- **2 steps show a fine woven pattern at 100% zoom** and flatten skin, so they are for previews. **With 3 steps the
  grid disappears and hair strands and knit stitches appear, nearly matching 6 steps**, so 3 steps is the default.
- The LoRA does not touch AdaLN, so it is independent of the table in 1., but the σ schedule changes, so the table
  (cache) is separate.

### 5. Generate 1024² at the requested size (2–3× faster)

The official implementation generates on a minimum 4MP (2048²-equivalent) canvas and then downscales, so asking for
1024² takes as long as 2048². With `generation_area: "requested"` (the default) the image is generated at the
requested size and 1024² goes from 3.5 s to 1.5 s. Quality is good, but use `"native"` if you want the officially
recommended quality.

### 6. W4A8 (4-bit weights, int8 activations) brings the transformer to ~11GB (for the 32GB class, 1.5× faster GEMM)

`H3IMG_TRANSFORMER_QUANT=ck-w4a8` (automatic in the `32gb` profile and below). Using
[comfy-kitchen](https://pypi.org/project/comfy-kitchen/) (Apache-2.0, Triton/CUDA kernels; the implementation behind
ComfyUI's W4A8), the 312 layers from 2. become "4-bit weights (group-16 scales + a 16-value codebook), int8 activations
(with rotation)". The 19GiB of int8wo becomes 10.7GiB; the GEMM alone is 1.5× faster than bf16 and faster than int8wo
(2048²: 4.1 s → 2.8 s). The pDMD LoRA is merged in fp32 before quantization (as in 2.). Caches are
`models/cache/ckw4a8_adaln_table*`.

- Quality: equivalent in side-by-side visual comparisons with int8wo at the same seed (skin, hair and signage text
  all intact). However, with a 3-step distilled model the quantization error shifts the composition as if the seed
  changed, so pixel equality is not expected.
- Verified on sm_120 (Blackwell) only. The kernels target Ampere and newer, but other generations are unconfirmed.

### 7. Stream the text encoder's 51 layers one at a time (resident 17GB → 0.3GB, for the 16GB class)

`H3IMG_TE_STREAM=1` (automatic in the `16gb` profile and below). The 51 nf4 decoder layers are loaded on the CPU from
the start (loading them on the GPU first and moving them out would exceed 16GB at load time) and streamed to the GPU
one layer (~0.3GB) at a time via forward hooks, from host RAM. The bitsandbytes nf4 quantization state (absmax / code)
is streamed together with the weights, so **hidden_states[50] is bit-identical to the fully resident case** (measured
maximum error 0.0). In addition:

- **Drop the lm_head (1.45GB)**: H3-Image reads only hidden_states[50], so it is never used.
- **Keep the embedding table (bf16, 1.45GB) on the CPU**: only the looked-up rows are sent to the GPU (a few ms).
- What stays on the GPU is the vision tower, norm and RoPE, ~0.3GB.
- **The VAE goes to the GPU only during encode / decode**: in edits with reference images the text encoder uses
  activations for the image tokens, and with the VAE resident the 16GB class ran out of memory during text encoding
  (measured). The round trip takes 0.1–0.2 s.

The prompt passes through only once, so streaming costs less than a second per generation. The 16GB configuration
renders 2048² in 3.4 s, faster than int8wo's 4.1 s (48GB configuration) thanks to W4A8's faster GEMMs. Host RAM used
by the process is about 19GB.

### 8. Stream the 50 transformer blocks with prefetching (resident 11GB → 1GB, for the 8GB class)

`H3IMG_TRANSFORMER_STREAM=1` (automatic in the `8gb` profile). The W4A8 transformer_blocks (~0.2GB each) are kept in
pinned host RAM, and forward hooks copy block i+1 on a separate CUDA stream while block i computes. What stays on the
GPU is the token refiner, input/output projections and so on, ~1GB. The copies hide behind the computation, so
**the speed is about the same as the 16GB configuration** (2048² 3.5 s), and the output is **pixel-identical** to the
resident case. Host RAM used by the process is about 29GB (below).

- Weights never change, so nothing is copied back from GPU to CPU. The pre-hook assigns the tensor copied to the GPU
  to `param.data`, and the post-hook just switches back to the pinned master.
- Pitfall: the prefetch runs on another stream, so if the GPU tensor is released without
  `record_stream(compute stream)`, the allocator reuses that memory for the next prefetch before the computation has
  finished, and **the output turns completely black** (measured).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (default in run.sh) keeps the reserved memory from growing
  (process usage 6.8GB → 4.2GB).

### Host RAM: pinning without round-up, and streaming without copy-back

Whatever leaves the GPU is kept in host RAM. The first implementation used **51GB** in the 8gb configuration. There
were three causes, all fixed (`pin_module` / `stream_layers` in `app/pruned_image.py`):

1. **`Tensor.pin_memory()` rounds up**: PyTorch's pinned-memory allocator rounds allocation sizes up (33.1GB for 24.6GB
   of tensors). Now each layer/block gets one anonymous buffer, pinned in place with `cudaHostRegister`, and each tensor
   is replaced by a view into it. No rounding occurs. The registration is removed before unloading.
2. **The text encoder was round-tripped with `.to()`**: `mod.to("cpu")` creates a new, unpinned CPU tensor every time,
   so transfers after the first one were slower too. It now uses the same scheme as the transformer: keep the pinned
   master and simply drop the GPU copy. The bitsandbytes quantization state (absmax / code) is streamed along with it.
3. **The safetensors mmap was still mapped**: once the text encoder's weights move to the pinned masters, the mmap
   loses its references and is released (the embedding table is copied to detach it as well). The quantization cache is
   loaded with `torch.load(mmap=True)` so it is never fully materialized in host RAM.

Process memory (RSS) after generating 2048²:

| Profile | Before | After |
|---|---|---|
| `48gb` | 2.1GB | 2.2GB |
| `16gb` | 32–35GB | **18.8GB** |
| `8gb` | 50–51GB | **29.2GB** |

During loading, only the mmap page cache grows (which the OS can reclaim); the anonymous-memory peak equals the value
after generation. As a side effect, text-encoder transfers now always come from pinned memory, and the 16gb / 8gb
configurations got faster (1024²: 2.6 s → 1.2 s / 1.8 s; 2048²: 4.1–4.2 s → 3.4–3.5 s).

PyTorch's allocator is also limited to the VRAM budget at load time (`torch.cuda.set_per_process_memory_fraction`),
so even when streaming delays frees, it does not hold more VRAM than the budget.

### VRAM reserve and a setting for PCs with little RAM

**VRAM reserve (`H3IMG_VRAM_RESERVE_GB`, default 1GB)**: the budget is computed as "free − reserve", always leaving
1GB for the display and other applications. If the budget has room to spare, the VAE (1.4GB) and the text encoder's
embedding table (1.45GB) are placed on the GPU even in the streaming configurations (fewer round trips, less host
RAM).

**Where the text encoder's weights live (`H3IMG_TE_HOST`)**:

- `pinned`: pinned host RAM (always uses ~13GB). Fast.
- `file`: each layer is read straight from the safetensors file into a small pinned staging area (2 × 0.26GB) by
  8 threads in parallel, then copied to the GPU. The page cache is bypassed (Linux drops the range with
  `fadvise(DONTNEED)`, Windows opens the file with `FILE_FLAG_SEQUENTIAL_SCAN`), so **the required host RAM drops by
  about 13GB** and nothing else gets evicted. Works on any OS; speed depends on the SSD (a Gen4 NVMe streams the 13GB
  text encoder in 4 s).
- `mmap`: stream from the safetensors file's page cache (Linux only; used layers are dropped with `madvise`). Slower
  than `file` (see below); kept for comparison.
- `auto` (default): `pinned` if the machine has 48GB of RAM or more, otherwise `file`.

Measured on simulated 8GB / 16GB cards (GPU free memory capped at 7.3GB / 15.5GB, 1GB reserve; all outputs
identical):

| Card | TE weights | Process VRAM | Minimum free GPU memory while running | Required host RAM | 1024² | 2048² | Edit 1664×2496 |
|---|---|---|---|---|---|---|---|
| 8GB | mmap | 5.9GB (VAE on the GPU too) | 1.5GB | **12.4GB** (+14GB page cache) | 2.6 s | 4.2 s | 7.0 s |
| 8GB | pinned | 5.9GB | 1.5GB | 27.2GB | 1.7 s | 3.4 s | 6.2 s |
| 16GB | mmap | 14.1GB | 1.7GB | **3.5GB** (+14GB page cache) | 2.2 s | 4.4 s | 7.2 s |
| 16GB | pinned | 14.1GB | 1.7GB | 18.3GB | 1.3 s | 3.5 s | 6.2 s |

The `mmap` rows above need the same RAM with the current default `file`, which is faster (see below).
**On a PC with 32GB of RAM, `file` is used (selected automatically)**; with 48GB or more, `pinned` is faster.

### A PC with 16GB of RAM and a 6GB GPU (streaming the transformer from the file / page cache too)

With `H3IMG_TRANSFORMER_HOST=file` (the default below 48GB of RAM), the 50 transformer blocks are also streamed straight
from the quantization cache (safetensors) in the page cache, and the required host RAM drops to **about 2GB** (the
process's anonymous memory). In exchange, whatever does not fit in RAM is **re-read from disk on every generation**,
so speed is set by the SSD's sequential read rate.

Measured on an RTX 4000 SFF Ada 20GB with its free memory capped at 6GB and the process limited to 12GB of memory
(equivalent to a 16GB PC), on a Gen4 NVMe reading at 2.3GB/s, with a 1GB reserve; outputs are pixel-identical to the
pinned configuration:

| Configuration | 1024² | 2048² | Edit 1664×2496 | Disk read per image | Required RAM |
|---|---|---|---|---|---|
| pinned (no RAM cap, same Ada card) | 8.2 s | 12.7 s | 24.3 s | 0 | 27GB |
| **file × file, 12GB cap (current default; 5 trailing blocks read from the file)** | **10.5 s** | **14.4 s** | — | 16GB | 2.3GB |
| same, 11GB cap (10 trailing blocks) | 10.6 s | 14.5 s | — | 19GB | 2.3GB |
| same, 10GB cap (15 trailing blocks) | 11.1 s | 14.5 s | — | 22GB | 2.3GB |
| **Windows, real machine: RTX 3050 6GB, 16GB RAM, Gen3 DRAM-less NVMe (1.3GB/s), file × file** | **~23 s** (6 steps: 42.5 s) | — | — | 16GB | 2.3GB |
| mmap × mmap, 12GB cap (previous Linux-only default) | 12.5–12.8 s | 16.6 s | 28.9 s | 13–14GB | 1.8GB |
| same, 11GB cap (6 trailing blocks dropped) | 13.5 s | 17.3 s | — | 16.7GB | 1.8GB |
| same, 10GB cap (11 trailing blocks dropped) | 13.9 s | 17.6 s | — | 19.7GB | 1.8GB |
| reference: without dropping used layers (mmap) | 17.8–21.3 s | 20.5–23.5 s | 31.2 s | 24–27GB | 1.8GB |

- `file` beats `mmap` because of how it reads. Going through the page cache (mmap page faults, or a single-threaded
  `read`) topped out at 1.6GB/s, whereas reading all tensors of a layer as 32MB chunks on 8 threads at once reaches
  4GB/s on the same NVMe (the 13GB text-encoder pass went from 8.4 s to 4.0 s). It reads slightly more than `mmap`
  (the trailing blocks) yet finishes sooner. On a SATA SSD expect about +30 s (bytes read ÷ 0.5GB/s).
- The `file` staging buffers are ordinary memory pinned with `cudaHostRegister`, not `pin_memory()` (cudaHostAlloc):
  the latter is charged to the cgroup as "file" memory, which escaped the page-cache budget (it subtracts anonymous
  memory) and pushed the transformer out of the cache (re-reads rose to 28GB in measurements).
- **Each text-encoder layer is read only once**, so it is never kept in the page cache (`file` drops the range it read
  with `fadvise`, `mmap` uses `madvise(MADV_PAGEOUT)`). Without this, the 13GB text encoder pushed the 10GB transformer out of the cache on every
  pass and 24GB were re-read each time. With it, the transformer stays cached and only the 13GB of the text encoder is
  re-read (this holds even with a 12GB cap; output unchanged).
- **When the transformer does not fit in the cache, only the overflow is dropped**: the trailing K blocks are
  evicted after use (`H3IMG_TRANSFORMER_DROP_BLOCKS`, default `auto`). Re-reading a fixed K blocks per step
  (0.2GB × K × 3 steps) is far cheaper than letting the LRU evict everything and re-read it every step (24.6GB with an
  11GB cap), so each 1GB less of RAM costs only +0.3–0.5 s. K is derived from the RAM available to the page cache
  (cgroup limits are honored). With `file` the K blocks are read straight from the file (the rest stay mmap'd in the
  page cache); with `mmap` they are dropped after use. The mmap pages of parts placed on the GPU (token refiner etc.)
  are dropped after loading to free cache space as well (Linux).
- Tensor positions inside the file come from the safetensors header: for an mmap'd tensor, "address − start of the
  mapping" is the file offset, so no names are needed. The W4A8 quantization cache is therefore saved as safetensors
  (an older `.pt` cache is converted automatically on first load).
- The Ada numbers above are this project's first measurements on a non-Blackwell, 70W small card (x4 link). The
  W4A8 kernels (comfy-kitchen) run on sm_89, 1.9× faster than bf16 per layer. The arithmetic differs from
  Blackwell, so outputs are not pixel-identical across the two (mean difference ~7/255, same composition).
- `file` works on Windows as well; only the `mmap` variant relies on `madvise` and is Linux-only.

### Things tried and not adopted

| Idea | Result |
|---|---|
| sage attention | The token count shrinks 32× (VAE 16× + 2×2 patches): about 1.1k at 1024² and 4.2k at 2048². Attention is only about 6% of the time even at 2048², so the gain is small. At 1024² the bottleneck is CPU-side kernel launches rather than the GPU (CUDA Graphs are the next candidate) |
| torchao int8wo + `torch.compile(dynamic=True)` | Recompiled every call and was unusable (200–3400 ms per layer). Needs a static-shape design |
| Community GGUF / nvfp4 | Require ComfyUI-specific formats or kernels, so not used on the diffusers path. nvfp4 is reported +31% over int8 on Blackwell (a future candidate). W4A8 was adopted by calling comfy-kitchen directly (6) |

### VRAM breakdown (3 steps, generating 2048²)

| | GB |
|---|---|
| transformer int8wo (312 layers) | ~17.5 |
| AdaLN tables, fp32 layers, norms | ~0.5 |
| VAE (bf16) | 2.3 |
| Activations and decoding during generation | +1–2 |
| **GPU0 total (peak)** | **~21** |
| text_encoder nf4 (can be on another GPU) | ~17 |

W4A8 configuration: transformer ck-w4a8 ~10.7 + tables etc. 0.5 + VAE 2.3 + activations → **~13–14GB**. With the TE
resident on the same GPU, +16.3GB (peak 29.1GB, the 32GB class). With layer streaming + lm_head removal + CPU
embedding + VAE swap, the TE adds only 0.3–0.9GB: **peak 12.5–13.1GB (the 16GB class)**. Adding transformer block
streaming leaves ~1GB of transformer + 2 prefetched blocks: **torch peak 2.4–2.9GB, 4.2GB actually used by the process
(the 8GB class)**.

## License notes

| Item | License |
|---|---|
| **The code in this repository** | **MIT** ([LICENSE](LICENSE)) |
| The model MiniMaxAI/MiniMax-H3-Image (transformer / VAE) | **MiniMax-H3-Image Community License** (the official [LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)) |
| text_encoder (the Qwen3-VL-32B of MiniMaxAI/MiniMax-H3) | The MiniMax-H3 license |
| pDMD 2-step LoRA (akhaliq/minimax-h3-image-pdmd-2step) | Apache-2.0 (outputs still follow the model's license) |

- **Even though the code is MIT, the model weights and generated outputs are subject to MiniMax's license.** The
  official license has conditions such as commercial use (separate written permission), regional restrictions and
  attribution requirements. Always read the original on the official page before use.
- This repository **does not contain any model weights**. The quantized text encoder and the transformer caches are
  meant to be generated in your own environment (whether derivatives may be redistributed is governed by the official
  license).
- Use generated images (commercially or publicly) only within the scope of the model's license.

## Investigation notes

Before implementing, the official model was investigated and quantization methods were compared on real hardware.
Highlights only.

### What was learned about the model

- The text encoder (Qwen3-VL-32B) **matches MiniMax-H3 (video) hash-for-hash across all 14 shards**. Both use
  `hidden_states[50]` for conditioning, so a text encoder quantized for the video model works as is.
- In diffusers it is provided only as a Modular Pipeline (`MiniMaxH3ImageModularPipeline`), and at the time of
  writing only on the main branch (not in 0.42.1).
- Sampling is fixed at 6 steps (the σ schedule is in the checkpoint; each step has its own output head). No CFG.
- By default it generates on a minimum 4MP canvas and downscales, so without `min_generation_area` a 1024² request
  takes as long as 2048².
- The token count shrinks 32× (VAE 16× and 2×2 patches): 1024² ≈ 1.1k, 2048² ≈ 4.2k, 4096² ≈ 16.5k.

### Time breakdown (int8wo, model loaded)

| Case | Total | text_encoder | transformer | VAE decode |
|---|---|---|---|---|
| T2I 1024² | 2.05 s (6 steps) | 0.11 | 1.74 | 0.17 |
| T2I 2048² | 7.04 s (6 steps) | 0.12 | 6.02 | 0.83 |
| Edit 2048² | 11.4 s (6 steps) | 0.45 | 9.76 | 0.82 (+ encode 0.25) |

70–90% of the time is the transformer (denoising). The text encoder and VAE are small; slimming the text encoder is for
VRAM only. At 1024², CPU-side kernel launches rather than the GPU become the bottleneck (CUDA Graphs are a future
candidate).

### Quality observations (100% zoom)

- Text-to-image is good for both photographs and anime styles. Reference editing preserves the identity of people and
  characters as well as clothing and accessories well.
- In edits, **small background text can break** (large text is preserved). Whether this comes from quantization or is a
  limit of the original model has not been isolated.
- int8wo outputs match bnb int8 at the same seed in composition and detail (only minor differences from the
  quantization method).
- 2 steps show a fine woven pattern; it disappears at 3 steps, which come close to 6.

### Implementation pitfalls (encountered on real hardware)

- `ModularPipeline._execution_device` returns the device of the first nn.Module among the components, so putting the
  text encoder on another GPU makes that the execution device → override the property to pin it.
- Always pass data between GPUs through the host (in one environment P2P copies silently returned all zeros).
- Newer huggingface_hub versions check for missing files even with `snapshot_download(local_files_only=True)` and raise.
  For an intentionally partial snapshot, reference `snapshots/<commit>` directly.
- `accelerate.set_module_tensor_to_device` casts to fp32 when no dtype is given (activations become fp32, 2× slower).
- `init_empty_weights` leaves buffers (`rope.inv_freq`) on the CPU → move them explicitly.
- Plain `torch.compile(dynamic=True)` on torchao int8wo recompiles every call.
- When switching steps, the pipeline attribute and the forward wrapper's closure kept the old transformer alive, so two
  copies were resident → drop all references, then run gc.

### Not yet verified

- Quality comparison against the bf16 model (to isolate the text breakage), VRAM and time at 4096² and above, edits
  with multiple (2+) references, nvfp4, CUDA Graphs, measurements on real 12GB / 24GB cards, and an end-to-end run
  of `prepare_models.py` (quantization) on a real Windows machine (verified on Linux).

## Background

This was implemented as the third backend of diffusers-movie-server (an integrated server for MiniMax-H3 video and
LTX-2.5) and then split out as a standalone image server.
