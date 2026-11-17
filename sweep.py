"""
day3_coarse_sweep_fast.py  —  memory-safe, fast coarse sweep

Configuration summary
- FLUX.1-schnell transformer: on GPU (bf16), streamed only if free VRAM < 24 GB.
- Text encoder: pre-encoded per poem once, on CPU. Never touches the GPU.
- Long-CLIP: fp16, on a different GPU with >= 4 GB free if available, else CPU.
- Attention slicing + VAE slicing/tiling on.
- PYTORCH_CUDA_ALLOC_CONF set before torch import to reduce fragmentation.
- OOM recovery: writes a nan row, rebuilds the pipeline in the safest mode,
  regenerates the donor cache for that poem, and continues.
- Resume: de-duplicated by (poem_id, region, step); fully-complete poems skipped.
- Sidecar JSONL logs every written row, so "fast, nothing saved" is impossible.
"""

import os
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)

import csv
import gc
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from diffusers import FluxPipeline
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPProcessor

from flux_hooks import ActivationCache, FluxActivationHooker, HookMode
from patch_and_generate import _generate
from regions import build_regions, region_sites_for_step

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


# =============================================================================
# CONFIG
# =============================================================================
CLIP_MODEL_ID = "zer0int/LongCLIP-GmP-ViT-L-14"
MAX_TEXT_LEN = 248

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
NUM_STEPS = 4
SEED = 42

POEM_PAIRS_PATH = "fig200_all_converted.json"
OUT_DIR = "day3_clean_out"

SAVE_IMAGES = False
LOG_EVERY = 6

FLUX_DEVICE_INDEX = 0
DEVICE = f"cuda:{FLUX_DEVICE_INDEX}" if torch.cuda.is_available() else "cpu"

# fp16 for the transformer is the single most effective OOM fix on 24 GB cards.
# Set to torch.bfloat16 on >= 40 GB GPUs for bit-exact parity with the paper.
DTYPE = (
    torch.float16 if torch.cuda.is_available() and
    (torch.cuda.get_device_properties(FLUX_DEVICE_INDEX).total_memory / 1024**3) < 32.0
    else torch.bfloat16 if torch.cuda.is_available()
    else torch.float32
)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# =============================================================================
# DEVICE / MEMORY HELPERS
# =============================================================================
def _free_gb(device_index: int) -> float:
    try:
        free, _ = torch.cuda.mem_get_info(device_index)
        return free / 1024**3
    except Exception:
        return 0.0


def pick_clip_device(flux_device_index: int) -> str:
    n = torch.cuda.device_count()
    if n == 0:
        return "cpu"
    for i in range(n):
        if i == flux_device_index:
            continue
        if _free_gb(i) >= 4.0:
            print(f"[clip] Using cuda:{i} (free {_free_gb(i):.1f} GB)")
            return f"cuda:{i}"
    if _free_gb(flux_device_index) >= 6.0:
        print(f"[clip] Sharing cuda:{flux_device_index} with FLUX")
        return f"cuda:{flux_device_index}"
    print("[clip] No GPU with >= 4 GB free; Long-CLIP on CPU")
    return "cpu"


CLIP_DEVICE = pick_clip_device(FLUX_DEVICE_INDEX)


def log_memory(tag=""):
    parts = []
    if torch.cuda.is_available():
        parts.append(
            f"GPU{FLUX_DEVICE_INDEX} alloc={torch.cuda.memory_allocated()/1024**3:.2f}GB "
            f"reserved={torch.cuda.memory_reserved()/1024**3:.2f}GB"
        )
        for i in range(torch.cuda.device_count()):
            if i == FLUX_DEVICE_INDEX:
                continue
            parts.append(f"GPU{i} free={_free_gb(i):.1f}GB")
    if _HAVE_PSUTIL:
        parts.append(f"RAM={psutil.virtual_memory().percent:.1f}%")
    if parts:
        print(f"  [mem {tag}] " + "  ".join(parts))


def error_cleanup(hooker=None, donor_cache=None):
    """Aggressive cleanup — only on the error path."""
    clear_donor_cache(hooker, donor_cache)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def clear_donor_cache(hooker, donor_cache):
    if donor_cache is None:
        return
    try:
        for v in donor_cache.values():
            if isinstance(v, torch.Tensor):
                v.data = torch.empty(0)
        donor_cache.clear()
    except Exception:
        pass
    try:
        if getattr(hooker, "cache", None) is not None:
            if hasattr(hooker.cache, "clear"):
                hooker.cache.clear()
            hooker.cache = None
    except Exception:
        pass
    try:
        if hasattr(hooker, "patch_source"):
            hooker.patch_source = None
    except Exception:
        pass


# =============================================================================
# LONG-CLIP (fp16, cached text features, batched-free scoring)
# =============================================================================
_clip_model = None
_clip_processor = None
_TEXT_FEAT_CACHE = {}


def load_clip(device: str = CLIP_DEVICE):
    global _clip_model, _clip_processor
    if _clip_model is None:
        print(f"[clip] Loading {CLIP_MODEL_ID} (fp16) on {device} ...")
        config = CLIPConfig.from_pretrained(CLIP_MODEL_ID)
        config.text_config.max_position_embeddings = MAX_TEXT_LEN

        dtype = torch.float16 if device != "cpu" else torch.float32
        _clip_model = CLIPModel.from_pretrained(
            CLIP_MODEL_ID, config=config, torch_dtype=dtype
        ).to(device).eval()

        _clip_processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_ID, padding="max_length", max_length=MAX_TEXT_LEN
        )
        _clip_processor.tokenizer.model_max_length = MAX_TEXT_LEN
    return _clip_model, _clip_processor


@torch.inference_mode()
def _text_features(texts):
    """Normalized text features (B, D). Cached per string."""
    model, proc = load_clip()
    missing = [t for t in texts if t not in _TEXT_FEAT_CACHE]
    if missing:
        inputs = proc(
            text=missing, return_tensors="pt",
            padding="max_length", max_length=MAX_TEXT_LEN, truncation=True,
        )
        ids = inputs["input_ids"].to(CLIP_DEVICE)
        mask = inputs["attention_mask"].to(CLIP_DEVICE)
        out = model.get_text_features(input_ids=ids, attention_mask=mask)
        if hasattr(out, "text_embeds"):
            feats = out.text_embeds
        elif hasattr(out, "pooler_output"):
            feats = out.pooler_output
        else:
            feats = out
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.float()
        for t, f in zip(missing, feats):
            _TEXT_FEAT_CACHE[t] = f.detach()
    return torch.stack([_TEXT_FEAT_CACHE[t] for t in texts], dim=0)


@torch.inference_mode()
def _image_features(images):
    model, proc = load_clip()
    inputs = proc(images=images, return_tensors="pt")
    px = inputs["pixel_values"].to(CLIP_DEVICE, dtype=next(model.parameters()).dtype)
    out = model.get_image_features(pixel_values=px)
    if hasattr(out, "image_embeds"):
        feats = out.image_embeds
    elif hasattr(out, "pooler_output"):
        feats = out.pooler_output
    else:
        feats = out
    return (feats / feats.norm(dim=-1, keepdim=True)).float()


def score_one(image, figurative_target, literal_text):
    img_f = _image_features([image])
    txt_f = _text_features([figurative_target, literal_text])
    sims = (img_f @ txt_f.T).squeeze(0) * 100.0
    fig = sims[0].item()
    lit = sims[1].item()
    return {
        "figurative_alignment": fig,
        "literal_alignment": lit,
        "gap": lit - fig,
    }


def get_figurative_target(poem, warned=None):
    if warned is None:
        warned = set()
    gloss = poem.get("figurative_meaning_gloss")
    if gloss and gloss != "nan":
        return gloss
    if poem["id"] not in warned:
        print(f"  [metric] '{poem['id']}' has no gloss; scoring against raw poem.")
        warned.add(poem["id"])
    return poem["poem"]


# =============================================================================
# FLUX
# =============================================================================
def load_pipeline(device_index: int = FLUX_DEVICE_INDEX):
    """Adaptive loader: picks a strategy that fits *free* VRAM at call time."""
    if not torch.cuda.is_available():
        print("[pipeline] No CUDA; using CPU")
        return FluxPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )

    free = _free_gb(device_index)
    total = torch.cuda.get_device_properties(device_index).total_memory / 1024**3
    print(f"[pipeline] GPU{device_index}: free={free:.1f}GB / total={total:.1f}GB dtype={DTYPE}")

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True
    )

    if free >= 32.0:
        pipe.to(f"cuda:{device_index}")
        print("[pipeline] strategy = full GPU")
    elif free >= 22.0:
        # Transformer + VAE resident, text encoder kept on CPU (we pre-encode).
        pipe.transformer.to(f"cuda:{device_index}")
        pipe.vae.to(f"cuda:{device_index}")
        pipe.text_encoder.to("cpu")
        pipe.enable_model_cpu_offload()
        print("[pipeline] strategy = model offload (TE on CPU)")
    else:
        pipe.enable_sequential_cpu_offload()
        print("[pipeline] strategy = sequential offload (memory-safe)")

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    try:
        pipe.transformer.set_attention_slice("auto")
    except Exception:
        pass
    return pipe


def load_poem_pairs(path):
    with open(path) as f:
        pairs = json.load(f)
    req = {"id", "poem", "literal"}
    for p in pairs:
        miss = req - set(p.keys())
        if miss:
            raise ValueError(f"{p.get('id','?')} missing {miss}")
    return pairs


@torch.inference_mode()
def encode_prompt_once(pipe, prompt: str):
    """Pre-encode the prompt on CPU so the GPU never sees the text encoder."""
    try:
        return pipe.encode_prompt(
            prompt=prompt, prompt_2=None,
            device="cpu",
            num_images_per_prompt=1,
            max_sequence_length=256,
        )
    except TypeError:
        # Older diffusers doesn't accept device= for encode_prompt; encode on CPU
        # by temporarily moving the text encoder.
        te_was_on = next(pipe.text_encoder.parameters()).device
        pipe.text_encoder.to("cpu")
        try:
            out = pipe.encode_prompt(
                prompt=prompt, prompt_2=None,
                num_images_per_prompt=1,
                max_sequence_length=256,
            )
        finally:
            if te_was_on.type == "cuda":
                pipe.text_encoder.to(te_was_on)
        return out


def get_baseline_and_donor(pipe, hooker, poem, seed):
    base_img, _ = _generate(pipe, hooker, poem["poem"], seed, HookMode.OFF)
    donor_img, donor_cache = _generate(pipe, hooker, poem["literal"], seed, HookMode.CACHE)
    return base_img, donor_img, donor_cache


# =============================================================================
# CSV / SIDECAR
# =============================================================================
FIELDNAMES = [
    "poem_id", "region", "step",
    "baseline_gap", "patched_gap", "effect",
    "patched_figurative_alignment", "patched_literal_alignment",
]


def _write_rows(csv_file, writer, rows, sidecar):
    if not rows:
        return
    writer.writerows(rows)
    csv_file.flush()
    os.fsync(csv_file.fileno())
    for r in rows:
        sidecar.write(json.dumps({
            "poem_id": r["poem_id"], "region": r["region"], "step": r["step"],
            "effect": r["effect"],
        }) + "\n")
    sidecar.flush()


def _nan_row(poem, region_name, step, base_m, reason=""):
    return {
        "poem_id": poem["id"], "region": region_name, "step": step,
        "baseline_gap": base_m["gap"],
        "patched_gap": float("nan"), "effect": float("nan"),
        "patched_figurative_alignment": float("nan"),
        "patched_literal_alignment": float("nan"),
    }


# =============================================================================
# SWEEP
# =============================================================================
def _run_one_cell(pipe, hooker, poem, region_name, step, sites,
                  donor_cache, base_m, fig_target, pe, pp, gen):
    hooker.reset_step_counter()
    hooker.set_mode(HookMode.PATCH)
    hooker.set_patch_sites(sites, donor_cache)

    call_kwargs = dict(
        height=512, width=512,
        num_inference_steps=NUM_STEPS,
        guidance_scale=0.0,
        generator=gen,
    )
    if pe is not None:
        call_kwargs["prompt_embeds"] = pe
        call_kwargs["pooled_prompt_embeds"] = pp
    else:
        call_kwargs["prompt"] = poem["poem"]

    result = pipe(**call_kwargs)
    img = result.images[0]
    m = score_one(img, fig_target, poem["literal"])
    return {
        "poem_id": poem["id"], "region": region_name, "step": step,
        "baseline_gap": base_m["gap"], "patched_gap": m["gap"],
        "effect": base_m["gap"] - m["gap"],
        "patched_figurative_alignment": m["figurative_alignment"],
        "patched_literal_alignment": m["literal_alignment"],
    }


def run_sweep():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading {MODEL_ID} on {DEVICE} (dtype={DTYPE}) ...")
    pipe = load_pipeline(FLUX_DEVICE_INDEX)
    hooker = FluxActivationHooker(pipe.transformer)
    regions = build_regions(pipe.transformer)
    region_names = [r.name for r in regions]
    sites_by_cell = {
        (r.name, s): region_sites_for_step(r, s)
        for r in regions for s in range(NUM_STEPS)
    }
    poems = load_poem_pairs(POEM_PAIRS_PATH)
    expected_per_poem = len(region_names) * NUM_STEPS
    print(f"{len(region_names)} regions x {NUM_STEPS} steps x {len(poems)} poems")
    log_memory("post-load")

    csv_path = os.path.join(OUT_DIR, "day3_results.csv")
    sidecar_path = os.path.join(OUT_DIR, "sweep_sidecar.jsonl")

    already_done = set()
    file_exists = os.path.exists(csv_path)
    if file_exists:
        existing = pd.read_csv(csv_path)
        if {"poem_id", "region", "step"} <= set(existing.columns):
            existing = existing.drop_duplicates(subset=["poem_id", "region", "step"])
            counts = existing.groupby("poem_id").size()
            already_done = set(counts[counts >= expected_per_poem].index)
            print(f"[resume] {len(already_done)} complete, "
                  f"{len(counts) - len(already_done)} partial, "
                  f"expected {expected_per_poem} rows/poem")

    # Load CLIP *before* opening CSV handles, so a CLIP load failure leaves
    # no half-open state.
    load_clip()
    log_memory("post-clip")

    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()
    sidecar = open(sidecar_path, "a")

    warned = set()
    total_cells = 0

    for poem in poems:
        if poem["id"] in already_done:
            continue

        print(f"\n--- {poem['id']} ---")
        fig_target = get_figurative_target(poem, warned)

        # Pre-encode prompt embeddings once for this poem.
        try:
            pe, pp = encode_prompt_once(pipe, poem["poem"])
        except Exception as e:
            print(f"  !! prompt encode failed: {e}; will fall back to prompt=...")
            pe, pp = None, None

        # Baseline + donor.
        try:
            base_img, donor_img, donor_cache = get_baseline_and_donor(
                pipe, hooker, poem, SEED
            )
        except (torch.cuda.OutOfMemoryError, MemoryError) as e:
            print(f"  !! OOM baseline/donor: {e}; rebuilding pipeline in safe mode.")
            error_cleanup(hooker, None)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            pipe = load_pipeline(FLUX_DEVICE_INDEX)
            hooker = FluxActivationHooker(pipe.transformer)
            try:
                base_img, donor_img, donor_cache = get_baseline_and_donor(
                    pipe, hooker, poem, SEED
                )
            except Exception as e2:
                print(f"  !! still failing: {e2}; skipping poem.")
                error_cleanup(hooker, None)
                continue
        except Exception as e:
            print(f"  !! baseline/donor failed: {e}; skipping poem.")
            error_cleanup(hooker, None)
            continue

        try:
            base_m = score_one(base_img, fig_target, poem["literal"])
        except Exception as e:
            print(f"  !! base scoring failed: {e}; skipping poem.")
            error_cleanup(hooker, donor_cache)
            continue
        print(f"  baseline gap {base_m['gap']:.3f}")

        if SAVE_IMAGES:
            d = os.path.join(OUT_DIR, "images", poem["id"])
            os.makedirs(d, exist_ok=True)
            base_img.save(os.path.join(d, "baseline.png"))
            donor_img.save(os.path.join(d, "donor_literal.png"))

        rows = []
        n_cells = 0
        for rname in region_names:
            for step in range(NUM_STEPS):
                sites = sites_by_cell[(rname, step)]
                try:
                    gen = torch.Generator(device="cpu").manual_seed(SEED)
                    row = _run_one_cell(
                        pipe, hooker, poem, rname, step, sites,
                        donor_cache, base_m, fig_target, pe, pp, gen,
                    )
                    rows.append(row)
                    print(f"    {rname:14s} step={step}  effect={row['effect']:+.3f}")
                except (torch.cuda.OutOfMemoryError, MemoryError) as e:
                    print(f"    !! OOM {rname}/s{step}: {e}")
                    rows.append(_nan_row(poem, rname, step, base_m, "oom"))
                    error_cleanup(hooker, donor_cache)
                    # Rebuild pipeline and regenerate donor cache for this poem.
                    try:
                        del pipe
                        gc.collect()
                        torch.cuda.empty_cache()
                        pipe = load_pipeline(FLUX_DEVICE_INDEX)
                        hooker = FluxActivationHooker(pipe.transformer)
                        _, _, donor_cache = get_baseline_and_donor(
                            pipe, hooker, poem, SEED
                        )
                    except Exception as e2:
                        print(f"    !! rebuild failed: {e2}; skipping remainder of poem.")
                        break
                except Exception as e:
                    print(f"    !! err {rname}/s{step}: {e}")
                    rows.append(_nan_row(poem, rname, step, base_m, "err"))
                finally:
                    gen = None
                    n_cells += 1
                    if n_cells % LOG_EVERY == 0:
                        log_memory(f"{rname}/s{step}")

        _write_rows(csv_file, writer, rows, sidecar)
        total_cells += len(rows)
        print(f"  [checkpoint] {len(rows)} rows, total cells {total_cells}")

        clear_donor_cache(hooker, donor_cache)
        donor_cache = None
        del base_img, donor_img, base_m
        pe = pp = None

    csv_file.close()
    sidecar.close()
    print(f"Saved {csv_path}")
    _make_heatmap(csv_path, region_names)
    try:
        hooker.remove()
    except Exception:
        pass
    return pd.read_csv(csv_path)


def _make_heatmap(csv_path, region_names):
    print("Generating heatmap ...")
    df = pd.read_csv(csv_path)
    heatmap = np.full((len(region_names), NUM_STEPS), np.nan)
    for i, r in enumerate(region_names):
        for s in range(NUM_STEPS):
            v = df[(df["region"] == r) & (df["step"] == s)]["effect"]
            if len(v) and not v.isna().all():
                heatmap[i, s] = v.mean()

    m = np.nanmax(np.abs(heatmap)) if not np.isnan(heatmap).all() else 1.0
    if not m or np.isnan(m):
        m = 1.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heatmap, cmap="RdBu_r", aspect="auto", vmin=-m, vmax=m)
    ax.set_xticks(range(NUM_STEPS))
    ax.set_xticklabels([f"step {i}" for i in range(NUM_STEPS)])
    ax.set_yticks(range(len(region_names)))
    ax.set_yticklabels(region_names)
    ax.set_title("Mean effect (baseline_gap - patched_gap)\n"
                 "positive = shift toward figurative fidelity")
    for i in range(len(region_names)):
        for j in range(NUM_STEPS):
            if not np.isnan(heatmap[i, j]):
                ax.text(j, i, f"{heatmap[i, j]:.2f}",
                        ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "day3_heatmap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    try:
        run_sweep()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except MemoryError as e:
        print(f"\nAborted to avoid OOM kill: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)