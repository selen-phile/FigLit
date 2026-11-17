"""
day5_validation.py (Updated for 500-sample Fine Sweep)

1. Automatically extracts the Top-5 sites from your new 500-sample fine sweep
   (day4_all_blocks_results.csv).
2. Reverse-Direction Check: Patches figurative (poem) activations into a 
   literal run. Effect should flip sign.
3. Fresh-Poem Check: Applies the top sites to completely unseen poems.
"""

import os
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)

import csv
import gc
import json
import pandas as pd
import torch
from diffusers import FluxPipeline
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPProcessor

from flux_hooks import FluxActivationHooker, HookMode, Site, Stream
from patch_and_generate import _generate

# =============================================================================
# CONFIG
# =============================================================================
CLIP_MODEL_ID = "zer0int/LongCLIP-GmP-ViT-L-14"
MAX_TEXT_LEN = 248
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
NUM_STEPS = 4
SEED = 42

# Use the NEW 500-sample fine sweep results
FINE_CSV_PATH = "day4_out/day4_all_blocks_results.csv" 
SWEEP_POEMS_PATH = "fig200_fine_sweep_500.json"   # The 500 poems used for the fine sweep
FRESH_POEMS_PATH = "fig200_fresh_200.json"        # 200 disjoint poems never used before
OUT_DIR = "day5_out"
TOP_N_SITES = 5

# =============================================================================
# CLIP METRICS (Cached Text Features for Speed)
# =============================================================================
CLIP_DEVICE = "cpu"
_clip_model = None
_clip_processor = None
_TEXT_FEAT_CACHE = {}

def load_clip(device: str = CLIP_DEVICE):
    global _clip_model, _clip_processor
    if _clip_model is None:
        print(f"[clip] Loading {CLIP_MODEL_ID} on {device} ...")
        config = CLIPConfig.from_pretrained(CLIP_MODEL_ID)
        config.text_config.max_position_embeddings = MAX_TEXT_LEN
        _clip_model = CLIPModel.from_pretrained(
            CLIP_MODEL_ID, config=config, torch_dtype=torch.float32
        ).to(device).eval()
        _clip_processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_ID, padding="max_length", max_length=MAX_TEXT_LEN
        )
        _clip_processor.tokenizer.model_max_length = MAX_TEXT_LEN
    return _clip_model, _clip_processor

@torch.inference_mode()
def _text_features(texts):
    model, proc = load_clip()
    missing = [t for t in texts if t not in _TEXT_FEAT_CACHE]
    if missing:
        inputs = proc(text=missing, return_tensors="pt", padding="max_length", max_length=MAX_TEXT_LEN, truncation=True)
        out = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        feats = out.text_embeds if hasattr(out, "text_embeds") else out.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        for t, f in zip(missing, feats.float()):
            _TEXT_FEAT_CACHE[t] = f.detach()
    return torch.stack([_TEXT_FEAT_CACHE[t] for t in texts], dim=0)

@torch.inference_mode()
def _image_features(images):
    model, proc = load_clip()
    inputs = proc(images=images, return_tensors="pt")
    out = model.get_image_features(pixel_values=inputs["pixel_values"])
    feats = out.image_embeds if hasattr(out, "image_embeds") else out.pooler_output
    return (feats / feats.norm(dim=-1, keepdim=True)).float()

def compute_metrics(image, figurative_target, literal_text):
    img_f = _image_features([image])
    txt_f = _text_features([figurative_target, literal_text])
    sims = (img_f @ txt_f.T).squeeze(0) * 100.0
    fig, lit = sims[0].item(), sims[1].item()
    return {"figurative_alignment": fig, "literal_alignment": lit, "gap": lit - fig}

def get_figurative_target(poem):
    gloss = poem.get("figurative_meaning_gloss")
    return gloss if gloss else poem["poem"]

# =============================================================================
# PIPELINE & SITE SELECTION
# =============================================================================
def load_pipeline():
    print(f"Loading {MODEL_ID} on {DEVICE} ...")
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True)
    
    # FORCE memory safe offload. Do not check total VRAM, because other users
    # might be occupying space on the GPU.
    print("[pipeline] strategy = sequential offload (memory-safe)")
    pipe.enable_sequential_cpu_offload()
    
    # Optional but highly recommended memory savers
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    return pipe

def select_top_sites(fine_csv_path: str, top_n: int):
    print(f"Reading fine sweep results from {fine_csv_path}...")
    df = pd.read_csv(fine_csv_path)
    
    # 1. Clean NaNs and dead cells
    df = df[(df['effect'] != 0.0) & (df['effect'].notna())]
    
    # 2. CRITICAL FIX: Only look at poems that actually suffered from Literalism Collapse
    # (i.e., poems where the baseline was already literal)
    df = df[df['baseline_gap'] > 2.0]
    print(f"Filtered to {df['poem_id'].nunique()} poems with baseline_gap > 2.0")
    
    # 3. Calculate consistency score
    agg = df.groupby(["region", "block_idx", "step"])["effect"].agg(['mean', 'std', 'count']).reset_index()
    agg = agg[agg['count'] > 100] # Lower threshold since we filtered the dataset
    agg['score'] = agg['mean'].abs() / (agg['std'] + 1e-5)
    
    top = agg.sort_values("score", ascending=False).head(top_n)
    
    print(f"\n=== Top {top_n} CONSISTENT Sites ===")
    print(top[["region", "block_idx", "step", "mean"]].to_string(index=False))

    sites = []
    for _, row in top.iterrows():
        stream = Stream.DOUBLE if "double" in row["region"] else Stream.SINGLE
        sites.append(Site(step=int(row["step"]), stream=stream, block_idx=int(row["block_idx"])))
    return sites
# =============================================================================
# VALIDATION RUNNER
# =============================================================================
def _run_direction(pipe, hooker, poems, sites, direction, out_dir, csv_name):
    csv_path = os.path.join(out_dir, csv_name)
    fieldnames = ["poem_id", "direction", "baseline_gap", "patched_gap", "effect"]

    already_done = set()
    file_exists = os.path.exists(csv_path)
    if file_exists:
        existing = pd.read_csv(csv_path)
        already_done = set(existing["poem_id"])
        if already_done:
            print(f"Resuming {direction}: {len(already_done)} poems already done.")

    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()

    for idx, poem in enumerate(poems):
        if poem["id"] in already_done:
            continue

        fig_target = get_figurative_target(poem)

        try:
            if direction == "forward":
                baseline_prompt, donor_prompt, target_prompt = poem["poem"], poem["literal"], poem["poem"]
            else:  # reverse
                baseline_prompt, donor_prompt, target_prompt = poem["literal"], poem["poem"], poem["literal"]

            baseline_image, _ = _generate(pipe, hooker, baseline_prompt, SEED, HookMode.OFF)
            _, donor_cache = _generate(pipe, hooker, donor_prompt, SEED, HookMode.CACHE)
            baseline_metrics = compute_metrics(baseline_image, fig_target, poem["literal"])

            hooker.reset_step_counter()
            hooker.set_mode(HookMode.PATCH)
            hooker.set_patch_sites(sites, donor_cache)
            
            generator = torch.Generator(device="cpu").manual_seed(SEED)
            result = pipe(prompt=target_prompt, height=512, width=512,
                          num_inference_steps=NUM_STEPS, guidance_scale=0.0, generator=generator)
            
            patched_metrics = compute_metrics(result.images[0], fig_target, poem["literal"])
            effect = baseline_metrics["gap"] - patched_metrics["gap"]

            writer.writerow({
                "poem_id": poem["id"], "direction": direction,
                "baseline_gap": baseline_metrics["gap"],
                "patched_gap": patched_metrics["gap"], "effect": effect
            })
            csv_file.flush()
            os.fsync(csv_file.fileno())
            print(f"  [{direction}] {poem['id']}: effect={effect:+.3f}")
            
        except Exception as e:
            print(f"  !! Error on {poem['id']}: {e}")
        finally:
            # Memory cleanup
            del baseline_image, donor_cache, result
            hooker.cache = None
            hooker.patch_source = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    csv_file.close()
    return pd.read_csv(csv_path)

# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Get Top Sites from the 500-sample Fine Sweep
    sites = select_top_sites(FINE_CSV_PATH, TOP_N_SITES)

    # 2. Load Model
    pipe = load_pipeline()
    hooker = FluxActivationHooker(pipe.transformer)
    load_clip()

    # 3. Load Poetry splits
    with open(SWEEP_POEMS_PATH) as f: sweep_poems = json.load(f)
    with open(FRESH_POEMS_PATH) as f: fresh_poems = json.load(f)

    # 4. Run Reverse Direction (Poetic into Literal)
    print(f"\n--- Reverse-direction check on {len(sweep_poems)} sweep poems ---")
    reverse_df = _run_direction(pipe, hooker, sweep_poems, sites, "reverse", OUT_DIR, "reverse_direction_results.csv")

    # 5. Run Fresh Poems (Forward Direction)
    print(f"\n--- Fresh-poem check (forward) on {len(fresh_poems)} unseen poems ---")
    fresh_df = _run_direction(pipe, hooker, fresh_poems, sites, "forward", OUT_DIR, "fresh_poem_results.csv")

    # 6. Summary
    reverse_mean = reverse_df["effect"].mean()
    fresh_mean = fresh_df["effect"].mean()

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Reverse-direction mean effect: {reverse_mean:+.3f} (Should be positive, flipping the sign)")
    print(f"Fresh-poem mean effect:       {fresh_mean:+.3f} (Should match fine sweep magnitude ~ -5.0)")
    
    if reverse_mean <= 0:
        print("WARNING: Reverse effect did not flip sign. The localized pathway may not be bidirectional.")
    if abs(fresh_mean) < 0.5 * 5.0:
        print("WARNING: Fresh-poem effect is much weaker than expected. Possible overfitting to the fine sweep.")

    hooker.remove()

if __name__ == "__main__":
    main()