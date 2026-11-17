"""
day7.py - Final 4-Way Comparison (Vanilla, FigSteer, Norm-Matched Global, Random 5-Site)
"""
import gc
import json
import os
import pickle
import time
import torch
import pandas as pd
import numpy as np
from diffusers import FluxPipeline
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPProcessor
from tqdm import tqdm

from flux_hooks import ActivationCache, FluxActivationHooker, HookMode, Site, Stream
from patch_and_generate import _generate
from regions import build_regions

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

# =============================================================================
# CLIP METRICS
# =============================================================================
CLIP_MODEL_ID = "zer0int/LongCLIP-GmP-ViT-L-14"
MAX_TEXT_LEN = 248
CLIP_DEVICE = "cpu"
_clip_model, _clip_processor, _TEXT_FEAT_CACHE = None, None, {}

def load_clip(device: str = CLIP_DEVICE):
    global _clip_model, _clip_processor
    if _clip_model is None:
        config = CLIPConfig.from_pretrained(CLIP_MODEL_ID)
        config.text_config.max_position_embeddings = MAX_TEXT_LEN
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, config=config, torch_dtype=torch.float32).to(device).eval()
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, padding="max_length", max_length=MAX_TEXT_LEN)
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
# CONFIG
# =============================================================================
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
NUM_STEPS = 4
SEED = 42

VECTOR_POEMS_PATH = "fig200_fine_sweep_500.json"
SANITY_POEMS_PATH = "fig200_fresh_200.json"
SANITY_N_POEMS = 20
LOCAL_VECTOR_PATH = "day6_out/steering_vectors.pkl"
OUT_DIR = "day7_out"
GLOBAL_VECTOR_CACHE_PATH = os.path.join(OUT_DIR, "global_steering_vectors.pkl")
CHOSEN_ALPHA = 1.0

def load_pipeline():
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True)
    pipe.enable_sequential_cpu_offload()
    return pipe

def load_poem_pairs(path):
    with open(path) as f:
        return json.load(f)

def get_all_sites(transformer):
    regions = build_regions(transformer)
    sites = []
    for region in regions:
        for step in range(NUM_STEPS):
            for block_idx in region.block_indices:
                sites.append(Site(step=step, stream=region.stream, block_idx=block_idx))
    return sites

# =============================================================================
# GLOBAL VECTOR CONSTRUCTION
# =============================================================================
def build_global_steering_vectors(pipe, hooker, poems, sites, out_dir):
    checkpoint_path = os.path.join(out_dir, "global_vector_accum_checkpoint.pkl")
    accum = {s.key(): None for s in sites}
    n = 0
    done_ids = set()

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        accum, n, done_ids = ckpt["accum"], ckpt["n"], ckpt["done_ids"]
        print(f"Resuming global vector construction: {n} poems accumulated.")

    print(f"Building GLOBAL vectors over {len(poems)} poems. This will take time.")
    transformer = pipe.transformer
    double_blocks = transformer.transformer_blocks
    single_blocks = transformer.single_transformer_blocks

    for poem in tqdm(poems, desc="Global Vector Build"):
        if poem["id"] in done_ids:
            continue

        poem_acts = {}
        lit_acts = {}

        def make_capture_hook(site, storage):
            def hook(module, inputs, output):
                if site.step == 0: 
                    if isinstance(output, tuple):
                        storage[site.key()] = output[1].clone().detach().cpu()
                    else:
                        storage[site.key()] = output.clone().detach().cpu()
            return hook

        handles = []
        for s in sites:
            block = double_blocks[s.block_idx] if s.stream == Stream.DOUBLE else single_blocks[s.block_idx]
            handles.append(block.register_forward_hook(make_capture_hook(s, poem_acts)))
            
        _generate(pipe, hooker, poem["poem"], SEED, HookMode.OFF) 
        
        for h in handles: h.remove()
        handles = []
        for s in sites:
            block = double_blocks[s.block_idx] if s.stream == Stream.DOUBLE else single_blocks[s.block_idx]
            handles.append(block.register_forward_hook(make_capture_hook(s, lit_acts)))
            
        _generate(pipe, hooker, poem["literal"], SEED, HookMode.OFF)
        
        for h in handles: h.remove()

        for s in sites:
            k = s.key()
            if k in poem_acts and k in lit_acts:
                diff = poem_acts[k] - lit_acts[k]
                if accum[k] is None:
                    accum[k] = diff
                else:
                    accum[k] = accum[k] + diff

        n += 1
        done_ids.add(poem["id"])

        if n % 5 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"accum": accum, "n": n, "done_ids": done_ids}, f)

        gc.collect()
        torch.cuda.empty_cache()

    vectors = {}
    for s in sites:
        acc = accum[s.key()]
        if acc is not None:
            mean_vec = acc / n
            vectors[s.key()] = mean_vec / mean_vec.norm() 

    return vectors

# =============================================================================
# STEERING HOOKS
# =============================================================================
def apply_steering_hooks(hooker, sites, vectors, alpha):
    handles = []
    double_blocks = hooker.transformer.transformer_blocks
    single_blocks = hooker.transformer.single_transformer_blocks

    sites_by_block = {}
    for s in sites:
        sites_by_block.setdefault((s.stream, s.block_idx), []).append(s)

    if isinstance(vectors, list):
        vec_dict = {s.key(): v for s, v in vectors}
    else:
        vec_dict = vectors

    for (stream, block_idx), block_sites in sites_by_block.items():
        block = double_blocks[block_idx] if stream == Stream.DOUBLE else single_blocks[block_idx]
        step_to_vec = {s.step: vec_dict[s.key()] for s in block_sites if s.key() in vec_dict}

        def make_hook():
            def hook(module, inputs, output):
                current_step = hooker._step_idx
                if current_step not in step_to_vec:
                    return output
                vec = step_to_vec[current_step]
                
                def align_and_add(out_tensor, vec_tensor):
    # 1. Ensure vector is on the same device and dtype
                    vec_tensor = vec_tensor.to(out_tensor.device, out_tensor.dtype)
    
    # 2. If dimensions don't match, align them
                    if out_tensor.dim() == 2 and vec_tensor.dim() == 3:
                        vec_tensor = vec_tensor.squeeze(0) # Remove batch dim from vec
                    elif out_tensor.dim() == 3 and vec_tensor.dim() == 2:
                        vec_tensor = vec_tensor.unsqueeze(0) # Add batch dim to vec
                        
                    # 3. Match the sequence length by slicing
                    if out_tensor.shape[1] != vec_tensor.shape[1]:
                        # If vec is longer than out, slice vec. If out is longer, slice out (or pad).
                        # Most likely vec is 1024 and out is 512
                        seq_len = min(out_tensor.shape[1], vec_tensor.shape[1])
                        out_tensor = out_tensor[:, :seq_len, :]
                        vec_tensor = vec_tensor[:, :seq_len, :]
                        
                    return out_tensor + (alpha * vec_tensor)

                if isinstance(output, tuple):
                    new_out = list(output)
                    for i, v in enumerate(vec):
                        if isinstance(v, torch.Tensor) and isinstance(new_out[i], torch.Tensor):
                            new_out[i] = align_and_add(new_out[i], v)
                    return tuple(new_out)
                else:
                    if isinstance(output, torch.Tensor) and isinstance(vec, torch.Tensor):
                        return align_and_add(output, vec)
                    return output
            return hook
        handles.append(block.register_forward_hook(make_hook()))
    return handles

# =============================================================================
# COMPARISON
# =============================================================================
def run_comparison(pipe, hooker, poems, local_sites, local_vectors, global_sites, global_vectors, out_dir):
    csv_path = os.path.join(out_dir, "three_way_comparison.csv")
    fieldnames = ["poem_id", "condition", "figurative_alignment", "literal_alignment", "gap"]

    already_done = set()
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            existing = pd.read_csv(csv_path)
            if "poem_id" in existing.columns:
                counts = existing.groupby("poem_id").size()
                already_done = set(counts[counts >= 4].index)
                print(f"Resuming comparison: {len(already_done)} poem(s) already complete.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV ({e}). Starting fresh.")

    import csv as csv_module
    csv_file = open(csv_path, "a", newline="")
    writer = csv_module.DictWriter(csv_file, fieldnames=fieldnames)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        writer.writeheader()
        csv_file.flush()

    for poem in tqdm(poems[:SANITY_N_POEMS], desc="Comparison"):
        if poem["id"] in already_done:
            continue

        fig_target = get_figurative_target(poem)

        # 1. Vanilla
        img, _ = _generate(pipe, hooker, poem["poem"], SEED, HookMode.OFF)
        m = compute_metrics(img, fig_target, poem["literal"])
        writer.writerow({"poem_id": poem["id"], "condition": "Vanilla", **m})
        csv_file.flush()
        os.fsync(csv_file.fileno())
        print(f"  [Vanilla] {poem['id']}: gap={m['gap']:+.3f}")

        # 2. Local FigSteer
        hooker.reset_step_counter()
        handles = apply_steering_hooks(hooker, local_sites, local_vectors, CHOSEN_ALPHA)
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        res = pipe(prompt=poem["poem"], height=512, width=512, num_inference_steps=NUM_STEPS, guidance_scale=0.0, generator=gen)
        m = compute_metrics(res.images[0], fig_target, poem["literal"])
        writer.writerow({"poem_id": poem["id"], "condition": "FigSteer (Local)", **m})
        csv_file.flush()
        os.fsync(csv_file.fileno())
        print(f"  [FigSteer] {poem['id']}: gap={m['gap']:+.3f}")
        for h in handles: h.remove()

        # 3. Global Steering (Norm-Matched)
        global_alpha = CHOSEN_ALPHA * np.sqrt(len(local_sites) / len(global_sites))
        hooker.reset_step_counter()
        handles = apply_steering_hooks(hooker, global_sites, global_vectors, global_alpha)
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        res = pipe(prompt=poem["poem"], height=512, width=512, num_inference_steps=NUM_STEPS, guidance_scale=0.0, generator=gen)
        m = compute_metrics(res.images[0], fig_target, poem["literal"])
        writer.writerow({"poem_id": poem["id"], "condition": "Global (Norm-Matched)", **m})
        csv_file.flush()
        os.fsync(csv_file.fileno())
        print(f"  [Global] {poem['id']}: gap={m['gap']:+.3f}")
        for h in handles: h.remove()

        # 4. Random 5-Site Baseline
        random_sites = list(np.random.choice(global_sites, size=5, replace=False))
        random_vectors = []
        for s in random_sites:
            k = s.key()
            if k in global_vectors:
                random_vectors.append((s, global_vectors[k]))
        
        hooker.reset_step_counter()
        handles = apply_steering_hooks(hooker, random_sites, random_vectors, CHOSEN_ALPHA)
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        res = pipe(prompt=poem["poem"], height=512, width=512, num_inference_steps=NUM_STEPS, guidance_scale=0.0, generator=gen)
        m = compute_metrics(res.images[0], fig_target, poem["literal"])
        writer.writerow({"poem_id": poem["id"], "condition": "Random 5-Site", **m})
        csv_file.flush()
        os.fsync(csv_file.fileno())
        print(f"  [Random] {poem['id']}: gap={m['gap']:+.3f}")
        for h in handles: h.remove()

        gc.collect()
        torch.cuda.empty_cache()

    csv_file.close()
    df = pd.read_csv(csv_path)
    print("\n=== Mean Metrics by Condition ===")
    print(df.groupby("condition")[["figurative_alignment", "literal_alignment", "gap"]].mean().to_string())
    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pipe = load_pipeline()
    hooker = FluxActivationHooker(pipe.transformer)
    
    all_sites = get_all_sites(pipe.transformer)
    if os.path.exists(GLOBAL_VECTOR_CACHE_PATH):
        with open(GLOBAL_VECTOR_CACHE_PATH, "rb") as f:
            global_vectors = pickle.load(f)
    else:
        vector_poems = load_poem_pairs(VECTOR_POEMS_PATH)[:50]
        global_vectors = build_global_steering_vectors(pipe, hooker, vector_poems, all_sites, OUT_DIR)
        with open(GLOBAL_VECTOR_CACHE_PATH, "wb") as f:
            pickle.dump(global_vectors, f)

    with open(LOCAL_VECTOR_PATH, "rb") as f:
        local_vectors = pickle.load(f)
    local_sites = [s for s, v in local_vectors]

    sanity_poems = load_poem_pairs(SANITY_POEMS_PATH)
    run_comparison(pipe, hooker, sanity_poems, local_sites, local_vectors, all_sites, global_vectors, OUT_DIR)

if __name__ == "__main__":
    main()
