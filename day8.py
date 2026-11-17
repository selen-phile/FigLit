"""
day8_full_eval.py - Final 4-Way Comparison (Vanilla, FigSteer, Global)
"""
import csv as csv_module
import gc
import json
import os
import pickle
import torch
import pandas as pd
import numpy as np
from diffusers import FluxPipeline
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPProcessor
from tqdm import tqdm

from flux_hooks import FluxActivationHooker, HookMode, Site, Stream
from patch_and_generate import _generate

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

# =============================================================================
# CLIP METRICS (Cached)
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
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
NUM_STEPS = 4
SEED = 42

HELDOUT_POEMS_PATH = "fig200_fresh_200.json"
LOCAL_VECTOR_PATH = "day6_out/steering_vectors.pkl"
GLOBAL_VECTOR_PATH = "day7_out/global_steering_vectors.pkl"

# TEST VALUES: Change to 20 / 0.1 for final run
CHOSEN_ALPHA = 1.0
NUM_EVAL_POEMS = 20
OUT_DIR = "day8_out"
SAVE_IMAGES = True

def load_pipeline():
    print(f"Loading {MODEL_ID} directly to {DEVICE} (Eager Mode) ...")
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True)
    pipe.to(DEVICE)
    
    # THE SURGICAL FIX: Bypass torch.compile completely
    import diffusers
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
    
    # Unwrap the forward method
    if hasattr(FluxTransformer2DModel.forward, "__wrapped__"):
        FluxTransformer2DModel.forward = FluxTransformer2DModel.forward.__wrapped__
        
    try:
        torch._dynamo.reset()
        torch.compiler.reset()
    except:
        pass
        
    if hasattr(pipe.transformer, "compile"):
        pipe.transformer.compile = False
        
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe

def load_poem_pairs(path):
    with open(path) as f:
        return json.load(f)

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =============================================================================
# STEERING HOOKS
# =============================================================================
def apply_steering(hooker, sites, vectors, alpha):
    transformer = hooker.transformer
    double_blocks = transformer.transformer_blocks
    single_blocks = transformer.single_transformer_blocks

    sites_by_block = {}
    for s in sites:
        sites_by_block.setdefault((s.stream, s.block_idx), []).append(s)

    if isinstance(vectors, list):
        vec_dict = {s.key(): v for s, v in vectors}
    else:
        vec_dict = vectors

    print(f"\n[Steering] Attaching hooks to {len(sites_by_block)} blocks (alpha={alpha})...")
    
    handles = []

    for (stream, block_idx), block_sites in sites_by_block.items():
        block = double_blocks[block_idx] if stream == Stream.DOUBLE else single_blocks[block_idx]
        step_to_vec = {s.step: vec_dict[s.key()] for s in block_sites if s.key() in vec_dict}

        for s in block_sites:
            if s.key() in vec_dict:
                v_norm = vec_dict[s.key()].norm().item()
                if v_norm < 1e-6:
                    raise ValueError(f"CRITICAL: Vector for block {block_idx} is ALL ZEROS.")
                print(f"  [Verified] Block {block_idx} Step {s.step} | Norm: {v_norm:.4f}")

        def make_hook(step_dict, b_idx):
            def hook(module, inputs, output):
                current_step = hooker._step_idx
                if current_step not in step_dict:
                    return output
                vec = step_dict[current_step]
                
                def align_and_add(out_tensor, vec_tensor):
                    if out_tensor.dim() == 2 and vec_tensor.dim() == 3:
                        vec_tensor = vec_tensor.squeeze(0)
                    elif out_tensor.dim() == 3 and vec_tensor.dim() == 2:
                        vec_tensor = vec_tensor.unsqueeze(0)
                    if out_tensor.shape[1] != vec_tensor.shape[1]:
                        seq_len = min(out_tensor.shape[1], vec_tensor.shape[1])
                        out_tensor = out_tensor[:, :seq_len, :]
                        vec_tensor = vec_tensor[:, :seq_len, :]
                    vec_tensor = vec_tensor.to(out_tensor.device, out_tensor.dtype)
                    
                    new_tensor = out_tensor + (alpha * vec_tensor)
                    diff = (new_tensor - out_tensor).abs().mean().item()
                    print(f"    [Hook Fired] Block {b_idx} Step {current_step} | Change: {diff:.6f}")
                    return new_tensor

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

        handle = block.register_forward_hook(make_hook(step_to_vec, block_idx))
        handles.append(handle)

    return handles

def remove_steering(handles):
    for h in handles:
        h.remove()

# =============================================================================
# MAIN EVALUATION
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img_dir = os.path.join(OUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    with open(LOCAL_VECTOR_PATH, "rb") as f:
        local_vectors = pickle.load(f)
    local_sites = [s for s, v in local_vectors]

    with open(GLOBAL_VECTOR_PATH, "rb") as f:
        global_vectors = pickle.load(f)
    global_sites = [Site(step=k[0], stream=Stream(k[1]), block_idx=k[2]) for k in global_vectors.keys()]

    poems = load_poem_pairs(HELDOUT_POEMS_PATH)[:NUM_EVAL_POEMS]

    csv_path = os.path.join(OUT_DIR, "day8_full_comparison.csv")
    fieldnames = ["poem_id", "condition", "figurative_alignment", "literal_alignment", "gap"]
    
    csv_file = open(csv_path, "a", newline="")
    writer = csv_module.DictWriter(csv_file, fieldnames=fieldnames)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        writer.writeheader()
        csv_file.flush()

    pipe = load_pipeline()
    hooker = FluxActivationHooker(pipe.transformer)

    print(f"Running evaluation on {len(poems)} poems...")
    for poem in tqdm(poems, desc="Evaluation"):
        fig_target = get_figurative_target(poem)
        
        # --- 1. Vanilla ---
        print(f"\n[{poem['id']}] Generating Vanilla...")
        img, _ = _generate(pipe, hooker, poem["poem"], SEED, HookMode.OFF)
        m = compute_metrics(img, fig_target, poem["literal"])
        writer.writerow({"poem_id": poem["id"], "condition": "Vanilla", **m})
        csv_file.flush(); os.fsync(csv_file.fileno())
        if SAVE_IMAGES: img.save(os.path.join(img_dir, f"{poem['id']}_vanilla.png"))
        del img
        cleanup()

        # --- 2. FigSteer (Local) ---
        print(f"[{poem['id']}] Generating FigSteer...")
        hooker.reset_step_counter()
        handles = apply_steering(hooker, local_sites, local_vectors, CHOSEN_ALPHA)
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        res = pipe(prompt=poem["poem"], height=512, width=512, num_inference_steps=NUM_STEPS, guidance_scale=0.0, generator=gen)
        m = compute_metrics(res.images[0], fig_target, poem["literal"])
        writer.writerow({"poem_id": poem["id"], "condition": "FigSteer", **m})
        csv_file.flush(); os.fsync(csv_file.fileno())
        if SAVE_IMAGES: res.images[0].save(os.path.join(img_dir, f"{poem['id']}_figsteer.png"))
        remove_steering(handles)
        del res
        cleanup()

        # --- 3. Global (Norm-Matched) ---
        print(f"[{poem['id']}] Generating Global...")
        global_alpha = CHOSEN_ALPHA * np.sqrt(len(local_sites) / len(global_sites))
        hooker.reset_step_counter()
        handles = apply_steering(hooker, global_sites, global_vectors, global_alpha)
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        res = pipe(prompt=poem["poem"], height=512, width=512, num_inference_steps=NUM_STEPS, guidance_scale=0.0, generator=gen)
        m = compute_metrics(res.images[0], fig_target, poem["literal"])
        writer.writerow({"poem_id": poem["id"], "condition": "Global", **m})
        csv_file.flush(); os.fsync(csv_file.fileno())
        if SAVE_IMAGES: res.images[0].save(os.path.join(img_dir, f"{poem['id']}_global.png"))
        remove_steering(handles)
        del res
        cleanup()

    csv_file.close()
    
    df = pd.read_csv(csv_path)
    print("\n=== Mean Metrics by Condition ===")
    print(df.groupby("condition")[["figurative_alignment", "literal_alignment", "gap"]].mean().to_string())

if __name__ == "__main__":
    main()