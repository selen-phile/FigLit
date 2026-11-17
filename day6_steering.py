"""
rebuild_vectors.py - Fast, Bulletproof Vector Construction
"""
import os
import json
import pickle
import torch
import pandas as pd
import gc
from diffusers import FluxPipeline
from tqdm import tqdm
from flux_hooks import FluxActivationHooker, HookMode, Site, Stream
from patch_and_generate import _generate

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
NUM_STEPS = 4
SEED = 42

POEMS_PATH = "fig200_fine_sweep_500.json"
FINE_CSV = "day4_out/day4_all_blocks_results.csv"
OUT_PATH = "day6_out/steering_vectors.pkl"

def load_pipeline():
    print(f"Loading {MODEL_ID} to GPU (Eager Mode)...")
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True)
    pipe.to(DEVICE)
    try:
        torch._dynamo.reset()
        if hasattr(torch, "compiler"):
            torch.compiler.reset()
    except:
        pass
    if hasattr(pipe.transformer, "compile"):
        pipe.transformer.compile = False
    return pipe

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    
    print("Reading fine sweep results...")
    df = pd.read_csv(FINE_CSV)
    df = df[(df['effect'] != 0.0) & (df['effect'].notna())]
    agg = df.groupby(["region", "block_idx", "step"])["effect"].agg(['mean', 'std', 'count']).reset_index()
    agg = agg[agg['count'] > 100]
    agg['score'] = agg['mean'].abs() / (agg['std'] + 1e-5)
    top = agg.sort_values("score", ascending=False).head(5)
    
    sites = []
    for _, row in top.iterrows():
        stream = Stream.DOUBLE if "double" in row["region"] else Stream.SINGLE
        sites.append(Site(step=int(row["step"]), stream=stream, block_idx=int(row["block_idx"])))
    
    print("\nTop sites being used:")
    for s in sites:
        print(f"  {s.stream.name} block {s.block_idx} step {s.step}")

    with open(POEMS_PATH) as f:
        poems = json.load(f)[:20]
    
    pipe = load_pipeline()
    hooker = FluxActivationHooker(pipe.transformer)
    
    accum = {s.key(): None for s in sites}
    n = 0

    print("\nExtracting activations...")
    for poem in tqdm(poems):
        poem_acts, lit_acts = {}, {}
        
        def make_capture_hook(site, storage):
            def hook(module, inputs, output):
                if hooker._step_idx == site.step: 
                    if isinstance(output, tuple):
                        storage[site.key()] = output[1].clone().detach().cpu()
                    else:
                        storage[site.key()] = output.clone().detach().cpu()
            return hook

        handles = []
        for s in sites:
            block = pipe.transformer.transformer_blocks[s.block_idx] if s.stream == Stream.DOUBLE else pipe.transformer.single_transformer_blocks[s.block_idx]
            handles.append(block.register_forward_hook(make_capture_hook(s, poem_acts)))
        _generate(pipe, hooker, poem["poem"], SEED, HookMode.OFF)
        for h in handles: h.remove()

        handles = []
        for s in sites:
            block = pipe.transformer.transformer_blocks[s.block_idx] if s.stream == Stream.DOUBLE else pipe.transformer.single_transformer_blocks[s.block_idx]
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
        gc.collect()
        torch.cuda.empty_cache()

    vectors = []
    print("\nFinal Vector Magnitudes (must be > 0):")
    
    total_mag = 0
    count = 0
    for s in sites:
        acc = accum[s.key()]
        if acc is not None:
            total_mag += acc.norm().item()
            count += 1
    avg_mag = total_mag / max(count, 1)
    print(f"Average activation magnitude: {avg_mag:.4f}")
    
    for s in sites:
        acc = accum[s.key()]
        if acc is not None:
            mean_vec = acc / n
            norm = mean_vec.norm()
            print(f"  Block {s.block_idx} Step {s.step}: raw |v| = {norm:.4f}")
            if norm > 1e-5:
                # SCALE BY THE AVERAGE ACTIVATION MAGNITUDE
                # FLIP THE SIGN
                vectors.append((s, -(mean_vec / norm) * avg_mag))
    
    if len(vectors) == 0:
        print("CRITICAL ERROR: All vectors are zero.")
        return

    with open(OUT_PATH, "wb") as f:
        pickle.dump(vectors, f)
    print(f"\nSaved {len(vectors)} scaled vectors to {OUT_PATH}")

if __name__ == "__main__":
    main()