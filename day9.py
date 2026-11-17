"""
day9_qualitative_figure.py - Build the qualitative figure for the paper.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Config
CSV_PATH = "day8_out/day8_full_comparison.csv"
IMG_DIR = "day8_out/images"
HELDOUT_POEMS_PATH = "fig200_fresh_200.json" 
OUT_PATH = "day9_out/qualitative_comparison.png"
N_ROWS = 3  # 3 examples is usually enough for a good figure

def truncate(text, max_chars=60):
    text = text.replace("\n", " ")
    return text[:max_chars] + ("..." if len(text) > max_chars else "")

def select_best_poems():
    """Selects poems where FigSteer improved the gap the most compared to Vanilla."""
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Could not find {CSV_PATH}")
        return []
        
    df = pd.read_csv(CSV_PATH)
    
    # Pivot so each row is a poem, columns are conditions
    pivot = df.pivot(index="poem_id", columns="condition", values="gap")
    
    # We want poems where Vanilla Gap is positive (model was literal)
    # and FigSteer Gap is lower (model became figurative)
    pivot["improvement"] = pivot["Vanilla"] - pivot["FigSteer"]
    
    # Sort by biggest improvement
    top_poems = pivot.sort_values("improvement", ascending=False).head(N_ROWS)
    
    print(f"\nTop {N_ROWS} poems by FigSteer improvement over Vanilla:")
    print(top_poems[["Vanilla", "FigSteer", "Global", "improvement"]].to_string())
    
    # Load poem texts
    with open(HELDOUT_POEMS_PATH) as f:
        all_poems = json.load(f)
    poem_dict = {p["id"]: p for p in all_poems}
    
    selected = []
    for p_id in top_poems.index:
        if p_id in poem_dict:
            selected.append(poem_dict[p_id])
            
    return selected

def build_figure(selected_poems):
    n_rows = len(selected_poems)
    if n_rows == 0:
        print("No poems selected. Aborting figure generation.")
        return
        
    # 3 Columns: Vanilla, FigSteer, Global
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4.5 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
        
    col_titles = ["Vanilla (Literal)", "FigSteer (Ours)", "Global (Fails)"]
    
    for row, poem in enumerate(selected_poems):
        p_id = poem["id"]
        prompt_text = truncate(poem["poem"])
        
        # Column 0: Vanilla
        img_van_path = os.path.join(IMG_DIR, f"{p_id}_vanilla.png")
        if os.path.exists(img_van_path):
            axes[row, 0].imshow(Image.open(img_van_path))
        axes[row, 0].set_ylabel(f"\"{prompt_text}\"", fontsize=9, rotation=0, ha="right", va="center", labelpad=20)
        
        # Column 1: FigSteer
        img_fig_path = os.path.join(IMG_DIR, f"{p_id}_figsteer.png")
        if os.path.exists(img_fig_path):
            axes[row, 1].imshow(Image.open(img_fig_path))
            
        # Column 2: Global
        img_glob_path = os.path.join(IMG_DIR, f"{p_id}_global.png")
        if os.path.exists(img_glob_path):
            axes[row, 2].imshow(Image.open(img_glob_path))
        
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=12, fontweight="bold")
                
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT_PATH}")

def main():
    selected = select_best_poems()
    if selected:
        build_figure(selected)

if __name__ == "__main__":
    main()