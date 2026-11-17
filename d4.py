import pandas as pd
import numpy as np

df = pd.read_csv("day4_all_blocks_results.csv")

# 1. Clean dead cells (effect == 0) and NaNs
df = df[df['effect'] != 0.0]
df = df.dropna(subset=['effect'])
print(f"Rows after cleaning: {len(df)}")

# 2. Find the top 10 absolute sites across all blocks
top_sites = df.groupby(['region', 'block_idx'])['effect'].agg(['mean', 'median', 'std', 'count']).reset_index()
top_sites['abs_mean'] = top_sites['mean'].abs()
top_sites = top_sites.sort_values('abs_mean', ascending=False)

print("\n=== TOP 10 SITES (Fine Sweep) ===")
print(top_sites.head(10).to_string())

# 3. Check the consistency of the top site across poems
best_site = top_sites.iloc[0]
print(f"\n=== Consistency of Top Site: {best_site['region']} block {best_site['block_idx']} ===")
poem_effects = df[(df['region'] == best_site['region']) & 
                  (df['block_idx'] == best_site['block_idx'])]['effect']

# How many poems have a negative effect (steering literal) vs positive?
neg_poems = (poem_effects < -0.5).sum()
pos_poems = (poem_effects > 0.5).sum()
neutral = (poem_effects.abs() <= 0.5).sum()

print(f"Poems strongly shifted to figurative (neg): {neg_poems}")
print(f"Poems strongly shifted to literal (pos): {pos_poems}")
print(f"Poems unaffected: {neutral}")