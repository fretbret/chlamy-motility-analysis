"""
Motility Assay Statistical Testing – NI vs CC-2344 MA
-----------------------------------------------------
Author: Ayesha Syeda
Lab: Ness Lab, University of Toronto Mississauga
Date: YYYY-MM-DD

Purpose
-------
Compare motility (Maximum Distance) between Natural Isolates (NI) 
and CC-2344 Mutation Accumulation (MA) strains.

These analyses are descriptive because the samples were not randomized.
Tests evaluate whether NI and CC-2344 MA differ in their mean motility values
or distributions.

Input
-----
- motility_assay_clean.csv (from preprocessing script)

Output
------
- anova_results_NI_vs_CC2344MA.csv
- permutation_results_NI_vs_CC2344MA.csv
- pairwise_results_NI_vs_CC2344MA.csv

Usage
-----
    python 03d_NI_vs_CC2344MA.py
"""

# ==============================================================
# 1. Import Packages
# ==============================================================

import pandas as pd
import numpy as np
from itertools import combinations
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# ==============================================================
# 2. Load Data
# ==============================================================

df = pd.read_csv("motility_assay_clean.csv", sep="\t")
subset = df[df["distance"] == "Maximum Distance"].copy()
subset = subset[subset["type"].isin(["NI", "CC-2344 MA"])].copy()
print(f"Subset loaded: {subset.shape[0]} observations ({subset['type'].unique()})")

# ==============================================================
# 3. Parametric Test: One-way ANOVA
# ==============================================================

"""
Test: One-way ANOVA
-------------------
Purpose:
    Tests whether NI and CC-2344 MA differ in mean motility (Maximum Distance),
    adjusting for experimental week ('variable') as a blocking factor.

Model:
    value ~ C(type) + C(variable)

Interpretation:
    A non-significant type effect indicates no strong evidence that 
    NI and CC-2344 MA differ in average motility.
"""

model = smf.ols("value ~ C(type) + C(variable)", data=subset).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table.to_csv("anova_results_NI_vs_CC2344MA.csv", index=True)
print("\nOne-way ANOVA results:")
print(anova_table)

# ==============================================================
# 4. Non-Parametric: Global Permutation Test
# ==============================================================

"""
Test: Global Permutation (Nonparametric ANOVA analog)
----------------------------------------------------
Purpose:
    Tests whether the observed between-group variance (NI vs CC-2344 MA)
    is greater than expected by chance.

Logic:
    - Compute observed between-group variance.
    - Shuffle type labels 10,000 times.
    - Compute variance each time to build null distribution.
    - p-value = proportion of permuted variances ≥ observed variance.

Interpretation:
    A high p-value indicates no significant difference between NI and CC-2344 MA.
"""

df_subset = subset[["type", "value"]].copy()
group_means = df_subset.groupby("type")["value"].mean()
overall_mean = df_subset["value"].mean()
group_sizes = df_subset.groupby("type").size()
between_group_var = np.sum(group_sizes * (group_means - overall_mean) ** 2)

# Permutation test
n_permutations = 10000
perm_stats = []
for _ in range(n_permutations):
    shuffled_type = np.random.permutation(df_subset["type"].values)
    df_shuffled = df_subset.copy()
    df_shuffled["shuffled_type"] = shuffled_type
    means_perm = df_shuffled.groupby("shuffled_type")["value"].mean()
    sizes_perm = df_shuffled.groupby("shuffled_type").size()
    perm_var = np.sum(sizes_perm * (means_perm - overall_mean) ** 2)
    perm_stats.append(perm_var)

perm_stats = np.array(perm_stats)
p_value_perm = np.mean(perm_stats >= between_group_var)
print("\nGlobal permutation test:")
print(f"Observed between-group variance: {between_group_var:.2f}")
print(f"P-value: {p_value_perm:.4f}")

# ==============================================================
# 5. Pairwise Permutation Test (Mean Differences)
# ==============================================================

"""
Test: Pairwise Permutation (Mean Differences)
---------------------------------------------
Purpose:
    Directly tests whether the difference in mean motility between 
    NI and CC-2344 MA is greater than expected by random chance.

Logic:
    - Compute observed mean difference.
    - Shuffle pooled values 10,000 times.
    - Compute mean difference for each permutation.
    - p-value = fraction of permuted differences ≥ observed difference.
    - Adjust for multiple comparisons (FDR).
"""

df_subset = subset[["type", "value"]].copy()
types = df_subset["type"].unique()
results = []

for t1, t2 in combinations(types, 2):
    v1 = df_subset[df_subset["type"] == t1]["value"].values
    v2 = df_subset[df_subset["type"] == t2]["value"].values
    observed_diff = np.abs(np.mean(v1) - np.mean(v2))
    combined = np.concatenate([v1, v2])
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        new_v1 = combined[:len(v1)]
        new_v2 = combined[len(v1):]
        perm_diff = np.abs(np.mean(new_v1) - np.mean(new_v2))
        if perm_diff >= observed_diff:
            count += 1
    p_value = count / n_permutations
    results.append({"type1": t1, "type2": t2, "mean_diff": observed_diff, "p_value": p_value})

pairwise_results = pd.DataFrame(results)
pairwise_results["p_adj"] = multipletests(pairwise_results["p_value"], method="fdr_bh")[1]
pairwise_results.to_csv("pairwise_results_NI_vs_CC2344MA.csv", index=False)

sig_pairs = pairwise_results[pairwise_results["p_adj"] < 0.05]
print("\nPairwise permutation results:")
print(f"Significant pairs: {len(sig_pairs)}")
print(sig_pairs[["type1", "type2", "mean_diff", "p_value", "p_adj"]] if not sig_pairs.empty else "None detected.")
