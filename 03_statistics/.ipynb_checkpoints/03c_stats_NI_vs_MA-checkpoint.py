"""
Motility Assay Statistical Testing – NI vs Pooled MA
---------------------------------------------------
Author: Ayesha Syeda
Lab: Ness Lab, University of Toronto Mississauga

Purpose
-------
Compare motility (Maximum Distance) between Natural Isolates (NI) 
and pooled Mutation Accumulation (MA) lines.

Since samples were not randomized, these results are descriptive.
The tests evaluate whether NI and MA lines differ in average 
or distributional motility patterns.

Input
-----
- motility_assay_clean.csv (from preprocessing script)

Output
------
- anova_results_NI_vs_MA.csv
- mixedlm_results_NI_vs_MA.txt
- permutation_results_NI_vs_MA.csv
- pairwise_results_NI_vs_MA.csv

Usage
-----
    python 03c_stats_NI_vs_MA.py
"""

# ==============================================================
# 1. Import Packages
# ==============================================================

import pandas as pd
import numpy as np
from itertools import combinations
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# ==============================================================
# 2. Load Data
# ==============================================================

df = pd.read_csv("motility_assay_clean.csv", sep="\t")
subset = df[df["distance"] == "Maximum Distance"].copy()
subset = subset[subset["broad_type"].isin(["NI", "MA"])].copy()
print(f" Subset loaded: {subset.shape[0]} observations ({subset['broad_type'].unique()})")


# ==============================================================
# 3. Parametric Test: One-way ANOVA
# ==============================================================

"""
Test: One-way ANOVA
-------------------
Purpose:
    Tests whether NI and MA lines differ in mean motility (Maximum Distance),
    adjusting for experiment (variable) as a nuisance factor.

Model:
    value ~ C(broad_type) + C(variable)

Interpretation:
    A non-significant type effect indicates no strong evidence of a difference
    between NI and pooled MA means.
"""

model = smf.ols("value ~ C(broad_type) + C(variable)", data=subset).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table.to_csv("anova_results_NI_vs_MA.csv")
print("\n One-way ANOVA (NI vs MA):")
print(anova_table)


# ==============================================================
# 4. Linear Mixed-Effects Model (nested strain)
# ==============================================================

"""
Test: Linear Mixed-Effects Model
--------------------------------
Purpose:
    Incorporates strain as a nested random effect within type
    to account for within-type heterogeneity.

Model:
    value ~ broad_type + variable + (1 | broad_type:strain2)

Notes:
    Python lacks the exact equivalent of R's aov(... + Error(...)).
    MixedLM approximates it. Significance of fixed effects (type) can
    be assessed by comparing full vs reduced models (not shown here).
"""

subset["broad_type"] = subset["broad_type"].astype("category")
subset["variable"] = subset["variable"].astype("category")
subset["strain2"] = subset["strain2"].astype("category")
subset["broad_type_strain2"] = subset["broad_type"].astype(str) + ":" + subset["strain2"].astype(str)

model_mixed = smf.mixedlm("value ~ broad_type + variable", 
                          data=subset, 
                          groups=subset["broad_type_strain2"])
result = model_mixed.fit()
print("\n Mixed-effects model summary:")
print(result.summary())
with open("mixedlm_results_NI_vs_MA.txt", "w") as f:
    f.write(result.summary().as_text())


# ==============================================================
# 5. Non-Parametric: Global Permutation Test
# ==============================================================

"""
Test: Global Permutation
------------------------
Purpose:
    Tests whether NI and MA differ in mean motility more than expected by chance.
Logic:
    - Compute between-group variance of means.
    - Shuffle type labels, recompute variance 10,000×.
    - p-value = proportion of permutations ≥ observed variance.
"""

df_subset = subset[["broad_type", "value"]].copy()
group_means = df_subset.groupby("broad_type")["value"].mean()
overall_mean = df_subset["value"].mean()
group_sizes = df_subset.groupby("broad_type").size()
between_group_var = np.sum(group_sizes * (group_means - overall_mean) ** 2)

n_permutations = 10000
perm_stats = []
for _ in range(n_permutations):
    shuffled = np.random.permutation(df_subset["broad_type"].values)
    df_shuffled = df_subset.copy()
    df_shuffled["shuffled_broad_type"] = shuffled
    means_perm = df_shuffled.groupby("shuffled_broad_type")["value"].mean()
    sizes_perm = df_shuffled.groupby("shuffled_broad_type").size()
    perm_var = np.sum(sizes_perm * (means_perm - overall_mean) ** 2)
    perm_stats.append(perm_var)

perm_stats = np.array(perm_stats)
p_value_perm = np.mean(perm_stats >= between_group_var)
print(f"\n Global permutation p-value: {p_value_perm:.4f}")


# ==============================================================
# 6. Pairwise Permutation (Mean Differences)
# ==============================================================

"""
Test: Pairwise Permutation (Mean Differences)
---------------------------------------------
Purpose:
    Compare NI vs MA mean differences under random shuffling.

Logic:
    - Compute observed mean difference.
    - Shuffle pooled values 10,000×, recompute difference.
    - p-value = proportion of permuted diffs ≥ observed.
    - Adjust for multiple testing (FDR).
"""

broad_types = df_subset["broad_type"].unique()
results = []
for t1, t2 in combinations(broad_types, 2):
    v1 = df_subset[df_subset["broad_type"] == t1]["value"].values
    v2 = df_subset[df_subset["broad_type"] == t2]["value"].values
    obs_diff = np.abs(np.mean(v1) - np.mean(v2))
    combined = np.concatenate([v1, v2])
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        new_v1, new_v2 = combined[:len(v1)], combined[len(v1):]
        perm_diff = np.abs(np.mean(new_v1) - np.mean(new_v2))
        if perm_diff >= obs_diff:
            count += 1
    p_value = count / n_permutations
    results.append({"broad_type1": t1, "broad_type2": t2,
                    "mean_diff": obs_diff, "p_value": p_value})

pairwise_df = pd.DataFrame(results)
pairwise_df["p_adj"] = multipletests(pairwise_df["p_value"], method="fdr_bh")[1]
pairwise

