"""
Motility Assay Statistical Testing – CC-2931 MA vs CC-2931 ANC
--------------------------------------------------------------
Author: Ayesha Syeda
Lab: Ness Lab, University of Toronto Mississauga

Purpose
-------
Compare motility (Maximum Distance) between the CC-2931 Mutation 
Accumulation (MA) strain and its ancestral (ANC) counterpart.

These tests assess whether accumulated mutations led to measurable 
differences in motility relative to the ancestor. 
Because sampling was not randomized, results are descriptive rather than inferential.

Input
-----
- motility_assay_clean.csv (from preprocessing script)

Output
------
- anova_results_CC2931MA_vs_CC2931ANC.csv
- permutation_results_CC2931MA_vs_CC2931ANC.csv
- pairwise_results_CC2931MA_vs_CC2931ANC.csv

Usage
-----
    python 03h_stats_CC2931MA_vs_CC2931ANC.py
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
subset = subset[subset["type"].isin(["CC-2931 MA", "CC-2931 ANC"])].copy()
print(f"Subset loaded: {subset.shape[0]} observations ({subset['type'].unique()})")

# ==============================================================
# 3. Parametric Test: One-way ANOVA
# ==============================================================

"""
Test: One-way ANOVA
-------------------
Purpose:
    Test whether mean motility (Maximum Distance) differs between 
    CC-2931 MA and CC-2931 ANC strains, adjusting for 'variable' (experimental week)
    as a blocking factor.

Model:
    value ~ C(type) + C(variable)

Interpretation:
    A non-significant type effect indicates no evidence of a difference
    in average motility between the MA and ancestral strains.
"""

model = smf.ols("value ~ C(type) + C(variable)", data=subset).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table.to_csv("anova_results_CC2931MA_vs_CC2931ANC.csv", index=True)
print("\nOne-way ANOVA results:")
print(anova_table)

# ==============================================================
# 4. Non-Parametric: Global Permutation Test
# ==============================================================

"""
Test: Global Permutation Test
-----------------------------
Purpose:
    Assess whether CC-2931 MA and CC-2931 ANC differ in mean motility 
    more than expected by chance.

Logic:
    1. Compute observed between-group variance (weighted by group size).
    2. Shuffle type labels 10,000 times.
    3. Recompute variance for each permutation.
    4. p-value = fraction of permuted variances ≥ observed variance.

Interpretation:
    A high p-value indicates no significant difference between the two groups.
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
    Test whether the mean motility difference between CC-2931 MA and CC-2931 ANC 
    is greater than expected under random reassignment of values.

Logic:
    1. Compute the observed mean difference.
    2. Shuffle pooled values 10,000 times.
    3. Recompute mean differ
