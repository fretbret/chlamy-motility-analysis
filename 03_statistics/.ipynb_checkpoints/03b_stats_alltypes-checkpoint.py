"""
Motility Assay Statistical Testing – Type-wise
----------------------------------------------
Author: Ayesha Syeda
Lab: Ness Lab, University of Toronto Mississauga

Purpose
-------
Tests for differences in motility (maximum distance) between *types* of
Chlamydomonas samples (NI, ANC, MA, and subtypes such as CC-2344 MA, CC-2931 MA, etc.).

Since this dataset involves fixed measurements (no random allocation or sampling),
results are interpreted as descriptive differences rather than inferential
generalizations.

Tests included:
1. Parametric tests (ANOVA-based)
2. Nonparametric tests (permutation-based, Kruskal–Wallis, and Dunn)
3. Pairwise permutation tests for mean and variance
4. Levene’s test for equality of variances

All analyses use *Maximum Distance* as the response variable.

Input
-----
- motility_assay_clean.csv (from preprocessing script)

Output
------
- permutation_results_type.csv
- pairwise_permutation_results_type.csv
- dunn_results_type.csv
- variance_posthoc_type.csv
- printed test summaries

Usage
-----
Run this script in the same directory as motility_assay_clean.csv:
    python 03b_stats_alltypes.py
"""

# ==============================================================
# 1. Import Required Packages
# ==============================================================

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import kruskal, levene
import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests


# ==============================================================
# 2. Load Data
# ==============================================================

df = pd.read_csv("motility_assay_clean.csv", sep="\t")
subset = df[df["distance"] == "Maximum Distance"].copy()
print(f" Loaded dataset: {subset.shape[0]} rows, {subset['type'].nunique()} types")


# ==============================================================
# 3. Global Permutation Test (Nonparametric ANOVA Analog)
# ==============================================================

"""
Test: Global Permutation (Nonparametric ANOVA Analog)
----------------------------------------------------
Purpose:
    To assess whether the mean motility (maximum distance) differs among types
    more than would be expected by chance.

Logic:
    1. Compute observed between-group variance of type means.
    2. Randomly shuffle type labels across all observations (break association between type and value).
    3. Recompute the between-group variance for each permutation.
    4. Compare observed variance to null distribution.

Interpretation:
    A low p-value suggests that at least one type’s mean differs significantly.
"""

df_subset = subset[["type", "value"]].copy()
group_means = df_subset.groupby("type")["value"].mean()
overall_mean = df_subset["value"].mean()
group_sizes = df_subset.groupby("type").size()
between_group_var = np.sum(group_sizes * (group_means - overall_mean) ** 2)

# Permutation
n_permutations = 10000
perm_stats = []
values = df_subset["value"].to_numpy()
group_sizes_array = group_sizes.to_numpy()

for _ in range(n_permutations):
    shuffled_values = np.random.permutation(values)
    start = 0
    perm_group_means = []
    for size in group_sizes_array:
        end = start + size
        group_vals = shuffled_values[start:end]
        perm_group_means.append(np.mean(group_vals))
        start = end
    perm_group_means = np.array(perm_group_means)
    perm_stat = np.sum(group_sizes_array * (perm_group_means - overall_mean) ** 2)
    perm_stats.append(perm_stat)

p_value_global = np.mean(np.array(perm_stats) >= between_group_var)
print(f"\n Global permutation test (type-level): p = {p_value_global:.4f}")


# ==============================================================
# 4. Permutation Test Using F-statistic as Test Statistic
# ==============================================================

"""
Extension: Permutation Test Using F-statistic
---------------------------------------------
Purpose:
    Analogous to an ANOVA F-test, but nonparametric.
    Compares between-group variance to within-group variance
    under repeated random shuffling of observations.

Interpretation:
    A low p-value indicates strong evidence that type means differ.
"""

# Group stats
df_between = len(group_sizes) - 1
df_within = len(df_subset) - len(group_sizes)
SSB = np.sum(group_sizes * (group_means - overall_mean) ** 2)
SSW = np.sum((df_subset["value"] - df_subset.groupby("type")["value"].transform("mean")) ** 2)
F_obs = (SSB / df_between) / (SSW / df_within)

# Permutation
perm_stats = []
values = df_subset["value"].to_numpy()

for _ in range(n_permutations):
    shuffled_values = np.random.permutation(values)
    start = 0
    perm_group_means = []
    SSW_perm = 0
    for size in group_sizes_array:
        end = start + size
        group_vals = shuffled_values[start:end]
        group_mean = np.mean(group_vals)
        SSW_perm += np.sum((group_vals - group_mean) ** 2)
        start = end
    SSB_perm = np.sum(group_sizes_array * (np.array(perm_group_means) - overall_mean) ** 2)
    F_perm = (SSB_perm / df_between) / (SSW_perm / df_within)
    perm_stats.append(F_perm)

p_value_F = np.mean(np.array(perm_stats) >= F_obs)
print(f"Observed F-statistic: {F_obs:.4f}")
print(f"P-value (permutation F-test): {p_value_F:.4f}")


# ==============================================================
# 5. Levene’s Test for Homogeneity of Variance
# ==============================================================

"""
Test: Levene’s Test
-------------------
Purpose:
    To evaluate whether the variance of motility (maximum distance)
    is equal across all type groups.

Logic:
    Compares absolute deviations from the median across groups.

Interpretation:
    A low p-value suggests that at least one type has a significantly
    different level of variance (heteroscedasticity).
"""

grouped_values = [g["value"].values for _, g in df_subset.groupby("type")]
stat, p_levene = levene(*grouped_values, center="median")
print(f"\n Levene’s test statistic: {stat:.4f}, p = {p_levene:.4e}")


# ==============================================================
# 6. Dunn’s Post-hoc Test
# ==============================================================

"""
Test: Dunn’s Test (Post-hoc after Kruskal–Wallis or Permutation)
----------------------------------------------------------------
Purpose:
    Pairwise nonparametric comparisons between type groups,
    adjusting for multiple testing (Bonferroni correction).

Interpretation:
    Identifies which pairs of types differ significantly in their
    distribution of motility values.
"""

dunn_type = sp.posthoc_dunn(df_subset, val_col="value", group_col="type", p_adjust="bonferroni")
dunn_type.to_csv("dunn_results_type.csv")

print(f"\n Dunn’s test complete. Saved to dunn_results_type.csv")


# ==============================================================
# 7. Pairwise Permutation Tests – Mean Differences
# ==============================================================

"""
Test: Pairwise Permutation (Mean Differences)
---------------------------------------------
Purpose:
    To test, for each pair of types, whether the observed mean difference
    is larger than expected under random reassignment of type labels.

Logic:
    - Compute observed mean difference.
    - Shuffle all values and recompute the difference many times.
    - P-value = proportion of permuted differences ≥ observed.
    - Adjust p-values using FDR (Benjamini–Hochberg).
"""

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

pairwise_df = pd.DataFrame(results)
pairwise_df["p_adj"] = multipletests(pairwise_df["p_value"], method="fdr_bh")[1]
pairwise_df.to_csv("pairwise_permutation_results_type.csv", index=False)

sig_pairs = pairwise_df[pairwise_df["p_adj"] < 0.05]
print(f"\n Pairwise permutation: {len(sig_pairs)} significant type pairs (FDR < 0.05).")


# ==============================================================
# 8. Pairwise Permutation Tests – Variance Differences
# ==============================================================

"""
Test: Pairwise Permutation (Variance Differences)
-------------------------------------------------
Purpose:
    To assess whether any two types differ significantly in the
    variability of motility (variance), independent of mean differences.

Logic:
    - Compute observed difference in variances between each pair.
    - Shuffle and recompute 10,000 times.
    - P-value = proportion of permuted variance differences ≥ observed.
"""

grouped = df_subset.groupby("type")
types = df_subset["type"].unique()
results_var = []

for g1, g2 in combinations(types, 2):
    v1 = grouped.get_group(g1)["value"].values
    v2 = grouped.get_group(g2)["value"].values
    observed_diff = np.abs(np.var(v1, ddof=1) - np.var(v2, ddof=1))
    combined = np.concatenate([v1, v2])
    n1, n2 = len(v1), len(v2)
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        g1_perm = combined[:n1]
        g2_perm = combined[n1:]
        perm_diff = np.abs(np.var(g1_perm, ddof=1) - np.var(g2_perm, ddof=1))
        perm_diffs.append(perm_diff)
    p_value = np.mean(np.array(perm_diffs) >= observed_diff)
    results_var.append({"group1": g1, "group2": g2,
                        "observed_diff": observed_diff, "p_value": p_value})

variance_df = pd.DataFrame(results_var)
variance_df.to_csv("variance_posthoc_type.csv", index=False)
sig_var_pairs = variance_df[variance_df["p_value"] < 0.05]
print(f" Pairwise variance differences: {len(sig_var_pairs)} significant pairs.")
print(sig_var_pairs.head())


# ==============================================================
# 9. Kruskal–Wallis Test
# ==============================================================

"""
Test: Kruskal–Wallis H Test (Nonparametric ANOVA Alternative)
-------------------------------------------------------------
Purpose:
    Tests whether at least one type differs from others in the
    distribution (median) of motility values.

Assumptions:
    - Independent observations
    - Ordinal or continuous response variable
    - Similar distribution shapes across groups

Interpretation:
    A significant result indicates at least one group differs;
    post-hoc tests (e.g., Dunn’s) identify which ones.
"""

groups = [group["value"].values for _, group in subset.groupby("type")]
stat, p_kw = kruskal(*groups)
print(f"\n Kruskal–Wallis H = {stat:.4f}, p = {p_kw:.4e}")

if p_kw < 0.05:
    print("Significant overall difference detected across types.")
else:
    print("No significant global difference detected across types.")
