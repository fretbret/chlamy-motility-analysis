"""
Motility Assay Statistical Testing – Strain-wise
-----------------------------------------------
Author: Ayesha Syeda
Lab: Ness Lab, University of Toronto Mississauga

Purpose
-------
Performs statistical analyses to test for differences in motility (maximum distance)
across individual Chlamydomonas strains.

Tests included:
1. One-way ANOVA with strain as fixed factor and experiment week (variable) as blocking factor
2. Tukey’s HSD post-hoc test for pairwise strain comparisons
3. Global permutation test for nonparametric validation
4. Pairwise permutation tests (strain-by-strain)
5. Kruskal–Wallis test with Dunn’s post-hoc comparisons

All tests use *Maximum Distance* as the response variable.

Input
-----
- motility_assay_clean.csv (from preprocessing script)

Output
------
- anova_results.csv
- tukey_results.csv
- pairwise_permutation_results.csv
- dunn_results.csv
- Printed summaries of test statistics and p-values

Usage
-----
Run this script in the same directory as motility_assay_clean.csv:
    python 03a_stats_strainwise.py
"""

# ==============================================================
# 1. Import Required Packages
# ==============================================================

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import kruskal
import scikit_posthocs as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt


# ==============================================================
# 2. Load Data
# ==============================================================

df = pd.read_csv("motility_assay_clean.csv", sep="\t")
subset = df[df["distance"] == "Maximum Distance"].copy()


# ==============================================================
# 3. Parametric: One-way ANOVA and Tukey HSD
# ==============================================================

"""
Test: One-way ANOVA
-------------------
Purpose:
    To assess whether mean motility (maximum distance) differs among strains,
    accounting for experiment week (variable) as a blocking/nuisance factor.

Model:
    value ~ C(strain) + C(variable)

Assumptions:
    - Normality of residuals
    - Homogeneity of variance (equal variance across strains)
    - Independence of observations (partially violated here since no random sampling)

Interpretation:
    A significant overall F-statistic suggests that at least one strain
    differs in mean motility from the others.
"""

model = smf.ols("value ~ C(strain) + C(variable)", data=subset).fit()
anova_table = sm.stats.anova_lm(model, typ=2)  # Type-II ANOVA aligns with R default
anova_table.to_csv("anova_results.csv")

print("\n One-way ANOVA results:")
print(anova_table)

"""
Post-hoc Test: Tukey’s Honestly Significant Difference (HSD)
-------------------------------------------------------------
Purpose:
    Conduct pairwise comparisons between all strain means
    to identify which pairs differ significantly.

Details:
    Controls the family-wise error rate (FWER).
    More conservative than unadjusted t-tests.
"""

tukey = pairwise_tukeyhsd(endog=subset["value"],
                          groups=subset["strain"],
                          alpha=0.05)

tukey.plot_simultaneous()
plt.title("Tukey HSD: Differences in Max Distance Across Strains")
plt.tight_layout()
plt.show()

tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
tukey_df.to_csv("tukey_results.csv", index=False)

significant_tukey = tukey_df[tukey_df["reject"] == True]
print(f" Tukey HSD: {len(significant_tukey)} significant pairwise differences detected.")


# ==============================================================
# 4. Non-Parametric: Global Permutation Test
# ==============================================================

"""
Test: Global Permutation (Nonparametric ANOVA Analog)
----------------------------------------------------
Purpose:
    Tests whether the observed between-group variance
    (variation among strain means) is greater than expected by chance.

Logic:
    - Compute observed between-group variance.
    - Randomly permute strain labels many times (e.g., 10,000).
    - Recompute variance each time to create a null distribution.
    - P-value = proportion of permuted variances ≥ observed variance.

Interpretation:
    A low p-value indicates that differences among strains are unlikely due to random chance.
"""

group_means = df_subset.groupby("strain")["value"].mean()
overall_mean = df_subset["value"].mean()
group_sizes = df_subset.groupby("strain").size()
between_group_var = np.sum(group_sizes * (group_means - overall_mean) ** 2)

n_permutations = 10000
perm_stats = []
for _ in range(n_permutations):
    shuffled = np.random.permutation(df_subset["strain"])
    df_shuffled = df_subset.copy()
    df_shuffled["shuffled_strain"] = shuffled
    group_means_perm = df_shuffled.groupby("shuffled_strain")["value"].mean()
    group_sizes_perm = df_shuffled.groupby("shuffled_strain").size()
    perm_var = np.sum(group_sizes_perm * (group_means_perm - overall_mean) ** 2)
    perm_stats.append(perm_var)

p_value_global = np.mean(np.array(perm_stats) >= between_group_var)
print(f"\n Global permutation test p-value = {p_value_global:.4f}")


# ==============================================================
# 5. Non-Parametric: Pairwise Permutation Tests
# ==============================================================

"""
Test: Pairwise Permutation (Nonparametric Alternative to Tukey/Dunn)
-------------------------------------------------------------------
Purpose:
    To compare each pair of strains directly using a resampling-based test.

Logic:
    - Compute the observed mean difference between two strains.
    - Pool their data, shuffle labels, and recompute the difference.
    - Repeat thousands of times to simulate the null distribution.
    - P-value = fraction of permutations where the permuted difference ≥ observed.

Notes:
    - With 42 strains, there are 861 pairwise comparisons.
    - 10,000 permutations per pair → ~8.6 million total tests.
    - P-values adjusted for multiple testing using FDR (Benjamini–Hochberg).
"""

strains = df_subset["strain"].unique()
results = []
n_permutations = 10000

print("\n Running pairwise permutation tests...")
for s1, s2 in combinations(strains, 2):
    v1 = df_subset[df_subset["strain"] == s1]["value"].values
    v2 = df_subset[df_subset["strain"] == s2]["value"].values
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
    results.append({"strain1": s1, "strain2": s2,
                    "mean_diff": observed_diff, "p_value": p_value})

pairwise_df = pd.DataFrame(results)
pairwise_df["p_adj"] = multipletests(pairwise_df["p_value"], method="fdr_bh")[1]
pairwise_df.to_csv("pairwise_permutation_results.csv", index=False)

sig_pairs = pairwise_df[pairwise_df["p_adj"] < 0.05]
print(f" Pairwise permutation: {len(sig_pairs)} significant strain pairs (FDR < 0.05).")


# ==============================================================
# 6. Non-Parametric: Kruskal–Wallis + Dunn’s Test
# ==============================================================

"""
Test: Kruskal–Wallis H Test
---------------------------
Purpose:
    Nonparametric analog of one-way ANOVA that compares medians among groups.

Assumptions:
    - Independent samples
    - Ordinal or continuous data
    - Similar shapes of distributions (but not equal variances)

Interpretation:
    Tests whether at least one strain’s distribution differs from the others.

Post-hoc: Dunn’s Test
---------------------
Purpose:
    Performs pairwise nonparametric comparisons between groups after a significant
    Kruskal–Wallis result. Adjusts for multiple testing using Bonferroni correction.
"""

groups = [g["value"].values for _, g in subset.groupby("strain")]
stat, p_kw = kruskal(*groups)
print(f"\n Kruskal–Wallis H = {stat:.4f}, p = {p_kw:.4e}")

# Dunn’s test for posthoc pairwise comparisons
dunn_strain = sp.posthoc_dunn(subset, val_col="value", group_col="strain", p_adjust="bonferroni")

# Reformat and extract significant results
long_strain = dunn_strain.reset_index().melt(id_vars="index",
                                             var_name="strain_2",
                                             value_name="p_value")
long_strain = long_strain.rename(columns={"index": "strain_1"})
sig_dunn = long_strain[long_strain["p_value"] < 0.05].sort_values("p_value")
sig_dunn.to_csv("dunn_results.csv", index=False)

print(f" Dunn’s test: {len(sig_dunn)} significant pairwise differences detected.")

