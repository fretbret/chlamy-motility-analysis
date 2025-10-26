"""
Evolutionary Statistics – Fitness, Mutations, and Phototaxis
------------------------------------------------------------
Author: Ayesha Syeda
Lab: Ness Lab, University of Toronto Mississauga

Purpose
-------
Integrate phototaxis measures with evolutionary parameters (fitness, mutation count)
to explore how genetic changes influence motility.

Analyses include:
1. Correlations between phototaxis and relative fitness.
2. Correlations between phototaxis and mutation count.
3. Combined linear models examining fitness and mutations jointly.

All relationships are descriptive; no causal inference is implied.

Inputs
------
- mutation+fitness.csv : contains strain-level mutation counts and average relative fitness.
- motility_assay_clean.csv : summarized phototaxis metrics from Section 03.

Outputs
-------
- fitness+mut+pheno.csv : merged dataset
- regression plots and printed model summaries
"""

# ==============================================================
# 1. Import Packages
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from natsort import natsorted
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# 2. Import and Clean Data
# ==============================================================

fitness = pd.read_csv("mutation+fitness.csv", sep="\t")
fitness = fitness.loc[natsorted(fitness["Unnamed: 0"])].reset_index(drop=True)

# Clean column names
fitness.columns = fitness.columns.str.strip()
fitness = fitness.rename(columns={"Unnamed: 0": "strain", "Average_realtive": "Average_relative"})

# Standardize strain naming
fitness["strain"] = (
    fitness["strain"]
    .str.replace("-", " ")
    .str.replace("CC2344", "CC-2344", regex=False)
    .str.replace("CC2931", "CC-2931", regex=False)
    .str.strip()
)

# Merge phototaxis summaries
fitness["strain"] = fitness["strain"].astype(str)
average_max["strain"] = average_max["strain"].astype(str)
average_weighted["strain"] = average_weighted["strain"].astype(str)

merged_df = (
    fitness[["strain", "mutations", "Average_relative"]]
    .merge(average_max[["strain", "mean", "std"]], on="strain", how="left", suffixes=("", "_max"))
    .merge(average_weighted[["strain", "mean", "std"]], on="strain", how="left", suffixes=("_max", "_weighted"))
)

# Extract group ID (CC-2344, CC-2931)
merged_df["group"] = merged_df["strain"].str.extract(r"(CC-\d{4})")

# Map ancestral mean values per group
anc_means = (
    merged_df[merged_df["strain"].str.contains("ANC")]
    .set_index("group")[["mean_max", "mean_weighted"]]
    .rename(columns={"mean_max": "anc_mean_max", "mean_weighted": "anc_mean_weighted"})
)

# Merge back to scale phototaxis relative to ancestor
merged_df = merged_df.merge(anc_means, on="group", how="left")
merged_df["scaled_mean_max"] = merged_df["mean_max"] / merged_df["anc_mean_max"]
merged_df["scaled_mean_weighted"] = merged_df["mean_weighted"] / merged_df["anc_mean_weighted"]

# Save merged dataset
merged_df.to_csv("fitness+mut+pheno.csv", sep="\t", header=True, index=False)
print("Merged dataset saved as fitness+mut+pheno.csv")

# ==============================================================
# 3. Phototaxis vs Fitness
# ==============================================================

clean_df = merged_df.dropna(subset=["Average_relative"])
x = clean_df["Average_relative"]

for col, label in [("mean_max", "Mean Max Distance"), ("mean_weighted", "Mean Weighted Distance")]:
    y = clean_df[col]
    slope, intercept = np.polyfit(x, y, 1)
    corr, _ = pearsonr(x, y)

    print(f"\n{label}: Slope={slope:.4f}, Intercept={intercept:.4f}, r={corr:.4f}")

    sns.scatterplot(x=x, y=y)
    plt.plot(x, slope * x + intercept, color="red")
    plt.xlabel("Fitness (Average_relative)")
    plt.ylabel(label)
    plt.title(f"{label} vs Fitness (r={corr:.3f})")
    plt.show()

# Linear models
for yvar in ["mean_max", "mean_weighted"]:
    model = smf.ols(f"{yvar} ~ Average_relative", data=clean_df).fit()
    print(f"\nLinear model: {yvar} ~ Fitness")
    print(model.summary())

# ==============================================================
# 4. Scaled Phototaxis vs Fitness
# ==============================================================

for col, label in [("scaled_mean_max", "Scaled Mean Max Distance"), ("scaled_mean_weighted", "Scaled Mean Weighted Distance")]:
    y = clean_df[col]
    slope, intercept = np.polyfit(x, y, 1)
    corr, _ = pearsonr(x, y)

    print(f"\n{label}: Slope={slope:.4f}, Intercept={intercept:.4f}, r={corr:.4f}")

    sns.regplot(x=x, y=y, ci=None)
    plt.title(f"{label} vs Fitness (r={corr:.3f})")
    plt.xlabel("Fitness (Average_relative)")
    plt.ylabel(label)
    plt.show()

for yvar in ["scaled_mean_max", "scaled_mean_weighted"]:
    model = smf.ols(f"{yvar} ~ Average_relative", data=clean_df).fit()
    print(f"\nLinear model: {yvar} ~ Fitness")
    print(model.summary())

# ==============================================================
# 5. Phototaxis vs Mutation Count
# ==============================================================

clean_df = merged_df.dropna(subset=["mutations"])
x = clean_df["mutations"]

for col, label in [("scaled_mean_max", "Scaled Mean Max Distance"), ("scaled_mean_weighted", "Scaled Mean Weighted Distance")]:
    y = clean_df[col]
    slope, intercept = np.polyfit(x, y, 1)
    corr, _ = pearsonr(x, y)
    print(f"\n{label}: Slope={slope:.4f}, Intercept={intercept:.4f}, r={corr:.4f}")

    sns.regplot(x=x, y=y, ci=None)
    plt.title(f"{label} vs Mutation Count (r={corr:.3f})")
    plt.xlabel("Mutation Count")
    plt.ylabel(label)
    plt.show()

# Linear models
for yvar in ["scaled_mean_max", "scaled_mean_weighted"]:
    model = smf.ols(f"{yvar} ~ mutations", data=clean_df).fit()
    print(f"\nLinear model: {yvar} ~ Mutations")
    print(model.summary())

# ==============================================================
# 6. Combined Models (Fitness + Mutations)
# ==============================================================

clean_df = merged_df.dropna(subset=["Average_relative", "mutations"])

# Additive model
print("\nAdditive Models: Fitness + Mutations")
for yvar in ["mean_max", "mean_weighted"]:
    model = smf.ols(f"{yvar} ~ Average_relative + mutations", data=clean_df).fit()
    print(f"\nModel: {yvar} ~ Fitness + Mutations")
    print(model.summary())

# Interaction model
print("\nInteraction Models: Fitness * Mutations")
for yvar in ["mean_max", "mean_weighted"]:
    model = smf.ols(f"{yvar} ~ Average_relative * mutations", data=clean_df).fit()
    print(f"\nModel: {yvar} ~ Fitness * Mutations")
    print(model.summary())
