"""
Motility Assay Plotting Script
------------------------------
Author: Ayesha Syeda
Lab: Ness Lab, University of Toronto Mississauga

Purpose
-------
Generates visual summaries of Chlamydomonas motility assay results.  
This script:
1. Reads raw Excel files and produces individual strain line plots.
2. Reads the cleaned dataset (`motility_assay_clean.csv`) for summary statistics.
3. Produces boxplots and violin plots comparing strain motility across experiments.

Input
-----
- Excel files named like: '2024-11-16_MotA1.xlsx', ..., '2024-12-24_MotA5_Reread.xlsx'
- 'Distance_df.xlsx' (contains distance column)
- 'motility_assay_clean.csv' (from preprocessing script)

Output
------
- Line plots of absorbance vs. distance for each strain (displayed interactively)
- Boxplots and violin plots comparing maximum and weighted averages (displayed, optionally savable)
- `average_max.csv` and `average_weighted.csv` containing per-strain averages

Usage
-----
Run this script in the directory containing both the raw MotA Excel files and
the cleaned `motility_assay_clean.csv` output from preprocessing.

Example:
    python 02_plotting.py
"""

# ==============================================================
# 1. Import Required Packages
# ==============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted


# ==============================================================
# 2. Helper Functions
# ==============================================================

def upload_file_lineplot(file_path: str, distance_file: str = "Distance_df.xlsx") -> pd.DataFrame:
    """
    Reads a motility assay Excel file and combines it with distance data
    for visualization of absorbance along distance.

    Parameters
    ----------
    file_path : str
        Path to a MotA Excel file (e.g., '2024-11-16_MotA1.xlsx').
    distance_file : str
        Path to the Excel file containing the 'Distance (mm)' column.

    Returns
    -------
    pd.DataFrame
        A dataframe with columns for each strain and a leading 'Distance (mm)' column.
    """
    data_dict = {}
    num_sheets = len(pd.read_excel(file_path, sheet_name=None))

    for sheet in range(num_sheets):
        sheet_df = pd.read_excel(file_path, sheet_name=sheet)
        sheet_df.columns = [i for i in range(len(sheet_df.columns))]

        strain_names = sheet_df.loc[21, [3, 4]].tolist()
        sample1 = sheet_df.iloc[[24, 26, 28], 2:14]
        sample2 = sheet_df.iloc[[32, 34, 36], 2:14]

        s1 = sum([sample1.loc[i].tolist() for i in sample1.index], [])
        s2 = sum([sample2.loc[i].tolist() for i in sample2.index], [])
        s1.reverse()
        s2.reverse()

        data_dict[strain_names[0]] = s1
        data_dict[strain_names[1]] = s2

    motility_df = pd.DataFrame({k: pd.Series(v) for k, v in data_dict.items()})
    motility_df = motility_df.reindex(natsorted(motility_df.columns), axis=1)

    # Add distance column
    distance_data = pd.read_excel(distance_file, sheet_name='Sheet1')
    motility_df.insert(0, 'Distance (mm)', distance_data)

    return motility_df


def plot_absorbance_profiles(motility_df: pd.DataFrame, blank_threshold: float = 0.07) -> None:
    """
    Generates line plots for absorbance vs. distance for each strain.

    Parameters
    ----------
    motility_df : pd.DataFrame
        DataFrame containing 'Distance (mm)' and strain absorbance columns.
    blank_threshold : float
        Horizontal line threshold (OD blank).
    """
    for column in motility_df.columns:
        if "Distance" not in column:
            plt.figure(figsize=(6, 4))
            plt.plot(motility_df["Distance (mm)"], motility_df[column], linewidth=2)
            plt.title(column, fontsize=12)
            plt.xlabel("Distance (mm)", fontsize=12)
            plt.ylabel("Absorbance (OD 680 nm)", fontsize=12)
            plt.axhline(y=blank_threshold, color='r', linestyle='--', label='Blank (0.07)')
            plt.legend()
            plt.tight_layout()
            plt.show()


# ==============================================================
# 3. Load Data
# ==============================================================

# Identify MotA files
motility_files = [
    f for f in os.listdir(os.getcwd())
    if f.endswith(".xlsx") and any(m in f for m in [
        "2024-11-16_MotA1", "2024-11-30_MotA2", "2024-12-07_MotA3",
        "2024-12-17_MotA4_Reread", "2024-12-24_MotA5_Reread"
    ])
]

# Load cleaned dataset (from preprocessing)
df = pd.read_csv("motility_assay_clean.csv", sep='\t')

# ==============================================================
# 4. Generate Line Plots (per MotA file)
# ==============================================================

for mot_file in motility_files:
    print(f"\n Plotting absorbance profiles for: {mot_file}")
    motility_df = upload_file_lineplot(mot_file)
    plot_absorbance_profiles(motility_df, blank_threshold=0.07)


# ==============================================================
# 5. Summary Statistics and Averages
# ==============================================================

def compute_average(df: pd.DataFrame, distance_type: str) -> pd.DataFrame:
    """
    Computes mean, standard deviation, and sample counts for each strain
    across all MotA experiments.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with columns ['strain', 'distance', 'variable', 'value'].
    distance_type : str
        One of ['Maximum Distance', 'Weighted Average'].

    Returns
    -------
    pd.DataFrame
        Summary statistics per strain.
    """
    averages = pd.DataFrame()
    for strain in df["strain"].unique():
        for mot in ["MotA1", "MotA2", "MotA3", "MotA4", "MotA5"]:
            subset = df[(df["strain"] == strain) &
                        (df["distance"] == distance_type) &
                        (df["variable"] == mot)]
            averages.loc[strain, mot] = subset["value"].mean()

    averages["mean"] = averages.mean(axis=1)
    averages["std"] = averages.std(axis=1)
    averages["n_MotA"] = averages[["MotA1", "MotA2", "MotA3", "MotA4", "MotA5"]].count(axis=1)
    averages["n_raw_rows"] = averages.index.map(
        lambda s: df[(df["strain"] == s) &
                     (df["distance"] == distance_type) &
                     (df["variable"].isin(["MotA1", "MotA2", "MotA3", "MotA4", "MotA5"]))].shape[0]
    )

    averages = averages.reset_index().rename(columns={"index": "strain"})
    averages.sort_values("mean", ascending=False, inplace=True)
    return averages


average_max = compute_average(df, "Maximum Distance")
average_weighted = compute_average(df, "Weighted Average")

average_max.to_csv("average_max.csv", sep="\t", index=False)
average_weighted.to_csv("average_weighted.csv", sep="\t", index=False)


# ==============================================================
# 6. Visualization: Boxplots and Violin Plots
# ==============================================================

sns.set(style="whitegrid", font_scale=1.2)
sorted_strains = natsorted(df["strain"].unique())

# Boxplot: Maximum Distance
plt.figure(figsize=(22, 20))
sns.boxplot(
    data=df[df["distance"] == "Maximum Distance"],
    y="strain", x="value", hue="type",
    order=sorted_strains, width=0.75
)
plt.xlabel("Mean Maximum Distance (mm)", fontsize=28)
plt.ylabel("Strains", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(title="Type", fontsize=20, loc="center right", bbox_to_anchor=(1, 0.45))
plt.tight_layout()
plt.show()

# Boxplot: Weighted Average
plt.figure(figsize=(22, 20))
sns.boxplot(
    data=df[df["distance"] == "Weighted Average"],
    y="strain", x="value", hue="type",
    order=sorted_strains, width=0.75
)
plt.xlabel("Mean Weighted Average (mm)", fontsize=28)
plt.ylabel("Strains", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(title="Type", fontsize=20, loc="center right", bbox_to_anchor=(1, 0.45))
plt.tight_layout()
plt.show()

# Violin Plots
plt.figure(figsize=(13, 10))
sns.violinplot(data=df[df["distance"] == "Maximum Distance"], y="value", x="type", width=0.9)
plt.xlabel("Strain Type", fontsize=24)
plt.ylabel("Mean Maximum Distance (mm)", fontsize=24)
plt.tight_layout()
plt.show()

plt.figure(figsize=(13, 10))
sns.violinplot(data=df[df["distance"] == "Weighted Average"], y="value", x="type", width=0.9)
plt.xlabel("Strain Type", fontsize=24)
plt.ylabel("Mean Weighted Average (mm)", fontsize=24)
plt.tight_layout()
plt.show()
