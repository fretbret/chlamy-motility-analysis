"""
Motility Assay Preprocessing Script
----------------------------------
Author: Ayesha Syeda, Eniolaye Balogun
Lab: Ness Lab, University of Toronto Mississauga

Purpose
-------
Reads multiple raw motility assay Excel files (MotA1–MotA5), cleans and merges them,
applies filtering rules, and outputs a tab-separated CSV file containing strain-level
motility statistics (maximum distance and weighted average).

Input
-----
- Excel files in the current working directory matching:
  '2024-11-16_MotA1', '2024-11-30_MotA2', '2024-12-07_MotA3',
  '2024-12-17_MotA4_Reread', '2024-12-24_MotA5_Reread.xlsx'

Output
------
- motility_assay_clean.csv : cleaned, combined dataset for downstream analysis

Usage
-----
Run this script from the folder containing the raw Excel files.
Adjust the `blank_threshold` or filenames as needed.
"""

# ==============================================================
# 1. Import Required Packages
# ==============================================================

import os
import pandas as pd
import numpy as np
from natsort import natsorted


# ==============================================================
# 2. Helper Functions
# ==============================================================

def upload_file(file_path: str) -> pd.DataFrame:
    """
    Load a motility assay Excel file and extract optical density (OD) readings.

    Parameters
    ----------
    file_path : str
        Path to the Excel file.

    Returns
    -------
    pd.DataFrame
        Columns correspond to strains, rows correspond to distances.
        Column order is naturally sorted.
    """
    data_dict = {}
    num_sheets = len(pd.read_excel(file_path, sheet_name=None))

    for sheet in range(num_sheets):
        sheet_df = pd.read_excel(file_path, sheet_name=sheet)
        sheet_df.columns = [i for i in range(len(sheet_df.columns))]

        strain_names = sheet_df.loc[21, [3, 4]].tolist()
        sample1 = sheet_df.iloc[[24, 26, 28], 2:14]
        sample2 = sheet_df.iloc[[32, 34, 36], 2:14]

        s1 = list(filter(lambda x: x >= 0.05, sum([sample1.loc[i].tolist() for i in sample1.index], [])))
        s2 = list(filter(lambda x: x >= 0.05, sum([sample2.loc[i].tolist() for i in sample2.index], [])))

        s1.reverse()
        s2.reverse()

        data_dict[strain_names[0]] = s1
        data_dict[strain_names[1]] = s2

    motility_df = pd.DataFrame({k: pd.Series(v) for k, v in data_dict.items()})
    motility_df = motility_df.reindex(natsorted(motility_df.columns), axis=1)

    return motility_df


def motility_stats(data: pd.DataFrame, blank_threshold: float) -> pd.DataFrame:
    """
    Compute motility statistics for each strain.

    Parameters
    ----------
    data : pd.DataFrame
        OD readings per distance for each strain.
    blank_threshold : float
        OD threshold for filtering background noise.

    Returns
    -------
    pd.DataFrame
        Motility statistics with 'Maximum Distance' and 'Weighted Average'.
    """
    stats_df = pd.DataFrame()
    data.index = range(len(data))
    data = data.astype(float)

    # ----------------------------------------------------------
    # Remove background (blank) readings
    # ----------------------------------------------------------
    for strain in data.columns:
        for idx in data[strain].dropna().index:
            try:
                if ((data.loc[idx, strain] <= blank_threshold and data.loc[idx + 1, strain] <= blank_threshold)
                        or (data.loc[idx, strain] <= blank_threshold and data.loc[idx - 1, strain] <= blank_threshold)):
                    data.loc[idx, strain] = 0
            except IndexError:
                continue
    data = data.replace(0, np.nan)

    # ----------------------------------------------------------
    # Compute maximum motility distance (each well = 0.8 mm)
    # ----------------------------------------------------------
    for strain in data.columns:
        non_blank = data[strain].dropna()
        stats_df.loc[strain, 'Maximum Distance'] = len(non_blank) * 0.8

    # ----------------------------------------------------------
    # Compute weighted average motility
    # ----------------------------------------------------------
    weighted_df = data.fillna(0)
    weighted_df.index = np.arange(1, len(weighted_df) + 1)

    total = weighted_df.sum(axis=0)
    weighted_sum = (weighted_df.T * (weighted_df.index * 0.8)).T.sum(axis=0)
    weighted_average = weighted_sum / total
    weighted_average.name = 'Weighted Average'

    stats_df = pd.concat([stats_df, weighted_average], axis=1)
    return stats_df


# ==============================================================
# 3. File Discovery
# ==============================================================

pwd = os.getcwd()
files = [
    f for f in os.listdir(pwd)
    if os.path.isfile(os.path.join(pwd, f))
    and any(keyword in f for keyword in [
        '2024-11-16_MotA1', '2024-11-30_MotA2',
        '2024-12-07_MotA3', '2024-12-17_MotA4_Reread',
        '2024-12-24_MotA5_Reread.xlsx'
    ])
    and f.endswith('.xlsx')
]


# ==============================================================
# 4. Data Import and Preprocessing
# ==============================================================

blank_threshold = 0.07
merged_df = pd.DataFrame(columns=['strain', 'distance'])

for file in files:
    motility_raw = upload_file(file)
    motility_processed = motility_stats(motility_raw, blank_threshold)

    motility_processed.reset_index(inplace=True)
    motility_long = pd.melt(motility_processed, id_vars='index', value_vars=motility_processed.columns)
    motility_long.columns = ['strain', 'distance', file[11:-5]]

    merged_df = pd.concat([merged_df, motility_long], ignore_index=True)

# Combine all experiments into a single long-format DataFrame
df = pd.melt(merged_df, id_vars=['strain', 'distance'], value_vars=merged_df.columns[2:]).dropna()


# ==============================================================
# 5. Feature Engineering and Filtering
# ==============================================================

# Columns tracking sample metadata
df['diluted'] = np.where(df['strain'].str.contains('Undiluted'), 'no', 'yes')
df['reread_strainwise'] = np.where(df['variable'].str.contains('Reread'), 'yes', 'no')
df['redone_strainwise'] = np.where(df['strain'].str.contains(' Redone'), 'yes', 'no')
df['again_strainwise'] = np.where(df['strain'].str.contains(' Again'), 'yes', 'no')

# Standardize strain naming
df['variable'] = df['variable'].str.replace('_Reread', '', regex=False)
df['strain'] = (df['strain']
                .str.replace(' Redone', '', regex=False)
                .str.replace(' Undiluted', '', regex=False)
                .str.replace(' Again', '', regex=False))

# Identify ancestor type (EB vs AS)
df['ANC_type'] = 'none'
df.loc[df['strain'].str.contains('EB'), 'ANC_type'] = 'EB'
df.loc[df['strain'].str.contains('AS'), 'ANC_type'] = 'AS'
df['strain'] = df['strain'].str.replace('-EB', '').str.replace('-AS', '')

# Label broad strain type
df['type'] = 'NI'
df.loc[df['strain'].str.contains('CC-2344 ANC'), 'type'] = 'CC-2344 ANC'
df.loc[df['strain'].str.contains('CC-2931 ANC'), 'type'] = 'CC-2931 ANC'
df.loc[df['strain'].str.contains('CC-2344 L'), 'type'] = 'CC-2344 MA'
df.loc[df['strain'].str.contains('CC-2931 L'), 'type'] = 'CC-2931 MA'

df['broad_type'] = 'NI'
df.loc[df['strain'].str.contains('ANC'), 'broad_type'] = 'ANC'
df.loc[df['strain'].str.contains('CC-2344 L|CC-2931 L'), 'broad_type'] = 'MA'

# Remove irrelevant entries
df = df[~df['strain'].str.contains('remove', case=False, na=False)]
df = df[df['value'] != 0.0]

# Drop temporary columns and standardize ANC naming
df.drop(columns=['diluted', 'reread_strainwise', 'redone_strainwise', 'again_strainwise'], inplace=True)
df.loc[df['strain'].str.contains('CC-2344 ANC'), 'strain'] = 'CC-2344 ANC'
df.loc[df['strain'].str.contains('CC-2931 ANC'), 'strain'] = 'CC-2931 ANC'

# Enumerate strains within experiment (nested random factor)
df['strain2'] = None
for variable in df['variable'].unique():
    subset = df[df['variable'] == variable]
    for strain_type in subset['type'].unique():
        mask = (df['variable'] == variable) & (df['type'] == strain_type)
        if strain_type in ['CC-2344 ANC', 'CC-2931 ANC']:
            for dist in ['Maximum Distance', 'Weighted Average']:
                dist_mask = mask & (df['distance'] == dist)
                for i, idx in enumerate(df.loc[dist_mask].index, start=1):
                    df.at[idx, 'strain2'] = i
        else:
            strain_map = {s: i + 1 for i, s in enumerate(sorted(df.loc[mask, 'strain'].unique()))}
            df.loc[mask, 'strain2'] = df.loc[mask, 'strain'].map(strain_map)

df['strain2'] = df['strain2'].astype(int)
df = df.sort_values(by=['type'])

# ==============================================================
# 6. Save Cleaned Output
# ==============================================================

output_path = os.path.join(os.getcwd(), 'motility_assay_clean.csv')
df.to_csv(output_path, sep='\t', header=True, index=False)
