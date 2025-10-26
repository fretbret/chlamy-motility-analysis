---

## **Chlamydomonas Motility and Evolutionary Analysis**

This repository contains reproducible scripts and notebooks for a quantitative and evolutionary analysis of motility and phototaxis in *Chlamydomonas reinhardtii*.
The project integrates experimental measurements, statistical testing, and evolutionary modeling to examine how mutational variance and fitness relate to phototactic performance across mutation accumulation (MA) and natural isolate (NI) strains.

---

### **Repository Structure**

```
motility_analysis/
│
├── 01_preprocessing/          # Data cleaning and formatting
│   ├── 01_preprocess_motility.py
│
├── 02_visualization/          # Exploratory plots and summary figures
│   ├── 02_plot_motility.py
│
├── 03_statistics/             # Parametric and non-parametric tests
│   ├── 03a_stats_strainwise.py
│   ├── 03b_stats_typewise_alltypes.py
│   ├── 03c_stats_typewise_comparisons/
│       ├── 03d_stats_NI_vs_CC2344MA.py
│       ├── 03e_stats_NI_vs_CC2931MA.py
│       ├── 03f_stats_CC2344MA_vs_CC2344ANC.py
│       ├── 03g_stats_CC2931MA_vs_CC2931ANC.py
│       └── ...
│
├── 04_evolutionary/           # Linking phototaxis to fitness and mutation
│   ├── 04_evolutionary_stats.py
│
├── 05_candidate_genes/        # GO-based candidate gene exploration
│   ├── Candidate_Genes_458.ipynb
│
└── README.md
```

---

### **Analysis Overview**

* **Data preprocessing:** Cleaning and harmonizing phototaxis assay outputs.
* **Visualization:** Generating summary figures of distance metrics by strain and week.
* **Statistical testing:** Comparing motility across types using ANOVA, mixed-effects models, and permutation tests.
* **Evolutionary modeling:** Testing associations between phototaxis, mutational load, and fitness.
* **Candidate gene exploration:** Integrating GO term and literature data to highlight vision- and motility-related genes.

---

### **Software Environment**

| Component                      | Description                                                                                                                                                                 |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Python**                     | 3.9–3.11 (tested locally in JupyterLab)                                                                                                                                     |
| **Core Packages**              | `pandas`, `numpy`, `scipy`, `statsmodels`, `seaborn`, `matplotlib`, `natsort`, `itertools`, `statsmodels.formula.api`, `multipletests` (from `statsmodels.stats.multitest`) |
| **Optional (for GO analysis)** | `gffutils`, `Bio` (Biopython), `requests`, `re`                                                                                                                             |
| **R (optional)**               | `tidyverse`, `ggplot2` — used for supplementary nested ANOVA validation                                                                                                     |
| **Environment**                | JupyterLab or VS Code (recommended for running scripts interactively)                                                                                                       |

If you wish to replicate the environment:

```bash
pip install pandas numpy scipy statsmodels seaborn matplotlib natsort biopython gffutils
```

---

### **Reproducibility**

To reproduce or extend the analysis:

1. Clone this repository:

   ```bash
   git clone https://github.com/fretbret/chlamy-motility-analysis.git
   cd chlamy-motility-analysis
   ```
2. Open JupyterLab or VSCode.
3. Run each script sequentially or explore the notebook interactively:

   ```bash
   jupyter lab 05_candidate_genes/Candidate_Genes_458.ipynb
   ```
4. All generated plots and CSV outputs will appear in the respective directories.

---

### **Citation and Attribution**

Developed as part of undergraduate research in the
**Ness Lab, Department of Biology, University of Toronto Mississauga**.

---
