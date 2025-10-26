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
├── 02_visualization/          # Plotting and exploratory visualizations
│   ├── 02_plot_motility.py
│
├── 03_statistics/             # Difference testing (strainwise, typewise)
│   ├── 03a_stats_strainwise.py
│   ├── 03b_stats_typewise_alltypes.py
│   ├── 03c_stats_typewise_comparisons/
│       ├── 03d_stats_NI_vs_CC2344MA.py
│       ├── 03e_stats_CC2344MA_vs_CC2344ANC.py
│       └── ...
│
├── 04_evolutionary/           # Fitness and mutation-based evolutionary analysis
│   ├── 04_evolutionary_stats.py
│
├── 05_candidate_genes/        # Exploratory gene-level analyses
│   ├── Candidate_Genes_458.ipynb
│
└── README.md
```

---

### **Analysis Overview**

* **Data preprocessing:** Cleaning and standardizing raw absorbance data from weekly motility assays.
* **Visualization:** Generating line plots, boxplots, and violin plots to summarize strain-level variation.
* **Statistical testing:** Parametric and non-parametric analyses comparing motility across strains and genetic types.
* **Evolutionary modeling:** Linking phototaxis phenotypes to mutational load and relative fitness measures.
* **Candidate gene exploration:** Integrating GO annotations and literature to identify genes related to motility and vision.

---

### **Tools & Environment**

* **Python:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `natsort`
* **R (optional):** `tidyverse`, `ggplot2` (for supplementary tests)
* **JupyterLab / VSCode:** for interactive analysis and documentation

A `requirements.txt` file can be generated using:

```bash
pip freeze > requirements.txt
```

---

### **Reproducibility**

To reproduce the analysis:

1. Clone this repository.
2. Navigate to the desired analysis stage (e.g., `04_evolutionary/`).
3. Run each Python script sequentially, or open the corresponding Jupyter notebook:

   ```bash
   jupyter lab 05_candidate_genes/Candidate_Genes_458.ipynb
   ```
4. Outputs (figures, summary tables, CSVs) will be generated in the same directory.

---

### **Acknowledgements**

Developed during undergraduate research in the **Ness Lab**,
Department of Biology, University of Toronto Mississauga.

---
