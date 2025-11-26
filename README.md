# NSDUH 2021–2023 CausalML Project
### Reliable and Fair Causal Machine Learning for Sparse Subpopulations in Survey Data
**Author: Yifan Xu (许逸凡)** 
MS in Biostatistics, New York University (NYU GPH), 2023 
Email: yifan1003a@gmail.com

--- 

# 1. Overview 

This repository contains the full reproducible pipeline for my ongoing research project:

> **Reliable and Fair Causal Machine Learning for Sparse Subpopulations in Survey Data:
> Substance Use and Major Depressive Episode among Reproductive-Age Women
> (NSDUH 2021–2023)**

For a quick tour, see the demo notebook:
[`notebooks/01_demo_pipeline.ipynb`](notebooks/01_demo_pipeline.ipynb).

This project is the core research component of my PhD applications (CS / Data Science / AI / Biostatistics/ Computaional Social Science).

It is designed to demonstrate: 
- my ability to build **end-to-end causal machine learning pipelines**,
- perform **survey-design–aware inference**,
- conduct **heterogeneous treatment effect** estimation,
- evaluate **fairness across subgroups**, and
- maintain **reproducible scientific workflows**.

This repository is designed to be fully reproducible and modular, enabling independent validation of each component (data processing, weighted GLM, DML ATE, Causal Forest CATE, and fairness diagnostics).

---

# 2. Research Motivation

Major Depressive Episode (MDE) remains a significant public-health burden among U.S. women of reproductive age. 

Traditional regression methods struggle with:
- sparse exposure patterns（e.g., stimulant misuse）
- heterogeneous effects（pregnancy, income, race）
- fairness and subgroup disparity
- survey weights and complex sampling design

This project uses **survey-weighted causal inference + modern causal ML** to study the relationship between substance use and MDE.

---

# 3. Data

**Dataset:** National Survey on Drug Use and Health (NSDUH), 2021–2023 public-use files
**Population:** Women aged **18–49**, including pregnant women 
**N** ≈ varies by year, pooled as harmonized dataset 

### **Outcome** 
- **IRAMDEYR** — Past-year Major Depressive Episode (binary)

### **Exposures** 
- **ILLYR** — Any illicit drug use (past year)
- **PNRNMYR** — Pain-reliever misuse (past year)
- **STMNMYR** — Stimulant misuse (past year)
- **ANY_CANNA_EVER** — Lifetime cannabis use (derived from MJEVER, CBDHMPEVR)

### **Covariates** 
- AGE3, NEWRACE2, IRMARIT, EDUHIGHCAT, INCOME
- IRINSUR4, ANY_NIC_EVER, ALCMON
- IRPREG, YEAR (2021/2022/2023)

### **Survey Weights** 
- ANALWT2_C1 / C2 / C3 (divided by 3 for pooled analysis)
  
---

# 4. Methods Pipeline (A + B + C + D) 

This repository follows a four-stage causal ML pipeline. 
The pipeline is structured as A (Weighted GLM) + B (DML for ATE) + C (Causal Forest for CATE) + D (Fairness Diagnostics).

--- 

## **A. Baseline: Weighted GLM (Logit)** 
- Survey-weighted logistic regression
- Construct a pooled weight `W_NORM` from ANALWT2_C1/C2/C3 (dividing by 3 and re-normalizing to mean 1)  
- Exposure → MDE(Major Depressive Episode) with full controls
- Robust SEs, domain corrections
- Produces baseline ATE and adjusted predicted probabilities

---

## **B. Double Machine Learning (ATE)** 
- Orthogonal DML
- Flexible nuisance learners (Random Forest, Gradient Boosting, regularized GLM)
- Cross-fitting to avoid overfitting
- ATE estimate for per exposure

---

## **C. Causal Forest (CATE)** 
- Generalized Random Forest
- Subgroup heterogeneity by IRPREG, income, race
- Variable importance, partial dependence
- Outputs: CATE distribution, Variable importance, Partial dependence functions, Heterogeneity heatmaps

---

## **D. Fairness Diagnostics** 
- TPR/FPR/PPV by pregnancy, race, income
- Fairness disparity plots
- Compare predictive disparity vs causal heterogeneity
- Identify whether ML models amplify or reduce subgroup inequality

---

# 5. Repository Structure
NSDUH-2021-2023-CausalML/
│
├── README.md
├── requirements.txt
│
├── src/
│ ├── 01_data_cleaning.py
│ ├── 02_weighted_glm.py
│ ├── 03_dml_ate.py
│ ├── 04_causal_forest_cate.py
│ └── 05_fairness_evaluation.py
│
├── notebooks/
│ └── 01_demo_pipeline.ipynb
│
├── figures/
│ ├── ate_result.png
│ ├── cate_heatmap.png
│ ├── fairness_plot.png
│ └── ate_compare_any_canna.png
│
├── results/
│ ├── analysis_dfs.pkl
│ ├── ate_table_B_weighted_logit.csv
│ ├── ate_table_DML.csv
│ └── fairness_table_boot.csv
│ └── fairness_table.csv
│ └── nsduh_analysis.csv
│ └── subgroup_ate_table.csv
│
└── docs/
│ └── NSDUH_CausalML_WorkingPaper.pdf

---
# 6. Quick Start 

### Clone the repo 
git clone https://github.com/basxxzz/NSDUH-2021-2023-CausalML.git 

### Install dependencies 
pip install -r requirements.txt 

### Demo Notebook (recommended entry point)

- **End-to-end pipeline**: [`notebooks/01_demo_pipeline.ipynb`](notebooks/01_demo_pipeline.ipynb)

  This notebook:
  - Runs the **A + B + C + D** pipeline on NSDUH 2021–2023
  - Calls the modular scripts in `src/`
  - Produces key outputs in `results/`
  - Saves paper-ready figures to `figures/` (ATE, CATE heatmap, fairness plots)


### Run the pipeline 
python src/01_data_cleaning.py 
python src/02_weighted_glm.py 
python src/03_dml_ate.py 
python src/04_causal_forest_cate.py 
python src/05_fairness_evaluation.py 

### Open the demo 
notebook notebooks/01_demo_pipeline.ipynb

---

# 7. Expected Outputs 

- Survey-weighted GLM results
- DML ATE results tables
- Causal Forest CATE distributions & heatmaps
- Fairness disparity metrics
- Publication-ready figures (stored in /figures)
- A 5–6 page working paper PDF (stored in /docs)

### Key Figures(Figures preview)

#### ATE (DML)
<img src="figures/ate_result.png" width="450"/>

#### GLM vs DML (ANY_CANNA_EVER)
<img src="figures/ate_compare_any_canna.png" width="450"/>

#### Subgroup CATE Heatmap
<img src="figures/cate_heatmap.png" width="450"/>

#### Fairness Metrics by Group
<img src="figures/fairness_plot.png" width="450"/>

### Working Paper

A PDF draft of the working paper is available at:

📄 [`docs/NSDUH_CausalML_WorkingPaper.pdf`](docs/NSDUH_CausalML_WorkingPaper.pdf)

---

# 8. Research Impact

This project demonstrates:
- strong causal inference training
- ability to execute full ML pipelines
- fairness + CATE + survey design competency
- reproducibility and documentation
- cross-disciplinary work bridging CS / Data Science / Biostatistics / Public Health / Computational Social Science

Ideal for PhD programs in: 
- Computer Science
- Data Science
- AI/ML
- Biostatistics
- Computational Social Science

This project forms the methodological foundation for my future doctoral research in causal machine learning, fairness-aware AI, heterogeneous treatment effects, and AI for health and social data.

---

# 9. Contact 

**Yifan Xu (许逸凡)** 
Email: **yifan1003a@gmail.com** 
Interests: causal ML, fair ML, heterogeneous treatment effects, survey inference, AI for health data.
