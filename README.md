# ARIA — Adaptive Reasoning Intelligence for Analytics

> A data science agent that questions its own answers.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What is ARIA?

Upload any dataset. Describe your problem in one sentence. ARIA figures out the right ML task, runs and compares models, explains the results in plain English, and produces three different reports for three different readers — all without you touching a single line of code.

The part that makes it different from standard AutoML: **after classifying the task type, ARIA spawns a Devil's Advocate check that actively tries to disprove its own classification before confirming it.** No silent misclassification.

Built for **ABB EngineeredX 2.0 | Problem Statement #8** — Design and Evaluate Data Science Language Models That Can Adapt to Different Data Science Models.

---

## Demo

![ARIA Screenshot](assets/demo.png)

Try it with any of these datasets:

| Dataset | Task ARIA detects | Download |
|---|---|---|
| Telco Customer Churn | Classification | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| House Prices | Regression | [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| Mall Customer Segmentation | Clustering | [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial) |
| Air Passenger Traffic | Time-Series | [Kaggle](https://www.kaggle.com/datasets/rakannimer/air-passengers) |
| COVID-19 Tweet Sentiment | Classification | [Kaggle](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification) |

---

## Features

**Adapts to any task type automatically**

| Task | How it detects | Models compared |
|---|---|---|
| Classification | Categorical target column | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| Regression | Continuous numeric target | Ridge, Random Forest, Gradient Boosting |
| Clustering | No target column specified | KMeans k=2..6, auto-selects best by Silhouette score |
| Time-Series | DateTime column + continuous target | Ridge + Random Forest on lag features, MAE/MAPE |
| Anomaly Detection | Sensor keywords + no labels | Isolation Forest |

**Devil's Advocate reasoning**

Every task classification runs a two-pass check:
1. Primary agent votes based on target type, problem text, column names, and ABB ontology
2. Devil's Advocate agent looks for counter-evidence against the winner
3. Confirms only if it fails to disprove

**ABB Industrial Ontology**

Pre-loaded keyword map routes industrial data to the right context before training:
- Motor vibration + current → Anomaly Detection
- Energy consumption logs → Time-Series Forecasting
- Quality inspection records → Classification
- Process parameters → Regression

**Tri-Persona Output Engine**

Same analysis, three structurally different documents:

| Persona | Audience | What they get |
|---|---|---|
| Engineer Mode | Data Scientist / ML Engineer | Leaderboard, metrics, top features, preprocessing log, Devil's Advocate audit |
| Manager Mode | Plant Manager / Business Head | 5-line summary, risk level, recommended action, confidence score |
| Regulator Mode | Compliance / Audit team | Model card, bias check note, data lineage, deployment recommendation |

---

## Architecture

```
Dataset + Problem Description
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 1 — Profile                  │
│  Schema detection · missing values  │
│  Skew · correlations · drift scan   │
│  ABB ontology match                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 2 — Classify                 │
│  4-signal voting system             │
│  Devil's Advocate check             │
│  Confidence score + reasoning chain │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 3 — AutoML                   │
│  Task-specific model comparison     │
│  Cross-validated leaderboard        │
│  Consequence-weighted metrics       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 4 — Explain                  │
│  Random Forest feature importance   │
│  Plain-English interpretation       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 5 — Report                   │
│  Engineer · Manager · Regulator     │
└─────────────────────────────────────┘
```

---

## Project structure

```
aria/
├── app.py                      ← Streamlit UI (5 tabs, tri-persona report)
├── ARIA_Demo_Notebook.ipynb    ← Jupyter walkthrough on sample data
├── requirements.txt
├── README.md
└── utils/
    └── aria_pipeline.py        ← All 5 pipeline stages
```

---

## Setup

```bash
git clone https://github.com/nehachinnam956/aria-analytics
cd aria-analytics
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Or run the notebook:**
```bash
jupyter notebook ARIA_Demo_Notebook.ipynb
```

---

## How to use

1. Upload any CSV or Excel file from the sidebar
2. Type a one-sentence problem description
3. Enter the target column name if you have one (leave blank for clustering)
4. Select consequence of being wrong, computation budget, and expertise level
5. Hit **Run ARIA**

Results appear across five tabs: Data Profile → Task Classification → AutoML Results → Feature Importance → Tri-Persona Report.

---

## Tech stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| ML models | scikit-learn — RF, GBM, Logistic Regression, Ridge, KMeans |
| Visualisation | Plotly |
| Data processing | pandas, numpy |
| Language | Python 3.9+ |

---

## Evaluation criteria — ABB EngineeredX 2.0

| Criterion | How ARIA addresses it |
|---|---|
| Innovation & Originality | Devil's Advocate metacognitive check — no existing AutoML tool implements this |
| Technical Implementation | 5 stages, all 5 task types, cross-validated leaderboard, feature importance, tri-persona output |
| Industrial Relevance | ABB ontology pre-loaded, consequence-weighted metrics, Manager and Regulator outputs built for real decision-makers |
| Scalability & Robustness | Modular stages, each independently swappable, handles any CSV dataset across any domain |

---

## Author

**Bhagavathi Neha Chinnam**
B.Tech Big Data & Analytics | SRM University AP | CGPA 8.86
[github.com/nehachinnam956](https://github.com/nehachinnam956) · nehabr.2302@gmail.com
