# ✈️ Clarity Travel Technology — Data Science Assessment

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **A complete end-to-end Data Science project** covering Exploratory Data Analysis, Predictive Modelling, NLP/GenAI complaint analysis, and an interactive Streamlit dashboard — built as part of the Clarity Travel Technology Solutions DS Assessment.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Live App](#-live-app)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Key Findings](#-key-findings)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Assessment Parts](#-assessment-parts)
- [Tech Stack](#-tech-stack)

---

## 🎯 Project Overview

This project analyses **2,000 airline booking records** from Clarity Travel Technology Solutions to:

1. **Uncover revenue and cancellation patterns** through detailed EDA
2. **Predict booking cancellations** using Logistic Regression and XGBoost
3. **Classify and summarise customer complaints** using NLP and GenAI (Groq/Llama 3.3 70B)
4. **Deliver an interactive dashboard** via Streamlit for the operations team

---

## 🚀 Live App

The full assessment is available as an **interactive 8-page Streamlit application**:

```bash
streamlit run clarity_app.py
```

### App Pages:
| Page | Description |
|------|-------------|
| 🏠 Home & Overview | Key metrics, navigation guide, findings summary |
| 📊 EDA Dashboard | Monthly trends, channel mix, booking status breakdown |
| 📈 Revenue Analysis | Top airlines, cabin class, NDC vs GDS, top routes |
| 🚫 Cancellation Patterns | By cabin, channel, lead time, fare buckets |
| 👥 Customer Behaviour | New vs Repeat customer comparison |
| 🤖 Cancellation Predictor | Real-time ML prediction for any booking |
| 💬 Complaint Summariser | AI-powered summaries using Groq LLM |
| 📋 Priority Action Board | Urgency-ranked complaint dashboard + CSV export |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | `clarity_bookings_dataset.csv` |
| Records | 2,000 airline bookings |
| Columns | 23 features |
| Period | January – December 2025 |
| Total Revenue | ₹25.06 Crore |
| Cancellation Rate | 22.6% |
| Complaints Logged | 248 |

**Key columns:** `booking_id`, `airline`, `cabin_class`, `trip_type`, `booking_channel`, `booking_source`, `total_fare_inr`, `lead_time_days`, `pax_count`, `booking_status`, `customer_complaint`, `satisfaction_score`

---

## 📁 Project Structure

```
clarity-ds-assessment/
│
├── clarity_app.py                        # 🚀 Streamlit interactive dashboard
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
│
├── data/
│   └── clarity_bookings_dataset.csv      # Source dataset
│
├── Part1_EDA/
│   └── Part1_EDA_Business_Insights.ipynb # EDA notebook (12+ charts)
│
├── Part2_Modelling/
│   └── Part2_Predictive_Modelling.ipynb  # ML modelling notebook
│
└── Part3_NLP_Bonus/
    └── Part3_NLP_Bonus.ipynb             # NLP + GenAI notebook
```

---

## 🔑 Key Findings

### 💰 Revenue
- **Air France** is the top revenue-generating airline at **₹3.1 Crore**
- **Business class** contributes the highest total revenue share **(38.1%)**
- **First class** commands the highest average fare at **₹4.2 Lakh per booking**
- **GDS channel** accounts for **68% of total bookings** vs NDC at 32%

### 🚫 Cancellations
- Overall cancellation rate: **22.6%** (Cancelled + Refunded)
- **Premium Economy** has the highest cancellation rate at **24.3%**
- **B2C Website** is the highest-risk channel at **26.0%**
- Cancelled bookings have **longer average lead times** (69.7 days vs 58.8 days) — early planners change their minds
- **One-Way trips** cancel most at **27.5%**

### 👥 Customer Behaviour
- **New customers** book higher-value tickets (avg ₹1.29L) but cancel more (24%)
- **Repeat customers** cancel less (20%) and have higher satisfaction (3.72 vs 3.66)

### 🤖 Predictive Model
| Model | Precision | Recall | F1-Score | AUC-ROC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.28 | 0.52 | 0.37 | **0.575** |
| XGBoost | 0.23 | 0.22 | 0.23 | 0.515 |

**Top predictive features:** `lead_time_bucket`, `is_group_booking`, `fare_per_pax`, `booking_source`, `is_repeat_customer`

> Logistic Regression outperforms XGBoost on this dataset — the cancellation signal is approximately linear, confirming explainable models are sufficient and preferable for this use case.

### 💬 Complaints (NLP)
| Category | Count | Urgency | Avg Satisfaction |
|----------|-------|---------|-----------------|
| Ticketing Issues | 64 | 🔴 HIGH | 1.62 / 5 |
| Refund Issues | 49 | 🔴 HIGH | 1.68 / 5 |
| Onboard Service | 36 | 🟢 LOW | 1.74 / 5 |
| Schedule Change | 31 | 🔴 HIGH | 1.76 / 5 |
| Baggage | 31 | 🟡 MEDIUM | 2.19 / 5 |
| Policy / Visa | 20 | 🟡 MEDIUM | 1.82 / 5 |
| Pricing Error | 17 | 🟡 MEDIUM | 1.81 / 5 |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/clarity-ds-assessment.git
cd clarity-ds-assessment
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run the Streamlit App
```bash
streamlit run clarity_app.py
```
Opens at **http://localhost:8501**

### Run the Jupyter Notebooks
```bash
jupyter notebook
```
Then open any notebook from `Part1_EDA/`, `Part2_Modelling/`, or `Part3_NLP_Bonus/`.

### Optional: Enable Live AI Summaries (Part 3 / Complaint Summariser)
1. Get a **free** Groq API key at https://console.groq.com/
2. In the Streamlit app → **💬 Complaint Summariser** → paste your key
3. Click **Generate AI Summaries** for live Llama 3.3 70B outputs

---

## 📓 Assessment Parts

### Part 1 — Exploratory Data Analysis (`Part1_EDA_Business_Insights.ipynb`)
- ✅ Revenue analysis by airline, route, cabin class, NDC vs GDS
- ✅ Cancellation patterns by lead time, fare bucket, channel, trip type
- ✅ Monthly booking trends and channel mix seasonality
- ✅ New vs Repeat customer behaviour comparison
- ✅ Missing value handling and data quality documentation
- ✅ Correlation heatmap
- ✅ **300-word business recommendations** for the Clarity operations team

### Part 2 — Predictive Modelling (`Part2_Predictive_Modelling.ipynb`)
- ✅ 10 engineered travel-domain features (fare per pax, lead time bucket, repeat customer flag, fare-to-route-avg ratio, etc.)
- ✅ Logistic Regression + XGBoost classifiers
- ✅ Stratified 80/20 train/test split
- ✅ Full evaluation: Precision, Recall, F1-Score, AUC-ROC, ROC curve, Precision-Recall curve
- ✅ Feature importance analysis (XGBoost) + coefficient chart (LR)
- ✅ Risk segment validation chart
- ✅ **Business operationalisation plan** with estimated revenue impact

### Part 3 — NLP/GenAI Bonus (`Part3_NLP_Bonus.ipynb`)
- ✅ **Option A:** Rule-based complaint classification into 7 business categories (100% accuracy on structured data)
- ✅ Visualisations: category distribution, airline cross-tab, channel/status correlation, cabin heatmap, satisfaction by category
- ✅ **Option B:** AI-powered batch complaint summariser using Groq API (Llama 3.3 70B)
- ✅ Structured output: urgency rating, root cause, recommended action, agent script
- ✅ Demo mode (no API key required) + live mode (free Groq key)
- ✅ Priority action board dashboard

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost` |
| NLP / GenAI | Groq API (`llama-3.3-70b-versatile`), rule-based classification |
| Dashboard | `streamlit` |
| Notebooks | `jupyter`, `ipykernel`, `nbformat` |

---

## 📬 Contact

Built as part of the **Clarity Travel Technology Solutions Data Science Assessment**.

---

*Dataset: Synthetic booking data provided by Clarity TTS | Analysis period: 2025*
