# 🏦 Banking Credit Risk Dashboard (LightGBM Edition)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://banking-credit-risk-app.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/kalyanizade/Banking_Credit_Risk_Project_App)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Live Application URL:**  
> 👉 **[https://banking-credit-risk-app.streamlit.app/](https://banking-credit-risk-app.streamlit.app/)**

---

## 🌟 Overview

An interactive, executive-grade web dashboard designed for retail banking credit risk analytics, loan origination simulation, and capital provisioning.

This application translates complex risk models into an intuitive business tool. Stakeholders can adjust **credit policy sliders** (Min FICO, Max TDI, Max LTV, Min Income) in real time to simulate approval rates, portfolio delinquency, ML model performance (**LightGBM vs. Logistic Regression**), and **IFRS 9 Expected Credit Loss**.

---

## 📊 Dashboard Architecture & 4-Stage Lifecycle

```
                           BANKING CREDIT RISK DASHBOARD
        ┌─────────────────────────────────────────────────────────────┐
        │                 ⚙️ Credit Policy Sliders                    │
        │   • Min FICO (300-850)        • Max TDI (0.10-0.90)         │
        │   • Max LTV (0.40-0.95)       • Min Monthly Income (₹)      │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
        ┌──────────────┬───────────────┴──────────────┬───────────────┐
        ▼              ▼                              ▼               ▼
  [Stage 1:      [Stage 2:                      [Stage 3:       [Stage 4:
  Origination]   Performance]                   ML Model]       Recovery]
  • Funnel Chart • Delinquency Buckets          • LR vs LightGBM• Pay Rate
  • Approval %   • Risk Segmentation (Pie)      • ROC Curves    • Recovery %
  • FICO Hist    • FICO vs Balance Scatter      • Split Count   • Collections
  • Raw Table    • Balance Control Ratio (BCR)  • IFRS 9 EL     • Action Matrix
```

### 1. 📂 Stage 1: Origination
- **Live Underwriting Simulation**: Adjusting FICO, TDI, LTV, or Income sliders recalculates Total Applications, Approval Rate, Booking Rate, and Decline Rate in milliseconds.
- **Origination Funnel**: Visualizes conversion from Applications $\rightarrow$ Approved $\rightarrow$ Booked.
- **Approval Rate by Portfolio**: Breakdowns across Personal Loans, Mortgages, Credit Cards, and Overdrafts.
- **FICO Score Distribution**: Histogram with live cutoff dashed lines and prime borrower thresholds.

### 2. 📉 Stage 2: Performance & Delinquency
- **Delinquency Buckets**: Tracks accounts across `Current`, `X` (1–30 DPD) through `X+5` and `CO` (Charge-Off).
- **Balance Control Ratio (BCR)**: Benchmark collections metric ($\text{GLPA} / \text{BLPA}$).
- **Risk Segmentation**: Portfolio distribution into High Risk, Medium Risk, and Low Risk tiers.
- **FICO vs. Outstanding Balance Scatter**: Highlights concentration of high-balance, low-credit accounts.

### 3. 🤖 Stage 3: ML Model (LightGBM vs. Logistic Regression)
- **Champion vs. Challenger AUC**: Logistic Regression (auditable baseline) vs. LightGBM (non-linear challenger).
- **ROC Curves**: Side-by-side True Positive vs. False Positive rate trade-offs.
- **Feature Importance**: Ranked by LightGBM split count.
- **IFRS 9 Expected Loss**: $\text{EL} = \text{PD} \times \text{EAD} \times \text{LGD}$ broken down by loan portfolio with product-specific haircuts:
  - Mortgage / Home Loan: **20%**
  - Auto Loan: **35%**
  - Line of Credit / Overdraft: **55%**
  - Personal Loan: **65%**
  - Credit Card: **75%**

### 4. 💰 Stage 4: Recovery & Collections
- **Charge-Off Account Metrics**: Pay Rate and Recovery Rate.
- **Recovery by Risk Segment**: Secured loan recovery vs. unsecured card recovery.
- **Collections Priority Action Plan**: Operational action rules for High, Medium, and Low-risk tiers.

---

## 🚀 How to Run Locally

### 1. Clone the repository:
```bash
git clone https://github.com/kalyanizade/Banking_Credit_Risk_Project_App.git
cd Banking_Credit_Risk_Project_App
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Start the dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## ☁️ Deployment

This project is deployed live on **Streamlit Community Cloud**:
👉 **[https://banking-credit-risk-app.streamlit.app/](https://banking-credit-risk-app.streamlit.app/)**

To deploy your own copy:
1. Fork or push this repository to your GitHub account.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Connect the repository, set the main file path to `app.py`, and click **Deploy**.
