# 🏦 Banking Credit Risk Dashboard (LightGBM Edition)

> **Interactive Executive Streamlit Dashboard for Banking Credit Risk Lifecycle.**  
> **Machine Learning:** Logistic Regression (Champion) vs. LightGBM (Challenger)  
> **Design Reference:** [banking-credit-risk.streamlit.app](https://banking-credit-risk.streamlit.app/)

---

## 🌟 Overview

This dashboard translates complex credit risk data science models into an intuitive, interactive tool designed for credit policy teams, risk managers, and business stakeholders.

Stakeholders can adjust **credit policy sliders** (FICO cutoff, TDI limit, LTV cap, Min Income) in real time and immediately see the impact across approval rates, default rates, ML AUC curves (LightGBM vs. Logistic Regression), and IFRS 9 Expected Credit Loss.

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
- **Live Underwriting Simulation**: Moving FICO, TDI, LTV, or Income sliders recalculates Total Applications, Approval Rate, Booking Rate, and Decline Rate in milliseconds.
- **Origination Funnel**: Visualizes conversion from Applications $\rightarrow$ Approved $\rightarrow$ Booked.
- **Approval Rate by Portfolio**: Breakdowns across Personal Loans, Mortgages, Credit Cards, and Overdrafts.
- **FICO Score Distribution**: Histogram with policy cutoffs and prime thresholds.

### 2. 📉 Stage 2: Performance & Delinquency
- **Delinquency Buckets**: Tracks accounts across `Current`, `X` (1–30 DPD) through `X+5` and `CO` (Charge-Off).
- **Balance Control Ratio (BCR)**: Benchmark collections metric ($\text{GLPA} / \text{BLPA}$).
- **Risk Segmentation**: Portfolio distribution into High Risk, Medium Risk, and Low Risk tiers.
- **FICO vs. Outstanding Balance Scatter**: Exposes high-exposure, low-credit accounts.

### 3. 🤖 Stage 3: ML Model (LightGBM vs. Logistic Regression)
- **Champion vs. Challenger AUC**: Logistic Regression (auditable baseline) vs. LightGBM (non-linear challenger).
- **ROC Curves**: Side-by-side True Positive vs. False Positive rate trade-offs.
- **Feature Importance**: LightGBM split-count ranking of top default risk drivers.
- **IFRS 9 Expected Loss**: $\text{EL} = \text{PD} \times \text{EAD} \times \text{LGD}$ broken down by loan portfolio.

### 4. 💰 Stage 4: Recovery & Collections
- **Charge-Off Account Metrics**: Pay Rate and Recovery Rate.
- **Recovery by Risk Segment**: Secured loan recovery vs. unsecured card recovery.
- **Collections Priority Action Plan**: Operational rules for High, Medium, and Low-risk tiers.

---

## 🚀 How to Run Locally

```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## ☁️ How to Deploy to Streamlit Community Cloud (Free)

1. **Commit and push all files to your GitHub repository**:
   ```bash
   git add .
   git commit -m "Banking Credit Risk Dashboard with LightGBM"
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. **Open [share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
3. Click **"New App"** and choose:
   - **Repository:** `<your-username>/<your-repo-name>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**
