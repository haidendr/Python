# 📈 Capital Markets Profitability Simulator

A Python-based analytics project that models and visualizes the profitability of financial products across global capital markets using macroeconomic indicators.

This project simulates investment returns, calculates financial metrics like Net Present Value (NPV) and Internal Rate of Return (IRR), and provides strategic recommendations based on risk-adjusted performance across countries.

---

## 🎯 Project Objectives

- Simulate future cash flows under varying interest rate environments
- Calculate Net Present Value (NPV) and Internal Rate of Return (IRR)
- Analyze profitability across countries using macroeconomic data
- Visualize trends, returns, and political risks
- Generate actionable insights and exportable summaries

---

## 📊 Tools & Libraries Used

- **Python** (3.8+)
- `pandas`
- `numpy`, `numpy_financial`
- `matplotlib`, `seaborn`
- Jupyter Notebooks

---

## 🌍 Data Source

- **[Global Finance and Economic Indicators Dataset (2024) – Kaggle](https://www.kaggle.com/datasets/imaadmahmood/global-finance-and-economic-indicators-dataset-2024)**
  
This dataset includes macroeconomic indicators for 39 countries, such as:
- GDP growth
- Inflation
- Interest rates
- Political risk scores
- Stock indices, and more

---

## 🛠️ Key Features

### 1. **Data Cleaning & Transformation**
- Converts dates to datetime
- Renames columns to snake_case
- Selects relevant indicators
- Handles missing values with forward-fill strategy

### 2. **Scenario Definition**
- Defines economic profiles for selected countries (e.g., U.S., China, Japan)
- Extracts interest rate, inflation, GDP growth, and risk score

### 3. **Cash Flow Simulation**
- Models annual cash flows using compound interest formulas
- Adjustable for different principal amounts and time horizons

### 4. **Profitability Metrics**
- Calculates **NPV** using discount rates
- Computes **IRR** for each simulated investment
- Summarizes results in a comparison table

### 5. **Visualization**
- Bar plots for NPV and IRR by country
- Scatter plots showing political risk vs NPV
- Easy-to-read, presentation-ready charts

### 6. **Strategic Recommendations**
- Highlights countries with favorable risk-return profiles
- Uses **risk-adjusted NPV** to inform global investment strategy
- Recommends diversification targets

### 7. **Exportable Output**
- Saves profitability summary to `profitability_summary.csv` for use in:
  - Dashboards (Tableau, Power BI)
  - Reports
  - Presentations
