# 🚨 Quantitative Early Warning System for Corporate Financial Distress

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)

A comprehensive software system that analyzes multi-year financial data to detect early warning signs of corporate financial distress using quantitative methods, machine learning, and advanced analytics.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Example Output](#example-output)
- [Contributing](#contributing)
- [License](#license)
- [Future Roadmap](#future-roadmap)

---

## 🎯 Overview

This system helps financial analysts, investors, and corporate managers identify companies at risk of financial distress **before** it becomes critical. By analyzing historical financial data and computing 20+ financial ratios, the system:

- ✅ Detects anomalies in financial metrics
- ✅ Computes a Composite Risk Score (0-100)
- ✅ Classifies companies as **Stable**, **Caution**, or **Distress**
- ✅ Provides actionable consulting-style recommendations

**Real-world applications:**
- Credit risk assessment for banks
- Investment due diligence
- Internal corporate health monitoring
- Early bankruptcy prediction

---

## ✨ Features

### Core Analytics
- **20+ Financial Ratios**: Liquidity, Solvency, Profitability, Efficiency, Growth
- **Time-Series Analysis**: Moving averages, volatility, trend detection
- **Anomaly Detection**: Z-score analysis and Isolation Forest
- **Risk Scoring**: Weighted composite score with customizable thresholds
- **PCA Analysis**: Dimensionality reduction for complex datasets

### Output Interfaces
- **Interactive Streamlit Dashboard**: Upload data, visualize trends, view recommendations
- **RESTful API**: FastAPI endpoints for programmatic access
- **Power BI Integration**: Export data for enterprise reporting

### Professional Features
- Automated data validation and preprocessing
- Comprehensive logging for debugging
- Modular, extensible architecture
- Unit tested components
- Consulting-grade recommendations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA INPUT LAYER                         │
│           (CSV, Excel, API, Manual Entry)                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION MODULE                         │
│   • File Reader  • Validation  • Logging                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING MODULE                           │
│   • Missing Value Handling  • Outlier Detection                 │
│   • Normalization  • Data Quality Checks                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FINANCIAL RATIO ENGINE                          │
│   • Liquidity Ratios    • Solvency Ratios                       │
│   • Profitability Ratios • Efficiency Ratios                    │
│   • Growth Ratios       • Market Ratios                         │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              ANALYTICS & ANOMALY DETECTION                       │
│   • Time-Series Analysis  • Z-Score Detection                   │
│   • Isolation Forest      • Statistical Tests                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RISK SCORING ENGINE                             │
│   • Weighted Linear Combination                                  │
│   • Score Normalization (0-100)                                 │
│   • Classification: Stable/Caution/Distress                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONSULTING RECOMMENDATION ENGINE                    │
│   • Strategic Recommendations  • Action Items                   │
│   • Priority Matrix           • Implementation Roadmap          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                            │
│   • Streamlit Dashboard  • FastAPI Endpoints                    │
│   • Power BI Export      • PDF Reports                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Core Language** | Python 3.8+ | Main development language |
| **Data Processing** | pandas, NumPy | Data manipulation and numerical operations |
| **Machine Learning** | scikit-learn | Anomaly detection, PCA, clustering |
| **Statistics** | SciPy | Statistical tests, distributions |
| **Visualization** | Matplotlib, Seaborn, Plotly | Charts and graphs |
| **Dashboard** | Streamlit | Interactive web interface |
| **API** | FastAPI | RESTful API endpoints |
| **Data Validation** | Pydantic | Schema validation |
| **Testing** | pytest | Unit and integration tests |
| **File Handling** | openpyxl, xlrd | Excel file support |
| **Logging** | Python logging | Debug and audit trails |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/financial-distress-ews.git
cd financial-distress-ews
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import pandas; import sklearn; import streamlit; print('Installation successful!')"
```

---

## 🚀 Quick Start

### Option 1: Run the Streamlit Dashboard (Recommended)
```bash
streamlit run src/dashboard/app.py
```
Then open your browser to `http://localhost:8501`

### Option 2: Run Analysis via Command Line
```bash
python main.py --input data/raw/sample_data.csv --output results/
```

### Option 3: Start the FastAPI Server
```bash
uvicorn src.api.server:app --reload
```
API documentation available at `http://localhost:8000/docs`

---

## 📖 Usage

### 1. Prepare Your Data

Create a CSV file with the following columns:
```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,
total_debt,equity,inventory,cogs,operating_income,interest_expense,
accounts_receivable,cash
```

**Example:**
```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
TechCorp,2020,1000000,100000,2000000,500000,300000,800000,1200000,150000,600000,150000,50000,200000,150000
TechCorp,2021,1100000,110000,2200000,550000,320000,850000,1350000,160000,650000,165000,55000,220000,180000
TechCorp,2022,1200000,120000,2400000,600000,340000,900000,1500000,170000,700000,180000,60000,240000,210000
```

### 2. Upload to Dashboard

1. Start the Streamlit dashboard
2. Click "Upload CSV File"
3. View automatically calculated ratios
4. Explore time-series trends
5. Review anomaly alerts
6. Check risk score and classification
7. Read recommendations

### 3. Interpret Results

**Risk Score Interpretation:**
- **70-100 (Stable)**: Company is financially healthy
- **40-69 (Caution)**: Warning signs detected, monitor closely
- **0-39 (Distress)**: Critical condition, immediate action required

**Key Ratios to Monitor:**
- Current Ratio < 1.0 → Liquidity crisis
- Debt-to-Equity > 2.0 → Excessive leverage
- ROE < 5% → Poor profitability
- Interest Coverage < 1.5 → Debt service risk

---

## 📁 Project Structure

```
financial-distress-ews/
│
├── data/
│   ├── raw/                          # Original datasets
│   │   └── sample_data.csv           # Example financial data
│   └── processed/                    # Cleaned datasets
│       └── cleaned_data.csv
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb # Data exploration
│   ├── 02_ratio_analysis.ipynb       # Ratio calculations
│   └── 03_model_development.ipynb    # ML model experiments
│
├── src/
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py                 # File reading and validation
│   │   └── validator.py              # Data quality checks
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py                # Missing value handling
│   │   └── normalizer.py             # Data normalization
│   │
│   ├── ratio_engine/
│   │   ├── __init__.py
│   │   ├── ratios.py                 # Financial ratio calculations
│   │   └── definitions.py            # Ratio formulas and metadata
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── timeseries.py             # Trend analysis, moving averages
│   │   └── statistical.py            # Hypothesis tests, distributions
│   │
│   ├── anomaly_detection/
│   │   ├── __init__.py
│   │   ├── zscore.py                 # Z-score anomaly detection
│   │   └── isolation_forest.py       # ML-based anomaly detection
│   │
│   ├── risk_score/
│   │   ├── __init__.py
│   │   ├── score.py                  # Composite risk scoring
│   │   └── classifier.py             # Risk classification logic
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── charts.py                 # Plotting functions
│   │   └── reports.py                # PDF report generation
│   │
│   ├── consulting/
│   │   ├── __init__.py
│   │   └── recommend.py              # Strategic recommendations
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py                    # Streamlit dashboard
│   │
│   └── api/
│       ├── __init__.py
│       ├── server.py                 # FastAPI application
│       └── schemas.py                # Pydantic models
│
├── tests/
│   ├── __init__.py
│   ├── test_ratios.py                # Ratio calculation tests
│   ├── test_preprocessing.py         # Data cleaning tests
│   ├── test_anomaly.py               # Anomaly detection tests
│   └── test_api.py                   # API endpoint tests
│
├── main.py                           # Command-line entry point
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── CONTRIBUTING.md                   # Contribution guidelines
├── LICENSE                           # MIT License
├── .gitignore                        # Git ignore rules
└── setup.py                          # Package installation script
```

---

## 📊 Example Output

### Dashboard Screenshot (ASCII Representation)
```
╔═══════════════════════════════════════════════════════════════╗
║     FINANCIAL DISTRESS EARLY WARNING SYSTEM                   ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Company: TechCorp Inc.                                       ║
║  Analysis Period: 2020-2024                                   ║
║  Last Updated: 2024-02-11                                     ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║  RISK SCORE:  [████████░░] 78/100 - STABLE                   ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  KEY FINANCIAL RATIOS                                         ║
║  ┌─────────────────────┬──────┬──────┬──────┬──────┐        ║
║  │ Ratio               │ 2021 │ 2022 │ 2023 │ 2024 │        ║
║  ├─────────────────────┼──────┼──────┼──────┼──────┤        ║
║  │ Current Ratio       │ 1.72 │ 1.76 │ 1.81 │ 1.85 │ ✓      ║
║  │ Debt-to-Equity      │ 0.65 │ 0.63 │ 0.60 │ 0.58 │ ✓      ║
║  │ ROE                 │ 9.2% │ 9.8% │10.2% │10.5% │ ✓      ║
║  │ Net Profit Margin   │ 8.5% │ 8.8% │ 9.1% │ 9.3% │ ✓      ║
║  └─────────────────────┴──────┴──────┴──────┴──────┘        ║
║                                                               ║
║  ANOMALIES DETECTED: 0                                        ║
║                                                               ║
║  TREND ANALYSIS                                               ║
║  • Liquidity: Improving ↗                                     ║
║  • Solvency: Improving ↗                                      ║
║  • Profitability: Stable →                                    ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║  RECOMMENDATIONS                                              ║
║  ✓ Maintain current financial strategy                       ║
║  ✓ Continue debt reduction initiatives                       ║
║  ✓ Monitor profit margin sustainability                      ║
╚═══════════════════════════════════════════════════════════════╝
```

### API Response Example
```json
{
  "company": "TechCorp Inc.",
  "risk_score": 78,
  "classification": "Stable",
  "ratios": {
    "current_ratio": 1.85,
    "quick_ratio": 1.42,
    "debt_to_equity": 0.58,
    "roe": 0.105,
    "roa": 0.065,
    "net_profit_margin": 0.093
  },
  "anomalies": [],
  "trends": {
    "liquidity": "improving",
    "solvency": "improving",
    "profitability": "stable"
  },
  "recommendations": [
    "Maintain current financial strategy",
    "Continue debt reduction initiatives",
    "Monitor profit margin sustainability"
  ]
}
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🗺️ Future Roadmap

### Version 2.0
- [ ] Real-time data integration (Yahoo Finance API, Alpha Vantage)
- [ ] Machine learning prediction models (Random Forest, XGBoost)
- [ ] Industry benchmarking (compare against sector averages)
- [ ] Multi-company portfolio analysis
- [ ] Automated email alerts for risk threshold breaches

### Version 3.0
- [ ] Deep learning models (LSTM for time-series forecasting)
- [ ] Natural language processing for earnings call sentiment
- [ ] Cloud deployment (AWS, Azure, Google Cloud)
- [ ] Mobile application (iOS/Android)
- [ ] Integration with accounting software (QuickBooks, Xero)

---

## 📞 Support

- **Documentation**: [Wiki](https://github.com/yourusername/financial-distress-ews/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/financial-distress-ews/issues)
- **Email**: support@yourcompany.com

---

## 🙏 Acknowledgments

- Inspired by the Altman Z-Score model
- Financial ratio definitions from Corporate Finance Institute
- Built with ❤️ using open-source technologies

---

**⭐ If you find this project useful, please star it on GitHub!**

Last Updated: February 2024
