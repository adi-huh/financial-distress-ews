# 📄 PDF Extraction + Financial Analysis Integration Guide

## Overview

You now have a **complete integrated system** that:

1. **📄 Extracts financial metrics from PDF annual reports**
2. **📊 Calculates 40+ financial ratios**
3. **📈 Analyzes trends and time-series data**
4. **🔍 Detects financial anomalies**
5. **🎯 Computes comprehensive risk scores**
6. **💡 Generates AI-powered recommendations**
7. **📁 Exports everything to CSV**

---

## 🚀 Quick Start

### Option 1: Interactive Web Dashboard (Recommended)

```bash
cd /Users/adi/Documents/financial-distress-ews

# Launch the integrated PDF + Analysis app
streamlit run app_pdf.py
```

**Then:**
1. Open browser to `http://localhost:8501`
2. Upload a PDF annual report OR CSV file
3. View complete financial analysis results
4. Download CSV with all metrics and ratios

### Option 2: Command Line

```bash
# Extract from single PDF
python quickstart.py extract --pdf your_report.pdf

# Batch process PDFs
python quickstart.py batch --dir ./reports

# Run demonstration
python quickstart.py demo
```

### Option 3: Python Script

```python
from orchestrator import FinancialExtractionOrchestrator
from loader import DataLoader
from ratios import FinancialRatioEngine
from score import RiskScoreEngine

# Extract from PDF
orchestrator = FinancialExtractionOrchestrator()
result = orchestrator.extract_and_analyze_single('report.pdf', 'output')

# Get extracted data as DataFrame
import pandas as pd
df = pd.DataFrame([result['cleaned_metrics']])

# Calculate ratios
ratios_engine = FinancialRatioEngine()
ratios_df = ratios_engine.calculate_all_ratios(df)

# Calculate risk score
risk_engine = RiskScoreEngine()
risk_scores = risk_engine.calculate_risk_score(ratios_df, pd.DataFrame())

print(f"Company: {result['company']}")
print(f"Financial Health: {ratios_df}")
print(f"Risk Score: {risk_scores}")
```

---

## 📊 What Gets Calculated

### Financial Ratios (40+)

**Liquidity:**
- Current Ratio
- Quick Ratio
- Cash Ratio
- Working Capital

**Profitability:**
- Gross Margin
- Operating Margin
- Net Profit Margin
- ROA (Return on Assets)
- ROE (Return on Equity)
- ROIC (Return on Invested Capital)

**Leverage:**
- Debt-to-Equity Ratio
- Debt-to-Assets Ratio
- Interest Coverage Ratio
- Long-term Debt Ratio

**Efficiency:**
- Asset Turnover
- Inventory Turnover
- Receivables Turnover
- Payables Turnover
- Cash Conversion Cycle

**And many more...**

### Analysis Features

1. **Anomaly Detection**
   - Z-score analysis
   - Isolation Forest
   - Multi-method ensemble

2. **Risk Scoring**
   - Liquidity Risk (25%)
   - Profitability Risk (25%)
   - Leverage Risk (25%)
   - Operational Risk (25%)
   - Overall Score (0-100)

3. **Time-Series Analysis**
   - Trend analysis
   - Moving averages
   - Volatility metrics
   - Correlation analysis

4. **Recommendations**
   - Immediate actions
   - Short-term strategy (3-6 months)
   - Long-term strategy (6-18 months)
   - Priority classification

---

## 📁 File Structure

```
financial-distress-ews/
│
├── 📄 APP FILES (Choose one to run)
│   ├── app_pdf.py              ⭐ BEST: PDF + CSV + Full Analysis
│   ├── app_simple.py           ✅ Simple mode with PDF extraction
│   ├── app.py                  (Original - use app_pdf.py instead)
│   ├── quickstart.py           CLI entry point
│
├── 🧮 EXTRACTION MODULES
│   ├── orchestrator.py         Unified orchestrator
│   ├── intelligent_pdf_extractor.py    PDF metric extraction
│   ├── pattern_learner.py      Learn extraction patterns
│   ├── extraction_pipeline.py  End-to-end pipeline
│   ├── financial_analysis.py   Financial health analysis
│   ├── demo.py                 Comprehensive demo
│
├── 📊 ANALYSIS MODULES
│   ├── loader.py               Data loading & validation
│   ├── cleaner.py              Data cleaning & preprocessing
│   ├── ratios.py               Financial ratio calculations
│   ├── timeseries.py           Trend & time-series analysis
│   ├── zscore.py               Anomaly detection
│   ├── score.py                Risk scoring engine
│   ├── recommend.py            AI recommendations
│   ├── charts.py               Visualization & charts
│
├── 📁 DATA DIRECTORY
│   ├── sample_data.csv         Sample for testing
│   └── annual_reports_2024/    Training data (25 reports)
│
└── 📁 OUTPUT DIRECTORIES
    └── extracted_data/         Generated CSV files
    └── results/                Analysis results
```

---

## 💡 Usage Examples

### Example 1: Upload PDF → Get Everything

```
1. Run: streamlit run app_pdf.py
2. Select Mode: "📄 PDF → CSV → Analysis"
3. Upload: annual_report.pdf
4. Get: CSV with metrics + ratios + analysis + recommendations
```

### Example 2: Upload CSV → Get Everything

```
1. Run: streamlit run app_pdf.py
2. Select Mode: "📊 CSV Direct Analysis"
3. Upload: financial_data.csv
4. Get: 40+ ratios + risk scores + anomalies + recommendations
```

### Example 3: Batch Process Multiple PDFs

```bash
# Extract and analyze all PDFs in a directory
python quickstart.py batch --dir ./annual_reports

# Outputs:
# - all_companies_combined.csv
# - batch_analysis_summary.json
# - Individual CSV and JSON for each company
```

---

## 📥 CSV Output Format

### Extracted Metrics CSV

```
company,year,revenue,net_income,total_assets,current_ratio,roe,net_margin,...
Company A,2024,1000000000,100000000,500000000,1.5,0.33,0.10,...
Company B,2024,800000000,60000000,400000000,1.2,0.20,0.075,...
```

### Risk Scores CSV

```
company,risk_score,classification,priority,trend
Company A,75,Stable,LOW,Improving
Company B,45,Caution,MEDIUM,Stable
Company C,25,Distress,HIGH,Deteriorating
```

---

## 🔧 Troubleshooting

### Issue: PDF Module Not Found
**Solution:** Make sure `orchestrator.py` is in the same directory

### Issue: "ModuleNotFoundError: No module named..."
**Solution:** 
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: PDF extraction not working
**Solution:** 
```bash
# Make sure pdfplumber is installed
pip install pdfplumber PyPDF2
```

### Issue: Streamlit not found
**Solution:**
```bash
# Install streamlit
pip install streamlit

# Use virtual environment
.venv/bin/streamlit run app_pdf.py
```

---

## 📊 Key Metrics Explained

### Risk Score (0-100)
- **75-100:** Excellent financial health (GREEN)
- **60-74:** Good financial health (YELLOW)
- **40-59:** Adequate financial health (ORANGE)
- **<40:** Poor financial health (RED)

### Ratio Categories
1. **Liquidity** - Can company pay short-term debts?
2. **Profitability** - How efficiently is company earning?
3. **Leverage** - Is company over-leveraged?
4. **Efficiency** - How well are assets utilized?

### Anomaly Severity
- **Critical:** Extreme outliers (>3 std dev)
- **High:** Significant anomalies (2-3 std dev)
- **Medium:** Moderate anomalies (1-2 std dev)
- **Low:** Minor variations (<1 std dev)

---

## 🎯 Workflow

```
PDF/CSV Upload
     ↓
Extract Metrics (if PDF)
     ↓
Clean & Validate Data
     ↓
Calculate 40+ Ratios
     ↓
Analyze Trends
     ↓
Detect Anomalies
     ↓
Compute Risk Scores
     ↓
Generate Recommendations
     ↓
Export CSV & Display Results
```

---

## 🚀 Next Steps

1. **Test with your PDF:**
   ```bash
   streamlit run app_pdf.py
   ```

2. **Batch process multiple companies:**
   ```bash
   python quickstart.py batch --dir ./your_reports
   ```

3. **Export results for further analysis:**
   - Download CSVs from the Streamlit interface
   - Use in Excel, Tableau, or Power BI

4. **Automate in production:**
   - Use `orchestrator.py` in Python scripts
   - Schedule batch processing
   - Integrate with BI tools

---

## 📞 Support

For questions about:
- **PDF Extraction:** See `orchestrator.py` and `pattern_learner.py`
- **Financial Ratios:** See `ratios.py` and `DEVELOPER_GUIDE.md`
- **Risk Scoring:** See `score.py` and `recommend.py`
- **Streamlit App:** See `app_pdf.py` and `QUICK_START.md`

---

## ✅ Verification

To verify everything is working:

```bash
# Test extraction
.venv/bin/python quickstart.py demo

# Test analysis
.venv/bin/python main.py -i sample_data.csv

# Test Streamlit app
.venv/bin/streamlit run app_pdf.py
```

All should complete without errors! ✅

---

**Happy analyzing! 📊🚀**
