# Quick Reference Guide

## 🚀 Get Started in 60 Seconds

### Installation
```bash
# 1. Clone repo
git clone https://github.com/adi-huh/financial-distress-ews.git
cd financial-distress-ews

# 2. Setup environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Analysis
```bash
# Option 1: CLI (Command Line)
python main.py -i data/sample_data.csv

# Option 2: Web Dashboard
streamlit run app.py
# Open http://localhost:8501
```

---

## 📊 Input Data Format

**Required columns:**
- `company`: Company name
- `year`: Fiscal year
- `revenue`: Total revenue
- `net_income`: Net income/profit
- `total_assets`: Total assets
- `current_assets`: Current assets
- `current_liabilities`: Current liabilities
- `total_debt`: Total debt
- `equity`: Shareholders' equity

**Optional columns:**
- `inventory`, `cogs`, `operating_income`, `interest_expense`
- `accounts_receivable`, `cash`, `accounts_payable`

**Example:**
```
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity
TechCorp,2024,1400000,140000,2800000,700000,420000,1000000,1800000
FinanceCo,2024,600000,60000,1200000,300000,180000,480000,720000
```

---

## 📈 Output Explained

### Risk Score (0-100)
- **70-100**: 🟢 **Stable** - Low distress risk
- **40-69**: 🟡 **Caution** - Moderate risk
- **0-39**: 🔴 **Distress** - High risk

### Financial Ratios Calculated

| Category | Ratios |
|----------|--------|
| **Liquidity** | Current, Quick, Cash, Working Capital |
| **Solvency** | Debt-to-Equity, Interest Coverage, Debt Ratio |
| **Profitability** | Net Margin, ROA, ROE, ROIC, Gross Margin |
| **Efficiency** | Asset Turnover, Inventory Turnover, DSO |
| **Growth** | Revenue Growth, Income Growth, Asset Growth |

### Anomalies Detected
- **Severity Levels**: Critical (>5σ), High (>4σ), Medium (>3σ), Low (>2σ)
- **Method**: Z-score, Isolation Forest, or both
- **Output**: Flagged values with deviation from mean

### Recommendations
- **Liquidity Focus**: Manage working capital, improve cash flow
- **Solvency Focus**: Debt restructuring, interest coverage improvement
- **Profitability Focus**: Cost reduction, revenue optimization
- **Efficiency Focus**: Operational improvements, asset utilization
- **Growth Focus**: Expansion strategies, market share growth

---

## 🎯 Common Tasks

### Task 1: Analyze Your Data
```bash
python main.py -i your_data.csv -o results/ --export-format excel
```
**Outputs:** Ratios CSV, Risk scores, Recommendations, Charts

### Task 2: Detect Specific Anomalies
```bash
python main.py -i data.csv --anomaly-method zscore
```
**Options:** `zscore`, `isolation_forest`

### Task 3: Interactive Dashboard
```bash
streamlit run app.py
```
**Then:** Upload file → Configure options → View analysis

### Task 4: Run Tests
```bash
# All tests
pytest tests.py -v

# Specific test
pytest tests.py::TestFinancialRatioEngine -v

# With coverage
pytest tests.py --cov
```

---

## 🔧 Python API Quick Reference

### Load Data
```python
from loader import DataLoader

loader = DataLoader()
data = loader.load_file('data.csv')
```

### Clean Data
```python
from cleaner import DataCleaner

cleaner = DataCleaner()
clean_data = cleaner.clean(data)
```

### Calculate Ratios
```python
from ratios import FinancialRatioEngine

engine = FinancialRatioEngine()
ratios = engine.calculate_all_ratios(clean_data)
```

### Analyze Trends
```python
from timeseries import TimeSeriesAnalyzer

analyzer = TimeSeriesAnalyzer()
trends = analyzer.analyze_trends(ratios)
```

### Detect Anomalies
```python
from zscore import AnomalyDetectionEngine

detector = AnomalyDetectionEngine()
anomalies = detector.detect_all_anomalies(ratios)
```

### Score Risk
```python
from score import RiskScoreEngine

scorer = RiskScoreEngine()
scores = scorer.calculate_risk_score(ratios, anomalies)
```

### Get Recommendations
```python
from recommend import ConsultingEngine

consultant = ConsultingEngine()
recommendations = consultant.generate_recommendations(ratios, scores)
```

### Create Charts
```python
from charts import ChartGenerator

gen = ChartGenerator()
gen.create_dashboard(ratios, scores, 'output_dir/')
```

---

## 📁 Project Structure

```
financial-distress-ews/
├── data/
│   ├── raw/              # Your raw data files
│   ├── processed/        # Generated cleaned data
│   └── sample_data.csv   # Sample for testing
├── results/              # Generated outputs
│   ├── charts/           # PNG charts
│   └── *.csv             # Result files
├── app.py                # Streamlit dashboard
├── main.py               # CLI application
├── *.py                  # Core modules
├── tests.py              # Test suite
├── README.md             # Overview
├── ARCHITECTURE.md       # System design
├── DEVELOPER_GUIDE.md    # Development guide
└── requirements.txt      # Dependencies
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | Run `pip install -r requirements.txt` |
| "File not found" | Check file path, use absolute path |
| "Missing columns" | Verify CSV has required columns |
| "Division by zero" | Module handles this, check logs |
| "Memory error" | Process data in chunks for large files |
| Dashboard won't start | Check port 8501 isn't in use |

---

## 📚 Documentation Map

| Document | Purpose |
|----------|---------|
| **README.md** | Overview, features, installation |
| **QUICK_START.md** | 5-minute quick start |
| **SETUP_GUIDE.md** | Detailed installation |
| **ARCHITECTURE.md** | System design, data flow |
| **DEVELOPER_GUIDE.md** | Development standards |
| **PROJECT_STATUS.md** | Current project status |
| **CONTRIBUTING.md** | Contribution guidelines |

---

## 💡 Tips & Tricks

### Tip 1: Custom Weights
```python
from score import RiskScoreEngine

weights = {
    'liquidity': 0.40,      # More important
    'solvency': 0.20,
    'profitability': 0.20,
    'efficiency': 0.15,
    'growth': 0.05
}
scorer = RiskScoreEngine(weights=weights)
```

### Tip 2: Save Results
```bash
python main.py -i data.csv -o results/ --export-format excel
# Outputs: .xlsx file with all data
```

### Tip 3: Batch Processing
```python
import glob
from loader import DataLoader

for file in glob.glob('data/*.csv'):
    loader = DataLoader()
    data = loader.load_file(file)
    # Process each file
```

### Tip 4: Debug Mode
```bash
python main.py -i data.csv --verbose
# Shows detailed logs
```

---

## 🎓 Understanding the Scores

### Liquidity Score (0-100)
- **Measures:** Ability to pay short-term obligations
- **Key Ratios:** Current ratio, Quick ratio, Cash ratio
- **Good Score:** > 80

### Solvency Score (0-100)
- **Measures:** Long-term financial stability
- **Key Ratios:** Debt-to-Equity, Interest Coverage
- **Good Score:** > 75

### Profitability Score (0-100)
- **Measures:** Earning efficiency
- **Key Ratios:** Net Margin, ROA, ROE
- **Good Score:** > 70

### Efficiency Score (0-100)
- **Measures:** Operational effectiveness
- **Key Ratios:** Asset Turnover, Inventory Turnover
- **Good Score:** > 65

### Growth Score (0-100)
- **Measures:** Expansion trajectory
- **Key Ratios:** Revenue Growth, Income Growth
- **Good Score:** > 60

---

## 🔗 External Resources

- **Financial Ratios:** https://www.investopedia.com/
- **Statistical Methods:** https://www.khanacademy.org/
- **Pandas Documentation:** https://pandas.pydata.org/
- **Scikit-learn:** https://scikit-learn.org/
- **Streamlit:** https://streamlit.io/

---

## 📞 Support

- **Issues:** Check GitHub Issues
- **Questions:** Create Discussion
- **Bugs:** Report with reproducible example
- **Features:** Request via Issues

---

## 📋 Checklist: Before Analyzing

- [ ] Data is in CSV or Excel format
- [ ] All required columns present
- [ ] No special characters in company names
- [ ] Years are numeric (YYYY)
- [ ] Financial values are numeric (no commas/currency symbols)
- [ ] Data is sorted by company then year
- [ ] At least 2 years of data per company
- [ ] At least 2 companies in dataset

---

## ⚡ Performance Tips

1. **Use CSV instead of Excel** → Faster loading
2. **Pre-clean data** → Better anomaly detection
3. **Run on same timezone** → Consistent time-series
4. **Batch similar companies** → Better trend analysis
5. **Use appropriate thresholds** → Relevant anomalies
6. **Cache results** → Don't recalculate unnecessarily

---

## 🎯 Real-World Use Cases

### Use Case 1: Credit Risk Assessment
```python
# Get distress scores for loan applicants
scores = scorer.calculate_risk_score(ratios)
for company, data in scores.items():
    if data['overall_score'] < 40:
        print(f"Deny credit: {company}")
```

### Use Case 2: Portfolio Monitoring
```python
# Monitor existing portfolio companies
anomalies = detector.detect_all_anomalies(ratios)
# Alert if anomalies increase
```

### Use Case 3: M&A Due Diligence
```python
# Analyze acquisition target
recommendations = consultant.generate_recommendations(ratios, scores)
# Review recommendations before acquisition
```

### Use Case 4: Internal Analysis
```python
# Monitor own company health
trends = analyzer.analyze_trends(ratios)
# Track improvement/decline over time
```

---

## 📊 Sample Output Interpretation

```
Risk Score: 75/100 (Stable) → ✅ Healthy
├── Liquidity: 85/100 ✅
│   └── Recommendation: Maintain current liquidity management
├── Solvency: 70/100 🟡
│   └── Recommendation: Consider debt restructuring
├── Profitability: 78/100 ✅
│   └── Recommendation: Focus on cost optimization
├── Efficiency: 65/100 🟡
│   └── Recommendation: Improve asset utilization
└── Growth: 60/100 🟡
    └── Recommendation: Invest in growth initiatives
```

---

## 🚀 Next Steps

1. **Explore Data:** Run with sample data first
2. **Customize Analysis:** Adjust weights/thresholds for your needs
3. **Integrate:** Add to your workflow/system
4. **Scale:** Process multiple datasets
5. **Automate:** Schedule regular analyses
6. **Share:** Export reports and recommendations

---

**Ready? Start with:** `python main.py -i data/sample_data.csv` 🎉

