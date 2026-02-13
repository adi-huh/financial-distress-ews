# ⚡ QUICK START - Commands Reference

## 🎯 Get Started in 60 Seconds

### 1. Setup (One-time)
```bash
cd financial-distress-ews
python -m venv venv
source venv/bin/activate          # Mac/Linux
# OR
venv\Scripts\activate              # Windows
pip install -r requirements.txt
```

### 2. Run Dashboard
```bash
streamlit run src/dashboard/app.py
```
🌐 Opens automatically at http://localhost:8501

### 3. Upload Sample Data
- Click "Browse files" in sidebar
- Select `data/raw/sample_data.csv`
- View results instantly!

---

## 📋 Command Cheat Sheet

### Run Command-Line Analysis
```bash
python main.py --input data/raw/sample_data.csv --output results/
```

### Run with Specific Company
```bash
python main.py --input data/raw/sample_data.csv --company TechCorp
```

### Export to Excel
```bash
python main.py --input data/raw/sample_data.csv --export-format excel
```

### Verbose Output
```bash
python main.py --input data/raw/sample_data.csv --verbose
```

---

## 🧪 Testing Commands

### Test Individual Modules
```python
# Test data loading
python -c "from src.data_ingestion.loader import DataLoader; print('✓ Loader works')"

# Test preprocessing
python -c "from src.preprocessing.cleaner import DataCleaner; print('✓ Cleaner works')"

# Test ratio engine
python -c "from src.ratio_engine.ratios import FinancialRatioEngine; print('✓ Ratios work')"
```

### Run Tests (when created)
```bash
pytest tests/
pytest tests/test_ratios.py
pytest --cov=src
```

---

## 🔧 Useful Python Snippets

### Quick Analysis Script
```python
from src.data_ingestion.loader import DataLoader
from src.preprocessing.cleaner import DataCleaner
from src.ratio_engine.ratios import FinancialRatioEngine
from src.risk_score.score import RiskScoreEngine

# Load and analyze
loader = DataLoader()
data = loader.load_file("data/raw/sample_data.csv")

cleaner = DataCleaner()
clean_data = cleaner.clean(data)

ratio_engine = FinancialRatioEngine()
ratios = ratio_engine.calculate_all_ratios(clean_data)

risk_engine = RiskScoreEngine()
scores = risk_engine.calculate_risk_score(ratios)

print(scores)
```

### Fetch Live Data from Yahoo Finance
```python
import yfinance as yf
import pandas as pd

# Get Apple financials
ticker = yf.Ticker("AAPL")
income = ticker.financials
balance = ticker.balance_sheet

# Extract key metrics
data = {
    'company': 'AAPL',
    'year': 2023,
    'revenue': income.loc['Total Revenue'][0],
    'net_income': income.loc['Net Income'][0],
    'total_assets': balance.loc['Total Assets'][0],
    # ... add more fields
}

# Save to CSV
df = pd.DataFrame([data])
df.to_csv('apple_data.csv', index=False)
```

---

## 📦 Git Commands

### Initialize Repository
```bash
git init
git add .
git commit -m "Initial commit: Financial Distress EWS"
```

### Connect to GitHub
```bash
git remote add origin https://github.com/yourusername/financial-distress-ews.git
git branch -M main
git push -u origin main
```

### Create Feature Branch
```bash
git checkout -b feature/new-feature
# Make changes
git add .
git commit -m "Add new feature"
git push origin feature/new-feature
```

---

## 🎨 Customization Tips

### Change Risk Thresholds
Edit `src/risk_score/score.py`:
```python
DISTRESS_THRESHOLD = 35  # Change from 40
CAUTION_THRESHOLD = 65   # Change from 70
```

### Change Ratio Weights
```python
engine = RiskScoreEngine(weights={
    'liquidity': 0.30,      # Increase from 0.25
    'solvency': 0.30,
    'profitability': 0.20,  # Decrease from 0.25
    'efficiency': 0.15,
    'growth': 0.05
})
```

### Add Custom Ratio
Edit `src/ratio_engine/ratios.py`:
```python
def _calculate_custom_ratio(self, data: pd.DataFrame):
    """Calculate your custom ratio."""
    if 'field1' in data.columns and 'field2' in data.columns:
        data['custom_ratio'] = data['field1'] / data['field2']
    return data
```

---

## 📊 Sample Data Formats

### Minimal CSV
```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity
CompanyA,2023,1000000,100000,2000000,500000,300000,800000,1200000
```

### Full CSV (Recommended)
```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
CompanyA,2023,1000000,100000,2000000,500000,300000,800000,1200000,150000,600000,150000,50000,200000,150000
```

---

## 🚨 Error Solutions

### ModuleNotFoundError
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
# OR
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

### Streamlit Port Already in Use
```bash
streamlit run src/dashboard/app.py --server.port 8502
```

### Permission Denied (Mac/Linux)
```bash
chmod +x main.py
```

### Missing Dependencies
```bash
pip install -r requirements.txt --force-reinstall
```

---

## 📈 Performance Tips

### Speed Up Analysis
```python
# Use smaller date ranges
filtered = data[data['year'] >= 2020]

# Analyze one company at a time
single = data[data['company'] == 'TechCorp']
```

### Reduce Memory Usage
```python
# Use specific dtypes
data = pd.read_csv('file.csv', dtype={'year': 'int16'})
```

---

## 🎯 What to Do Next

1. ✅ **Run the dashboard** - See it in action
2. ✅ **Upload sample data** - Test with provided data
3. ✅ **Try your own data** - Analyze real companies
4. 📝 **Read documentation** - Understand the system
5. 🔧 **Customize** - Make it your own
6. 📤 **Deploy** - Share with others
7. 🌟 **Star on GitHub** - Support the project

---

## 📞 Help & Resources

- **Documentation**: README.md
- **Setup Guide**: SETUP_GUIDE.md
- **Complete Guide**: PROJECT_COMPLETE.md
- **Contributing**: CONTRIBUTING.md

---

**Status**: ✅ READY TO USE

**Time to First Result**: < 60 seconds

**Difficulty**: Easy

**Requirements**: Python 3.8+

---

Last Updated: February 2024
