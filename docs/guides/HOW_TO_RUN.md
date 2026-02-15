# 🚀 How to Run Locally

## Quick Start (3 Steps)

### 1️⃣ Activate Virtual Environment
```bash
cd /Users/adi/Documents/financial-distress-ews
source .venv/bin/activate
```

### 2️⃣ Run Analysis with Sample Data
```bash
python main.py -i sample_data.csv
```

### 3️⃣ View Results
- Check `results/` folder for CSV files
- Check `results/charts/` for visualizations

---

## 📋 Running Options

### Option A: Basic Analysis (Recommended First)
```bash
python main.py -i sample_data.csv
```
**What it does:**
- Loads sample data
- Calculates all financial ratios
- Detects anomalies
- Computes risk scores
- Generates recommendations
- Creates visualizations
- Exports to CSV

**Output:**
- `results/financial_ratios.csv`
- `results/risk_scores.csv`
- `results/anomalies.csv`
- `results/recommendations.csv`
- `results/charts/*.png`

---

### Option B: Custom Data File
```bash
python main.py -i path/to/your/file.csv
```

**Supported formats:**
- CSV (.csv)
- Excel (.xlsx, .xls)

**Required columns:**
- Date
- Company
- Revenue
- Net Income
- Total Assets
- Total Liabilities
- Inventory
- Accounts Receivable
- (See sample_data.csv for all required columns)

---

### Option C: Analyze Specific Company
```bash
python main.py -i sample_data.csv -c "TechCorp"
```
**Output:** Results for TechCorp only

---

### Option D: Change Output Directory
```bash
python main.py -i sample_data.csv -o my_results/
```
**Output:** Saves to `my_results/` instead of `results/`

---

### Option E: Different Export Format
```bash
# Export as Excel
python main.py -i sample_data.csv --export-format excel

# Export as JSON
python main.py -i sample_data.csv --export-format json
```

---

### Option F: Change Anomaly Detection Method
```bash
# Using Isolation Forest (advanced ML)
python main.py -i sample_data.csv --anomaly-method isolation_forest

# Using Z-Score (statistical)
python main.py -i sample_data.csv --anomaly-method zscore
```

---

### Option G: Verbose Output
```bash
python main.py -i sample_data.csv --verbose
```
**Shows:** Detailed logging of each step

---

## 🎯 Complete Example Commands

### Run Everything
```bash
cd /Users/adi/Documents/financial-distress-ews
source .venv/bin/activate
python main.py -i sample_data.csv --verbose
```

### Run with Custom Settings
```bash
python main.py \
  -i sample_data.csv \
  -o my_analysis/ \
  -c "RetailCo" \
  --export-format excel \
  --anomaly-method isolation_forest \
  --verbose
```

### Run Tests
```bash
python -m pytest tests.py -v
```

### Run Dashboard (Streamlit)
```bash
streamlit run app.py
```

---

## 📊 Output Files Explained

### 1. Financial Ratios (`financial_ratios.csv`)
- 40+ calculated ratios
- Liquidity, Solvency, Profitability, Efficiency, Growth
- Per company, per year

### 2. Risk Scores (`risk_scores.csv`)
- Overall risk score (0-100)
- Category scores
- Classification (Stable/Caution/Distress)

### 3. Anomalies (`anomalies.csv`)
- Detected anomalies
- Severity levels
- Affected ratios

### 4. Recommendations (`recommendations.csv`)
- Strategic recommendations
- By company and category
- Action items

### 5. Visualizations (`charts/`)
- Risk comparison charts
- Category breakdowns
- Trend analysis
- Heatmaps

---

## 🛠️ Troubleshooting

### Issue: "command not found: python"
**Solution:**
```bash
# Use the virtual environment Python
.venv/bin/python main.py -i sample_data.csv
```

### Issue: "ModuleNotFoundError"
**Solution:**
```bash
# Activate virtual environment first
source .venv/bin/activate
python main.py -i sample_data.csv
```

### Issue: "File not found"
**Solution:**
```bash
# Make sure file path is correct
# Use absolute path if needed
python main.py -i /full/path/to/file.csv
```

### Issue: "results/ directory not found"
**Solution:**
```bash
# The script creates it automatically
# If not, create manually:
mkdir -p results/charts
```

### Issue: Dependencies missing
**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

---

## 📈 What Each Step Does

When you run `python main.py -i sample_data.csv`:

1. **Load Data** 📂
   - Reads CSV file
   - Validates schema
   - Checks data integrity

2. **Clean Data** 🧹
   - Handles missing values
   - Removes outliers
   - Normalizes data

3. **Calculate Ratios** 📊
   - Liquidity ratios (5)
   - Solvency ratios (5)
   - Profitability ratios (5)
   - Efficiency ratios (5)
   - Growth ratios (3+)

4. **Time-Series Analysis** 📈
   - Trends over time
   - Moving averages
   - Volatility
   - Correlations

5. **Detect Anomalies** 🚨
   - Z-score method OR
   - Isolation Forest ML
   - Severity classification

6. **Calculate Risk Score** ⚠️
   - Composite score (0-100)
   - Category breakdown
   - Classification tier

7. **Generate Recommendations** 💡
   - Strategic insights
   - Action items
   - By category

8. **Create Visualizations** 📸
   - Charts and graphs
   - Save as PNG

---

## 💻 Real Example

```bash
# Step 1: Navigate to project
cd /Users/adi/Documents/financial-distress-ews

# Step 2: Activate environment
source .venv/bin/activate

# Step 3: Run analysis
python main.py -i sample_data.csv

# Expected output:
# 📂 Loaded 34 records (6 companies)
# 🧹 Data cleaned: 34 records retained
# 📊 Calculated 40 financial ratios
# 📈 Trend analysis completed
# 🚨 Detected 9 anomalies
# ⚠️ Risk scores computed (all 6 companies)
# 💡 Generated 6 recommendations
# 📸 Created 4+ visualization charts
# ✅ Analysis completed successfully

# Step 4: Check results
ls -la results/
# financial_ratios.csv
# risk_scores.csv
# anomalies.csv
# recommendations.csv
# charts/

# Step 5: View outputs
cat results/risk_scores.csv
open results/charts/  # macOS
```

---

## 🌐 Dashboard Version

### Run Streamlit Dashboard
```bash
streamlit run app.py
```

**Features:**
- File upload
- Interactive analysis
- Real-time visualizations
- Download results
- Responsive design

**Access at:** http://localhost:8501

---

## ⚡ Quick Reference

| Task | Command |
|------|---------|
| Basic run | `python main.py -i sample_data.csv` |
| Verbose | `python main.py -i sample_data.csv --verbose` |
| Custom output | `python main.py -i sample_data.csv -o my_results/` |
| Specific company | `python main.py -i sample_data.csv -c "TechCorp"` |
| Excel export | `python main.py -i sample_data.csv --export-format excel` |
| ML anomaly detection | `python main.py -i sample_data.csv --anomaly-method isolation_forest` |
| Run tests | `python -m pytest tests.py -v` |
| Dashboard | `streamlit run app.py` |
| Get help | `python main.py --help` |

---

## 🎯 Next Steps

1. ✅ Run with sample data first
2. ✅ Check outputs in `results/`
3. ✅ Review visualizations
4. ✅ Try with your own data
5. ✅ Explore dashboard with `streamlit run app.py`

---

*Created: February 13, 2026*
*Status: Ready to use locally*
