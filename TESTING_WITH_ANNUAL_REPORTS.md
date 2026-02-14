# 📊 How to Test with Annual Report Data

## 🎯 Quick Overview

Your system accepts **financial data in CSV format**. If you have an annual report from any company, you can extract the key financial figures and create a CSV file to analyze it.

---

## 📋 Required Columns (Must Have)

Your CSV file needs these 15 columns:

| Column | Source | Example |
|--------|--------|---------|
| `company` | Company name | "Apple Inc" |
| `year` | Fiscal year | 2024 |
| `revenue` | Total Revenue | 383285000 |
| `net_income` | Net Income | 93736000 |
| `total_assets` | Total Assets | 352755000 |
| `current_assets` | Current Assets | 135405000 |
| `current_liabilities` | Current Liabilities | 123137000 |
| `total_debt` | Total Debt (Long + Short term) | 108949000 |
| `equity` | Total Shareholders' Equity | 62411000 |
| `inventory` | Inventory | 6331000 |
| `cogs` | Cost of Goods Sold | 214301000 |
| `operating_income` | Operating Income | 119437000 |
| `interest_expense` | Interest Expense | 3933000 |
| `accounts_receivable` | Accounts Receivable | 28184000 |
| `cash` | Cash & Cash Equivalents | 29941000 |

---

## 📖 Where to Find These Numbers

### From Annual Report (10-K / Financial Statements):

**Income Statement:**
- `revenue` → Line: "Total Revenue" or "Net Sales"
- `net_income` → Line: "Net Income" (bottom of P&L)
- `cogs` → Line: "Cost of Goods Sold" or "Cost of Revenue"
- `operating_income` → Line: "Operating Income"
- `interest_expense` → Line: "Interest Expense"

**Balance Sheet:**
- `total_assets` → Line: "Total Assets"
- `current_assets` → Line: "Total Current Assets"
- `current_liabilities` → Line: "Total Current Liabilities"
- `total_debt` → Line: "Total Debt" (or: Long-term Debt + Short-term Borrowings)
- `equity` → Line: "Total Shareholders' Equity"
- `inventory` → Line: "Inventory"
- `accounts_receivable` → Line: "Accounts Receivable"
- `cash` → Line: "Cash and Cash Equivalents"

---

## 🔧 Example: Creating CSV from Annual Report

### Step 1: Extract from Annual Report

Let's say you have **Apple Inc 2024 Annual Report** (10-K):

**From Income Statement:**
```
Total Revenue: $383,285 million
Net Income: $93,736 million
Cost of Revenue: $214,301 million
Operating Income: $119,437 million
Interest Expense: $3,933 million
```

**From Balance Sheet:**
```
Total Assets: $352,755 million
Current Assets: $135,405 million
Current Liabilities: $123,137 million
Total Debt: $108,949 million
Shareholders' Equity: $62,411 million
Inventory: $6,331 million
Accounts Receivable: $28,184 million
Cash: $29,941 million
```

### Step 2: Create CSV File

Create a file called `apple_2024.csv`:

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Apple Inc,2024,383285,93736,352755,135405,123137,108949,62411,6331,214301,119437,3933,28184,29941
```

### Step 3: Run Analysis

```bash
.venv/bin/python main.py -i apple_2024.csv
```

---

## 📊 Multi-Year Example

For better trend analysis, add **multiple years** of data:

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Apple Inc,2022,394328,99803,352755,135405,123137,108949,62411,6331,214301,119437,3933,28184,29941
Apple Inc,2023,383285,96995,352755,135405,123137,108949,62411,6331,214301,119437,3933,28184,29941
Apple Inc,2024,383285,93736,352755,135405,123137,108949,62411,6331,214301,119437,3933,28184,29941
```

**Benefits of multi-year:**
- ✅ Trend analysis works better
- ✅ More accurate anomaly detection
- ✅ Better risk scoring
- ✅ Clearer recommendations

---

## 🔢 Important Notes About Numbers

### Units Matter!
All numbers should be in the **same units**. Choose one:

**Option 1: Thousands** (Most Common)
```
revenue,net_income,total_assets
383285,93736,352755    ← All in thousands
```

**Option 2: Millions**
```
revenue,net_income,total_assets
383285000,93736000,352755000    ← All in actual dollars
```

**Option 3: Actual Dollars**
```
revenue,net_income,total_assets
383285000000,93736000000,352755000000    ← All in actual dollars
```

⚠️ **Important:** Mix any units = wrong results!

**Recommendation:** Use thousands to keep numbers manageable

---

## 📝 Step-by-Step: Your Company's Data

### 1. Get Annual Report
- Download 10-K from SEC.gov (US companies)
- Download Annual Report from company website
- Or use financial databases (Yahoo Finance, Bloomberg, etc.)

### 2. Extract Key Numbers

Create a text file with these values:

```
Company Name: [Your Company]
Year: 2024

FROM INCOME STATEMENT:
Revenue: 
Net Income:
Cost of Goods Sold:
Operating Income:
Interest Expense:

FROM BALANCE SHEET:
Total Assets:
Current Assets:
Current Liabilities:
Total Debt:
Total Equity:
Inventory:
Accounts Receivable:
Cash:
```

### 3. Create CSV File

Use a text editor or spreadsheet:

```
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Your Company Name,2024,[value],[value],...
```

### 4. Test

```bash
.venv/bin/python main.py -i your_company.csv
```

---

## 💡 Quick Template (Copy & Use)

### Save as `my_company.csv`:

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Company Name,2024,1000000,100000,2000000,500000,300000,800000,1200000,150000,600000,150000,50000,200000,150000
```

**Then run:**
```bash
.venv/bin/python main.py -i my_company.csv
```

---

## 🎯 Real-World Examples

### Example 1: Small Company (2024)

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
TechStartup Inc,2024,5000,250,20000,8000,4000,8000,12000,2000,2500,750,400,1500,2000
```

### Example 2: Mid-Size Company (Multi-Year)

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Manufacturing Co,2022,50000,5000,100000,30000,20000,40000,60000,8000,30000,7500,2000,8000,5000
Manufacturing Co,2023,55000,5500,110000,33000,22000,44000,66000,9000,33000,8250,2200,8800,5500
Manufacturing Co,2024,60000,6000,120000,36000,24000,48000,72000,10000,36000,9000,2400,9600,6000
```

### Example 3: Large Company (Multi-Year)

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Fortune500 Corp,2022,500000,50000,1000000,300000,200000,400000,600000,50000,300000,75000,20000,80000,50000
Fortune500 Corp,2023,550000,55000,1100000,330000,220000,440000,660000,55000,330000,82500,22000,88000,55000
Fortune500 Corp,2024,600000,60000,1200000,360000,240000,480000,720000,60000,360000,90000,24000,96000,60000
```

---

## 📊 What You'll Get After Running

When you run the analysis on your company data:

### 1. Financial Ratios (CSV)
- 40+ calculated ratios
- Liquidity, Solvency, Profitability, Efficiency, Growth

### 2. Risk Score (0-100)
- Overall financial health score
- Classification: Stable / Caution / Distress
- Category breakdown

### 3. Anomalies Detected
- Any unusual patterns
- Severity levels (Critical/High/Medium/Low)

### 4. Strategic Recommendations
- Immediate actions needed
- Short-term improvements
- Long-term strategies

### 5. Beautiful Charts
- Risk comparison
- Category breakdown
- Trend analysis
- Heatmaps

---

## 🔄 Multi-Company Comparison

Want to compare **multiple companies**? Just add more rows to the CSV:

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Tech Company A,2024,1000000,100000,2000000,500000,300000,800000,1200000,150000,600000,150000,50000,200000,150000
Tech Company B,2024,950000,95000,1900000,475000,285000,760000,1140000,140000,570000,142500,47500,190000,140000
Tech Company C,2024,1100000,110000,2200000,550000,320000,850000,1350000,160000,650000,165000,55000,220000,180000
```

**Run:**
```bash
.venv/bin/python main.py -i companies.csv
```

**You'll get:**
- Risk scores for all 3 companies
- Comparison charts
- Individual recommendations

---

## ⚠️ Important Tips

### Data Accuracy
✅ Use official financial statements (10-K, Annual Reports)  
❌ Avoid estimates or unofficial sources  
✅ Keep all numbers in same currency (USD, EUR, etc.)  
❌ Don't mix currencies without converting

### Consistency
✅ Use same fiscal year definitions for all companies  
✅ Use same units (thousands, millions, etc.) throughout  
❌ Don't change units between rows

### Completeness
✅ Fill in all 15 required columns  
❌ Don't leave blank cells - use 0 if data unavailable  
✅ Include multiple years for better analysis  
❌ Don't include only 1 year of data

---

## 🚀 Quick Start (Your Company)

### 1. Create file `my_company.csv`:
```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Your Company,2024,1000000,100000,2000000,500000,300000,800000,1200000,150000,600000,150000,50000,200000,150000
```

### 2. Run analysis:
```bash
cd /Users/adi/Documents/financial-distress-ews
.venv/bin/python main.py -i my_company.csv
```

### 3. Check results:
```bash
ls results/
open results/charts/risk_comparison.png
```

---

## 📖 Additional Resources

### Finding Financial Data Online:
- **SEC.gov** (US companies) - Search for 10-K filings
- **Yahoo Finance** - Summary financials available
- **Company Website** - Download official annual reports
- **Google Finance** - Quick financial snapshots
- **Financial Databases** - Bloomberg, CapitalIQ, etc.

### CSV Format Help:
- Use any spreadsheet program (Excel, Google Sheets)
- Save as CSV format
- Or use text editor with proper formatting

---

## ✅ Checklist Before Running

- [ ] Have annual report or financial statements
- [ ] Extracted all 15 required numbers
- [ ] All numbers in same units (thousands/millions/dollars)
- [ ] All numbers in same currency
- [ ] Created CSV file with proper format
- [ ] File has `.csv` extension
- [ ] Numbers only (no $ signs or commas)
- [ ] Company name filled in
- [ ] Year filled in correctly

---

## 🎉 Ready to Test?

Once your CSV is ready:

```bash
.venv/bin/python main.py -i your_file.csv
```

You'll get complete financial analysis in ~2 seconds! 🚀

---

*Guide Created: February 13, 2026*
*Status: Ready to test with real company data*
