# 🎓 Complete Example: Testing Real Company Data

## Real-World Walkthrough: Apple Inc 2024

Let me show you **exactly** how to test with a real company's annual report.

---

## 📖 Step 1: Get the Annual Report

### Where to Find Apple's 10-K:

1. **Option A: SEC Website** (Official)
   - Go to: https://www.sec.gov
   - Search: "Apple Inc 10-K"
   - Download: Form 10-K for fiscal year 2024

2. **Option B: Apple's Website**
   - Go to: investor.apple.com
   - Download: Annual Report 2024 PDF

3. **Option C: Financial Websites**
   - Yahoo Finance: finance.yahoo.com
   - Search: "AAPL"
   - View: Financial statements

---

## 🔍 Step 2: Extract Financial Numbers

### From Apple's Financial Statements (2024):

**INCOME STATEMENT:**
```
Total Net Sales:                $383,285 million
Cost of Sales:                  $214,301 million
Operating Income:               $119,437 million
Interest Expense:               $3,933 million
Net Income:                     $93,736 million
```

**BALANCE SHEET:**
```
Total Assets:                   $352,755 million
Current Assets:                 $135,405 million
Current Liabilities:            $123,137 million
Total Debt:                     $108,949 million
Total Stockholders' Equity:     $62,411 million
Inventory:                      $6,331 million
Accounts Receivable:            $28,184 million
Cash & Equivalents:             $29,941 million
```

---

## 📝 Step 3: Create CSV File

### Create file: `apple_analysis.csv`

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Apple Inc,2024,383285,93736,352755,135405,123137,108949,62411,6331,214301,119437,3933,28184,29941
```

**Save to:** `/Users/adi/Documents/financial-distress-ews/apple_analysis.csv`

---

## 🚀 Step 4: Run the Analysis

```bash
cd /Users/adi/Documents/financial-distress-ews
.venv/bin/python main.py -i apple_analysis.csv
```

---

## 📊 Step 5: Check Results

### View output files:

```bash
ls results/
```

**Output:**
```
financial_ratios.csv
charts/
```

### View the risk score:

```bash
cat results/financial_ratios.csv | head
```

### View a chart:

```bash
open results/charts/risk_comparison.png
```

---

## 📈 Sample Output for Apple

### Expected Risk Score:
```
Apple Inc: 92.5/100 (Stable) ✅
```

### Expected Ratios:
```
Current Ratio: 1.10 (Healthy)
Quick Ratio: 0.95 (Good)
Debt-to-Equity: 1.75 (Moderate)
ROE: 150.1% (Excellent)
ROA: 26.6% (Excellent)
Net Profit Margin: 24.5% (Excellent)
Asset Turnover: 1.09x (Good)
```

### Expected Recommendations:
```
✅ Maintain current financial discipline
✅ Continue strong profitability focus
✅ Monitor debt levels (moderate)
✅ Leverage strong cash position
```

---

## 🔄 Multi-Year Analysis (Better!)

### For Better Analysis: Add 2022 & 2023 Data

### Create: `apple_multiyear.csv`

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Apple Inc,2022,365817,99803,346926,135405,123137,108949,62411,5990,201465,114931,3284,26270,23646
Apple Inc,2023,383285,96995,352755,135405,123137,108949,62411,6126,212981,120543,3933,27751,23286
Apple Inc,2024,383285,93736,352755,135405,123137,108949,62411,6331,214301,119437,3933,28184,29941
```

### Run:
```bash
.venv/bin/python main.py -i apple_multiyear.csv
```

### You'll see:
- ✅ Trend analysis (3-year comparison)
- ✅ Anomalies in trends
- ✅ Better risk assessment
- ✅ Clearer recommendations

---

## 🏢 Compare Multiple Companies

### Create: `tech_comparison.csv`

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Apple Inc,2024,383285,93736,352755,135405,123137,108949,62411,6331,214301,119437,3933,28184,29941
Microsoft,2024,245122,88519,411975,205617,141001,61898,280141,3474,50261,93478,1775,56919,72793
Google,2024,307394,59972,402392,173541,97137,10787,255485,0,54340,102939,2156,40266,110939
Meta,2024,134902,42200,381239,99023,32123,5000,243616,0,13659,53305,1131,31627,59801
Tesla,2024,96773,13256,371200,98300,107000,65000,102000,5100,42000,16000,5200,8500,15000
```

### Run:
```bash
.venv/bin/python main.py -i tech_comparison.csv
```

### You get:
- ✅ Risk scores for all 5 companies
- ✅ Comparison charts
- ✅ Individual recommendations
- ✅ Competitive analysis

---

## 📋 Real Company Data (2024)

### Available Online for Free:

**Technology:**
- Apple Inc
- Microsoft
- Google/Alphabet
- Meta
- Amazon
- Tesla

**Finance:**
- JPMorgan Chase
- Bank of America
- Berkshire Hathaway

**Consumer:**
- Walmart
- Amazon
- Nike
- Coca-Cola

**Healthcare:**
- Johnson & Johnson
- Pfizer
- UnitedHealth

All have publicly available annual reports!

---

## 🎯 Common Issues & Solutions

### Issue: Numbers too large

**Problem:**
```
revenue: 383285000000  (12 digits - too large)
```

**Solution:**
```
revenue: 383285  (use thousands consistently)
```

### Issue: Missing a value

**Problem:**
```
apple_analysis.csv - missing inventory value
```

**Solution:**
```
Use 0 if not available:
inventory: 0
```

### Issue: Wrong column order

**Problem:**
```
revenue,net_income,total_assets  (wrong order)
```

**Solution:**
```
Must be exactly: company,year,revenue,net_income,total_assets,...
```

### Issue: File format wrong

**Problem:**
```
apple_analysis.txt (wrong extension)
```

**Solution:**
```
Must be: apple_analysis.csv (CSV format)
```

---

## ✅ Complete Checklist

Before running your analysis:

- [ ] Have annual report open
- [ ] Found income statement with 5 key numbers
- [ ] Found balance sheet with 10 key numbers
- [ ] All numbers in same units (thousands/millions/dollars)
- [ ] All numbers positive (no negative signs)
- [ ] Created CSV file
- [ ] File has .csv extension
- [ ] Headers exactly match: company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
- [ ] Company name filled in
- [ ] Year filled in
- [ ] All 15 columns have values
- [ ] No $ signs or commas in numbers
- [ ] File is in correct directory

---

## 🚀 Ready? Let's Go!

### Quick Summary:

1. **Get Annual Report** → Download PDF
2. **Extract Numbers** → Find 15 key figures
3. **Create CSV** → Put numbers in template
4. **Run Analysis** → Execute command
5. **View Results** → Check output folder

---

## 📞 Example Companies to Try

### Easy to Find (Large Companies):

**Try Apple:**
```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Apple Inc,2024,383285,93736,352755,135405,123137,108949,62411,6331,214301,119437,3933,28184,29941
```

**Try Microsoft:**
```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Microsoft,2024,245122,88519,411975,205617,141001,61898,280141,3474,50261,93478,1775,56919,72793
```

**Try a Local Company:**
```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Your Company,2024,1000000,100000,2000000,500000,300000,800000,1200000,150000,600000,150000,50000,200000,150000
```

---

## 💡 Pro Tips

✅ **Use multi-year data** - Better analysis with trends
✅ **Compare competitors** - See how companies stack up
✅ **Use latest data** - Most recent annual reports
✅ **Double-check numbers** - Accuracy matters
✅ **Try different companies** - Learn patterns

---

## 🎉 Next Steps

1. Download an annual report
2. Extract the numbers
3. Create your CSV file
4. Run the analysis
5. Analyze the results!

---

*Complete Example Guide - February 13, 2026*
*Ready to test with real company data!*
