# 📋 Simple Example - Test Your Company

## Quickest Way to Test

### Step 1: Create a CSV file with your data

**Filename:** `my_company.csv`

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Apple Inc,2024,383285,93736,352755,135405,123137,108949,62411,6331,214301,119437,3933,28184,29941
```

### Step 2: Run the analysis

```bash
cd /Users/adi/Documents/financial-distress-ews
.venv/bin/python main.py -i my_company.csv
```

### Step 3: View results

```bash
ls results/
open results/charts/risk_comparison.png
```

---

## 🎯 Where to Get Numbers (Easy Way)

### From Annual Report PDF:

**Income Statement Page:**
1. Find "Revenue" or "Total Sales" → Put in `revenue` column
2. Find "Cost of Goods Sold" → Put in `cogs` column
3. Find "Operating Income" → Put in `operating_income` column
4. Find "Interest Expense" → Put in `interest_expense` column
5. Find "Net Income" (bottom line) → Put in `net_income` column

**Balance Sheet Page:**
1. Find "Total Assets" → Put in `total_assets` column
2. Find "Current Assets" → Put in `current_assets` column
3. Find "Current Liabilities" → Put in `current_liabilities` column
4. Find "Total Debt" → Put in `total_debt` column (or add Long-term + Short-term debt)
5. Find "Total Stockholders' Equity" → Put in `equity` column
6. Find "Inventory" → Put in `inventory` column
7. Find "Accounts Receivable" → Put in `accounts_receivable` column
8. Find "Cash" or "Cash & Equivalents" → Put in `cash` column

**Company Info:**
1. Company name → Put in `company` column
2. Fiscal year → Put in `year` column

---

## 💡 Real Example (Copy & Use)

### Microsoft 2024:

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Microsoft,2024,245122,88519,411975,205617,141001,61898,280141,3474,50261,93478,1775,56919,72793
```

**Run:**
```bash
.venv/bin/python main.py -i microsoft.csv
```

---

## 📊 Compare Multiple Companies

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Microsoft,2024,245122,88519,411975,205617,141001,61898,280141,3474,50261,93478,1775,56919,72793
Apple Inc,2024,383285,93736,352755,135405,123137,108949,62411,6331,214301,119437,3933,28184,29941
Google,2024,307394,59972,402392,173541,97137,10787,255485,0,54340,102939,2156,40266,110939
```

**Run:**
```bash
.venv/bin/python main.py -i tech_companies.csv
```

**Get:**
- Risk scores for all 3
- Comparison charts
- Individual recommendations

---

## 📈 Multi-Year Analysis (Better!)

```csv
company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
Tesla,2022,81462,12587,298695,84997,97000,60000,90000,4410,36659,13686,4735,7686,13256
Tesla,2023,81462,12587,298695,84997,97000,60000,90000,4410,36659,13686,4735,7686,13256
Tesla,2024,96773,13256,371200,98300,107000,65000,102000,5100,42000,16000,5200,8500,15000
```

**Benefits:**
- Shows trend over time
- Better anomaly detection
- More accurate risk scoring
- Clearer recommendations

---

## ⚠️ Important: Numbers Format

### ✅ DO THIS (Numbers only):
```
revenue,net_income
245122,88519
383285,93736
```

### ❌ DON'T DO THIS (With $ or commas):
```
revenue,net_income
$245,122,000,$88,519,000
383,285,000,93,736,000
```

### 📝 Units don't matter as long as consistent:
```
Thousands:     revenue,net_income
              245122,88519

Millions:      revenue,net_income
              245122,88519

Dollars:       revenue,net_income
              245122000000,88519000000
```

Just pick ONE and stick with it for all numbers!

---

## 🚀 Ready? Let's Go!

1. **Open a text editor** (Notepad, VS Code, Google Sheets)
2. **Copy this template:**
   ```csv
   company,year,revenue,net_income,total_assets,current_assets,current_liabilities,total_debt,equity,inventory,cogs,operating_income,interest_expense,accounts_receivable,cash
   Your Company Name,2024,1000000,100000,2000000,500000,300000,800000,1200000,150000,600000,150000,50000,200000,150000
   ```
3. **Fill in your numbers** from annual report
4. **Save as** `my_company.csv`
5. **Run:**
   ```bash
   cd /Users/adi/Documents/financial-distress-ews
   .venv/bin/python main.py -i my_company.csv
   ```

---

## 📞 Need Help?

Check: **TESTING_WITH_ANNUAL_REPORTS.md** for detailed guide

---

*Easy Quick Start Guide*
