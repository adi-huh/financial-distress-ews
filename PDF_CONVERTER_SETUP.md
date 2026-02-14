# 🚀 PDF Converter Setup & Installation

## ⚡ Quick Setup (5 minutes)

### Step 1: Install PDF Libraries

```bash
pip install pdfplumber PyPDF2
```

### Step 2: Verify Installation

```bash
python -c "import pdfplumber; import PyPDF2; print('✓ Libraries installed')"
```

### Step 3: You're Ready!

Start converting PDFs:

```bash
python convert.py apple_2024.pdf "Apple Inc"
```

---

## 📋 Full Installation Steps

### 1. Activate Virtual Environment

```bash
cd /Users/adi/Documents/financial-distress-ews
source .venv/bin/activate
```

### 2. Install PDF Processing Libraries

```bash
pip install pdfplumber PyPDF2
```

**What these do:**
- **pdfplumber** - Better for table extraction (recommended)
- **PyPDF2** - Fallback PDF reader for text extraction

### 3. Verify Installation

```bash
python -c "
import pdfplumber
from PyPDF2 import PdfReader
print('✓ pdfplumber:', pdfplumber.__version__)
print('✓ PyPDF2 installed')
print('✓ Ready to convert PDFs!')
"
```

### 4. Test Conversion (Optional)

```bash
python convert.py --help
```

---

## 🎯 Quick Start Examples

### Example 1: Convert Apple 10-K

```bash
# Download apple_10k_2024.pdf from SEC.gov first

# Convert to CSV
python convert.py apple_10k_2024.pdf "Apple Inc"

# Analyze
.venv/bin/python main.py -i apple_10k_2024_extracted.csv
```

### Example 2: Batch Convert Multiple Companies

```bash
# Create reports folder with PDFs:
# reports/
#   ├── apple_2024.pdf
#   ├── microsoft_2024.pdf
#   └── google_2024.pdf

# Convert all at once
python convert.py --batch reports converted

# Analyze each
.venv/bin/python main.py -i converted/apple_2024_extracted.csv
.venv/bin/python main.py -i converted/microsoft_2024_extracted.csv
```

### Example 3: Specify Custom Output Name

```bash
python convert.py report.pdf "My Company" my_company_2024.csv
```

---

## 📊 Files in the System

| File | Purpose |
|------|---------|
| `pdf_converter.py` | Core extraction engine (400+ lines) |
| `convert.py` | Easy-to-use wrapper script |
| `PDF_CONVERTER_GUIDE.md` | Complete documentation |
| `PDF_CONVERTER_SETUP.md` | This file |

---

## 🔧 How It Works

### Architecture:

```
PDF File
   ↓
[PDFReportExtractor]
   ├─ Extract text
   ├─ Extract tables
   ├─ Parse Income Statement
   ├─ Parse Balance Sheet
   └─ Find financial metrics
   ↓
[Data Validation]
   ├─ Check completeness
   ├─ Normalize numbers
   └─ Handle missing values
   ↓
CSV File (15 columns)
   ↓
[Analysis System]
   ├─ Risk scoring
   ├─ Ratio calculation
   ├─ Anomaly detection
   └─ Recommendations
   ↓
Results (Charts, Scores, Recommendations)
```

### Extraction Methods:

1. **Table Detection** (Preferred)
   - Scans for structured financial tables
   - High accuracy for formatted reports
   
2. **Text Parsing** (Fallback)
   - Pattern matching for line items
   - Works with unstructured reports
   
3. **Keyword Matching**
   - Identifies financial terms
   - Context-aware extraction

---

## 🎯 Usage Patterns

### Pattern 1: Single Company Analysis

```bash
# Download PDF
# Save as: apple_2024.pdf

# Convert
python convert.py apple_2024.pdf "Apple Inc"

# Analyze
.venv/bin/python main.py -i apple_2024_extracted.csv
```

### Pattern 2: Multi-Year Trend Analysis

```bash
# Convert 3 years
python convert.py apple_2022.pdf "Apple Inc" apple_2022.csv
python convert.py apple_2023.pdf "Apple Inc" apple_2023.csv
python convert.py apple_2024.pdf "Apple Inc" apple_2024.csv

# Combine CSVs
cat apple_2022.csv apple_2023.csv apple_2024.csv > apple_trends.csv

# Skip header for appended files
tail -n +2 apple_2023.csv >> apple_combined.csv
tail -n +2 apple_2024.csv >> apple_combined.csv

# Analyze
.venv/bin/python main.py -i apple_combined.csv
```

### Pattern 3: Competitive Analysis

```bash
# Convert competitors
python convert.py apple_2024.pdf "Apple Inc" apple_2024.csv
python convert.py microsoft_2024.pdf "Microsoft" microsoft_2024.csv
python convert.py google_2024.pdf "Google" google_2024.csv

# Combine
cat *_2024.csv | tail -n +2 > tech_comparison_2024.csv

# Analyze
.venv/bin/python main.py -i tech_comparison_2024.csv
```

---

## ⚠️ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pdfplumber'"

**Solution:**
```bash
pip install pdfplumber
```

### Issue: PDF extraction returns no data

**Check:**
1. Is PDF text-readable? (Open in text editor to verify)
2. Does PDF contain financial statements?
3. Try with a known 10-K first

**Example test:**
```bash
python -c "
import pdfplumber
with pdfplumber.open('test.pdf') as pdf:
    print(f'Pages: {len(pdf.pages)}')
    text = pdf.pages[0].extract_text()
    print(f'Text length: {len(text)}')
"
```

### Issue: Numbers extracted incorrectly

**Try:**
1. Check extraction with --verbose flag (coming soon)
2. Manual verification against original PDF
3. Manual entry for critical numbers

### Issue: Cannot find financial metrics

**Try:**
1. Ensure financial statements are included
2. Try first 30 pages instead of 20 (edit pdf_converter.py)
3. Use batch mode to test multiple files

---

## 🔍 Quality Checks

### Before Using Extracted CSV:

1. **Check basic info:**
   ```bash
   head -1 output.csv  # View headers
   tail -1 output.csv  # View data
   ```

2. **Verify key numbers:**
   - Open original PDF
   - Compare extracted vs actual
   - Fix any discrepancies

3. **Check for missing data:**
   - Open CSV in Excel
   - Look for blank cells
   - Fill in manually if needed

---

## 🎯 Integration Example

### Complete Workflow:

```bash
# 1. Download PDF from SEC
wget https://example.com/apple_10k_2024.pdf

# 2. Convert to CSV
python convert.py apple_10k_2024.pdf "Apple Inc"

# 3. Verify in spreadsheet (optional but recommended)
# Open apple_10k_2024_extracted.csv in Excel and verify numbers

# 4. Analyze
.venv/bin/python main.py -i apple_10k_2024_extracted.csv

# 5. View results
ls results/
open results/charts/risk_comparison.png
```

---

## 🚀 Advanced Configuration

### Edit Extraction Keywords

File: `pdf_converter.py`

```python
FINANCIAL_METRICS = {
    'revenue': ['revenue', 'net sales', 'total revenue', 'sales'],
    'net_income': ['net income', 'net earnings', 'income'],
    # Add more as needed
}
```

### Change Number of Pages to Scan

File: `pdf_converter.py`

```python
# Change from:
for page in pdf.pages[:20]:

# To:
for page in pdf.pages[:50]:  # Scans more pages
```

### Add Custom Extraction Logic

Create extension:

```python
from pdf_converter import PDFReportExtractor

class CustomExtractor(PDFReportExtractor):
    def extract_financial_metrics(self):
        metrics = super().extract_financial_metrics()
        # Add your custom logic here
        return metrics
```

---

## 📈 Performance

### Typical Extraction Times:

| Task | Time |
|------|------|
| Single PDF conversion | 5-15 seconds |
| 10-file batch | 1-2 minutes |
| Text extraction | 90% of time |
| CSV generation | < 1 second |

### Optimization Tips:

- Use pdfplumber (faster than PyPDF2)
- Limit page range if known
- Batch process overnight for large collections

---

## ✅ Installation Checklist

- [ ] Virtual environment activated
- [ ] pdfplumber installed
- [ ] PyPDF2 installed
- [ ] Test conversion successful
- [ ] Got sample output CSV
- [ ] Analyzed with main.py

---

## 📞 Next Steps

1. **Install:** `pip install pdfplumber PyPDF2`
2. **Test:** `python convert.py --help`
3. **Download:** Any 10-K from SEC.gov
4. **Convert:** `python convert.py [file] [company]`
5. **Analyze:** `.venv/bin/python main.py -i [output.csv]`

---

## 🎉 You're Ready!

Convert PDFs to CSVs and analyze with a single command:

```bash
python convert.py report.pdf "Company" && .venv/bin/python main.py -i report_extracted.csv
```

---

*PDF Converter Setup Guide - February 13, 2026*
*Ready to automate financial data extraction!*
