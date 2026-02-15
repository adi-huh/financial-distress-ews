"""
PDF Annual Report to CSV Converter
Extracts financial data from PDF annual reports and converts to CSV format.
Supports 10-K filings, annual reports, and financial statements.
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("pdfplumber not installed. Install with: pip install pdfplumber")

try:
    from PyPDF2 import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")


class PDFReportExtractor:
    """Extract financial data from PDF annual reports."""
    
    # Financial statement keywords
    INCOME_STATEMENT_KEYWORDS = [
        'income statement', 'statement of earnings', 'statement of income',
        'consolidated statements of earnings', 'operations', 'revenues', 'sales'
    ]
    
    BALANCE_SHEET_KEYWORDS = [
        'balance sheet', 'statement of financial position', 'consolidated balance sheet',
        'assets', 'liabilities', 'stockholders equity'
    ]
    
    FINANCIAL_METRICS = {
        'revenue': ['revenue', 'net sales', 'total revenue', 'sales', 'operating revenue'],
        'net_income': ['net income', 'net earnings', 'income from operations', 'net loss', 'bottom line'],
        'total_assets': ['total assets', 'total current assets', 'assets'],
        'current_assets': ['current assets', 'total current assets'],
        'current_liabilities': ['current liabilities', 'total current liabilities'],
        'total_debt': ['total debt', 'long-term debt', 'short-term debt', 'borrowings', 'total liabilities'],
        'equity': ['stockholders equity', 'shareholders equity', 'total equity', 'total stockholders'],
        'inventory': ['inventory', 'inventories'],
        'cogs': ['cost of goods sold', 'cost of sales', 'cost of revenue', 'cogs'],
        'operating_income': ['operating income', 'income from operations', 'operating earnings'],
        'interest_expense': ['interest expense', 'interest paid'],
        'accounts_receivable': ['accounts receivable', 'receivables'],
        'cash': ['cash', 'cash and equivalents', 'cash and cash equivalents']
    }
    
    def __init__(self, pdf_path: str):
        """Initialize PDF extractor."""
        self.pdf_path = Path(pdf_path)
        self.company_name = None
        self.fiscal_year = None
        self.extracted_text = ""
        self.tables = []
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Initialized PDFReportExtractor for: {self.pdf_path.name}")
    
    def extract_text(self) -> str:
        """Extract text from PDF."""
        logger.info("Extracting text from PDF...")
        
        try:
            if PDF_AVAILABLE:
                return self._extract_with_pdfplumber()
            elif PYPDF_AVAILABLE:
                return self._extract_with_pypdf()
            else:
                raise ImportError("No PDF library available. Install: pip install pdfplumber")
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise
    
    def _extract_with_pdfplumber(self) -> str:
        """Extract text using pdfplumber."""
        text = ""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                logger.info(f"PDF has {len(pdf.pages)} pages")
                
                # Extract from first 20 pages (usually contains financial statements)
                for i, page in enumerate(pdf.pages[:20]):
                    text += page.extract_text() or ""
                    if i < 5:  # Log first few pages
                        logger.info(f"Extracted page {i + 1}")
                
                # Also extract tables
                for i, page in enumerate(pdf.pages[:20]):
                    tables = page.extract_tables()
                    if tables:
                        self.tables.extend(tables)
                        logger.info(f"Found {len(tables)} table(s) on page {i + 1}")
        
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            raise
        
        self.extracted_text = text
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
    
    def _extract_with_pypdf(self) -> str:
        """Extract text using PyPDF2."""
        text = ""
        try:
            reader = PdfReader(self.pdf_path)
            logger.info(f"PDF has {len(reader.pages)} pages")
            
            for i, page in enumerate(reader.pages[:20]):
                text += page.extract_text() or ""
                if i < 5:
                    logger.info(f"Extracted page {i + 1}")
        
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise
        
        self.extracted_text = text
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
    
    def extract_tables(self) -> List[List[List[str]]]:
        """Extract tables from PDF."""
        logger.info("Extracting tables from PDF...")
        
        try:
            if PDF_AVAILABLE:
                with pdfplumber.open(self.pdf_path) as pdf:
                    all_tables = []
                    for page in pdf.pages[:20]:
                        tables = page.extract_tables()
                        if tables:
                            all_tables.extend(tables)
                    logger.info(f"Found {len(all_tables)} total tables")
                    return all_tables
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        return []
    
    def extract_company_info(self) -> Tuple[str, int]:
        """Extract company name and fiscal year from text."""
        logger.info("Extracting company information...")
        
        text = self.extracted_text.lower()
        
        # Try to extract fiscal year
        year_patterns = [
            r'fiscal year ended\s+(\d{4})',
            r'year ended\s+[a-z]+\s+\d+,?\s+(\d{4})',
            r'for the year ended.*?(\d{4})',
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text)
            if match:
                self.fiscal_year = int(match.group(1))
                logger.info(f"Found fiscal year: {self.fiscal_year}")
                break
        
        # Default to current year if not found
        if not self.fiscal_year:
            import datetime
            self.fiscal_year = datetime.datetime.now().year
            logger.warning(f"Fiscal year not found, using current year: {self.fiscal_year}")
        
        return self.company_name, self.fiscal_year
    
    def extract_financial_metrics(self) -> Dict[str, Optional[float]]:
        """Extract financial metrics from text and tables."""
        logger.info("Extracting financial metrics...")
        
        metrics = {}
        
        # Try table extraction first (more reliable)
        table_metrics = self._extract_from_tables()
        if table_metrics:
            metrics.update(table_metrics)
            logger.info(f"Extracted {len(table_metrics)} metrics from tables")
        
        # Then try text extraction for missing metrics
        text_metrics = self._extract_from_text()
        for key, value in text_metrics.items():
            if key not in metrics or metrics[key] is None:
                metrics[key] = value
        
        logger.info(f"Total metrics extracted: {len([v for v in metrics.values() if v is not None])}")
        return metrics
    
    def _extract_from_tables(self) -> Dict[str, Optional[float]]:
        """Extract metrics from PDF tables."""
        metrics = {}
        
        try:
            if not PDF_AVAILABLE:
                return metrics
            
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages[:20]):
                    tables = page.extract_tables()
                    
                    for table in tables:
                        for row in table:
                            for cell in row:
                                if not cell:
                                    continue
                                
                                cell_text = str(cell).lower().strip()
                                cell_value = self._extract_number_from_cell(str(cell))
                                
                                if cell_value is None:
                                    continue
                                
                                # Match against known metrics
                                for metric_key, keywords in self.FINANCIAL_METRICS.items():
                                    for keyword in keywords:
                                        if keyword in cell_text:
                                            if metric_key not in metrics or metrics[metric_key] is None:
                                                metrics[metric_key] = cell_value
                                            logger.info(f"Found {metric_key}: {cell_value}")
                                            break
        
        except Exception as e:
            logger.warning(f"Table extraction error: {e}")
        
        return metrics
    
    def _extract_from_text(self) -> Dict[str, Optional[float]]:
        """Extract metrics from text."""
        metrics = {}
        text = self.extracted_text.lower()
        
        for metric_key, keywords in self.FINANCIAL_METRICS.items():
            for keyword in keywords:
                # Look for pattern: keyword followed by number
                pattern = rf'{keyword}\s*(?:for|in|of)?\s*(?:the\s+)?(?:year|period)?[:\s]*\$?\s*([\d,\.]+)'
                match = re.search(pattern, text)
                
                if match and metric_key not in metrics:
                    try:
                        value = float(match.group(1).replace(',', ''))
                        metrics[metric_key] = value
                        logger.info(f"Found {metric_key}: {value} from text")
                        break
                    except ValueError:
                        continue
        
        return metrics
    
    def _extract_number_from_cell(self, text: str) -> Optional[float]:
        """Extract number from cell text."""
        try:
            # Remove common financial symbols
            text = text.replace('$', '').replace(',', '').strip()
            
            # Look for numbers (including decimals)
            match = re.search(r'-?\d+\.?\d*', text)
            if match:
                return float(match.group())
        except (ValueError, AttributeError):
            pass
        
        return None
    
    def to_csv(self, company_name: str, metrics: Dict[str, Optional[float]], 
               output_path: str = "extracted_report.csv") -> str:
        """Convert extracted metrics to CSV."""
        logger.info(f"Converting to CSV: {output_path}")
        
        # Prepare data row
        data = {
            'company': company_name,
            'year': self.fiscal_year,
            'revenue': metrics.get('revenue'),
            'net_income': metrics.get('net_income'),
            'total_assets': metrics.get('total_assets'),
            'current_assets': metrics.get('current_assets'),
            'current_liabilities': metrics.get('current_liabilities'),
            'total_debt': metrics.get('total_debt'),
            'equity': metrics.get('equity'),
            'inventory': metrics.get('inventory'),
            'cogs': metrics.get('cogs'),
            'operating_income': metrics.get('operating_income'),
            'interest_expense': metrics.get('interest_expense'),
            'accounts_receivable': metrics.get('accounts_receivable'),
            'cash': metrics.get('cash')
        }
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"✓ CSV saved to: {output_path}")
        
        return output_path


class ReportConverter:
    """Main converter class for PDF to CSV conversion."""
    
    def __init__(self):
        """Initialize converter."""
        logger.info("Initialized ReportConverter")
    
    def convert_pdf_to_csv(self, pdf_path: str, company_name: str, 
                          output_path: Optional[str] = None) -> str:
        """
        Convert PDF annual report to CSV.
        
        Args:
            pdf_path: Path to PDF file
            company_name: Company name for the CSV
            output_path: Output CSV path (optional)
        
        Returns:
            Path to generated CSV file
        """
        logger.info(f"Starting PDF to CSV conversion: {pdf_path}")
        
        # Initialize extractor
        extractor = PDFReportExtractor(pdf_path)
        
        # Extract text
        extractor.extract_text()
        
        # Extract company info
        extractor.extract_company_info()
        
        # Extract financial metrics
        metrics = extractor.extract_financial_metrics()
        
        # Generate output path if not provided
        if not output_path:
            base_name = Path(pdf_path).stem
            output_path = f"{base_name}_extracted.csv"
        
        # Convert to CSV
        csv_path = extractor.to_csv(company_name, metrics, output_path)
        
        logger.info(f"✓ Conversion complete: {csv_path}")
        return csv_path
    
    def batch_convert(self, pdf_folder: str, output_folder: str = "converted_reports") -> List[str]:
        """
        Convert multiple PDFs in a folder.
        
        Args:
            pdf_folder: Folder containing PDF files
            output_folder: Folder for output CSVs
        
        Returns:
            List of generated CSV paths
        """
        logger.info(f"Batch converting PDFs from: {pdf_folder}")
        
        pdf_folder = Path(pdf_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        
        pdf_files = list(pdf_folder.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        converted_files = []
        
        for pdf_file in pdf_files:
            try:
                # Extract company name from filename
                company_name = pdf_file.stem.replace("_", " ").replace("-", " ")
                
                output_path = output_folder / f"{pdf_file.stem}_extracted.csv"
                
                csv_path = self.convert_pdf_to_csv(
                    str(pdf_file),
                    company_name,
                    str(output_path)
                )
                converted_files.append(csv_path)
                logger.info(f"✓ Converted: {pdf_file.name} → {output_path.name}")
            
            except Exception as e:
                logger.error(f"Failed to convert {pdf_file.name}: {e}")
                continue
        
        logger.info(f"✓ Batch conversion complete: {len(converted_files)} files")
        return converted_files


def main():
    """Command-line interface for PDF to CSV conversion."""
    import sys
    
    if len(sys.argv) < 3:
        print("""
PDF Annual Report to CSV Converter

Usage:
    python pdf_converter.py <pdf_file> <company_name> [output_file]
    
    Or for batch processing:
    python pdf_converter.py --batch <pdf_folder> [output_folder]

Example:
    python pdf_converter.py apple_2024.pdf "Apple Inc" apple_2024.csv
    python pdf_converter.py --batch ./reports ./converted
        """)
        return
    
    try:
        if sys.argv[1] == "--batch":
            # Batch processing
            pdf_folder = sys.argv[2]
            output_folder = sys.argv[3] if len(sys.argv) > 3 else "converted_reports"
            
            converter = ReportConverter()
            converted = converter.batch_convert(pdf_folder, output_folder)
            
            print(f"\n✓ Converted {len(converted)} files")
            for csv_file in converted:
                print(f"  - {csv_file}")
        else:
            # Single file processing
            pdf_file = sys.argv[1]
            company_name = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            
            converter = ReportConverter()
            csv_path = converter.convert_pdf_to_csv(pdf_file, company_name, output_file)
            
            print(f"\n✓ Conversion complete!")
            print(f"  PDF: {pdf_file}")
            print(f"  CSV: {csv_path}")
    
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main()
