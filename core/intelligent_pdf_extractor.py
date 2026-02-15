"""
Intelligent Financial Metrics Extraction System

Trains on sample annual reports to extract financial metrics
from any PDF and converts to standardized CSV format.
"""

import pandas as pd
import numpy as np
import pdfplumber
import re
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractedMetric:
    """Extracted financial metric."""
    name: str
    value: float
    page: int
    confidence: float
    raw_text: str


class FinancialMetricsExtractor:
    """Intelligent extractor for financial metrics from PDFs."""
    
    # Financial metrics keywords (expanded)
    METRICS_KEYWORDS = {
        'revenue': ['revenue', 'net sales', 'total revenue', 'sales', 'turnover'],
        'gross_profit': ['gross profit', 'gross margin'],
        'operating_income': ['operating income', 'ebit', 'operating profit'],
        'ebitda': ['ebitda', 'ebit da'],
        'net_income': ['net income', 'net earnings', 'net profit', 'profit after tax'],
        'total_assets': ['total assets'],
        'current_assets': ['current assets'],
        'non_current_assets': ['non-current assets', 'fixed assets'],
        'cash': ['cash and cash equivalents', 'cash'],
        'total_liabilities': ['total liabilities'],
        'current_liabilities': ['current liabilities'],
        'non_current_liabilities': ['non-current liabilities', 'long-term liabilities'],
        'total_debt': ['total debt', 'total borrowings'],
        'shareholders_equity': ['shareholders equity', 'equity', 'shareholders funds'],
        'cost_of_goods_sold': ['cost of goods sold', 'cogs'],
        'operating_expenses': ['operating expenses', 'opex'],
        'interest_expense': ['interest expense', 'finance costs'],
        'tax_expense': ['tax expense', 'income tax'],
    }
    
    # Patterns for number extraction
    CURRENCY_PATTERNS = [
        r'[₹€$£¥]?\s?[\d,]+\.?\d*\s*(?:crore|lakhs?|million|billion|thousand|bn|mn|k)?',
        r'[\d,]+\.?\d*\s*(?:crore|lakhs?|million|billion|thousand|bn|mn|k)',
    ]
    
    def __init__(self, sample_pdfs_dir: Optional[str] = None):
        """Initialize with optional sample PDFs for training."""
        self.sample_pdfs_dir = sample_pdfs_dir
        self.trained_patterns = {}
        self.metric_positions = {}
        
        if sample_pdfs_dir:
            self._train_on_samples(sample_pdfs_dir)
    
    def _train_on_samples(self, pdf_dir: str):
        """Train extraction patterns on sample PDFs."""
        logger.info(f"Training on sample PDFs from {pdf_dir}")
        
        pdf_files = list(Path(pdf_dir).glob('*.pdf'))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files[:5]:  # Train on first 5
            try:
                logger.info(f"Training on {pdf_path.name}")
                self._analyze_pdf_structure(str(pdf_path))
            except Exception as e:
                logger.warning(f"Error training on {pdf_path.name}: {e}")
    
    def _analyze_pdf_structure(self, pdf_path: str):
        """Analyze PDF structure for metric positions."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages[:30]):  # Analyze first 30 pages
                    text = page.extract_text()
                    tables = page.extract_tables()
                    
                    # Find metric keywords
                    for metric, keywords in self.METRICS_KEYWORDS.items():
                        for keyword in keywords:
                            if keyword.lower() in text.lower():
                                if metric not in self.metric_positions:
                                    self.metric_positions[metric] = []
                                self.metric_positions[metric].append(page_num)
                    
                    # Extract tables structure
                    if tables:
                        for table in tables:
                            self._analyze_table_structure(table, page_num)
        except Exception as e:
            logger.warning(f"Error analyzing {pdf_path}: {e}")
    
    def _analyze_table_structure(self, table: List[List[str]], page_num: int):
        """Analyze table structure for financial data."""
        if not table:
            return
        
        # Look for common financial statement structure
        for row in table:
            row_text = ' '.join([str(cell) for cell in row if cell]).lower()
            for metric, keywords in self.METRICS_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in row_text:
                        # Store pattern for this metric
                        if metric not in self.trained_patterns:
                            self.trained_patterns[metric] = []
                        self.trained_patterns[metric].append({
                            'page': page_num,
                            'row': row,
                            'pattern': row_text
                        })
    
    def extract_metrics_from_pdf(self, pdf_path: str) -> Dict[str, ExtractedMetric]:
        """Extract financial metrics from a PDF."""
        extracted = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # First try table extraction
                for page_num, page in enumerate(pdf.pages[:40]):  # Check first 40 pages
                    tables = page.extract_tables()
                    text = page.extract_text()
                    
                    # Extract from tables
                    if tables:
                        for table in tables:
                            self._extract_from_table(table, extracted, page_num, pdf_path)
                    
                    # Extract from text
                    self._extract_from_text(text, extracted, page_num)
        
        except Exception as e:
            logger.error(f"Error extracting metrics from {pdf_path}: {e}")
        
        return extracted
    
    def _extract_from_table(self, table: List[List[str]], extracted: Dict, page_num: int, pdf_path: str):
        """Extract metrics from table data."""
        for row_idx, row in enumerate(table):
            if not row or len(row) < 2:
                continue
            
            row_text = ' '.join([str(cell) for cell in row if cell]).lower()
            
            # Find metric keywords in row
            for metric, keywords in self.METRICS_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in row_text:
                        # Try to extract number from row
                        for cell in row:
                            if cell:
                                value = self._extract_number(str(cell))
                                if value is not None:
                                    if metric not in extracted:
                                        extracted[metric] = ExtractedMetric(
                                            name=metric,
                                            value=value,
                                            page=page_num,
                                            confidence=0.8,
                                            raw_text=str(cell)
                                        )
                                    break
    
    def _extract_from_text(self, text: str, extracted: Dict, page_num: int):
        """Extract metrics from text."""
        if not text:
            return
        
        text_lower = text.lower()
        
        for metric, keywords in self.METRICS_KEYWORDS.items():
            if metric in extracted:
                continue
            
            for keyword in keywords:
                if keyword in text_lower:
                    # Find number near keyword
                    pattern = rf'{keyword}[:\s]*([₹€$£¥]?\s?[\d,]+\.?\d*\s*(?:crore|lakhs?|million|billion|thousand|bn|mn|k)?)'
                    matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                    
                    for match in matches:
                        value = self._extract_number(match.group(1))
                        if value is not None:
                            extracted[metric] = ExtractedMetric(
                                name=metric,
                                value=value,
                                page=page_num,
                                confidence=0.7,
                                raw_text=match.group(1)
                            )
                            break
                    
                    if metric in extracted:
                        break
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numeric value from text."""
        if not text:
            return None
        
        text = str(text).strip()
        
        # Handle different unit multipliers
        multipliers = {
            'crore': 10_000_000,
            'cr': 10_000_000,
            'lakh': 100_000,
            'lac': 100_000,
            'million': 1_000_000,
            'mn': 1_000_000,
            'billion': 1_000_000_000,
            'bn': 1_000_000_000,
            'thousand': 1_000,
            'k': 1_000,
        }
        
        # Remove currency symbols
        text = re.sub(r'[₹€$£¥]', '', text)
        
        # Extract number
        number_match = re.search(r'[\d,]+\.?\d*', text)
        if not number_match:
            return None
        
        number_str = number_match.group().replace(',', '')
        
        try:
            number = float(number_str)
        except ValueError:
            return None
        
        # Apply multiplier
        for unit, multiplier in multipliers.items():
            if unit in text.lower():
                number *= multiplier
                break
        
        return number
    
    def extract_and_generate_csv(self, pdf_path: str, output_csv: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Extract metrics and generate CSV."""
        
        # Extract company name and year from filename
        filename = Path(pdf_path).stem
        parts = filename.split('_')
        company_name = parts[0] if parts else 'Unknown'
        fiscal_year = parts[-1] if len(parts) > 1 else '2024'
        
        # Extract metrics
        metrics = self.extract_metrics_from_pdf(pdf_path)
        
        # Create dataframe
        data = {
            'company': [company_name],
            'fiscal_year': [fiscal_year],
        }
        
        for metric_name, metric in metrics.items():
            data[metric_name] = [metric.value]
        
        df = pd.DataFrame(data)
        
        # Save CSV if path provided
        if output_csv:
            df.to_csv(output_csv, index=False)
            logger.info(f"Saved CSV to {output_csv}")
        
        # Create report
        report = {
            'company': company_name,
            'fiscal_year': fiscal_year,
            'metrics_extracted': len(metrics),
            'metrics': {
                name: {
                    'value': metric.value,
                    'page': metric.page,
                    'confidence': metric.confidence
                }
                for name, metric in metrics.items()
            }
        }
        
        return df, report


class BatchPDFProcessor:
    """Process batch of PDFs."""
    
    def __init__(self, extractor: FinancialMetricsExtractor):
        self.extractor = extractor
    
    def process_directory(self, pdf_dir: str, output_dir: str = 'extracted_data') -> pd.DataFrame:
        """Process all PDFs in directory."""
        
        Path(output_dir).mkdir(exist_ok=True)
        all_data = []
        
        pdf_files = list(Path(pdf_dir).glob('*.pdf'))
        logger.info(f"Processing {len(pdf_files)} PDFs")
        
        for idx, pdf_path in enumerate(pdf_files):
            try:
                logger.info(f"[{idx+1}/{len(pdf_files)}] Processing {pdf_path.name}")
                
                output_csv = Path(output_dir) / f"{pdf_path.stem}_extracted.csv"
                df, report = self.extractor.extract_and_generate_csv(str(pdf_path), str(output_csv))
                
                all_data.append(df)
                
                logger.info(f"Extracted {report['metrics_extracted']} metrics from {report['company']}")
            
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_path = Path(output_dir) / 'all_companies_combined.csv'
            combined_df.to_csv(combined_path, index=False)
            logger.info(f"Saved combined data to {combined_path}")
            return combined_df
        
        return pd.DataFrame()


if __name__ == "__main__":
    # Initialize extractor with training on sample PDFs
    sample_dir = '/Users/adi/Documents/financial-distress-ews/annual_reports_2024'
    extractor = FinancialMetricsExtractor(sample_pdfs_dir=sample_dir)
    
    # Process all PDFs in directory
    processor = BatchPDFProcessor(extractor)
    combined_df = processor.process_directory(sample_dir, 'extracted_metrics')
    
    print("\nCombined Data Summary:")
    print(combined_df.head())
    print(f"\nShape: {combined_df.shape}")
    print(f"\nColumns: {combined_df.columns.tolist()}")
