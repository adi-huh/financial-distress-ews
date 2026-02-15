"""
Financial Metric Pattern Learner

Analyzes training PDFs to identify and learn metric extraction patterns.
Builds templates for automated extraction from new PDFs.
"""

import pdfplumber
import pandas as pd
import numpy as np
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MetricPattern:
    """Pattern for extracting a metric."""
    metric_name: str
    keywords: List[str]
    common_pages: List[int]
    table_positions: List[Dict]
    text_patterns: List[str]
    unit_type: str  # 'currency', 'percentage', 'count'
    typical_value_range: Tuple[float, float]
    confidence: float


class FinancialMetricsPatternLearner:
    """Learns extraction patterns from sample annual reports."""
    
    # Financial statement sections
    BALANCE_SHEET_KEYWORDS = [
        'balance sheet', 'statement of financial position',
        'assets', 'liabilities', 'equity', 'shareholders funds'
    ]
    
    INCOME_STATEMENT_KEYWORDS = [
        'income statement', 'profit and loss', 'p&l',
        'revenue', 'expenses', 'earnings'
    ]
    
    CASH_FLOW_KEYWORDS = [
        'cash flow', 'cash flows', 'operating activities',
        'investing activities', 'financing activities'
    ]
    
    # Core metrics to extract
    CORE_METRICS = {
        'revenue': {
            'keywords': ['revenue', 'net sales', 'total revenue', 'turnover', 'sales'],
            'unit': 'currency',
            'range': (0, 10_000_000_000),
        },
        'gross_profit': {
            'keywords': ['gross profit', 'gross margin'],
            'unit': 'currency',
            'range': (0, 10_000_000_000),
        },
        'operating_income': {
            'keywords': ['operating income', 'ebit', 'operating profit'],
            'unit': 'currency',
            'range': (0, 5_000_000_000),
        },
        'net_income': {
            'keywords': ['net income', 'net profit', 'net earnings', 'profit after tax', 'pat'],
            'unit': 'currency',
            'range': (0, 5_000_000_000),
        },
        'total_assets': {
            'keywords': ['total assets'],
            'unit': 'currency',
            'range': (0, 50_000_000_000),
        },
        'total_liabilities': {
            'keywords': ['total liabilities'],
            'unit': 'currency',
            'range': (0, 50_000_000_000),
        },
        'shareholders_equity': {
            'keywords': ['shareholders equity', 'shareholders funds', 'total equity'],
            'unit': 'currency',
            'range': (0, 50_000_000_000),
        },
        'cash': {
            'keywords': ['cash and cash equivalents', 'cash and equivalents'],
            'unit': 'currency',
            'range': (0, 10_000_000_000),
        },
        'ebitda': {
            'keywords': ['ebitda', 'ebit da'],
            'unit': 'currency',
            'range': (0, 5_000_000_000),
        },
        'debt': {
            'keywords': ['total debt', 'long-term debt', 'short-term debt'],
            'unit': 'currency',
            'range': (0, 20_000_000_000),
        },
    }
    
    def __init__(self):
        self.patterns: Dict[str, MetricPattern] = {}
        self.page_distribution: Dict[str, Counter] = defaultdict(Counter)
        self.table_structures: Dict[str, List[List]] = defaultdict(list)
        self.metric_values: Dict[str, List[float]] = defaultdict(list)
        self.training_samples = 0
    
    def learn_from_pdfs(self, pdf_dir: str) -> Dict[str, MetricPattern]:
        """Learn patterns from directory of PDFs."""
        
        pdf_files = list(Path(pdf_dir).glob('*.pdf'))
        logger.info(f"Learning from {len(pdf_files)} PDFs in {pdf_dir}")
        
        for pdf_idx, pdf_path in enumerate(pdf_files):
            try:
                logger.info(f"[{pdf_idx + 1}/{len(pdf_files)}] Learning from {pdf_path.name}")
                self._learn_from_single_pdf(str(pdf_path))
                self.training_samples += 1
            except Exception as e:
                logger.warning(f"Error learning from {pdf_path.name}: {e}")
        
        # Build patterns from learned data
        self._build_patterns()
        
        logger.info(f"Learned {len(self.patterns)} metric patterns from {self.training_samples} samples")
        return self.patterns
    
    def _learn_from_single_pdf(self, pdf_path: str):
        """Learn from a single PDF."""
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                tables = page.extract_tables()
                
                if not text:
                    continue
                
                # Determine which statement this page contains
                statement_type = self._identify_statement_type(text)
                
                # Learn from text
                self._learn_from_text(text, page_num, statement_type)
                
                # Learn from tables
                if tables:
                    for table in tables:
                        self._learn_from_table(table, page_num, statement_type)
    
    def _identify_statement_type(self, text: str) -> str:
        """Identify which financial statement this is."""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in self.BALANCE_SHEET_KEYWORDS):
            return 'balance_sheet'
        elif any(kw in text_lower for kw in self.INCOME_STATEMENT_KEYWORDS):
            return 'income_statement'
        elif any(kw in text_lower for kw in self.CASH_FLOW_KEYWORDS):
            return 'cash_flow'
        
        return 'unknown'
    
    def _learn_from_text(self, text: str, page_num: int, statement_type: str):
        """Learn metric patterns from text."""
        text_lower = text.lower()
        
        for metric, config in self.CORE_METRICS.items():
            for keyword in config['keywords']:
                if keyword in text_lower:
                    self.page_distribution[metric][page_num] += 1
                    
                    # Extract numeric values near keywords
                    pattern = rf'{keyword}[:\s]*([₹€$£¥]?\s?[\d,]+\.?\d*(?:\s*(?:crore|lakhs?|million|billion|thousand|bn|mn|k))?)'
                    matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                    
                    for match in matches:
                        value = self._parse_value(match.group(1))
                        if value and config['range'][0] <= value <= config['range'][1]:
                            self.metric_values[metric].append(value)
    
    def _learn_from_table(self, table: List[List], page_num: int, statement_type: str):
        """Learn patterns from table structures."""
        
        for row_idx, row in enumerate(table):
            if not row or len(row) < 2:
                continue
            
            row_text = ' '.join([str(cell) for cell in row if cell]).lower()
            
            for metric, config in self.CORE_METRICS.items():
                for keyword in config['keywords']:
                    if keyword in row_text:
                        # Store table structure
                        self.table_structures[metric].append({
                            'row_idx': row_idx,
                            'page': page_num,
                            'statement': statement_type,
                            'row': row,
                            'position': 'left' if row_idx < 5 else 'middle' if row_idx < 15 else 'right'
                        })
                        
                        # Try to extract value from next column
                        for col_idx in range(len(row)):
                            if keyword in row[col_idx].lower():
                                # Value is typically in next column
                                if col_idx + 1 < len(row):
                                    value = self._parse_value(row[col_idx + 1])
                                    if value and config['range'][0] <= value <= config['range'][1]:
                                        self.metric_values[metric].append(value)
    
    def _parse_value(self, value_str: str) -> Optional[float]:
        """Parse numeric value from string."""
        if not value_str:
            return None
        
        value_str = str(value_str).strip()
        
        # Remove currency symbols
        value_str = re.sub(r'[₹€$£¥]', '', value_str)
        
        # Handle multipliers
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
            't': 1_000,
            'k': 1_000,
        }
        
        # Extract number
        number_match = re.search(r'[\d,]+\.?\d*', value_str)
        if not number_match:
            return None
        
        number = float(number_match.group().replace(',', ''))
        
        # Apply multiplier
        for unit, multiplier in multipliers.items():
            if unit in value_str.lower():
                number *= multiplier
                break
        
        return number
    
    def _build_patterns(self):
        """Build MetricPattern objects from learned data."""
        
        for metric, config in self.CORE_METRICS.items():
            # Find most common page
            if metric in self.page_distribution:
                common_pages = sorted(
                    self.page_distribution[metric].most_common(3),
                    key=lambda x: x[1],
                    reverse=True
                )
                page_numbers = [p[0] for p in common_pages]
            else:
                page_numbers = []
            
            # Calculate value statistics
            if metric in self.metric_values:
                values = self.metric_values[metric]
                confidence = min(1.0, len(values) / max(self.training_samples, 1))
            else:
                values = []
                confidence = 0.0
            
            # Get table structures
            table_positions = self.table_structures.get(metric, [])[:5]
            
            pattern = MetricPattern(
                metric_name=metric,
                keywords=config['keywords'],
                common_pages=page_numbers,
                table_positions=table_positions,
                text_patterns=[f"{kw}[:\s]*\\d+" for kw in config['keywords']],
                unit_type=config['unit'],
                typical_value_range=(
                    min(values) if values else config['range'][0],
                    max(values) if values else config['range'][1]
                ),
                confidence=confidence
            )
            
            self.patterns[metric] = pattern
    
    def save_patterns(self, output_file: str):
        """Save learned patterns to JSON."""
        patterns_dict = {
            name: {
                'metric_name': p.metric_name,
                'keywords': p.keywords,
                'common_pages': p.common_pages,
                'unit_type': p.unit_type,
                'typical_value_range': p.typical_value_range,
                'confidence': p.confidence,
                'text_patterns': p.text_patterns,
                'num_table_positions': len(p.table_positions),
            }
            for name, p in self.patterns.items()
        }
        
        with open(output_file, 'w') as f:
            json.dump(patterns_dict, f, indent=2)
        
        logger.info(f"Saved patterns to {output_file}")
    
    def get_pattern_summary(self) -> pd.DataFrame:
        """Get summary of learned patterns."""
        data = []
        
        for metric_name, pattern in self.patterns.items():
            data.append({
                'metric': metric_name,
                'keywords': ', '.join(pattern.keywords[:2]),
                'confidence': f"{pattern.confidence:.2f}",
                'common_pages': str(pattern.common_pages),
                'value_range': f"[{pattern.typical_value_range[0]:.0f}, {pattern.typical_value_range[1]:.0f}]",
                'unit': pattern.unit_type,
                'sample_count': len(self.metric_values.get(metric_name, [])),
            })
        
        return pd.DataFrame(data)


class PatternMatchingExtractor:
    """Uses learned patterns to extract metrics."""
    
    def __init__(self, patterns: Dict[str, MetricPattern]):
        self.patterns = patterns
    
    def extract_using_patterns(self, pdf_path: str) -> Dict[str, float]:
        """Extract metrics using learned patterns."""
        extracted = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            for metric_name, pattern in self.patterns.items():
                # Try pages where metric is commonly found
                if pattern.common_pages:
                    for page_num in pattern.common_pages:
                        if page_num < len(pdf.pages):
                            page = pdf.pages[page_num]
                            value = self._extract_metric_from_page(page, pattern)
                            if value:
                                extracted[metric_name] = value
                                break
                
                # If not found, search all pages
                if metric_name not in extracted:
                    for page in pdf.pages[:40]:
                        value = self._extract_metric_from_page(page, pattern)
                        if value:
                            extracted[metric_name] = value
                            break
        
        return extracted
    
    def _extract_metric_from_page(self, page, pattern: MetricPattern) -> Optional[float]:
        """Extract metric from specific page using pattern."""
        text = page.extract_text()
        tables = page.extract_tables()
        
        if not text:
            return None
        
        # Try text extraction
        for keyword in pattern.keywords:
            pattern_str = rf'{keyword}[:\s]*([₹€$£¥]?\s?[\d,]+\.?\d*(?:\s*(?:crore|lakhs?|million|billion|thousand))?)'
            matches = re.finditer(pattern_str, text.lower(), re.IGNORECASE)
            for match in matches:
                value = self._parse_value(match.group(1))
                if value and pattern.typical_value_range[0] <= value <= pattern.typical_value_range[1]:
                    return value
        
        # Try table extraction
        if tables:
            for table in tables:
                for row in table:
                    row_text = ' '.join([str(cell) for cell in row if cell]).lower()
                    for keyword in pattern.keywords:
                        if keyword in row_text:
                            for cell in row:
                                value = self._parse_value(str(cell))
                                if value and pattern.typical_value_range[0] <= value <= pattern.typical_value_range[1]:
                                    return value
        
        return None
    
    def _parse_value(self, value_str: str) -> Optional[float]:
        """Parse numeric value."""
        if not value_str:
            return None
        
        value_str = str(value_str).strip().replace('₹', '').replace(',', '')
        
        multipliers = {
            'crore': 10_000_000,
            'lakh': 100_000,
            'million': 1_000_000,
            'billion': 1_000_000_000,
            'thousand': 1_000,
            'k': 1_000,
        }
        
        number_match = re.search(r'[\d.]+', value_str)
        if not number_match:
            return None
        
        number = float(number_match.group())
        
        for unit, mult in multipliers.items():
            if unit in value_str.lower():
                number *= mult
                break
        
        return number


if __name__ == "__main__":
    sample_dir = '/Users/adi/Documents/financial-distress-ews/annual_reports_2024'
    
    # Learn patterns
    learner = FinancialMetricsPatternLearner()
    patterns = learner.learn_from_pdfs(sample_dir)
    
    # Save patterns
    learner.save_patterns('metric_extraction_patterns.json')
    
    # Print summary
    print("\n=== Pattern Learning Summary ===")
    print(learner.get_pattern_summary())
