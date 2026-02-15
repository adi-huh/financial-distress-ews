"""
Automated Financial Metrics Extraction Pipeline

End-to-end pipeline: PDF → Extract → Clean → Validate → Score → Calculate Ratios → CSV
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, asdict
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of extraction process."""
    company: str
    fiscal_year: str
    pdf_path: str
    extracted_metrics: Dict[str, float]
    cleaned_metrics: Dict[str, float]
    validation_status: Dict[str, bool]
    quality_score: float
    calculated_ratios: Dict[str, float]
    csv_path: Optional[str] = None
    processing_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class FinancialMetricsCalculator:
    """Calculate financial ratios from extracted metrics."""
    
    @staticmethod
    def calculate_liquidity_ratios(metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate liquidity ratios."""
        ratios = {}
        
        try:
            if metrics.get('current_assets') and metrics.get('current_liabilities'):
                ratios['current_ratio'] = metrics['current_assets'] / metrics['current_liabilities']
            
            if metrics.get('cash') and metrics.get('current_liabilities'):
                ratios['cash_ratio'] = metrics['cash'] / metrics['current_liabilities']
        
        except (ZeroDivisionError, KeyError):
            pass
        
        return ratios
    
    @staticmethod
    def calculate_profitability_ratios(metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate profitability ratios."""
        ratios = {}
        
        try:
            if metrics.get('revenue'):
                if metrics.get('gross_profit'):
                    ratios['gross_margin'] = metrics['gross_profit'] / metrics['revenue']
                if metrics.get('operating_income'):
                    ratios['operating_margin'] = metrics['operating_income'] / metrics['revenue']
                if metrics.get('net_income'):
                    ratios['net_margin'] = metrics['net_income'] / metrics['revenue']
            
            if metrics.get('total_assets'):
                if metrics.get('net_income'):
                    ratios['roa'] = metrics['net_income'] / metrics['total_assets']
            
            if metrics.get('shareholders_equity'):
                if metrics.get('net_income'):
                    ratios['roe'] = metrics['net_income'] / metrics['shareholders_equity']
        
        except (ZeroDivisionError, KeyError):
            pass
        
        return ratios
    
    @staticmethod
    def calculate_leverage_ratios(metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate leverage ratios."""
        ratios = {}
        
        try:
            if metrics.get('total_liabilities') and metrics.get('total_assets'):
                ratios['debt_to_assets'] = metrics['total_liabilities'] / metrics['total_assets']
            
            if metrics.get('total_liabilities') and metrics.get('shareholders_equity'):
                ratios['debt_to_equity'] = metrics['total_liabilities'] / metrics['shareholders_equity']
            
            if metrics.get('debt') and metrics.get('shareholders_equity'):
                ratios['debt_to_equity_debt'] = metrics['debt'] / metrics['shareholders_equity']
            
            if metrics.get('ebitda') and metrics.get('debt'):
                ratios['debt_to_ebitda'] = metrics['debt'] / metrics['ebitda']
        
        except (ZeroDivisionError, KeyError):
            pass
        
        return ratios
    
    @staticmethod
    def calculate_efficiency_ratios(metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate efficiency ratios."""
        ratios = {}
        
        try:
            if metrics.get('revenue') and metrics.get('total_assets'):
                ratios['asset_turnover'] = metrics['revenue'] / metrics['total_assets']
        
        except (ZeroDivisionError, KeyError):
            pass
        
        return ratios


class DataValidator:
    """Validate extracted financial data."""
    
    # Reasonable ranges for financial metrics
    VALIDATION_RULES = {
        'current_ratio': (0.5, 10.0),
        'net_margin': (-1.0, 1.0),
        'roe': (-2.0, 2.0),
        'roa': (-0.5, 0.5),
    }
    
    @staticmethod
    def validate_metrics(metrics: Dict[str, float]) -> Tuple[Dict[str, bool], List[str]]:
        """Validate extracted metrics."""
        validation_status = {}
        errors = []
        
        # Check for negative values where not expected
        positive_metrics = [
            'revenue', 'total_assets', 'total_liabilities',
            'shareholders_equity', 'current_assets', 'current_liabilities'
        ]
        
        for metric in positive_metrics:
            if metric in metrics:
                if metrics[metric] < 0:
                    validation_status[f'{metric}_sign'] = False
                    errors.append(f"{metric} is negative: {metrics[metric]}")
                else:
                    validation_status[f'{metric}_sign'] = True
        
        # Check reasonable ratios
        for metric, (min_val, max_val) in DataValidator.VALIDATION_RULES.items():
            if metric in metrics:
                if min_val <= metrics[metric] <= max_val:
                    validation_status[metric] = True
                else:
                    validation_status[metric] = False
                    errors.append(f"{metric} out of range: {metrics[metric]}")
        
        # Check accounting equation: Assets = Liabilities + Equity
        if 'total_assets' in metrics and 'total_liabilities' in metrics and 'shareholders_equity' in metrics:
            assets = metrics['total_assets']
            liab_plus_equity = metrics['total_liabilities'] + metrics['shareholders_equity']
            
            if abs(assets - liab_plus_equity) / max(assets, 1) < 0.05:  # Within 5%
                validation_status['accounting_equation'] = True
            else:
                validation_status['accounting_equation'] = False
                errors.append(f"Accounting equation not balanced: {assets} != {liab_plus_equity}")
        
        return validation_status, errors


class DataCleaner:
    """Clean extracted financial data."""
    
    @staticmethod
    def clean_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
        """Clean and normalize extracted metrics."""
        cleaned = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Remove outliers using IQR method if we have enough data
                if not np.isnan(value) and not np.isinf(value):
                    cleaned[key] = value
        
        return cleaned
    
    @staticmethod
    def handle_missing_values(metrics: Dict[str, float], method: str = 'forward_fill') -> Dict[str, float]:
        """Handle missing values in metrics."""
        # For cross-sectional data, missing values stay as is
        # For time-series data, could implement forward fill or interpolation
        return metrics


class QualityScorer:
    """Score quality of extracted data."""
    
    @staticmethod
    def calculate_quality_score(
        metrics: Dict[str, float],
        validation_status: Dict[str, bool],
        extraction_confidence: float
    ) -> float:
        """Calculate overall quality score (0-100)."""
        
        # Completeness: percentage of expected metrics present
        expected_metrics = [
            'revenue', 'net_income', 'total_assets',
            'total_liabilities', 'shareholders_equity', 'cash'
        ]
        completeness = sum(1 for m in expected_metrics if m in metrics) / len(expected_metrics)
        
        # Validity: percentage of validation checks passed
        if validation_status:
            validity = sum(validation_status.values()) / len(validation_status)
        else:
            validity = 0.5
        
        # Confidence from extraction
        confidence = extraction_confidence
        
        # Weighted average
        score = (completeness * 0.4 + validity * 0.4 + confidence * 0.2) * 100
        
        return min(100, max(0, score))


class AutomatedExtractionPipeline:
    """End-to-end extraction pipeline."""
    
    def __init__(self, extractor=None, pattern_file: Optional[str] = None):
        """Initialize pipeline."""
        self.extractor = extractor
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.scorer = QualityScorer()
        self.calculator = FinancialMetricsCalculator()
        self.pattern_file = pattern_file
    
    def process_pdf(
        self,
        pdf_path: str,
        output_csv: Optional[str] = None,
        output_json: Optional[str] = None
    ) -> ExtractionResult:
        """Process single PDF through full pipeline."""
        
        import time
        start_time = time.time()
        
        try:
            pdf_path_obj = Path(pdf_path)
            company = pdf_path_obj.stem.split('_')[0]
            fiscal_year = pdf_path_obj.stem.split('_')[-1]
            
            # Step 1: Extract metrics
            logger.info(f"Extracting metrics from {pdf_path_obj.name}")
            extracted_metrics = self._extract_metrics(pdf_path)
            
            # Step 2: Clean data
            logger.info("Cleaning extracted metrics")
            cleaned_metrics = self.cleaner.clean_metrics(extracted_metrics)
            cleaned_metrics = self.cleaner.handle_missing_values(cleaned_metrics)
            
            # Step 3: Validate
            logger.info("Validating metrics")
            validation_status, errors = self.validator.validate_metrics(cleaned_metrics)
            
            # Step 4: Calculate quality
            confidence = min(1.0, len(extracted_metrics) / 10)  # Approximate confidence
            quality_score = self.scorer.calculate_quality_score(
                cleaned_metrics, validation_status, confidence
            )
            
            # Step 5: Calculate ratios
            logger.info("Calculating financial ratios")
            ratios = {}
            ratios.update(self.calculator.calculate_liquidity_ratios(cleaned_metrics))
            ratios.update(self.calculator.calculate_profitability_ratios(cleaned_metrics))
            ratios.update(self.calculator.calculate_leverage_ratios(cleaned_metrics))
            ratios.update(self.calculator.calculate_efficiency_ratios(cleaned_metrics))
            
            # Step 6: Generate CSV
            csv_path = None
            if output_csv:
                csv_path = self._generate_csv(
                    company, fiscal_year, cleaned_metrics, ratios, output_csv
                )
                logger.info(f"Saved CSV to {csv_path}")
            
            # Step 7: Generate JSON report
            if output_json:
                self._generate_json_report(
                    company, fiscal_year, extracted_metrics, cleaned_metrics,
                    validation_status, quality_score, ratios, output_json
                )
                logger.info(f"Saved JSON report to {output_json}")
            
            processing_time = time.time() - start_time
            
            return ExtractionResult(
                company=company,
                fiscal_year=fiscal_year,
                pdf_path=pdf_path,
                extracted_metrics=extracted_metrics,
                cleaned_metrics=cleaned_metrics,
                validation_status=validation_status,
                quality_score=quality_score,
                calculated_ratios=ratios,
                csv_path=csv_path,
                processing_time=processing_time,
                errors=errors
            )
        
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return ExtractionResult(
                company='Unknown',
                fiscal_year='2024',
                pdf_path=pdf_path,
                extracted_metrics={},
                cleaned_metrics={},
                validation_status={},
                quality_score=0.0,
                calculated_ratios={},
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _extract_metrics(self, pdf_path: str) -> Dict[str, float]:
        """Extract metrics from PDF."""
        if self.extractor:
            return self.extractor.extract_metrics_from_pdf(pdf_path)
        return {}
    
    def _generate_csv(
        self,
        company: str,
        fiscal_year: str,
        metrics: Dict[str, float],
        ratios: Dict[str, float],
        output_path: str
    ) -> str:
        """Generate CSV file."""
        
        row = {
            'company': company,
            'fiscal_year': fiscal_year,
        }
        row.update(metrics)
        row.update({f'ratio_{k}': v for k, v in ratios.items()})
        
        df = pd.DataFrame([row])
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def _generate_json_report(
        self,
        company: str,
        fiscal_year: str,
        extracted: Dict,
        cleaned: Dict,
        validation: Dict,
        quality: float,
        ratios: Dict,
        output_path: str
    ):
        """Generate detailed JSON report."""
        
        # Convert to JSON-serializable format
        def to_serializable(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        report = {
            'company': company,
            'fiscal_year': fiscal_year,
            'extracted_metrics': to_serializable(cleaned),  # Use cleaned metrics
            'cleaned_metrics': to_serializable(cleaned),
            'validation': to_serializable(validation),
            'quality_score': quality,
            'calculated_ratios': to_serializable(ratios),
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def process_batch(
        self,
        pdf_dir: str,
        output_dir: str = 'extracted_data'
    ) -> Tuple[pd.DataFrame, List[ExtractionResult]]:
        """Process batch of PDFs."""
        
        Path(output_dir).mkdir(exist_ok=True)
        results = []
        all_data = []
        
        pdf_files = sorted(Path(pdf_dir).glob('*.pdf'))
        logger.info(f"Processing {len(pdf_files)} PDFs from {pdf_dir}")
        
        for idx, pdf_path in enumerate(pdf_files):
            logger.info(f"[{idx+1}/{len(pdf_files)}] {pdf_path.name}")
            
            output_csv = Path(output_dir) / f"{pdf_path.stem}_extracted.csv"
            output_json = Path(output_dir) / f"{pdf_path.stem}_report.json"
            
            result = self.process_pdf(
                str(pdf_path),
                str(output_csv),
                str(output_json)
            )
            results.append(result)
            
            if result.cleaned_metrics:
                row = {
                    'company': result.company,
                    'fiscal_year': result.fiscal_year,
                    'quality_score': result.quality_score,
                }
                row.update(result.cleaned_metrics)
                row.update({f'ratio_{k}': v for k, v in result.calculated_ratios.items()})
                all_data.append(row)
        
        # Combine all data
        if all_data:
            combined_df = pd.DataFrame(all_data)
            combined_path = Path(output_dir) / 'all_companies_combined.csv'
            combined_df.to_csv(combined_path, index=False)
            logger.info(f"Saved combined CSV to {combined_path}")
        else:
            combined_df = pd.DataFrame()
        
        return combined_df, results


if __name__ == "__main__":
    from intelligent_pdf_extractor import FinancialMetricsExtractor
    
    # Initialize
    sample_dir = '/Users/adi/Documents/financial-distress-ews/annual_reports_2024'
    extractor = FinancialMetricsExtractor(sample_pdfs_dir=sample_dir)
    
    # Create pipeline
    pipeline = AutomatedExtractionPipeline(extractor=extractor)
    
    # Process batch
    combined_df, results = pipeline.process_batch(sample_dir, 'extracted_data')
    
    print("\n=== Processing Complete ===")
    print(f"Processed {len(results)} PDFs")
    print(f"\nCombined Data Shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")
    
    # Summary statistics
    print("\n=== Quality Scores ===")
    quality_scores = [r.quality_score for r in results if r.quality_score > 0]
    print(f"Average: {np.mean(quality_scores):.1f}")
    print(f"Min: {np.min(quality_scores):.1f}")
    print(f"Max: {np.max(quality_scores):.1f}")
