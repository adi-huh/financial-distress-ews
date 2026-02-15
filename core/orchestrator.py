"""
Main PDF Extraction and Analysis Orchestrator

Combines all modules to provide unified interface for PDF processing,
metrics extraction, and financial analysis.
"""

import sys
import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinancialExtractionOrchestrator:
    """Orchestrate complete PDF extraction and analysis workflow."""
    
    def __init__(self, sample_pdf_dir: Optional[str] = None):
        """Initialize orchestrator."""
        
        self.sample_pdf_dir = sample_pdf_dir
        
        # Import all modules
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all extraction and analysis components."""
        
        try:
            from intelligent_pdf_extractor import FinancialMetricsExtractor
            from pattern_learner import FinancialMetricsPatternLearner, PatternMatchingExtractor
            from extraction_pipeline import AutomatedExtractionPipeline
            from financial_analysis import (
                FinancialHealthAnalyzer,
                AnomalyDetector,
                DistressPredictor,
                CompanyComparer
            )
            
            self.extractor = FinancialMetricsExtractor(sample_pdfs_dir=self.sample_pdf_dir)
            self.pattern_learner = FinancialMetricsPatternLearner()
            self.pipeline = AutomatedExtractionPipeline(extractor=self.extractor)
            self.health_analyzer = FinancialHealthAnalyzer()
            self.anomaly_detector = AnomalyDetector()
            self.distress_predictor = DistressPredictor()
            self.company_comparer = CompanyComparer()
            
            logger.info("All components initialized successfully")
        
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise
    
    def train_on_samples(self, pdf_dir: str) -> Dict:
        """Train extraction patterns on sample PDFs."""
        
        logger.info(f"Training on sample PDFs from {pdf_dir}")
        
        patterns = self.pattern_learner.learn_from_pdfs(pdf_dir)
        
        # Save patterns
        patterns_file = 'learned_patterns.json'
        self.pattern_learner.save_patterns(patterns_file)
        
        logger.info(f"Training complete. Learned {len(patterns)} patterns")
        logger.info(f"Patterns saved to {patterns_file}")
        
        return {
            'patterns_learned': len(patterns),
            'training_samples': self.pattern_learner.training_samples,
            'patterns_file': patterns_file
        }
    
    def extract_and_analyze_single(
        self,
        pdf_path: str,
        output_dir: str = 'output'
    ) -> Dict:
        """Extract metrics and analyze single PDF."""
        
        Path(output_dir).mkdir(exist_ok=True)
        
        logger.info(f"Processing {Path(pdf_path).name}")
        
        # Step 1: Extract
        output_csv = Path(output_dir) / f"{Path(pdf_path).stem}_metrics.csv"
        output_json = Path(output_dir) / f"{Path(pdf_path).stem}_report.json"
        
        extraction_result = self.pipeline.process_pdf(
            pdf_path,
            str(output_csv),
            str(output_json)
        )
        
        # Step 2: Analyze
        if extraction_result.cleaned_metrics:
            analysis = self.health_analyzer.analyze_company(extraction_result.cleaned_metrics)
            analysis.company = extraction_result.company
            
            # Generate analysis report
            analysis_report = {
                'company': analysis.company,
                'financial_health_score': analysis.financial_health_score,
                'distress_risk_level': analysis.distress_risk_level,
                'key_strengths': analysis.key_strengths,
                'key_weaknesses': analysis.key_weaknesses,
                'recommendations': analysis.recommendations,
                'anomalies': analysis.anomalies,
            }
            
            analysis_json = Path(output_dir) / f"{Path(pdf_path).stem}_analysis.json"
            with open(analysis_json, 'w') as f:
                json.dump(analysis_report, f, indent=2)
        else:
            analysis_report = None
        
        return {
            'company': extraction_result.company,
            'fiscal_year': extraction_result.fiscal_year,
            'metrics_extracted': len(extraction_result.extracted_metrics),
            'quality_score': extraction_result.quality_score,
            'extracted_metrics': extraction_result.cleaned_metrics,
            'ratios_calculated': extraction_result.calculated_ratios,
            'analysis': analysis_report,
            'csv_file': str(output_csv),
            'json_file': str(output_json),
        }
    
    def extract_and_analyze_batch(
        self,
        pdf_dir: str,
        output_dir: str = 'output'
    ) -> Dict:
        """Extract and analyze batch of PDFs."""
        
        Path(output_dir).mkdir(exist_ok=True)
        
        logger.info(f"Batch processing PDFs from {pdf_dir}")
        
        # Step 1: Extract batch
        combined_df, extraction_results = self.pipeline.process_batch(pdf_dir, output_dir)
        
        logger.info(f"Extracted {len(extraction_results)} PDFs")
        
        # Step 2: Analyze each company
        analyses = []
        for result in extraction_results:
            if result.cleaned_metrics:
                analysis = self.health_analyzer.analyze_company(result.cleaned_metrics)
                analysis.company = result.company
                analyses.append(analysis)
        
        logger.info(f"Analyzed {len(analyses)} companies")
        
        # Step 3: Detect anomalies
        if not combined_df.empty:
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                self.anomaly_detector.fit(combined_df[numeric_cols].fillna(0))
                anomaly_predictions = self.anomaly_detector.predict(combined_df)
                combined_df['is_anomaly'] = (anomaly_predictions == -1).astype(int)
        
        # Step 4: Generate summary report
        summary = {
            'total_pdfs_processed': len(extraction_results),
            'successful_extractions': sum(1 for r in extraction_results if r.quality_score > 0),
            'avg_quality_score': np.mean([r.quality_score for r in extraction_results]) if extraction_results else 0,
            'companies_analyzed': len(analyses),
            'risk_distribution': self._get_risk_distribution(analyses),
            'top_performers': self._get_top_performers(analyses, 'health_score'),
            'at_risk_companies': self._get_at_risk_companies(analyses),
        }
        
        # Save summary
        summary_file = Path(output_dir) / 'batch_analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Batch processing complete. Summary saved to {summary_file}")
        
        return {
            'summary': summary,
            'combined_csv': str(Path(output_dir) / 'all_companies_combined.csv'),
            'summary_json': str(summary_file),
            'extraction_results': extraction_results,
            'analyses': analyses,
        }
    
    def _get_risk_distribution(self, analyses: List) -> Dict[str, int]:
        """Get distribution of risk levels."""
        distribution = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        for analysis in analyses:
            distribution[analysis.distress_risk_level] += 1
        return distribution
    
    def _get_top_performers(self, analyses: List, metric: str, top_n: int = 5) -> List[Dict]:
        """Get top performing companies."""
        if metric == 'health_score':
            sorted_analyses = sorted(
                analyses,
                key=lambda x: x.financial_health_score,
                reverse=True
            )[:top_n]
            return [
                {
                    'company': a.company,
                    'health_score': a.financial_health_score,
                    'risk_level': a.distress_risk_level
                }
                for a in sorted_analyses
            ]
        return []
    
    def _get_at_risk_companies(self, analyses: List, top_n: int = 5) -> List[Dict]:
        """Get companies at risk."""
        at_risk = [a for a in analyses if a.distress_risk_level in ['HIGH', 'CRITICAL']]
        at_risk = sorted(at_risk, key=lambda x: x.financial_health_score)[:top_n]
        return [
            {
                'company': a.company,
                'health_score': a.financial_health_score,
                'risk_level': a.distress_risk_level,
                'weaknesses': a.key_weaknesses
            }
            for a in at_risk
        ]
    
    def generate_comparison_report(
        self,
        csv_file: str,
        output_dir: str = 'output'
    ) -> Dict:
        """Generate comparison report for all companies."""
        
        Path(output_dir).mkdir(exist_ok=True)
        
        logger.info(f"Generating comparison report from {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # Calculate key metrics
        metrics_to_analyze = [
            'revenue', 'net_income', 'total_assets',
            'current_ratio', 'net_margin', 'roe', 'roa',
            'debt_to_equity'
        ]
        
        comparison_data = {}
        for metric in metrics_to_analyze:
            if metric in df.columns:
                percentiles = self.company_comparer.get_percentiles(df, metric)
                comparison_data[metric] = percentiles
        
        # Save comparison report
        comparison_file = Path(output_dir) / 'comparison_report.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info(f"Comparison report saved to {comparison_file}")
        
        return {
            'comparison_data': comparison_data,
            'comparison_file': str(comparison_file),
            'total_companies': len(df),
            'metrics_analyzed': len(comparison_data),
        }
    
    def export_analysis_report(
        self,
        output_dir: str = 'output'
    ) -> str:
        """Export comprehensive analysis report."""
        
        Path(output_dir).mkdir(exist_ok=True)
        
        report_file = Path(output_dir) / 'extraction_analysis_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Financial Metrics Extraction and Analysis Report\n\n")
            
            f.write("## System Overview\n")
            f.write("- **Module**: Financial Metrics Extraction and Analysis\n")
            f.write("- **Purpose**: Automated extraction of financial metrics from annual reports\n")
            f.write("- **Output**: CSV with metrics, financial analysis, risk assessment\n\n")
            
            f.write("## Key Components\n")
            f.write("1. **PDF Extractor**: Extract metrics from PDF annual reports\n")
            f.write("2. **Pattern Learner**: Learn extraction patterns from sample reports\n")
            f.write("3. **Extraction Pipeline**: End-to-end extraction workflow\n")
            f.write("4. **Financial Analyzer**: Assess financial health and risk\n")
            f.write("5. **Anomaly Detector**: Identify unusual financial patterns\n\n")
            
            f.write("## Extracted Metrics\n")
            f.write("- Revenue, Gross Profit, Operating Income, Net Income\n")
            f.write("- Total Assets, Liabilities, Shareholders' Equity\n")
            f.write("- Cash, EBITDA, Debt\n")
            f.write("- 20+ Financial Ratios (liquidity, profitability, leverage, efficiency)\n\n")
            
            f.write("## Analysis Features\n")
            f.write("- Financial Health Scoring (0-100)\n")
            f.write("- Distress Risk Assessment (Low/Medium/High/Critical)\n")
            f.write("- Strength and Weakness Identification\n")
            f.write("- Financial Recommendations\n")
            f.write("- Anomaly Detection\n")
            f.write("- Company Comparison and Ranking\n\n")
            
            f.write("## Usage\n\n")
            f.write("### Train on Sample Reports\n")
            f.write("```python\n")
            f.write("from orchestrator import FinancialExtractionOrchestrator\n\n")
            f.write("orchestrator = FinancialExtractionOrchestrator()\n")
            f.write("orchestrator.train_on_samples('path/to/annual_reports')\n")
            f.write("```\n\n")
            
            f.write("### Extract from Single PDF\n")
            f.write("```python\n")
            f.write("result = orchestrator.extract_and_analyze_single('path/to/report.pdf')\n")
            f.write("```\n\n")
            
            f.write("### Batch Process Multiple PDFs\n")
            f.write("```python\n")
            f.write("result = orchestrator.extract_and_analyze_batch('path/to/reports_dir')\n")
            f.write("```\n\n")
            
            f.write("## Output Files\n")
            f.write("- `all_companies_combined.csv`: Combined metrics for all companies\n")
            f.write("- `batch_analysis_summary.json`: Summary statistics and risk assessment\n")
            f.write("- `comparison_report.json`: Comparative metrics across companies\n")
            f.write("- Individual company CSV and analysis JSON files\n\n")
            
            f.write("## Quality Metrics\n")
            f.write("Each extraction is scored on:\n")
            f.write("- Completeness (0-100): % of expected metrics extracted\n")
            f.write("- Validity (0-100): % of validation checks passed\n")
            f.write("- Confidence (0-100): Extraction confidence level\n\n")
            
            f.write("## Financial Health Scoring\n")
            f.write("- **Liquidity** (25%): Ability to meet short-term obligations\n")
            f.write("- **Profitability** (35%): Earnings generation efficiency\n")
            f.write("- **Leverage** (25%): Debt and solvency position\n")
            f.write("- **Efficiency** (15%): Asset utilization\n\n")
            
            f.write("Score Interpretation:\n")
            f.write("- 75-100: Excellent financial health (LOW risk)\n")
            f.write("- 60-74: Good financial health (MEDIUM risk)\n")
            f.write("- 40-59: Adequate financial health (HIGH risk)\n")
            f.write("- <40: Poor financial health (CRITICAL risk)\n\n")
        
        logger.info(f"Report exported to {report_file}")
        return str(report_file)


def main():
    """Main execution function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Financial Metrics Extraction and Analysis Orchestrator'
    )
    parser.add_argument('--mode', choices=['train', 'extract-single', 'extract-batch', 'analyze'], 
                        help='Execution mode')
    parser.add_argument('--pdf-dir', help='Directory containing PDFs')
    parser.add_argument('--pdf-file', help='Single PDF file')
    parser.add_argument('--csv-file', help='CSV file for analysis')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    sample_dir = '/Users/adi/Documents/financial-distress-ews/annual_reports_2024'
    orchestrator = FinancialExtractionOrchestrator(sample_pdf_dir=sample_dir)
    
    if args.mode == 'train':
        result = orchestrator.train_on_samples(args.pdf_dir or sample_dir)
        print(f"\n✅ Training Complete")
        print(f"Patterns Learned: {result['patterns_learned']}")
        print(f"Training Samples: {result['training_samples']}")
    
    elif args.mode == 'extract-single':
        result = orchestrator.extract_and_analyze_single(args.pdf_file, args.output_dir)
        print(f"\n✅ Extraction Complete")
        print(f"Company: {result['company']}")
        print(f"Quality Score: {result['quality_score']:.1f}/100")
        print(f"Health Score: {result['analysis']['financial_health_score']:.1f}")
    
    elif args.mode == 'extract-batch':
        result = orchestrator.extract_and_analyze_batch(args.pdf_dir, args.output_dir)
        print(f"\n✅ Batch Processing Complete")
        print(f"Processed: {result['summary']['total_pdfs_processed']} PDFs")
        print(f"Risk Distribution: {result['summary']['risk_distribution']}")
    
    elif args.mode == 'analyze':
        result = orchestrator.generate_comparison_report(args.csv_file, args.output_dir)
        print(f"\n✅ Analysis Complete")
        print(f"Companies Analyzed: {result['total_companies']}")
        print(f"Metrics: {result['metrics_analyzed']}")
    
    # Export report
    orchestrator.export_analysis_report(args.output_dir)


if __name__ == '__main__':
    main()
