"""
Demonstration and Testing Script

Shows how to use the PDF extraction and analysis system.
"""

import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_single_extraction():
    """Demo: Extract from single PDF."""
    
    print("\n" + "="*80)
    print("DEMO 1: Single PDF Extraction")
    print("="*80)
    
    from orchestrator import FinancialExtractionOrchestrator
    
    sample_dir = '/Users/adi/Documents/financial-distress-ews/annual_reports_2024'
    
    # Initialize
    orchestrator = FinancialExtractionOrchestrator(sample_pdf_dir=sample_dir)
    
    # Get first PDF
    pdf_files = list(Path(sample_dir).glob('*.pdf'))
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    pdf_file = str(pdf_files[0])
    print(f"\n📄 Processing: {Path(pdf_file).name}")
    
    # Extract and analyze
    result = orchestrator.extract_and_analyze_single(pdf_file, 'demo_output_single')
    
    print(f"\n✅ Extraction Complete")
    print(f"   Company: {result['company']}")
    print(f"   Fiscal Year: {result['fiscal_year']}")
    print(f"   Quality Score: {result['quality_score']:.1f}/100")
    print(f"   Metrics Extracted: {result['metrics_extracted']}")
    print(f"   Ratios Calculated: {len(result['ratios_calculated'])}")
    
    if result['analysis']:
        analysis = result['analysis']
        print(f"\n📊 Financial Analysis")
        print(f"   Health Score: {analysis['financial_health_score']:.1f}/100")
        print(f"   Risk Level: {analysis['distress_risk_level']}")
        print(f"   Strengths: {', '.join(analysis['key_strengths'][:2])}")
        print(f"   Weaknesses: {', '.join(analysis['key_weaknesses'][:2])}")
    
    print(f"\n📁 Output:")
    print(f"   CSV: {result['csv_file']}")
    print(f"   JSON Report: {result['json_file']}")


def demo_batch_extraction():
    """Demo: Extract from batch of PDFs."""
    
    print("\n" + "="*80)
    print("DEMO 2: Batch PDF Extraction and Analysis")
    print("="*80)
    
    from orchestrator import FinancialExtractionOrchestrator
    
    sample_dir = '/Users/adi/Documents/financial-distress-ews/annual_reports_2024'
    
    # Initialize
    orchestrator = FinancialExtractionOrchestrator(sample_pdf_dir=sample_dir)
    
    print(f"\n📦 Batch Processing PDFs from {sample_dir}")
    
    # Batch process
    result = orchestrator.extract_and_analyze_batch(sample_dir, 'demo_output_batch')
    
    summary = result['summary']
    
    print(f"\n✅ Batch Processing Complete")
    print(f"   Total PDFs: {summary['total_pdfs_processed']}")
    print(f"   Successful: {summary['successful_extractions']}")
    print(f"   Avg Quality: {summary['avg_quality_score']:.1f}/100")
    print(f"   Companies Analyzed: {summary['companies_analyzed']}")
    
    print(f"\n📊 Risk Distribution")
    for risk_level, count in summary['risk_distribution'].items():
        print(f"   {risk_level}: {count} companies")
    
    if summary['top_performers']:
        print(f"\n🏆 Top Performers")
        for company in summary['top_performers'][:3]:
            print(f"   {company['company']}: {company['health_score']:.1f}")
    
    if summary['at_risk_companies']:
        print(f"\n⚠️  At Risk Companies")
        for company in summary['at_risk_companies'][:3]:
            print(f"   {company['company']} (Risk: {company['risk_level']})")
            print(f"      Issues: {', '.join(company['weaknesses'][:2])}")
    
    print(f"\n📁 Output:")
    print(f"   Combined CSV: {result['combined_csv']}")
    print(f"   Summary: {result['summary_json']}")


def demo_training():
    """Demo: Train pattern learner."""
    
    print("\n" + "="*80)
    print("DEMO 3: Train Pattern Learner")
    print("="*80)
    
    from orchestrator import FinancialExtractionOrchestrator
    
    sample_dir = '/Users/adi/Documents/financial-distress-ews/annual_reports_2024'
    
    # Initialize
    orchestrator = FinancialExtractionOrchestrator()
    
    print(f"\n🎓 Training on sample PDFs from {sample_dir}")
    
    # Train
    training_result = orchestrator.train_on_samples(sample_dir)
    
    print(f"\n✅ Training Complete")
    print(f"   Patterns Learned: {training_result['patterns_learned']}")
    print(f"   Training Samples: {training_result['training_samples']}")
    print(f"   Patterns File: {training_result['patterns_file']}")


def demo_comparison():
    """Demo: Generate comparison report."""
    
    print("\n" + "="*80)
    print("DEMO 4: Generate Comparison Report")
    print("="*80)
    
    from orchestrator import FinancialExtractionOrchestrator
    
    # Find combined CSV from previous batch run
    csv_file = 'demo_output_batch/all_companies_combined.csv'
    
    if not Path(csv_file).exists():
        print(f"❌ CSV file not found: {csv_file}")
        print("   Run DEMO 2 first to generate batch extraction data")
        return
    
    # Initialize
    orchestrator = FinancialExtractionOrchestrator()
    
    print(f"\n📊 Generating comparison report from {csv_file}")
    
    # Compare
    comparison_result = orchestrator.generate_comparison_report(csv_file, 'demo_output_comparison')
    
    print(f"\n✅ Comparison Complete")
    print(f"   Total Companies: {comparison_result['total_companies']}")
    print(f"   Metrics Analyzed: {comparison_result['metrics_analyzed']}")
    
    print(f"\n📁 Output:")
    print(f"   Comparison Report: {comparison_result['comparison_file']}")


def demo_data_flow():
    """Demo: Show data flow through pipeline."""
    
    print("\n" + "="*80)
    print("DEMO 5: Data Flow Through Pipeline")
    print("="*80)
    
    print("""
    📊 Data Flow Architecture
    
    1. INPUT
       └─ PDF Annual Report
    
    2. EXTRACTION
       ├─ pdfplumber (extract text)
       ├─ Table extraction
       ├─ Pattern matching
       └─ → Raw Financial Metrics
    
    3. CLEANING
       ├─ Remove outliers
       ├─ Handle missing values
       ├─ Normalize units
       └─ → Cleaned Metrics
    
    4. VALIDATION
       ├─ Data type validation
       ├─ Range checking
       ├─ Accounting equation
       ├─ Business logic rules
       └─ → Validation Status
    
    5. QUALITY SCORING
       ├─ Completeness (40%)
       ├─ Validity (40%)
       ├─ Confidence (20%)
       └─ → Quality Score (0-100)
    
    6. RATIO CALCULATION
       ├─ Liquidity Ratios
       ├─ Profitability Ratios
       ├─ Leverage Ratios
       ├─ Efficiency Ratios
       └─ → Financial Ratios
    
    7. ANALYSIS
       ├─ Financial Health Scoring
       ├─ Risk Assessment
       ├─ Strength/Weakness Analysis
       ├─ Anomaly Detection
       └─ → Company Analysis
    
    8. OUTPUT
       ├─ CSV with metrics and ratios
       ├─ JSON report with analysis
       ├─ Health score and risk level
       └─ Recommendations
    """)


def demo_modules_overview():
    """Show overview of all modules."""
    
    print("\n" + "="*80)
    print("DEMO 6: Modules Overview")
    print("="*80)
    
    modules = {
        'intelligent_pdf_extractor.py': {
            'purpose': 'Extract financial metrics from PDFs',
            'classes': ['FinancialMetricsExtractor', 'BatchPDFProcessor'],
            'key_methods': ['extract_metrics_from_pdf', 'extract_and_generate_csv'],
        },
        'pattern_learner.py': {
            'purpose': 'Learn metric extraction patterns from training PDFs',
            'classes': ['FinancialMetricsPatternLearner', 'PatternMatchingExtractor'],
            'key_methods': ['learn_from_pdfs', 'get_pattern_summary'],
        },
        'extraction_pipeline.py': {
            'purpose': 'End-to-end extraction pipeline with cleaning and validation',
            'classes': [
                'AutomatedExtractionPipeline',
                'DataValidator',
                'DataCleaner',
                'QualityScorer'
            ],
            'key_methods': ['process_pdf', 'process_batch'],
        },
        'financial_analysis.py': {
            'purpose': 'Financial health analysis and risk assessment',
            'classes': [
                'FinancialHealthAnalyzer',
                'AnomalyDetector',
                'DistressPredictor'
            ],
            'key_methods': ['analyze_company', 'predict'],
        },
        'orchestrator.py': {
            'purpose': 'Unified interface for all functionality',
            'classes': ['FinancialExtractionOrchestrator'],
            'key_methods': [
                'extract_and_analyze_single',
                'extract_and_analyze_batch',
                'generate_comparison_report'
            ],
        },
        'extraction_cli.py': {
            'purpose': 'Command-line interface for all operations',
            'commands': [
                'train',
                'extract-single',
                'batch-extract',
                'analyze',
                'compare'
            ],
        },
    }
    
    for module, info in modules.items():
        print(f"\n📄 {module}")
        print(f"   Purpose: {info['purpose']}")
        if 'classes' in info:
            print(f"   Classes: {', '.join(info['classes'][:2])}{'...' if len(info['classes']) > 2 else ''}")
            print(f"   Methods: {', '.join(info.get('key_methods', [])[:2])}...")
        if 'commands' in info:
            print(f"   Commands: {', '.join(info['commands'])}")


def run_all_demos():
    """Run all demonstrations."""
    
    print("\n" + "="*80)
    print("FINANCIAL METRICS EXTRACTION AND ANALYSIS SYSTEM")
    print("="*80)
    print("\nRunning Comprehensive Demonstration\n")
    
    try:
        # Show architecture
        demo_data_flow()
        
        # Show modules
        demo_modules_overview()
        
        # Run demos
        print("\n" + "="*80)
        print("Running Live Demonstrations")
        print("="*80)
        
        # Demo 1: Training
        demo_training()
        
        # Demo 2: Single extraction
        demo_single_extraction()
        
        # Demo 3: Batch extraction
        demo_batch_extraction()
        
        # Demo 4: Comparison
        demo_comparison()
        
        # Summary
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print("""
✅ All demonstrations completed successfully!

📊 Generated Output:
   - demo_output_single/: Single PDF extraction results
   - demo_output_batch/: Batch processing results
   - demo_output_comparison/: Comparison analysis results
   - Extracted CSV files with all metrics and ratios
   - Analysis JSON reports with financial assessments

🚀 Next Steps:
   1. Review generated CSV files for extracted metrics
   2. Check analysis JSON for financial health scores
   3. Use orchestrator.py for your own PDFs
   4. Integrate with your financial analysis pipeline
        """)
    
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_demos()
