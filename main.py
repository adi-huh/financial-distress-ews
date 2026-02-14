"""
Main entry point for Financial Distress Early Warning System.
Provides command-line interface for analyzing financial data.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Import core modules (flat structure)
from loader import DataLoader
from cleaner import DataCleaner
from ratios import FinancialRatioEngine
from timeseries import TimeSeriesAnalyzer
from zscore import AnomalyDetectionEngine
from score import RiskScoreEngine
from recommend import ConsultingEngine
from charts import ChartGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Financial Distress Early Warning System - Analyze corporate financial health'
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        type=str,
        help='Path to input CSV or Excel file containing financial data'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='results/',
        help='Output directory for results (default: results/)'
    )
    
    parser.add_argument(
        '-c', '--company',
        type=str,
        help='Filter analysis for specific company name'
    )
    
    parser.add_argument(
        '--export-format',
        type=str,
        choices=['csv', 'excel', 'json'],
        default='csv',
        help='Export format for results (default: csv)'
    )
    
    parser.add_argument(
        '--anomaly-method',
        type=str,
        choices=['zscore', 'isolation_forest'],
        default='zscore',
        help='Anomaly detection method (default: zscore)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 70)
    logger.info("Financial Distress Early Warning System - Starting Analysis")
    logger.info("=" * 70)
    
    try:
        # Step 1: Load Data
        logger.info(f"Loading data from: {args.input}")
        loader = DataLoader()
        raw_data = loader.load_file(args.input)
        logger.info(f"✓ Loaded {len(raw_data)} records")
        
        # Filter by company if specified
        if args.company:
            raw_data = raw_data[raw_data['company'] == args.company]
            logger.info(f"✓ Filtered for company: {args.company}")
        
        # Step 2: Preprocess Data
        logger.info("Preprocessing data...")
        cleaner = DataCleaner()
        clean_data = cleaner.clean(raw_data)
        logger.info(f"✓ Data cleaned: {len(clean_data)} valid records")
        
        # Step 3: Calculate Financial Ratios
        logger.info("Calculating financial ratios...")
        ratio_engine = FinancialRatioEngine()
        ratios_df = ratio_engine.calculate_all_ratios(clean_data)
        logger.info(f"✓ Calculated {len(ratios_df.columns)} financial ratios")
        
        # Step 4: Time-Series Analysis
        logger.info("Performing time-series analysis...")
        ts_analyzer = TimeSeriesAnalyzer()
        trends = ts_analyzer.analyze_trends(ratios_df)
        logger.info("✓ Trend analysis completed")
        
        # Step 5: Anomaly Detection
        logger.info(f"Detecting anomalies using {args.anomaly_method}...")
        use_zscore = args.anomaly_method == 'zscore'
        use_if = args.anomaly_method == 'isolation_forest'
        detector = AnomalyDetectionEngine(
            use_zscore=use_zscore,
            use_isolation_forest=use_if,
            zscore_threshold=3.0
        )
        all_anomalies = detector.detect_all_anomalies(ratios_df)
        # Extract the anomalies from the selected method
        anomalies = all_anomalies.get(args.anomaly_method, pd.DataFrame())
        logger.info(f"✓ Detected {len(anomalies)} anomalies")
        
        # Step 6: Calculate Risk Score
        logger.info("Computing composite risk score...")
        risk_engine = RiskScoreEngine()
        risk_results = risk_engine.calculate_risk_score(ratios_df, anomalies)
        logger.info(f"✓ Risk scores calculated for {len(risk_results)} companies")
        
        # Step 7: Generate Recommendations
        logger.info("Generating strategic recommendations...")
        consultant = ConsultingEngine()
        recommendations = consultant.generate_recommendations(
            ratios_df, 
            risk_results
        )
        logger.info(f"✓ Generated {len(recommendations)} recommendations")
        
        # Step 8: Export Results
        logger.info("Exporting results...")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export ratios
        output_file = output_dir / f"financial_ratios.{args.export_format}"
        if args.export_format == 'csv':
            ratios_df.to_csv(output_file, index=False)
        elif args.export_format == 'excel':
            ratios_df.to_excel(output_file, index=False)
        elif args.export_format == 'json':
            ratios_df.to_json(output_file, orient='records', indent=2)
        
        logger.info(f"✓ Results exported to: {output_file}")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        chart_gen = ChartGenerator()
        chart_gen.create_dashboard(
            ratios_df, 
            risk_results, 
            output_dir / "charts"
        )
        logger.info(f"✓ Charts saved to: {output_dir / 'charts'}")
        
        # Print Summary
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Companies Analyzed: {len(risk_results)}")
        print(f"Period: {clean_data['year'].min()} - {clean_data['year'].max()}")
        print(f"Anomalies Detected: {len(anomalies)}")
        
        # Print risk scores for each company
        print("\nRisk Scores by Company:")
        for company, scores in risk_results.items():
            print(f"  {company}: {scores['overall_score']:.2f}/100 ({scores['classification']})")
        
        print("\nTop Recommendations:")
        for i, (company, rec_data) in enumerate(list(recommendations.items())[:3], 1):
            print(f"{i}. {company} ({rec_data['classification']}):")
            if rec_data.get('immediate_actions'):
                print(f"   Immediate: {rec_data['immediate_actions'][0]}")
        print("=" * 70)
        
        logger.info("Analysis completed successfully!")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
