#!/usr/bin/env python3
"""
Quick Start Entry Point for Financial Metrics Extraction System

Usage:
    python quickstart.py --help
    python quickstart.py extract --pdf <path>
    python quickstart.py batch --dir <path>
    python quickstart.py demo
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Financial Metrics Extraction System - Quick Start',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quickstart.py extract --pdf report.pdf
  python quickstart.py batch --dir ./reports
  python quickstart.py train --dir ./annual_reports_2024
  python quickstart.py demo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract from single PDF')
    extract_parser.add_argument('--pdf', required=True, help='Path to PDF file')
    extract_parser.add_argument('--output', default='output', help='Output directory')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch extract from directory')
    batch_parser.add_argument('--dir', required=True, help='Directory with PDFs')
    batch_parser.add_argument('--output', default='output', help='Output directory')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train pattern learner')
    train_parser.add_argument('--dir', required=True, help='Directory with sample PDFs')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch Streamlit dashboard')
    dashboard_parser.add_argument('--mode', choices=['simple', 'full'], default='simple')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        if args.command == 'extract':
            extract_single(args.pdf, args.output)
        
        elif args.command == 'batch':
            extract_batch(args.dir, args.output)
        
        elif args.command == 'train':
            train_patterns(args.dir)
        
        elif args.command == 'demo':
            run_demo()
        
        elif args.command == 'dashboard':
            launch_dashboard(args.mode)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def extract_single(pdf_path: str, output_dir: str):
    """Extract from single PDF."""
    
    from orchestrator import FinancialExtractionOrchestrator
    
    print(f"\n📄 Extracting from: {pdf_path}")
    
    sample_dir = '/Users/adi/Documents/financial-distress-ews/annual_reports_2024'
    orchestrator = FinancialExtractionOrchestrator(sample_pdf_dir=sample_dir)
    
    result = orchestrator.extract_and_analyze_single(pdf_path, output_dir)
    
    print(f"\n✅ Extraction Complete")
    print(f"   Company: {result['company']}")
    print(f"   Quality Score: {result['quality_score']:.1f}/100")
    
    if result['analysis']:
        print(f"   Health Score: {result['analysis']['financial_health_score']:.1f}")
        print(f"   Risk Level: {result['analysis']['distress_risk_level']}")
    
    print(f"\n📁 Output: {output_dir}")


def extract_batch(pdf_dir: str, output_dir: str):
    """Extract from batch of PDFs."""
    
    from orchestrator import FinancialExtractionOrchestrator
    
    print(f"\n📦 Batch extracting from: {pdf_dir}")
    
    orchestrator = FinancialExtractionOrchestrator(sample_pdf_dir=pdf_dir)
    
    result = orchestrator.extract_and_analyze_batch(pdf_dir, output_dir)
    
    summary = result['summary']
    
    print(f"\n✅ Batch Processing Complete")
    print(f"   Total PDFs: {summary['total_pdfs_processed']}")
    print(f"   Successful: {summary['successful_extractions']}")
    print(f"   Avg Quality: {summary['avg_quality_score']:.1f}/100")
    print(f"\n   Risk Distribution:")
    for risk_level, count in summary['risk_distribution'].items():
        print(f"      {risk_level}: {count}")
    
    print(f"\n📁 Output: {output_dir}")


def train_patterns(pdf_dir: str):
    """Train pattern learner."""
    
    from orchestrator import FinancialExtractionOrchestrator
    
    print(f"\n🎓 Training on: {pdf_dir}")
    
    orchestrator = FinancialExtractionOrchestrator()
    
    result = orchestrator.train_on_samples(pdf_dir)
    
    print(f"\n✅ Training Complete")
    print(f"   Patterns Learned: {result['patterns_learned']}")
    print(f"   Training Samples: {result['training_samples']}")
    print(f"   File: {result['patterns_file']}")


def run_demo():
    """Run demonstration."""
    
    print("\n" + "="*80)
    print("Running Comprehensive Demonstration")
    print("="*80)
    
    from demo import run_all_demos
    
    run_all_demos()


def launch_dashboard(mode: str = 'simple'):
    """Launch Streamlit dashboard."""
    
    import subprocess
    
    if mode == 'simple':
        app_file = 'app_simple.py'
    else:
        app_file = 'app.py'
    
    print(f"\n🚀 Launching {mode} dashboard...")
    print(f"   App: {app_file}")
    print(f"   URL: http://localhost:8501")
    
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', app_file,
        '--logger.level=warning'
    ])


if __name__ == '__main__':
    sys.exit(main())
