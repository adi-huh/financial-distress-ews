"""
Enhanced CLI for PDF Extraction and Financial Analysis

Train on annual reports, extract from new PDFs, generate CSV and analysis.
"""

import click
import sys
import os
import json
import time
import logging
from pathlib import Path
from tabulate import tabulate
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtractorCLI:
    """CLI interface for PDF extraction system."""
    
    def __init__(self):
        self.extractor = None
        self.pipeline = None
        self.patterns = None
    
    def initialize_extractor(self, sample_dir: str = None):
        """Initialize extraction system."""
        try:
            from intelligent_pdf_extractor import FinancialMetricsExtractor
            self.extractor = FinancialMetricsExtractor(sample_pdfs_dir=sample_dir)
            logger.info("Extractor initialized")
        except ImportError:
            logger.error("Could not import FinancialMetricsExtractor")
            sys.exit(1)
    
    def initialize_pipeline(self):
        """Initialize extraction pipeline."""
        try:
            from extraction_pipeline import AutomatedExtractionPipeline
            self.pipeline = AutomatedExtractionPipeline(extractor=self.extractor)
            logger.info("Pipeline initialized")
        except ImportError:
            logger.error("Could not import AutomatedExtractionPipeline")
            sys.exit(1)
    
    def initialize_pattern_learner(self):
        """Initialize pattern learner."""
        try:
            from pattern_learner import FinancialMetricsPatternLearner
            self.pattern_learner = FinancialMetricsPatternLearner()
            logger.info("Pattern learner initialized")
        except ImportError:
            logger.error("Could not import FinancialMetricsPatternLearner")
            sys.exit(1)


@click.group()
def cli():
    """Financial Metrics PDF Extraction System.
    
    Train on annual reports to automatically extract financial metrics from PDFs.
    """
    pass


@cli.command()
@click.option('--pdf-dir', type=click.Path(exists=True), required=True,
              help='Directory containing sample annual report PDFs')
@click.option('--output', type=click.Path(), default='metric_extraction_patterns.json',
              help='Output file for learned patterns')
def train(pdf_dir, output):
    """Learn extraction patterns from sample annual reports."""
    
    click.secho("🎓 Training Pattern Learner...", fg='blue', bold=True)
    
    cli_app = ExtractorCLI()
    cli_app.initialize_pattern_learner()
    
    start_time = time.time()
    
    # Learn patterns
    patterns = cli_app.pattern_learner.learn_from_pdfs(pdf_dir)
    
    # Save patterns
    cli_app.pattern_learner.save_patterns(output)
    
    elapsed = time.time() - start_time
    
    # Display summary
    summary_df = cli_app.pattern_learner.get_pattern_summary()
    
    click.secho("\n✅ Pattern Learning Complete", fg='green', bold=True)
    click.echo(f"\nLearned {len(patterns)} metric patterns from {cli_app.pattern_learner.training_samples} PDFs")
    click.echo(f"Time: {elapsed:.2f}s\n")
    
    click.echo(tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False))
    
    click.secho(f"\n📊 Patterns saved to: {output}", fg='cyan')


@cli.command()
@click.option('--pdf-file', type=click.Path(exists=True), required=True,
              help='PDF file to extract metrics from')
@click.option('--sample-dir', type=click.Path(exists=True),
              help='Directory with sample PDFs for training')
@click.option('--output-csv', type=click.Path(),
              help='Output CSV file path')
@click.option('--output-json', type=click.Path(),
              help='Output JSON report path')
def extract_single(pdf_file, sample_dir, output_csv, output_json):
    """Extract financial metrics from a single PDF."""
    
    click.secho("📄 Extracting Metrics...", fg='blue', bold=True)
    
    cli_app = ExtractorCLI()
    cli_app.initialize_extractor(sample_dir=sample_dir)
    cli_app.initialize_pipeline()
    
    # Set default output paths if not provided
    if not output_csv:
        output_csv = Path(pdf_file).stem + '_extracted.csv'
    if not output_json:
        output_json = Path(pdf_file).stem + '_report.json'
    
    start_time = time.time()
    
    # Process PDF
    result = cli_app.pipeline.process_pdf(pdf_file, output_csv, output_json)
    
    elapsed = time.time() - start_time
    
    # Display results
    click.secho("\n✅ Extraction Complete", fg='green', bold=True)
    click.echo(f"Company: {result.company}")
    click.echo(f"Fiscal Year: {result.fiscal_year}")
    click.echo(f"Quality Score: {result.quality_score:.1f}/100")
    click.echo(f"Metrics Extracted: {len(result.extracted_metrics)}")
    click.echo(f"Ratios Calculated: {len(result.calculated_ratios)}")
    click.echo(f"Time: {elapsed:.2f}s\n")
    
    if result.cleaned_metrics:
        click.echo("📊 Extracted Metrics:")
        metrics_data = [
            [k, f"{v:.2e}" if abs(v) > 1e6 else f"{v:.2f}"]
            for k, v in list(result.cleaned_metrics.items())[:10]
        ]
        click.echo(tabulate(metrics_data, headers=['Metric', 'Value'], tablefmt='grid'))
    
    if result.calculated_ratios:
        click.echo("\n📈 Calculated Ratios:")
        ratios_data = [
            [k, f"{v:.4f}"]
            for k, v in list(result.calculated_ratios.items())[:10]
        ]
        click.echo(tabulate(ratios_data, headers=['Ratio', 'Value'], tablefmt='grid'))
    
    if result.errors:
        click.secho("\n⚠️ Warnings:", fg='yellow')
        for error in result.errors[:5]:
            click.echo(f"  • {error}")
    
    click.secho(f"\n📁 CSV: {output_csv}", fg='cyan')
    click.secho(f"📋 Report: {output_json}", fg='cyan')


@cli.command()
@click.option('--pdf-dir', type=click.Path(exists=True), required=True,
              help='Directory containing PDF files to process')
@click.option('--sample-dir', type=click.Path(exists=True),
              help='Directory with sample PDFs for training')
@click.option('--output-dir', type=click.Path(), default='extracted_data',
              help='Output directory for extracted data')
def batch_extract(pdf_dir, sample_dir, output_dir):
    """Extract metrics from batch of PDFs."""
    
    click.secho("📦 Batch Extraction Started...", fg='blue', bold=True)
    
    cli_app = ExtractorCLI()
    cli_app.initialize_extractor(sample_dir=sample_dir)
    cli_app.initialize_pipeline()
    
    start_time = time.time()
    
    # Process batch
    combined_df, results = cli_app.pipeline.process_batch(pdf_dir, output_dir)
    
    elapsed = time.time() - start_time
    
    # Statistics
    successful = sum(1 for r in results if r.quality_score > 0)
    avg_quality = sum(r.quality_score for r in results) / max(len(results), 1)
    
    click.secho("\n✅ Batch Processing Complete", fg='green', bold=True)
    click.echo(f"Processed: {len(results)} PDFs")
    click.echo(f"Successful: {successful}/{len(results)}")
    click.echo(f"Avg Quality: {avg_quality:.1f}/100")
    click.echo(f"Time: {elapsed:.2f}s\n")
    
    if not combined_df.empty:
        click.echo("📊 Data Summary:")
        click.echo(f"Shape: {combined_df.shape}")
        click.echo(f"Columns: {len(combined_df.columns)}")
        
        # Show sample
        click.echo("\nSample Companies:")
        sample_cols = ['company', 'fiscal_year', 'quality_score']
        sample_cols = [c for c in sample_cols if c in combined_df.columns]
        if sample_cols:
            sample_data = combined_df[sample_cols].head(5).values.tolist()
            click.echo(tabulate(sample_data, headers=sample_cols, tablefmt='grid'))
    
    click.secho(f"\n📁 Output Directory: {output_dir}", fg='cyan')
    click.secho(f"📄 Combined CSV: {Path(output_dir) / 'all_companies_combined.csv'}", fg='cyan')


@cli.command()
@click.option('--csv-file', type=click.Path(exists=True), required=True,
              help='CSV file with extracted metrics')
@click.option('--output', type=click.Path(), default='analysis_report.json',
              help='Output analysis report')
def analyze(csv_file, output):
    """Analyze extracted financial metrics."""
    
    click.secho("📊 Analyzing Financial Metrics...", fg='blue', bold=True)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Calculate statistics
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    analysis = {
        'total_companies': len(df),
        'metrics_analyzed': len(numeric_cols),
        'statistics': {}
    }
    
    for col in numeric_cols:
        if col not in ['fiscal_year']:
            analysis['statistics'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
            }
    
    # Save report
    with open(output, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    click.secho("\n✅ Analysis Complete", fg='green', bold=True)
    click.echo(f"Companies Analyzed: {analysis['total_companies']}")
    click.echo(f"Metrics: {analysis['metrics_analyzed']}\n")
    
    # Show top metrics
    if analysis['statistics']:
        click.echo("📈 Top Metrics (by value range):")
        metrics_range = [
            [k, f"{v['min']:.2e}", f"{v['max']:.2e}", f"{v['mean']:.2e}"]
            for k, v in list(analysis['statistics'].items())[:10]
        ]
        click.echo(tabulate(
            metrics_range,
            headers=['Metric', 'Min', 'Max', 'Mean'],
            tablefmt='grid'
        ))
    
    click.secho(f"\n📋 Report: {output}", fg='cyan')


@cli.command()
@click.option('--csv-file', type=click.Path(exists=True), required=True,
              help='CSV file with extracted metrics')
@click.option('--metric', required=True,
              help='Metric to visualize (e.g., net_income, roe)')
def compare(csv_file, metric):
    """Compare metric across companies."""
    
    df = pd.read_csv(csv_file)
    
    if metric not in df.columns:
        click.secho(f"❌ Metric '{metric}' not found in CSV", fg='red')
        click.echo(f"Available metrics: {', '.join(df.columns)}")
        sys.exit(1)
    
    # Sort by metric
    df_sorted = df.sort_values(metric, ascending=False)
    
    click.secho(f"\n📊 {metric.upper()} Comparison", fg='blue', bold=True)
    
    # Display top and bottom
    click.echo("\n🔝 Top 5:")
    data = df_sorted.head(5)[['company', metric]].values.tolist()
    click.echo(tabulate(data, headers=['Company', metric], tablefmt='grid'))
    
    click.echo("\n🔻 Bottom 5:")
    data = df_sorted.tail(5)[['company', metric]].values.tolist()
    click.echo(tabulate(data, headers=['Company', metric], tablefmt='grid'))
    
    click.echo(f"\n📊 Statistics:")
    click.echo(f"  Mean: {df[metric].mean():.2e}")
    click.echo(f"  Median: {df[metric].median():.2e}")
    click.echo(f"  Std Dev: {df[metric].std():.2e}")


@cli.command()
def info():
    """Show system information."""
    
    click.secho("\n📋 Financial Metrics Extraction System", fg='blue', bold=True)
    
    info_data = [
        ['Version', '1.0'],
        ['Python', f"{sys.version.split()[0]}"],
        ['Location', os.path.dirname(__file__)],
        ['Status', '✅ Ready'],
    ]
    
    click.echo(tabulate(info_data, tablefmt='grid'))
    
    click.secho("\n📚 Available Commands:", fg='cyan')
    
    commands = [
        ['train', 'Learn patterns from sample annual reports'],
        ['extract-single', 'Extract metrics from one PDF'],
        ['batch-extract', 'Extract metrics from multiple PDFs'],
        ['analyze', 'Analyze extracted metrics'],
        ['compare', 'Compare metric across companies'],
    ]
    
    click.echo(tabulate(commands, headers=['Command', 'Description'], tablefmt='grid'))


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        click.secho(f"\n❌ Error: {e}", fg='red', bold=True)
        logger.exception("CLI Error")
        sys.exit(1)
