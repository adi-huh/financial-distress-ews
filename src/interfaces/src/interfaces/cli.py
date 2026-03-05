"""
Unified CLI for Financial Distress Early Warning System
Handles CSV analysis and PDF extraction via command-line.
"""

import click
import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.loader import DataLoader
from src.core.cleaner import DataCleaner
from src.core.ratios import FinancialRatioEngine
from src.core.score import RiskScoreEngine
from src.core.recommend import ConsultingEngine
from src.core.charts import ChartGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Financial Distress Early Warning System CLI."""
    pass


@cli.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input CSV or Excel file')
@click.option('-o', '--output', 'output_dir', default='results/',
              help='Output directory for results')
@click.option('-c', '--company', help='Filter by specific company')
@click.option('--format', 'export_format', type=click.Choice(['csv', 'excel', 'json']),
              default='csv', help='Export format')
@click.option('--verbose', is_flag=True, help='Verbose output')
def analyze(input_file, output_dir, company, export_format, verbose):
    """Analyze financial data from CSV/Excel file."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    click.echo("📊 Financial Distress Analysis Starting...\n")
    
    try:
        # Load data
        click.echo(f"Loading data from {input_file}...")
        loader = DataLoader()
        data = loader.load_file(input_file)
        click.echo(f"✅ Loaded {len(data)} records")
        
        # Filter by company if specified
        if company:
            data = data[data['company'] == company]
            click.echo(f"✅ Filtered for company: {company}")
        
        # Clean data
        click.echo("\nCleaning data...")
        cleaner = DataCleaner()
        clean_data = cleaner.clean(data)
        click.echo(f"✅ {len(clean_data)} valid records")
        
        # Calculate ratios
        click.echo("\nCalculating financial ratios...")
        ratio_engine = FinancialRatioEngine()
        ratios_df = ratio_engine.calculate_all_ratios(clean_data)
        click.echo(f"✅ Calculated {len(ratios_df.columns)} metrics")
        
        # Calculate risk scores
        click.echo("\nCalculating risk scores...")
        risk_engine = RiskScoreEngine()
        risk_results = risk_engine.calculate_risk_score(ratios_df, pd.DataFrame())
        click.echo(f"✅ Risk scores for {len(risk_results)} companies")
        
        # Generate recommendations
        click.echo("\nGenerating recommendations...")
        consultant = ConsultingEngine()
        recommendations = consultant.generate_recommendations(ratios_df, risk_results)
        
        # Export results
        click.echo(f"\nExporting to {output_dir}...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_file = Path(output_dir) / f"financial_ratios.{export_format}"
        if export_format == 'csv':
            ratios_df.to_csv(output_file, index=False)
        elif export_format == 'excel':
            ratios_df.to_excel(output_file, index=False)
        elif export_format == 'json':
            ratios_df.to_json(output_file, orient='records', indent=2)
        
        click.echo(f"✅ Results saved to: {output_file}")
        
        # Generate visualizations
        click.echo("\nGenerating visualizations...")
        chart_gen = ChartGenerator()
        chart_gen.create_dashboard(ratios_df, risk_results, Path(output_dir) / "charts")
        click.echo(f"✅ Charts saved to: {output_dir}/charts")
        
        # Print summary
        click.echo("\n" + "="*70)
        click.echo("ANALYSIS SUMMARY")
        click.echo("="*70)
        
        for company_name, result in risk_results.items():
            score = result['overall_score']
            classification = result['classification']
            
            if classification == 'Stable':
                color = 'green'
            elif classification == 'Caution':
                color = 'yellow'
            else:
                color = 'red'
            
            click.secho(f"{company_name}: {score:.1f}/100 ({classification})", fg=color)
        
        click.echo("="*70)
        click.secho("\n✅ Analysis Complete!", fg='green', bold=True)
        
        return 0
    
    except Exception as e:
        click.secho(f"\n❌ Error: {str(e)}", fg='red', bold=True)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


@cli.command()
@click.option('-i', '--input', 'pdf_file', required=True, type=click.Path(exists=True),
              help='Input PDF file')
@click.option('-o', '--output', 'output_dir', default='extraction_output/',
              help='Output directory')
def extract_pdf(pdf_file, output_dir):
    """Extract financial metrics from PDF annual report."""
    
    click.echo("📄 PDF Extraction Starting...\n")
    
    try:
        from src.pdf_extraction.orchestrator import FinancialExtractionOrchestrator
        
        orchestrator = FinancialExtractionOrchestrator()
        
        click.echo(f"Processing {pdf_file}...")
        result = orchestrator.extract_and_analyze_single(pdf_file, output_dir)
        
        click.echo("\n" + "="*70)
        click.echo("EXTRACTION RESULTS")
        click.echo("="*70)
        click.echo(f"Company: {result['company']}")
        click.echo(f"Fiscal Year: {result['fiscal_year']}")
        click.echo(f"Quality Score: {result['quality_score']:.1f}/100")
        click.echo(f"Metrics Extracted: {result['metrics_extracted']}")
        
        if result['analysis']:
            click.echo(f"\nHealth Score: {result['analysis']['financial_health_score']:.1f}/100")
            click.echo(f"Risk Level: {result['analysis']['distress_risk_level']}")
        
        click.echo("="*70)
        click.secho("\n✅ Extraction Complete!", fg='green', bold=True)
        click.echo(f"📁 Output: {output_dir}")
        
        return 0
    
    except ImportError:
        click.secho("\n❌ PDF extraction module not available", fg='red', bold=True)
        return 1
    except Exception as e:
        click.secho(f"\n❌ Error: {str(e)}", fg='red', bold=True)
        return 1


@cli.command()
@click.option('-i', '--input', 'pdf_dir', required=True, type=click.Path(exists=True),
              help='Directory with PDF files')
@click.option('-o', '--output', 'output_dir', default='batch_output/',
              help='Output directory')
def batch_extract(pdf_dir, output_dir):
    """Batch extract from multiple PDFs."""
    
    click.echo("📦 Batch Extraction Starting...\n")
    
    try:
        from src.pdf_extraction.orchestrator import FinancialExtractionOrchestrator
        
        orchestrator = FinancialExtractionOrchestrator()
        
        click.echo(f"Processing PDFs from {pdf_dir}...")
        result = orchestrator.extract_and_analyze_batch(pdf_dir, output_dir)
        
        summary = result['summary']
        
        click.echo("\n" + "="*70)
        click.echo("BATCH PROCESSING SUMMARY")
        click.echo("="*70)
        click.echo(f"Total PDFs: {summary['total_pdfs_processed']}")
        click.echo(f"Successful: {summary['successful_extractions']}")
        click.echo(f"Average Quality: {summary['avg_quality_score']:.1f}/100")
        click.echo("="*70)
        click.secho("\n✅ Batch Processing Complete!", fg='green', bold=True)
        click.echo(f"📁 Output: {output_dir}")
        
        return 0
    
    except ImportError:
        click.secho("\n❌ PDF extraction module not available", fg='red', bold=True)
        return 1
    except Exception as e:
        click.secho(f"\n❌ Error: {str(e)}", fg='red', bold=True)
        return 1


@cli.command()
def version():
    """Show version information."""
    click.echo("Financial Distress Early Warning System")
    click.echo("Version: 1.0.0")
    click.echo("Built with ❤️ by Adi")


if __name__ == '__main__':
    sys.exit(cli())