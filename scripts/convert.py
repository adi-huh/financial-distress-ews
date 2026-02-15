#!/usr/bin/env python3
"""
Easy PDF to CSV Converter Wrapper
Simplified interface for converting annual reports to CSV files.
"""

import sys
import os
from pathlib import Path
from pdf_converter import ReportConverter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def show_help():
    """Show help message."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║         PDF Annual Report to CSV Converter - Easy Wrapper         ║
╚═══════════════════════════════════════════════════════════════════╝

USAGE:

  Single PDF:
    python convert.py apple_2024.pdf "Apple Inc"
    
  With custom output:
    python convert.py apple_2024.pdf "Apple Inc" output.csv
    
  Batch processing:
    python convert.py --batch ./reports ./output

EXAMPLES:

  # Convert Apple 10-K
  python convert.py apple_10k_2024.pdf "Apple Inc"
  
  # Convert with custom name
  python convert.py report.pdf "My Company" my_company_2024.csv
  
  # Batch convert all PDFs in folder
  python convert.py --batch ./annual_reports ./converted
  
  # Then analyze
  python main.py -i apple_10k_2024_extracted.csv

REQUIREMENTS:
  pip install pdfplumber PyPDF2

═══════════════════════════════════════════════════════════════════
    """)


def main():
    """Main entry point."""
    
    if len(sys.argv) < 2:
        show_help()
        return 0
    
    # Show help
    if sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        return 0
    
    # Batch mode
    if sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("Error: --batch requires folder path")
            print("Usage: python convert.py --batch <folder> [output_folder]")
            return 1
        
        pdf_folder = sys.argv[2]
        output_folder = sys.argv[3] if len(sys.argv) > 3 else "converted_reports"
        
        print(f"\n📂 Batch converting PDFs from: {pdf_folder}")
        print(f"📁 Output folder: {output_folder}\n")
        
        try:
            converter = ReportConverter()
            converted = converter.batch_convert(pdf_folder, output_folder)
            
            print(f"\n✓ Successfully converted {len(converted)} file(s):")
            for i, csv_file in enumerate(converted, 1):
                print(f"   {i}. {csv_file}")
            
            return 0
        
        except Exception as e:
            print(f"✗ Error: {e}")
            return 1
    
    # Single file mode
    elif len(sys.argv) >= 3:
        pdf_file = sys.argv[1]
        company_name = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        
        # Validate PDF file exists
        if not Path(pdf_file).exists():
            print(f"✗ Error: PDF file not found: {pdf_file}")
            return 1
        
        print(f"\n📄 Converting PDF to CSV")
        print(f"   PDF: {pdf_file}")
        print(f"   Company: {company_name}\n")
        
        try:
            converter = ReportConverter()
            csv_path = converter.convert_pdf_to_csv(pdf_file, company_name, output_file)
            
            print(f"\n✓ Conversion successful!")
            print(f"   CSV: {csv_path}")
            print(f"\n📊 Next: Analyze the CSV with:")
            print(f"   .venv/bin/python main.py -i {csv_path}\n")
            
            return 0
        
        except Exception as e:
            print(f"✗ Error: {e}")
            return 1
    
    else:
        print("Error: Invalid arguments")
        show_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
