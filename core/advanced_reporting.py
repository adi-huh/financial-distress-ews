"""
Advanced Reporting Engine for Financial Distress Early Warning System
Day 17: Comprehensive report generation with PDF export, scheduling, templates, and versioning
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum
import hashlib
import pandas as pd

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from core.risk_prediction import EnsembleRiskPredictor, FeatureEngineer


logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Report type enumeration"""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    COMPARATIVE = "comparative"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """Report output format"""
    JSON = "json"
    PDF = "pdf"
    HTML = "html"
    CSV = "csv"


class ReportTemplate:
    """Template for report generation"""

    def __init__(self, name: str, report_type: ReportType, sections: List[str]):
        self.name = name
        self.report_type = report_type
        self.sections = sections
        self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict:
        """Convert template to dictionary"""
        return {
            'name': self.name,
            'report_type': self.report_type.value,
            'sections': self.sections,
            'created_at': self.created_at.isoformat()
        }


class ReportVersion:
    """Version control for reports"""

    def __init__(self, report_id: str, version: int, content: Dict, changes: str = ""):
        self.report_id = report_id
        self.version = version
        self.content = content
        self.changes = changes
        self.created_at = datetime.now(timezone.utc)
        self.hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate hash of content"""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        """Convert version to dictionary"""
        return {
            'report_id': self.report_id,
            'version': self.version,
            'hash': self.hash,
            'created_at': self.created_at.isoformat(),
            'changes': self.changes
        }


class ReportCache:
    """Cache for generated reports"""

    def __init__(self, max_age_hours: int = 24):
        self.cache = {}
        self.max_age_hours = max_age_hours

    def get(self, key: str) -> Optional[Dict]:
        """Retrieve cached report"""
        if key not in self.cache:
            return None

        report, timestamp = self.cache[key]
        age = (datetime.now(timezone.utc) - timestamp).total_seconds() / 3600

        if age > self.max_age_hours:
            del self.cache[key]
            return None

        return report

    def set(self, key: str, report: Dict) -> None:
        """Cache a report"""
        self.cache[key] = (report, datetime.now(timezone.utc))

    def clear_expired(self) -> int:
        """Clear expired reports"""
        current_time = datetime.now(timezone.utc)
        expired_keys = []

        for key, (_, timestamp) in self.cache.items():
            age = (current_time - timestamp).total_seconds() / 3600
            if age > self.max_age_hours:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)


class ExecutiveSummaryGenerator:
    """Generate executive summary reports"""

    def __init__(self):
        self.risk_predictor = EnsembleRiskPredictor()

    def generate(self, financial_data: Dict, company_name: str = "Company") -> Dict:
        """Generate executive summary"""
        report = {
            'report_type': 'executive_summary',
            'company_name': company_name,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'sections': {}
        }

        # Overview
        report['sections']['overview'] = self._generate_overview(financial_data)

        # Key metrics
        report['sections']['key_metrics'] = self._generate_key_metrics(financial_data)

        # Risk assessment
        report['sections']['risk_assessment'] = self._generate_risk_assessment(financial_data)

        # Recommendations
        report['sections']['recommendations'] = self._generate_recommendations(financial_data)

        return report

    def _generate_overview(self, financial_data: Dict) -> Dict:
        """Generate overview section"""
        return {
            'company_name': financial_data.get('company_id', 'Unknown'),
            'total_assets': financial_data.get('total_assets', 0),
            'total_liabilities': financial_data.get('total_liabilities', 0),
            'equity': financial_data.get('equity', financial_data.get('total_assets', 0) - financial_data.get('total_liabilities', 0)),
            'revenue': financial_data.get('revenue', 0)
        }

    def _generate_key_metrics(self, financial_data: Dict) -> Dict:
        """Generate key metrics section"""
        # Calculate ratios manually to avoid DataFrame conversion issues
        try:
            metrics = self._calculate_ratios_from_dict(financial_data)
        except Exception as e:
            logger.warning(f"Error calculating ratios: {str(e)}")
            metrics = {}

        return metrics

    def _calculate_ratios_from_dict(self, data: Dict) -> Dict:
        """Calculate financial ratios from dictionary"""
        metrics = {}

        # Liquidity ratios
        if data.get('current_assets') and data.get('current_liabilities'):
            metrics['current_ratio'] = data['current_assets'] / data['current_liabilities']
            if data.get('inventory'):
                metrics['quick_ratio'] = (data['current_assets'] - data['inventory']) / data['current_liabilities']

        # Profitability ratios
        if data.get('net_income'):
            if data.get('total_assets') and data['total_assets'] != 0:
                metrics['roa'] = data['net_income'] / data['total_assets']
            if data.get('equity') and data['equity'] != 0:
                metrics['roe'] = data['net_income'] / data['equity']
            if data.get('revenue') and data['revenue'] != 0:
                metrics['net_profit_margin'] = data['net_income'] / data['revenue']

        # Solvency ratios
        if data.get('total_liabilities') and data.get('equity'):
            if data['equity'] != 0:
                metrics['debt_to_equity'] = data['total_liabilities'] / data['equity']
        if data.get('ebit') and data.get('interest_expense'):
            if data['interest_expense'] != 0:
                metrics['interest_coverage'] = data['ebit'] / data['interest_expense']

        # Efficiency ratios
        if data.get('revenue') and data.get('total_assets'):
            if data['total_assets'] != 0:
                metrics['asset_turnover'] = data['revenue'] / data['total_assets']

        return metrics

    def _generate_risk_assessment(self, financial_data: Dict) -> Dict:
        """Generate risk assessment section"""
        prediction = self.risk_predictor.predict_comprehensive_risk(financial_data)

        return {
            'overall_risk_level': prediction.get('overall_risk_level', 'unknown'),
            'overall_risk_score': prediction.get('overall_risk_score', 0),
            'predictions': prediction.get('predictions', {})
        }

    def _generate_recommendations(self, financial_data: Dict) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        prediction = self.risk_predictor.predict_comprehensive_risk(financial_data)
        risk_level = prediction.get('overall_risk_level', 'low')

        if risk_level == 'high':
            recommendations.append("Urgent: Implement immediate financial recovery plan")
            recommendations.append("Review operating costs and reduce expenses")
            recommendations.append("Explore debt restructuring opportunities")
            recommendations.append("Increase focus on revenue generation")
        elif risk_level == 'medium':
            recommendations.append("Monitor financial metrics closely")
            recommendations.append("Improve operational efficiency")
            recommendations.append("Consider strategic partnerships")
        else:
            recommendations.append("Maintain current financial management practices")
            recommendations.append("Monitor industry trends")

        return recommendations


class DetailedAnalysisGenerator:
    """Generate detailed analysis reports"""

    def __init__(self):
        self.risk_predictor = EnsembleRiskPredictor()

    def generate(self, financial_data: Dict, company_name: str = "Company") -> Dict:
        """Generate detailed analysis"""
        report = {
            'report_type': 'detailed_analysis',
            'company_name': company_name,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'sections': {}
        }

        report['sections']['financial_overview'] = self._financial_overview(financial_data)
        report['sections']['ratio_analysis'] = self._ratio_analysis(financial_data)
        report['sections']['trend_analysis'] = self._trend_analysis(financial_data)
        report['sections']['risk_factors'] = self._risk_factors(financial_data)
        report['sections']['stress_testing'] = self._stress_testing(financial_data)

        return report

    def _financial_overview(self, financial_data: Dict) -> Dict:
        """Financial overview section"""
        return {
            'balance_sheet': {
                'total_assets': financial_data.get('total_assets', 0),
                'current_assets': financial_data.get('current_assets', 0),
                'total_liabilities': financial_data.get('total_liabilities', 0),
                'current_liabilities': financial_data.get('current_liabilities', 0),
                'equity': financial_data.get('equity', 0)
            },
            'income_statement': {
                'revenue': financial_data.get('revenue', 0),
                'net_income': financial_data.get('net_income', 0),
                'ebit': financial_data.get('ebit', 0),
                'gross_profit': financial_data.get('gross_profit', 0)
            }
        }

    def _ratio_analysis(self, financial_data: Dict) -> Dict:
        """Ratio analysis section"""
        try:
            all_ratios = self._calculate_ratios_from_dict(financial_data)
        except Exception as e:
            logger.warning(f"Error calculating ratios: {str(e)}")
            all_ratios = {}
        return all_ratios

    def _calculate_ratios_from_dict(self, data: Dict) -> Dict:
        """Calculate financial ratios from dictionary"""
        metrics = {}

        # Liquidity ratios
        if data.get('current_assets') and data.get('current_liabilities'):
            metrics['current_ratio'] = data['current_assets'] / data['current_liabilities']
            if data.get('inventory'):
                metrics['quick_ratio'] = (data['current_assets'] - data['inventory']) / data['current_liabilities']

        # Profitability ratios
        if data.get('net_income'):
            if data.get('total_assets') and data['total_assets'] != 0:
                metrics['roa'] = data['net_income'] / data['total_assets']
            if data.get('equity') and data['equity'] != 0:
                metrics['roe'] = data['net_income'] / data['equity']
            if data.get('revenue') and data['revenue'] != 0:
                metrics['net_profit_margin'] = data['net_income'] / data['revenue']

        # Solvency ratios
        if data.get('total_liabilities') and data.get('equity'):
            if data['equity'] != 0:
                metrics['debt_to_equity'] = data['total_liabilities'] / data['equity']
        if data.get('ebit') and data.get('interest_expense'):
            if data['interest_expense'] != 0:
                metrics['interest_coverage'] = data['ebit'] / data['interest_expense']

        # Efficiency ratios
        if data.get('revenue') and data.get('total_assets'):
            if data['total_assets'] != 0:
                metrics['asset_turnover'] = data['revenue'] / data['total_assets']

        return metrics

    def _trend_analysis(self, financial_data: Dict) -> Dict:
        """Trend analysis section"""
        return {
            'revenue_growth': financial_data.get('revenue_growth', 0),
            'earnings_growth': financial_data.get('earnings_growth', 0),
            'asset_growth': 0,  # Would need historical data
            'trend_direction': 'stable'  # Would be calculated from historical data
        }

    def _risk_factors(self, financial_data: Dict) -> Dict:
        """Risk factors section"""
        prediction = self.risk_predictor.predict_comprehensive_risk(financial_data)
        return {
            'bankruptcy_risk': prediction.get('predictions', {}).get('bankruptcy', {}),
            'default_risk': prediction.get('predictions', {}).get('default', {}),
            'stress_level': prediction.get('predictions', {}).get('stress', {}),
            'key_risk_indicators': self._calculate_kris(financial_data)
        }

    def _calculate_kris(self, financial_data: Dict) -> List[Dict]:
        """Calculate key risk indicators"""
        kris = []
        
        try:
            ratios = self._calculate_ratios_from_dict(financial_data)
        except Exception as e:
            logger.warning(f"Error calculating KRIs: {str(e)}")
            ratios = {}

        debt_to_equity = ratios.get('debt_to_equity', 0)
        if debt_to_equity > 2.0:
            kris.append({'indicator': 'High Leverage', 'value': debt_to_equity, 'severity': 'high'})

        current_ratio = ratios.get('current_ratio', 0)
        if current_ratio < 1.0:
            kris.append({'indicator': 'Liquidity Risk', 'value': current_ratio, 'severity': 'high'})

        roe = ratios.get('roe', 0)
        if roe < 0:
            kris.append({'indicator': 'Negative Returns', 'value': roe, 'severity': 'high'})

        return kris

    def _stress_testing(self, financial_data: Dict) -> Dict:
        """Stress testing scenarios"""
        base_revenue = financial_data.get('revenue', 0)

        scenarios = {
            'base_case': {'revenue_change': 0, 'description': 'Current trajectory'},
            'pessimistic': {'revenue_change': -20, 'description': '20% revenue decline'},
            'optimistic': {'revenue_change': 20, 'description': '20% revenue growth'},
            'severe_stress': {'revenue_change': -50, 'description': '50% revenue decline'}
        }

        results = {}
        for scenario_name, scenario in scenarios.items():
            stressed_revenue = base_revenue * (1 + scenario['revenue_change'] / 100)
            stressed_data = financial_data.copy()
            stressed_data['revenue'] = stressed_revenue

            prediction = self.risk_predictor.predict_comprehensive_risk(stressed_data)
            results[scenario_name] = {
                'description': scenario['description'],
                'revenue': stressed_revenue,
                'risk_level': prediction.get('overall_risk_level')
            }

        return results


class CustomReportBuilder:
    """Build custom reports from selected sections"""

    def __init__(self):
        self.exec_gen = ExecutiveSummaryGenerator()
        self.detail_gen = DetailedAnalysisGenerator()

    def build_custom_report(self, financial_data: Dict, sections: List[str], company_name: str = "Company") -> Dict:
        """Build custom report with selected sections"""
        report = {
            'report_type': 'custom',
            'company_name': company_name,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'sections': {}
        }

        exec_summary = self.exec_gen.generate(financial_data, company_name) if 'executive_summary' in sections else {}
        detailed = self.detail_gen.generate(financial_data, company_name) if 'detailed_analysis' in sections else {}

        if 'overview' in sections and exec_summary:
            report['sections']['overview'] = exec_summary['sections'].get('overview', {})

        if 'key_metrics' in sections and exec_summary:
            report['sections']['key_metrics'] = exec_summary['sections'].get('key_metrics', {})

        if 'risk_assessment' in sections and exec_summary:
            report['sections']['risk_assessment'] = exec_summary['sections'].get('risk_assessment', {})

        if 'financial_analysis' in sections and detailed:
            report['sections']['financial_analysis'] = detailed['sections'].get('financial_overview', {})

        if 'ratio_analysis' in sections and detailed:
            report['sections']['ratio_analysis'] = detailed['sections'].get('ratio_analysis', {})

        if 'stress_testing' in sections and detailed:
            report['sections']['stress_testing'] = detailed['sections'].get('stress_testing', {})

        return report


class PDFReportExporter:
    """Export reports to PDF format"""

    def __init__(self):
        if not HAS_REPORTLAB:
            logger.warning("ReportLab not installed. PDF export will be simulated.")

    def export_to_pdf(self, report: Dict, output_path: str) -> bool:
        """Export report to PDF"""
        if not HAS_REPORTLAB:
            logger.warning(f"PDF export simulated. Would create: {output_path}")
            return self._simulate_pdf_export(report, output_path)

        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f4788'),
                spaceAfter=30,
                alignment=1
            )
            elements.append(Paragraph(f"Financial Analysis Report", title_style))
            elements.append(Spacer(1, 0.3 * inch))

            # Company name
            company_style = ParagraphStyle('CompanyName', parent=styles['Heading2'], fontSize=16)
            elements.append(Paragraph(f"Company: {report.get('company_name', 'Unknown')}", company_style))
            elements.append(Spacer(1, 0.2 * inch))

            # Generated date
            date_style = ParagraphStyle('DateStyle', parent=styles['Normal'], fontSize=10, textColor=colors.grey)
            elements.append(Paragraph(f"Generated: {report.get('generated_at', 'N/A')}", date_style))
            elements.append(Spacer(1, 0.3 * inch))

            # Sections
            for section_name, section_content in report.get('sections', {}).items():
                elements.append(Paragraph(section_name.replace('_', ' ').title(), styles['Heading2']))
                elements.append(Spacer(1, 0.1 * inch))

                if isinstance(section_content, dict):
                    for key, value in section_content.items():
                        elements.append(Paragraph(f"<b>{key}:</b> {str(value)}", styles['Normal']))
                elif isinstance(section_content, list):
                    for item in section_content:
                        elements.append(Paragraph(f"• {str(item)}", styles['Normal']))

                elements.append(Spacer(1, 0.2 * inch))

            doc.build(elements)
            logger.info(f"PDF report exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting PDF: {str(e)}")
            return False

    def _simulate_pdf_export(self, report: Dict, output_path: str) -> bool:
        """Simulate PDF export (when ReportLab not available)"""
        try:
            metadata = {
                'company': report.get('company_name'),
                'generated_at': report.get('generated_at'),
                'sections_count': len(report.get('sections', {}))
            }
            logger.info(f"PDF export simulated for {output_path}: {metadata}")
            return True
        except Exception as e:
            logger.error(f"Error in PDF export simulation: {str(e)}")
            return False


class HTMLReportExporter:
    """Export reports to HTML format"""

    def export_to_html(self, report: Dict, output_path: str) -> bool:
        """Export report to HTML"""
        try:
            html_content = self._generate_html(report)

            with open(output_path, 'w') as f:
                f.write(html_content)

            logger.info(f"HTML report exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting HTML: {str(e)}")
            return False

    def _generate_html(self, report: Dict) -> str:
        """Generate HTML content"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1f4788; border-bottom: 3px solid #1f4788; padding-bottom: 10px; }}
        h2 {{ color: #2c5aa0; margin-top: 30px; }}
        .report-metadata {{ background-color: #f0f0f0; padding: 15px; border-radius: 4px; margin-bottom: 20px; }}
        .section {{ margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        table, th, td {{ border: 1px solid #ddd; }}
        th {{ background-color: #1f4788; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 10px; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ color: #666; font-size: 12px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1f4788; }}
        .risk-high {{ color: #d32f2f; }}
        .risk-medium {{ color: #f57c00; }}
        .risk-low {{ color: #388e3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Financial Analysis Report</h1>
        <div class="report-metadata">
            <strong>Company:</strong> {report.get('company_name', 'Unknown')}<br>
            <strong>Generated:</strong> {report.get('generated_at', 'N/A')}<br>
            <strong>Report Type:</strong> {report.get('report_type', 'N/A').replace('_', ' ').title()}
        </div>
"""

        for section_name, section_content in report.get('sections', {}).items():
            html += f'<div class="section"><h2>{section_name.replace("_", " ").title()}</h2>'

            if isinstance(section_content, dict):
                for key, value in section_content.items():
                    if isinstance(value, dict):
                        html += f'<h3>{key}</h3><table><tr><td>{self._dict_to_html_table(value)}</td></tr></table>'
                    else:
                        html += f'<div class="metric"><div class="metric-label">{key}</div><div class="metric-value">{str(value)}</div></div>'

            elif isinstance(section_content, list):
                html += '<ul>'
                for item in section_content:
                    html += f'<li>{str(item)}</li>'
                html += '</ul>'

            html += '</div>'

        html += '</div></body></html>'
        return html

    def _dict_to_html_table(self, data: Dict) -> str:
        """Convert dict to HTML table rows"""
        rows = ''
        for key, value in data.items():
            rows += f'<tr><td><strong>{key}</strong></td><td>{str(value)}</td></tr>'
        return rows


class CSVReportExporter:
    """Export reports to CSV format"""

    def export_to_csv(self, report: Dict, output_path: str) -> bool:
        """Export report to CSV"""
        try:
            csv_content = self._generate_csv(report)

            with open(output_path, 'w') as f:
                f.write(csv_content)

            logger.info(f"CSV report exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
            return False

    def _generate_csv(self, report: Dict) -> str:
        """Generate CSV content"""
        lines = []
        lines.append('Financial Analysis Report\n')
        lines.append(f'Company,{report.get("company_name", "Unknown")}\n')
        lines.append(f'Generated,{report.get("generated_at", "N/A")}\n')
        lines.append(f'Report Type,{report.get("report_type", "N/A")}\n')
        lines.append('\n')

        for section_name, section_content in report.get('sections', {}).items():
            lines.append(f'\n{section_name.replace("_", " ").upper()}\n')
            lines.append('Key,Value\n')

            if isinstance(section_content, dict):
                for key, value in section_content.items():
                    if isinstance(value, (dict, list)):
                        lines.append(f'{key},"{json.dumps(value)}"\n')
                    else:
                        lines.append(f'{key},{str(value)}\n')

            elif isinstance(section_content, list):
                for i, item in enumerate(section_content):
                    lines.append(f'Item {i+1},{str(item)}\n')

        return ''.join(lines)


class ReportScheduler:
    """Schedule report generation"""

    def __init__(self):
        self.scheduled_reports = []

    def schedule_report(self, report_config: Dict, schedule_type: str, interval: int) -> str:
        """Schedule report generation"""
        schedule_id = hashlib.md5(
            f"{datetime.now(timezone.utc).isoformat()}{report_config}".encode()
        ).hexdigest()[:12]

        scheduled = {
            'id': schedule_id,
            'config': report_config,
            'schedule_type': schedule_type,  # 'daily', 'weekly', 'monthly'
            'interval': interval,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'next_run': self._calculate_next_run(schedule_type, interval),
            'active': True
        }

        self.scheduled_reports.append(scheduled)
        logger.info(f"Report scheduled: {schedule_id}")
        return schedule_id

    def _calculate_next_run(self, schedule_type: str, interval: int) -> str:
        """Calculate next run time"""
        now = datetime.now(timezone.utc)

        if schedule_type == 'daily':
            next_run = now + timedelta(days=interval)
        elif schedule_type == 'weekly':
            next_run = now + timedelta(weeks=interval)
        elif schedule_type == 'monthly':
            next_run = now + timedelta(days=interval * 30)
        else:
            next_run = now + timedelta(hours=interval)

        return next_run.isoformat()

    def get_scheduled_reports(self) -> List[Dict]:
        """Get all scheduled reports"""
        return self.scheduled_reports

    def deactivate_schedule(self, schedule_id: str) -> bool:
        """Deactivate a schedule"""
        for schedule in self.scheduled_reports:
            if schedule['id'] == schedule_id:
                schedule['active'] = False
                return True
        return False


class AdvancedReportingEngine:
    """Main advanced reporting engine orchestrator"""

    def __init__(self):
        self.exec_summary_gen = ExecutiveSummaryGenerator()
        self.detail_analysis_gen = DetailedAnalysisGenerator()
        self.custom_builder = CustomReportBuilder()
        self.pdf_exporter = PDFReportExporter()
        self.html_exporter = HTMLReportExporter()
        self.csv_exporter = CSVReportExporter()
        self.scheduler = ReportScheduler()
        self.cache = ReportCache()
        self.versions = {}

    def generate_report(self, report_type: ReportType, financial_data: Dict, company_name: str = "Company") -> Dict:
        """Generate report based on type"""
        cache_key = f"{report_type.value}_{company_name}"
        cached = self.cache.get(cache_key)

        if cached:
            logger.info(f"Returning cached report: {cache_key}")
            return cached

        if report_type == ReportType.EXECUTIVE_SUMMARY:
            report = self.exec_summary_gen.generate(financial_data, company_name)
        elif report_type == ReportType.DETAILED_ANALYSIS:
            report = self.detail_analysis_gen.generate(financial_data, company_name)
        else:
            report = {'error': 'Unknown report type'}
            return report

        self.cache.set(cache_key, report)
        return report

    def export_report(self, report: Dict, format_type: ReportFormat, output_path: str) -> bool:
        """Export report in specified format"""
        if format_type == ReportFormat.PDF:
            return self.pdf_exporter.export_to_pdf(report, output_path)
        elif format_type == ReportFormat.HTML:
            return self.html_exporter.export_to_html(report, output_path)
        elif format_type == ReportFormat.CSV:
            return self.csv_exporter.export_to_csv(report, output_path)
        elif format_type == ReportFormat.JSON:
            return self._export_json(report, output_path)
        else:
            logger.error(f"Unknown export format: {format_type}")
            return False

    def _export_json(self, report: Dict, output_path: str) -> bool:
        """Export report to JSON"""
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"JSON report exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting JSON: {str(e)}")
            return False

    def save_report_version(self, report_id: str, report: Dict, changes: str = "") -> str:
        """Save report version"""
        if report_id not in self.versions:
            self.versions[report_id] = []

        version_num = len(self.versions[report_id]) + 1
        version = ReportVersion(report_id, version_num, report, changes)
        self.versions[report_id].append(version)

        logger.info(f"Report version saved: {report_id} v{version_num}")
        return version.hash

    def get_report_history(self, report_id: str) -> List[Dict]:
        """Get report version history"""
        if report_id not in self.versions:
            return []

        return [v.to_dict() for v in self.versions[report_id]]

    def generate_custom_report(self, financial_data: Dict, sections: List[str], company_name: str = "Company") -> Dict:
        """Generate custom report"""
        return self.custom_builder.build_custom_report(financial_data, sections, company_name)

    def get_cache_statistics(self) -> Dict:
        """Get cache statistics"""
        return {
            'cached_reports': len(self.cache.cache),
            'max_age_hours': self.cache.max_age_hours,
            'expired_cleared': self.cache.clear_expired()
        }

    def schedule_report_generation(self, report_type: str, schedule_type: str, interval: int) -> str:
        """Schedule report generation"""
        config = {
            'report_type': report_type,
            'schedule_type': schedule_type,
            'interval': interval
        }
        return self.scheduler.schedule_report(config, schedule_type, interval)

    def get_scheduled_reports(self) -> List[Dict]:
        """Get all scheduled reports"""
        return self.scheduler.get_scheduled_reports()

    def create_report_template(self, name: str, report_type: str, sections: List[str]) -> Dict:
        """Create report template"""
        try:
            rt = ReportType[report_type.upper()]
            template = ReportTemplate(name, rt, sections)
            return template.to_dict()
        except KeyError:
            logger.error(f"Invalid report type: {report_type}")
            return {'error': 'Invalid report type'}

    def generate_multi_company_report(self, companies_data: List[Dict]) -> Dict:
        """Generate comparative report for multiple companies"""
        report = {
            'report_type': 'comparative',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'companies_count': len(companies_data),
            'companies': []
        }

        for company_data in companies_data:
            company_report = self.exec_summary_gen.generate(
                company_data,
                company_data.get('company_id', 'Unknown')
            )
            report['companies'].append(company_report)

        return report
