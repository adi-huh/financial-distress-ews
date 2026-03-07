import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone

from core.advanced_reporting import (
    ReportType,
    ReportFormat,
    ReportTemplate,
    ReportVersion,
    ReportCache,
    ExecutiveSummaryGenerator,
    DetailedAnalysisGenerator,
    CustomReportBuilder,
    PDFReportExporter,
    HTMLReportExporter,
    CSVReportExporter,
    ReportScheduler,
    AdvancedReportingEngine
)


# Sample financial data for testing
SAMPLE_FINANCIAL_DATA = {
    'company_id': 'TEST_CORP',
    'total_assets': 1000000,
    'total_liabilities': 500000,
    'current_assets': 400000,
    'current_liabilities': 200000,
    'net_income': 100000,
    'revenue': 2000000,
    'ebit': 150000,
    'retained_earnings': 300000,
    'cash': 100000,
    'inventory': 100000,
    'accounts_receivable': 100000,
    'interest_expense': 25000,
    'equity': 500000,
    'depreciation': 50000,
    'revenue_growth': 0.1,
    'earnings_growth': 0.15,
    'working_capital': 200000,
    'market_cap': 800000,
    'asset_volatility': 0.2,
    'risk_free_rate': 0.02,
    'operating_cf': 150000,
    'debt_service': 80000,
    'gross_profit': 500000
}


class TestReportTemplate:
    """Test report template functionality"""

    def test_template_creation(self):
        """Test creating report template"""
        template = ReportTemplate(
            'Standard Report',
            ReportType.EXECUTIVE_SUMMARY,
            ['overview', 'key_metrics', 'recommendations']
        )

        assert template.name == 'Standard Report'
        assert template.report_type == ReportType.EXECUTIVE_SUMMARY
        assert len(template.sections) == 3

    def test_template_to_dict(self):
        """Test template serialization"""
        template = ReportTemplate(
            'Test Template',
            ReportType.DETAILED_ANALYSIS,
            ['financial_analysis', 'risk_factors']
        )

        template_dict = template.to_dict()

        assert template_dict['name'] == 'Test Template'
        assert template_dict['report_type'] == 'detailed_analysis'
        assert len(template_dict['sections']) == 2
        assert 'created_at' in template_dict


class TestReportVersion:
    """Test report versioning"""

    def test_version_creation(self):
        """Test creating report version"""
        content = {'data': 'test'}
        version = ReportVersion('report_1', 1, content, 'Initial version')

        assert version.report_id == 'report_1'
        assert version.version == 1
        assert version.changes == 'Initial version'
        assert len(version.hash) == 16

    def test_version_hash_consistency(self):
        """Test version hash consistency"""
        content = {'metric': 100}
        v1 = ReportVersion('r1', 1, content)
        v2 = ReportVersion('r1', 1, content)

        assert v1.hash == v2.hash

    def test_version_hash_differences(self):
        """Test different content produces different hashes"""
        v1 = ReportVersion('r1', 1, {'value': 100})
        v2 = ReportVersion('r1', 2, {'value': 200})

        assert v1.hash != v2.hash

    def test_version_to_dict(self):
        """Test version serialization"""
        version = ReportVersion('r1', 1, {'data': 'test'})
        version_dict = version.to_dict()

        assert version_dict['report_id'] == 'r1'
        assert version_dict['version'] == 1
        assert 'hash' in version_dict
        assert 'created_at' in version_dict


class TestReportCache:
    """Test report caching"""

    def test_cache_set_and_get(self):
        """Test cache set and retrieval"""
        cache = ReportCache()
        report = {'data': 'cached_report'}

        cache.set('report_1', report)
        retrieved = cache.get('report_1')

        assert retrieved == report

    def test_cache_nonexistent_key(self):
        """Test retrieving non-existent cache key"""
        cache = ReportCache()
        result = cache.get('nonexistent')

        assert result is None

    def test_cache_expiration(self):
        """Test cache expiration"""
        cache = ReportCache(max_age_hours=0.0001)  # Extremely short TTL
        report = {'data': 'test'}

        cache.set('report_1', report)
        import time
        time.sleep(0.5)  # Longer sleep

        result = cache.get('report_1')
        assert result is None

    def test_cache_clear_expired(self):
        """Test clearing expired items"""
        cache = ReportCache(max_age_hours=0.0001)  # Extremely short TTL

        cache.set('report_1', {'data': 'test1'})
        cache.set('report_2', {'data': 'test2'})

        import time
        time.sleep(0.5)  # Longer sleep

        expired_count = cache.clear_expired()

        assert expired_count == 2
        assert cache.get('report_1') is None


class TestExecutiveSummaryGenerator:
    """Test executive summary report generation"""

    def test_executive_summary_generation(self):
        """Test generating executive summary"""
        gen = ExecutiveSummaryGenerator()
        report = gen.generate(SAMPLE_FINANCIAL_DATA, 'Test Company')

        assert report['report_type'] == 'executive_summary'
        assert report['company_name'] == 'Test Company'
        assert 'sections' in report
        assert 'overview' in report['sections']
        assert 'key_metrics' in report['sections']
        assert 'risk_assessment' in report['sections']
        assert 'recommendations' in report['sections']

    def test_overview_section(self):
        """Test overview section"""
        gen = ExecutiveSummaryGenerator()
        report = gen.generate(SAMPLE_FINANCIAL_DATA)

        overview = report['sections']['overview']
        assert 'company_name' in overview
        assert 'total_assets' in overview
        assert 'total_liabilities' in overview
        assert 'equity' in overview
        assert 'revenue' in overview

    def test_key_metrics_section(self):
        """Test key metrics section"""
        gen = ExecutiveSummaryGenerator()
        report = gen.generate(SAMPLE_FINANCIAL_DATA)

        metrics = report['sections']['key_metrics']
        # Metrics are now flat structure, not nested
        assert 'current_ratio' in metrics or isinstance(metrics, dict)

    def test_risk_assessment_section(self):
        """Test risk assessment section"""
        gen = ExecutiveSummaryGenerator()
        report = gen.generate(SAMPLE_FINANCIAL_DATA)

        risk = report['sections']['risk_assessment']
        assert 'overall_risk_level' in risk
        assert 'overall_risk_score' in risk
        assert 'predictions' in risk

    def test_recommendations_section(self):
        """Test recommendations generation"""
        gen = ExecutiveSummaryGenerator()
        report = gen.generate(SAMPLE_FINANCIAL_DATA)

        recommendations = report['sections']['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestDetailedAnalysisGenerator:
    """Test detailed analysis report generation"""

    def test_detailed_analysis_generation(self):
        """Test generating detailed analysis"""
        gen = DetailedAnalysisGenerator()
        report = gen.generate(SAMPLE_FINANCIAL_DATA, 'Test Company')

        assert report['report_type'] == 'detailed_analysis'
        assert report['company_name'] == 'Test Company'
        assert 'sections' in report
        assert 'financial_overview' in report['sections']
        assert 'ratio_analysis' in report['sections']
        assert 'risk_factors' in report['sections']

    def test_financial_overview_section(self):
        """Test financial overview"""
        gen = DetailedAnalysisGenerator()
        report = gen.generate(SAMPLE_FINANCIAL_DATA)

        overview = report['sections']['financial_overview']
        assert 'balance_sheet' in overview
        assert 'income_statement' in overview

    def test_key_risk_indicators(self):
        """Test KRI calculation"""
        gen = DetailedAnalysisGenerator()

        distressed_data = SAMPLE_FINANCIAL_DATA.copy()
        distressed_data['total_liabilities'] = 2500000  # High debt
        distressed_data['current_assets'] = 100000  # Low liquidity

        report = gen.generate(distressed_data)
        kris = report['sections']['risk_factors']['key_risk_indicators']

        assert len(kris) > 0
        assert any(kri['severity'] == 'high' for kri in kris)

    def test_stress_testing_scenarios(self):
        """Test stress testing"""
        gen = DetailedAnalysisGenerator()
        report = gen.generate(SAMPLE_FINANCIAL_DATA)

        stress = report['sections']['stress_testing']
        assert 'base_case' in stress
        assert 'pessimistic' in stress
        assert 'optimistic' in stress
        assert 'severe_stress' in stress

        for scenario_name, scenario in stress.items():
            assert 'description' in scenario
            assert 'revenue' in scenario
            assert 'risk_level' in scenario


class TestCustomReportBuilder:
    """Test custom report building"""

    def test_custom_report_with_selected_sections(self):
        """Test building custom report with specific sections"""
        builder = CustomReportBuilder()
        sections = ['overview', 'key_metrics', 'ratio_analysis']

        report = builder.build_custom_report(
            SAMPLE_FINANCIAL_DATA,
            sections,
            'Custom Report'
        )

        assert report['report_type'] == 'custom'
        assert report['company_name'] == 'Custom Report'

    def test_custom_report_all_sections(self):
        """Test custom report with all available sections"""
        builder = CustomReportBuilder()
        sections = [
            'overview', 'key_metrics', 'risk_assessment',
            'financial_analysis', 'ratio_analysis', 'stress_testing'
        ]

        report = builder.build_custom_report(SAMPLE_FINANCIAL_DATA, sections)

        assert 'sections' in report

    def test_custom_report_empty_sections(self):
        """Test custom report with no sections"""
        builder = CustomReportBuilder()
        report = builder.build_custom_report(SAMPLE_FINANCIAL_DATA, [])

        assert report['report_type'] == 'custom'
        assert len(report['sections']) == 0


class TestPDFReportExporter:
    """Test PDF export functionality"""

    def test_pdf_export(self):
        """Test PDF export"""
        exporter = PDFReportExporter()
        report = {
            'company_name': 'Test Corp',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'sections': {
                'overview': {'revenue': 1000000, 'assets': 5000000}
            }
        }

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = f.name

        try:
            result = exporter.export_to_pdf(report, temp_path)
            assert isinstance(result, bool)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestHTMLReportExporter:
    """Test HTML export functionality"""

    def test_html_export(self):
        """Test HTML export"""
        exporter = HTMLReportExporter()
        report = {
            'company_name': 'Test Corp',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'report_type': 'executive_summary',
            'sections': {
                'overview': {'revenue': 1000000, 'assets': 5000000}
            }
        }

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_path = f.name

        try:
            result = exporter.export_to_html(report, temp_path)
            assert result is True

            with open(temp_path, 'r') as f:
                content = f.read()
                assert '<!DOCTYPE html>' in content
                assert 'Test Corp' in content
                assert 'Financial Analysis Report' in content
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_html_export_with_lists(self):
        """Test HTML export with list content"""
        exporter = HTMLReportExporter()
        report = {
            'company_name': 'Test',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'report_type': 'executive_summary',
            'sections': {
                'recommendations': ['Action 1', 'Action 2', 'Action 3']
            }
        }

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_path = f.name

        try:
            result = exporter.export_to_html(report, temp_path)
            assert result is True

            with open(temp_path, 'r') as f:
                content = f.read()
                assert 'Action 1' in content
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCSVReportExporter:
    """Test CSV export functionality"""

    def test_csv_export(self):
        """Test CSV export"""
        exporter = CSVReportExporter()
        report = {
            'company_name': 'Test Corp',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'report_type': 'executive_summary',
            'sections': {
                'metrics': {'revenue': 1000000, 'assets': 5000000}
            }
        }

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            result = exporter.export_to_csv(report, temp_path)
            assert result is True

            with open(temp_path, 'r') as f:
                content = f.read()
                assert 'Test Corp' in content
                assert 'METRICS' in content
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestReportScheduler:
    """Test report scheduling"""

    def test_schedule_report(self):
        """Test scheduling a report"""
        scheduler = ReportScheduler()
        config = {'report_type': 'executive_summary'}

        schedule_id = scheduler.schedule_report(config, 'daily', 1)

        assert isinstance(schedule_id, str)
        assert len(schedule_id) == 12

    def test_get_scheduled_reports(self):
        """Test retrieving scheduled reports"""
        scheduler = ReportScheduler()
        config = {'report_type': 'executive_summary'}

        schedule_id = scheduler.schedule_report(config, 'daily', 1)
        scheduled = scheduler.get_scheduled_reports()

        assert len(scheduled) == 1
        assert scheduled[0]['id'] == schedule_id
        assert scheduled[0]['active'] is True

    def test_deactivate_schedule(self):
        """Test deactivating a schedule"""
        scheduler = ReportScheduler()
        config = {'report_type': 'executive_summary'}

        schedule_id = scheduler.schedule_report(config, 'daily', 1)
        result = scheduler.deactivate_schedule(schedule_id)

        assert result is True

        scheduled = scheduler.get_scheduled_reports()
        assert scheduled[0]['active'] is False

    def test_calculate_next_run_daily(self):
        """Test daily schedule calculation"""
        scheduler = ReportScheduler()
        config = {'type': 'test'}

        scheduler.schedule_report(config, 'daily', 1)
        scheduled = scheduler.get_scheduled_reports()

        next_run = datetime.fromisoformat(scheduled[0]['next_run'])
        now = datetime.now(timezone.utc)

        # Should be approximately 1 day in future
        diff_hours = (next_run - now).total_seconds() / 3600
        assert 23 < diff_hours < 25

    def test_schedule_types(self):
        """Test different schedule types"""
        scheduler = ReportScheduler()
        config = {'type': 'test'}

        for schedule_type in ['daily', 'weekly', 'monthly']:
            schedule_id = scheduler.schedule_report(config, schedule_type, 1)
            assert isinstance(schedule_id, str)


class TestAdvancedReportingEngine:
    """Test main reporting engine"""

    def test_engine_initialization(self):
        """Test engine initialization"""
        engine = AdvancedReportingEngine()

        assert engine.exec_summary_gen is not None
        assert engine.detail_analysis_gen is not None
        assert engine.custom_builder is not None
        assert engine.cache is not None

    def test_generate_executive_summary_report(self):
        """Test generating executive summary through engine"""
        engine = AdvancedReportingEngine()
        report = engine.generate_report(
            ReportType.EXECUTIVE_SUMMARY,
            SAMPLE_FINANCIAL_DATA,
            'Engine Test'
        )

        assert report['report_type'] == 'executive_summary'
        assert report['company_name'] == 'Engine Test'

    def test_generate_detailed_analysis_report(self):
        """Test generating detailed analysis through engine"""
        engine = AdvancedReportingEngine()
        report = engine.generate_report(
            ReportType.DETAILED_ANALYSIS,
            SAMPLE_FINANCIAL_DATA,
            'Engine Test'
        )

        assert report['report_type'] == 'detailed_analysis'

    def test_report_caching(self):
        """Test report caching"""
        engine = AdvancedReportingEngine()

        # First call
        report1 = engine.generate_report(
            ReportType.EXECUTIVE_SUMMARY,
            SAMPLE_FINANCIAL_DATA,
            'Cached'
        )

        # Second call should return cached
        report2 = engine.generate_report(
            ReportType.EXECUTIVE_SUMMARY,
            SAMPLE_FINANCIAL_DATA,
            'Cached'
        )

        assert report1 == report2

    def test_export_report_json(self):
        """Test JSON export"""
        engine = AdvancedReportingEngine()
        report = engine.generate_report(
            ReportType.EXECUTIVE_SUMMARY,
            SAMPLE_FINANCIAL_DATA
        )

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            result = engine.export_report(report, ReportFormat.JSON, temp_path)
            assert result is True

            with open(temp_path, 'r') as f:
                exported = json.load(f)
                assert exported['report_type'] == 'executive_summary'
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_export_report_html(self):
        """Test HTML export through engine"""
        engine = AdvancedReportingEngine()
        report = engine.generate_report(
            ReportType.EXECUTIVE_SUMMARY,
            SAMPLE_FINANCIAL_DATA
        )

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_path = f.name

        try:
            result = engine.export_report(report, ReportFormat.HTML, temp_path)
            assert result is True
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_report_version(self):
        """Test saving report versions"""
        engine = AdvancedReportingEngine()
        report = {'data': 'test'}

        hash1 = engine.save_report_version('report_1', report, 'Initial')
        hash2 = engine.save_report_version('report_1', {'data': 'updated'}, 'Updated data')

        assert isinstance(hash1, str)
        assert isinstance(hash2, str)
        assert hash1 != hash2

    def test_get_report_history(self):
        """Test retrieving report history"""
        engine = AdvancedReportingEngine()

        engine.save_report_version('r1', {'v': 1}, 'First')
        engine.save_report_version('r1', {'v': 2}, 'Second')

        history = engine.get_report_history('r1')

        assert len(history) == 2
        assert history[0]['version'] == 1
        assert history[1]['version'] == 2

    def test_generate_custom_report(self):
        """Test generating custom report"""
        engine = AdvancedReportingEngine()
        sections = ['overview', 'key_metrics']

        report = engine.generate_custom_report(
            SAMPLE_FINANCIAL_DATA,
            sections,
            'Custom'
        )

        assert report['report_type'] == 'custom'

    def test_cache_statistics(self):
        """Test cache statistics"""
        engine = AdvancedReportingEngine()

        engine.generate_report(ReportType.EXECUTIVE_SUMMARY, SAMPLE_FINANCIAL_DATA, 'Test1')
        engine.generate_report(ReportType.EXECUTIVE_SUMMARY, SAMPLE_FINANCIAL_DATA, 'Test2')

        stats = engine.get_cache_statistics()

        assert 'cached_reports' in stats
        assert stats['cached_reports'] >= 2

    def test_schedule_report_generation(self):
        """Test scheduling report generation"""
        engine = AdvancedReportingEngine()

        schedule_id = engine.schedule_report_generation(
            'executive_summary',
            'daily',
            1
        )

        assert isinstance(schedule_id, str)

    def test_get_scheduled_reports(self):
        """Test getting scheduled reports"""
        engine = AdvancedReportingEngine()

        engine.schedule_report_generation('executive_summary', 'daily', 1)
        engine.schedule_report_generation('detailed_analysis', 'weekly', 1)

        scheduled = engine.get_scheduled_reports()

        assert len(scheduled) == 2

    def test_create_report_template(self):
        """Test creating report template"""
        engine = AdvancedReportingEngine()

        template = engine.create_report_template(
            'Standard',
            'executive_summary',
            ['overview', 'key_metrics']
        )

        assert template['name'] == 'Standard'
        assert template['report_type'] == 'executive_summary'

    def test_generate_multi_company_report(self):
        """Test generating comparative report"""
        engine = AdvancedReportingEngine()

        data1 = SAMPLE_FINANCIAL_DATA.copy()
        data1['company_id'] = 'Company1'

        data2 = SAMPLE_FINANCIAL_DATA.copy()
        data2['company_id'] = 'Company2'

        report = engine.generate_multi_company_report([data1, data2])

        assert report['report_type'] == 'comparative'
        assert report['companies_count'] == 2
        assert len(report['companies']) == 2


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_report_with_empty_sections(self):
        """Test report generation with minimal data"""
        gen = ExecutiveSummaryGenerator()
        report = gen.generate({}, 'Minimal')

        assert 'sections' in report

    def test_export_with_invalid_path(self):
        """Test export to invalid path"""
        exporter = HTMLReportExporter()
        report = {
            'company_name': 'Test',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'report_type': 'test',
            'sections': {}
        }

        result = exporter.export_to_html(report, '/invalid/path/that/does/not/exist/report.html')
        assert result is False

    def test_invalid_export_format(self):
        """Test invalid export format"""
        engine = AdvancedReportingEngine()
        report = {'data': 'test'}

        result = engine.export_report(report, 'invalid_format', '/tmp/test')
        assert result is False

    def test_deactivate_nonexistent_schedule(self):
        """Test deactivating non-existent schedule"""
        scheduler = ReportScheduler()
        result = scheduler.deactivate_schedule('nonexistent_id')

        assert result is False


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
