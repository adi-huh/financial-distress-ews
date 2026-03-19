"""
Day 22: Integration Testing & QA Tests
"""

import pytest
import tempfile
from datetime import datetime, timezone

from core.integration_testing_qa import (
    IntegrationTestSuite, CodeQualityAnalyzer, RegressionTester,
    PerformanceValidator, SecurityValidator, QualityAssuranceEngine,
    TestResult, CodeQualityMetrics
)


class TestIntegrationTestSuite:
    """Test integration test suite"""
    
    def test_pdf_to_analysis_pipeline(self):
        """Test PDF pipeline"""
        suite = IntegrationTestSuite()
        result = suite.test_pdf_to_analysis_pipeline()
        assert result.status == 'pass'
    
    def test_data_import_export(self):
        """Test data import/export"""
        suite = IntegrationTestSuite()
        result = suite.test_data_import_export()
        assert result.status == 'pass'
    
    def test_risk_scoring_workflow(self):
        """Test risk scoring"""
        suite = IntegrationTestSuite()
        result = suite.test_risk_scoring_workflow()
        assert result.status == 'pass'
    
    def test_multi_company_analysis(self):
        """Test multi-company analysis"""
        suite = IntegrationTestSuite()
        result = suite.test_multi_company_analysis()
        assert result.status == 'pass'
    
    def test_api_endpoints(self):
        """Test API endpoints"""
        suite = IntegrationTestSuite()
        result = suite.test_api_endpoints()
        assert result.status == 'pass'
    
    def test_summary(self):
        """Test summary generation"""
        suite = IntegrationTestSuite()
        suite.test_pdf_to_analysis_pipeline()
        suite.test_risk_scoring_workflow()
        
        summary = suite.get_test_summary()
        assert summary['passed'] == 2
        assert summary['success_rate'] == 100.0


class TestCodeQualityAnalyzer:
    """Test code quality analysis"""
    
    def test_analyzer_creation(self):
        """Test analyzer creation"""
        analyzer = CodeQualityAnalyzer()
        assert len(analyzer.metrics) == 0
    
    def test_analyze_file(self):
        """Test file analysis"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def hello():\n    return 'world'\n")
            f.flush()
            
            analyzer = CodeQualityAnalyzer()
            metrics = analyzer.analyze_file(f.name)
            
            assert metrics.total_lines > 0
            assert metrics.cyclomatic_complexity > 0
    
    def test_quality_report(self):
        """Test quality report"""
        analyzer = CodeQualityAnalyzer()
        report = analyzer.get_quality_report()
        
        assert isinstance(report, dict)


class TestRegressionTester:
    """Test regression testing"""
    
    def test_set_baseline(self):
        """Test setting baseline"""
        tester = RegressionTester()
        tester.set_baseline('test1', {'result': 100})
        
        assert 'test1' in tester.baselines
    
    def test_regression_pass(self):
        """Test passing regression"""
        tester = RegressionTester()
        tester.set_baseline('test1', {'result': 100})
        
        passed = tester.test_against_baseline('test1', {'result': 100})
        assert passed is True
    
    def test_regression_fail(self):
        """Test failing regression"""
        tester = RegressionTester()
        tester.set_baseline('test1', {'result': 100})
        
        passed = tester.test_against_baseline('test1', {'result': 200})
        assert passed is False
    
    def test_regression_report(self):
        """Test regression report"""
        tester = RegressionTester()
        tester.set_baseline('test1', 100)
        tester.test_against_baseline('test1', 100)
        
        report = tester.get_regression_report()
        assert report['passed'] == 1


class TestPerformanceValidator:
    """Test performance validation"""
    
    def test_set_benchmark(self):
        """Test setting benchmark"""
        validator = PerformanceValidator()
        validator.set_benchmark('operation1', 1000)
        
        assert 'operation1' in validator.benchmarks
    
    def test_measure_operation_pass(self):
        """Test passing performance"""
        validator = PerformanceValidator()
        validator.set_benchmark('operation1', 1000)
        
        passed = validator.measure_operation('operation1', 500)
        assert passed is True
    
    def test_measure_operation_fail(self):
        """Test failing performance"""
        validator = PerformanceValidator()
        validator.set_benchmark('operation1', 1000)
        
        passed = validator.measure_operation('operation1', 1500)
        assert passed is False
    
    def test_performance_report(self):
        """Test performance report"""
        validator = PerformanceValidator()
        validator.set_benchmark('op1', 100)
        validator.measure_operation('op1', 50)
        validator.measure_operation('op1', 60)
        
        report = validator.get_performance_report()
        assert 'op1' in report


class TestSecurityValidator:
    """Test security validation"""
    
    def test_sql_injection_detection(self):
        """Test SQL injection detection"""
        validator = SecurityValidator()
        
        safe_query = "SELECT * FROM users WHERE id = ?"
        malicious_query = "SELECT * FROM users WHERE id = ' OR '1'='1"
        
        assert validator.check_sql_injection(safe_query) is True
        assert validator.check_sql_injection(malicious_query) is False
    
    def test_input_validation(self):
        """Test input validation"""
        validator = SecurityValidator()
        
        safe_input = "Hello World"
        malicious_input = "<script>alert('xss')</script>"
        
        assert validator.check_input_validation(safe_input) is True
        assert validator.check_input_validation(malicious_input) is False
    
    def test_security_report(self):
        """Test security report"""
        validator = SecurityValidator()
        validator.check_sql_injection("SELECT * FROM users WHERE id = ' OR '1'='1")
        
        report = validator.get_security_report()
        assert report['security_passed'] is False


class TestQualityAssuranceEngine:
    """Test QA engine"""
    
    def test_engine_creation(self):
        """Test QA engine creation"""
        engine = QualityAssuranceEngine()
        
        assert engine.integration_tests is not None
        assert engine.code_quality is not None
        assert engine.regression_tester is not None
    
    def test_run_all_tests(self):
        """Test running all tests"""
        engine = QualityAssuranceEngine()
        results = engine.run_all_tests()
        
        assert 'integration_tests' in results
        assert 'timestamp' in results
    
    def test_qa_report(self):
        """Test QA report"""
        engine = QualityAssuranceEngine()
        report = engine.get_qa_report()
        
        assert 'integration_tests' in report
        assert 'security' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
