"""
Day 14: Multi-Company Analysis Tools Tests
Tests for peer comparisons, ranking systems, portfolio analysis, and group analysis
"""

import pytest
from datetime import datetime
import numpy as np
from core.multicompany_analysis import (
    CompanyProfile, PeerComparison, CompanyRanking, PortfolioMetrics,
    ComparativeReport, PeerComparisonEngine, RankingSystem, PortfolioAnalyzer,
    GroupAnalysis, ComparativeReportBuilder
)


class TestCompanyProfile:
    """Test CompanyProfile data class"""
    
    def test_profile_creation(self):
        """Test creating company profile"""
        profile = CompanyProfile(
            company_id=1,
            name='Test Corp',
            ticker='TST',
            industry='Technology',
            country='USA',
            risk_score=5.5,
            revenue=1000000,
            ebitda=200000,
            net_income=100000,
            total_assets=5000000,
            total_liabilities=2000000,
            equity=3000000,
            current_ratio=1.5,
            debt_ratio=0.4,
            roe=0.15,
            profitability_margin=0.1
        )
        
        assert profile.company_id == 1
        assert profile.name == 'Test Corp'
        assert profile.risk_score == 5.5
    
    def test_profile_to_dict(self):
        """Test profile to dictionary conversion"""
        profile = CompanyProfile(
            company_id=1,
            name='Test Corp',
            ticker='TST',
            industry='Technology',
            country='USA',
            risk_score=5.5,
            revenue=1000000,
            ebitda=200000,
            net_income=100000,
            total_assets=5000000,
            total_liabilities=2000000,
            equity=3000000,
            current_ratio=1.5,
            debt_ratio=0.4,
            roe=0.15,
            profitability_margin=0.1
        )
        
        profile_dict = profile.to_dict()
        assert profile_dict['company_id'] == 1
        assert profile_dict['name'] == 'Test Corp'
        assert 'created_at' in profile_dict


class TestPeerComparisonEngine:
    """Test peer comparison functionality"""
    
    @pytest.fixture
    def engine(self):
        """Create peer comparison engine"""
        return PeerComparisonEngine()
    
    @pytest.fixture
    def sample_companies(self):
        """Create sample companies"""
        companies = [
            CompanyProfile(
                company_id=1,
                name='TechCorp',
                ticker='TC',
                industry='Technology',
                country='USA',
                risk_score=5.5,
                revenue=1000000,
                ebitda=200000,
                net_income=100000,
                total_assets=5000000,
                total_liabilities=2000000,
                equity=3000000,
                current_ratio=1.5,
                debt_ratio=0.4,
                roe=0.15,
                profitability_margin=0.1
            ),
            CompanyProfile(
                company_id=2,
                name='TechInc',
                ticker='TI',
                industry='Technology',
                country='USA',
                risk_score=4.2,
                revenue=1200000,
                ebitda=250000,
                net_income=120000,
                total_assets=6000000,
                total_liabilities=1800000,
                equity=4200000,
                current_ratio=1.8,
                debt_ratio=0.3,
                roe=0.18,
                profitability_margin=0.12
            ),
            CompanyProfile(
                company_id=3,
                name='FinCorp',
                ticker='FC',
                industry='Finance',
                country='USA',
                risk_score=3.8,
                revenue=800000,
                ebitda=180000,
                net_income=90000,
                total_assets=4000000,
                total_liabilities=1600000,
                equity=2400000,
                current_ratio=1.3,
                debt_ratio=0.4,
                roe=0.12,
                profitability_margin=0.11
            )
        ]
        return companies
    
    def test_add_company(self, engine, sample_companies):
        """Test adding companies to engine"""
        engine.add_company(sample_companies[0])
        assert sample_companies[0].company_id in engine.companies
    
    def test_compare_with_peers(self, engine, sample_companies):
        """Test peer comparison"""
        for company in sample_companies[:2]:
            engine.add_company(company)
        
        comparisons = engine.compare_with_peers(1, 'risk_score')
        assert len(comparisons) > 0
        assert comparisons[0].company_id == 1
    
    def test_get_industry_benchmark(self, engine, sample_companies):
        """Test industry benchmark calculation"""
        for company in sample_companies[:2]:
            engine.add_company(company)
        
        benchmark = engine.get_industry_benchmark('Technology', 'risk_score')
        assert 'min' in benchmark
        assert 'max' in benchmark
        assert 'mean' in benchmark
        assert 'median' in benchmark
    
    def test_percentile_calculation(self, engine):
        """Test percentile calculation"""
        engine.add_company(CompanyProfile(
            company_id=1,
            name='Test',
            ticker='T',
            industry='Tech',
            country='USA',
            risk_score=5.0,
            revenue=1000000,
            ebitda=200000,
            net_income=100000,
            total_assets=5000000,
            total_liabilities=2000000,
            equity=3000000,
            current_ratio=1.5,
            debt_ratio=0.4,
            roe=0.15,
            profitability_margin=0.1
        ))
        
        percentile = engine._calculate_percentile(5.0, [3.0, 4.0, 5.0, 6.0])
        assert 0 <= percentile <= 100


class TestRankingSystem:
    """Test ranking system"""
    
    @pytest.fixture
    def system(self):
        """Create ranking system"""
        return RankingSystem()
    
    @pytest.fixture
    def sample_companies(self):
        """Create sample companies"""
        return [
            CompanyProfile(
                company_id=i,
                name=f'Company{i}',
                ticker=f'C{i}',
                industry='Technology',
                country='USA',
                risk_score=5.0 - i,
                revenue=1000000 + (i * 100000),
                ebitda=200000,
                net_income=100000,
                total_assets=5000000,
                total_liabilities=2000000,
                equity=3000000,
                current_ratio=1.5,
                debt_ratio=0.4,
                roe=0.15 + (i * 0.02),
                profitability_margin=0.1
            )
            for i in range(1, 6)
        ]
    
    def test_add_company(self, system, sample_companies):
        """Test adding companies to system"""
        system.add_company(sample_companies[0])
        assert sample_companies[0].company_id in system.companies
    
    def test_rank_by_metric(self, system, sample_companies):
        """Test ranking by metric"""
        for company in sample_companies:
            system.add_company(company)
        
        rankings = system.rank_by_metric('roe')
        assert len(rankings) == len(sample_companies)
        assert rankings[0].rank == 1
        assert rankings[-1].rank == len(sample_companies)
    
    def test_get_top_companies(self, system, sample_companies):
        """Test getting top companies"""
        for company in sample_companies:
            system.add_company(company)
        
        top = system.get_top_companies('roe', n=3)
        assert len(top) == 3
        assert top[0].rank == 1
    
    def test_get_worst_companies(self, system, sample_companies):
        """Test getting worst companies"""
        for company in sample_companies:
            system.add_company(company)
        
        worst = system.get_worst_companies('roe', n=2)
        assert len(worst) == 2
    
    def test_rank_by_industry(self, system, sample_companies):
        """Test ranking within industry"""
        for company in sample_companies:
            system.add_company(company)
        
        rankings = system.rank_by_metric('roe', industry='Technology')
        assert len(rankings) == len(sample_companies)


class TestPortfolioAnalyzer:
    """Test portfolio analysis"""
    
    @pytest.fixture
    def analyzer(self):
        """Create portfolio analyzer"""
        return PortfolioAnalyzer()
    
    @pytest.fixture
    def sample_companies(self):
        """Create sample companies"""
        companies = [
            CompanyProfile(
                company_id=i,
                name=f'Company{i}',
                ticker=f'C{i}',
                industry='Technology',
                country='USA',
                risk_score=5.0 - (i * 0.5),
                revenue=1000000,
                ebitda=200000,
                net_income=100000,
                total_assets=5000000 + (i * 1000000),
                total_liabilities=2000000,
                equity=3000000,
                current_ratio=1.5,
                debt_ratio=0.4,
                roe=0.15 + (i * 0.02),
                profitability_margin=0.1
            )
            for i in range(1, 6)
        ]
        return companies
    
    def test_add_company_metrics(self, analyzer, sample_companies):
        """Test adding company metrics"""
        analyzer.add_company_metrics(sample_companies[0])
        assert sample_companies[0].company_id in analyzer.company_metrics
    
    def test_create_portfolio(self, analyzer, sample_companies):
        """Test creating portfolio"""
        for company in sample_companies:
            analyzer.add_company_metrics(company)
        
        metrics = analyzer.create_portfolio('portfolio1', [1, 2, 3])
        assert metrics.portfolio_id == 'portfolio1'
        assert len(metrics.companies) == 3
        assert metrics.total_value > 0
    
    def test_portfolio_metrics_calculation(self, analyzer, sample_companies):
        """Test portfolio metrics calculation"""
        for company in sample_companies:
            analyzer.add_company_metrics(company)
        
        metrics = analyzer.create_portfolio('portfolio1', [1, 2])
        assert 0 <= metrics.average_risk_score <= 10
        assert 0 <= metrics.risk_concentration <= 1
        assert 0 <= metrics.diversification_index <= 1
    
    def test_optimize_portfolio(self, analyzer, sample_companies):
        """Test portfolio optimization"""
        for company in sample_companies:
            analyzer.add_company_metrics(company)
        
        optimal = analyzer.optimize_portfolio('Technology', target_risk=3.0, max_companies=3)
        assert len(optimal) <= 3
    
    def test_get_portfolio_metrics(self, analyzer, sample_companies):
        """Test retrieving portfolio metrics"""
        for company in sample_companies:
            analyzer.add_company_metrics(company)
        
        analyzer.create_portfolio('portfolio1', [1, 2, 3])
        metrics = analyzer.get_portfolio_metrics('portfolio1')
        assert metrics is not None
        assert metrics.portfolio_id == 'portfolio1'


class TestGroupAnalysis:
    """Test group analysis"""
    
    @pytest.fixture
    def analysis(self):
        """Create group analysis"""
        return GroupAnalysis()
    
    @pytest.fixture
    def sample_companies(self):
        """Create sample companies"""
        return [
            CompanyProfile(
                company_id=i,
                name=f'Company{i}',
                ticker=f'C{i}',
                industry='Technology' if i < 3 else 'Finance',
                country='USA',
                risk_score=5.0 - (i * 0.5),
                revenue=1000000 + (i * 100000),
                ebitda=200000,
                net_income=100000,
                total_assets=5000000,
                total_liabilities=2000000,
                equity=3000000,
                current_ratio=1.5,
                debt_ratio=0.4,
                roe=0.15,
                profitability_margin=0.1
            )
            for i in range(1, 6)
        ]
    
    def test_create_group(self, analysis, sample_companies):
        """Test creating analysis group"""
        analysis.create_group('group1', 'Tech Group', [1, 2, 3])
        assert 'group1' in analysis.groups
    
    def test_add_company(self, analysis, sample_companies):
        """Test adding company to analysis"""
        analysis.add_company(sample_companies[0])
        assert sample_companies[0].company_id in analysis.companies
    
    def test_analyze_group(self, analysis, sample_companies):
        """Test group analysis"""
        analysis.create_group('group1', 'Tech Group', [1, 2, 3])
        for company in sample_companies[:3]:
            analysis.add_company(company)
        
        group_analysis = analysis.analyze_group('group1')
        assert group_analysis['group_id'] == 'group1'
        assert group_analysis['company_count'] == 3
        assert 'total_revenue' in group_analysis
        assert 'average_risk_score' in group_analysis
    
    def test_get_industry_summary(self, analysis, sample_companies):
        """Test industry summary"""
        for company in sample_companies:
            analysis.add_company(company)
        
        summary = analysis.get_industry_summary('Technology')
        assert summary['industry'] == 'Technology'
        assert 'company_count' in summary
        assert 'average_risk_score' in summary
        assert 'healthiest_company' in summary


class TestComparativeReportBuilder:
    """Test comparative report building"""
    
    @pytest.fixture
    def builder(self):
        """Create report builder"""
        return ComparativeReportBuilder()
    
    @pytest.fixture
    def sample_companies(self):
        """Create sample companies"""
        return [
            CompanyProfile(
                company_id=i,
                name=f'Company{i}',
                ticker=f'C{i}',
                industry='Technology',
                country='USA',
                risk_score=5.0 - (i * 0.5),
                revenue=1000000 + (i * 100000),
                ebitda=200000,
                net_income=100000,
                total_assets=5000000,
                total_liabilities=2000000,
                equity=3000000,
                current_ratio=1.5,
                debt_ratio=0.4,
                roe=0.15 + (i * 0.02),
                profitability_margin=0.1
            )
            for i in range(1, 4)
        ]
    
    def test_build_peer_comparison_report(self, builder, sample_companies):
        """Test building peer comparison report"""
        for company in sample_companies:
            builder.peer_engine.add_company(company)
            builder.ranking_system.add_company(company)
        
        report = builder.build_peer_comparison_report(1, ['risk_score', 'roe'])
        assert 'peer_comp_' in report.report_id
        assert len(report.companies) == 3
        assert len(report.recommendations) > 0
    
    def test_build_industry_comparison_report(self, builder, sample_companies):
        """Test building industry comparison report"""
        for company in sample_companies:
            builder.peer_engine.add_company(company)
            builder.ranking_system.add_company(company)
        
        report = builder.build_industry_comparison_report('Technology')
        assert 'industry_' in report.report_id
        assert 'Technology' in report.title
        assert len(report.companies) == 3
    
    def test_report_to_dict(self, builder, sample_companies):
        """Test report to dictionary conversion"""
        for company in sample_companies:
            builder.peer_engine.add_company(company)
            builder.ranking_system.add_company(company)
        
        report = builder.build_industry_comparison_report('Technology')
        report_dict = report.to_dict()
        
        assert 'report_id' in report_dict
        assert 'title' in report_dict
        assert 'companies' in report_dict
        assert 'metrics' in report_dict


class TestPeerComparison:
    """Test peer comparison data class"""
    
    def test_comparison_creation(self):
        """Test creating peer comparison"""
        comparison = PeerComparison(
            company_id=1,
            peer_id=2,
            metric_name='risk_score',
            company_value=5.5,
            peer_value=4.2,
            industry_median=5.0,
            percentile=75.0,
            difference=1.3
        )
        
        assert comparison.company_id == 1
        assert comparison.peer_id == 2
        assert comparison.difference == 1.3
    
    def test_comparison_to_dict(self):
        """Test comparison to dictionary"""
        comparison = PeerComparison(
            company_id=1,
            peer_id=2,
            metric_name='risk_score',
            company_value=5.5,
            peer_value=4.2,
            industry_median=5.0,
            percentile=75.0,
            difference=1.3
        )
        
        comp_dict = comparison.to_dict()
        assert comp_dict['company_id'] == 1
        assert comp_dict['metric_name'] == 'risk_score'


class TestCompanyRanking:
    """Test company ranking data class"""
    
    def test_ranking_creation(self):
        """Test creating company ranking"""
        ranking = CompanyRanking(
            company_id=1,
            name='TestCorp',
            ticker='TC',
            rank=1,
            metric='roe',
            value=0.18,
            industry='Technology'
        )
        
        assert ranking.rank == 1
        assert ranking.metric == 'roe'
    
    def test_ranking_to_dict(self):
        """Test ranking to dictionary"""
        ranking = CompanyRanking(
            company_id=1,
            name='TestCorp',
            ticker='TC',
            rank=1,
            metric='roe',
            value=0.18,
            industry='Technology'
        )
        
        rank_dict = ranking.to_dict()
        assert rank_dict['rank'] == 1


class TestPortfolioMetrics:
    """Test portfolio metrics data class"""
    
    def test_metrics_creation(self):
        """Test creating portfolio metrics"""
        metrics = PortfolioMetrics(
            portfolio_id='portfolio1',
            companies=[1, 2, 3],
            total_value=15000000,
            average_risk_score=4.5,
            risk_concentration=0.45,
            diversification_index=0.55,
            portfolio_performance=0.15,
            volatility=1.2
        )
        
        assert metrics.portfolio_id == 'portfolio1'
        assert len(metrics.companies) == 3
    
    def test_metrics_to_dict(self):
        """Test metrics to dictionary"""
        metrics = PortfolioMetrics(
            portfolio_id='portfolio1',
            companies=[1, 2, 3],
            total_value=15000000,
            average_risk_score=4.5,
            risk_concentration=0.45,
            diversification_index=0.55,
            portfolio_performance=0.15,
            volatility=1.2
        )
        
        metrics_dict = metrics.to_dict()
        assert metrics_dict['portfolio_id'] == 'portfolio1'


class TestComparativeReport:
    """Test comparative report data class"""
    
    def test_report_creation(self):
        """Test creating comparative report"""
        report = ComparativeReport(
            report_id='report1',
            title='Test Report',
            companies=[{'id': 1, 'name': 'Corp1'}],
            metrics={'risk_score': [5.5, 4.2]},
            rankings=[],
            recommendations=['Test recommendation']
        )
        
        assert report.report_id == 'report1'
        assert len(report.companies) == 1
    
    def test_report_to_dict(self):
        """Test report to dictionary"""
        report = ComparativeReport(
            report_id='report1',
            title='Test Report',
            companies=[{'id': 1, 'name': 'Corp1'}],
            metrics={'risk_score': [5.5, 4.2]},
            rankings=[],
            recommendations=['Test recommendation']
        )
        
        report_dict = report.to_dict()
        assert report_dict['report_id'] == 'report1'
        assert 'created_at' in report_dict


class TestMultiCompanyIntegration:
    """Integration tests for multi-company analysis"""
    
    def test_full_peer_comparison_workflow(self):
        """Test complete peer comparison workflow"""
        engine = PeerComparisonEngine()
        
        companies = [
            CompanyProfile(
                company_id=i,
                name=f'Company{i}',
                ticker=f'C{i}',
                industry='Technology',
                country='USA',
                risk_score=5.0 - (i * 0.5),
                revenue=1000000,
                ebitda=200000,
                net_income=100000,
                total_assets=5000000,
                total_liabilities=2000000,
                equity=3000000,
                current_ratio=1.5,
                debt_ratio=0.4,
                roe=0.15,
                profitability_margin=0.1
            )
            for i in range(1, 4)
        ]
        
        for company in companies:
            engine.add_company(company)
        
        comparisons = engine.compare_with_peers(1, 'risk_score')
        assert len(comparisons) > 0
        assert all(c.company_id == 1 for c in comparisons)
    
    def test_full_ranking_workflow(self):
        """Test complete ranking workflow"""
        system = RankingSystem()
        
        companies = [
            CompanyProfile(
                company_id=i,
                name=f'Company{i}',
                ticker=f'C{i}',
                industry='Technology',
                country='USA',
                risk_score=5.0 - (i * 0.5),
                revenue=1000000,
                ebitda=200000,
                net_income=100000,
                total_assets=5000000,
                total_liabilities=2000000,
                equity=3000000,
                current_ratio=1.5,
                debt_ratio=0.4,
                roe=0.15 + (i * 0.02),
                profitability_margin=0.1
            )
            for i in range(1, 4)
        ]
        
        for company in companies:
            system.add_company(company)
        
        top = system.get_top_companies('roe', n=2)
        assert len(top) == 2
        assert top[0].rank < top[1].rank
    
    def test_full_portfolio_workflow(self):
        """Test complete portfolio workflow"""
        analyzer = PortfolioAnalyzer()
        
        companies = [
            CompanyProfile(
                company_id=i,
                name=f'Company{i}',
                ticker=f'C{i}',
                industry='Technology',
                country='USA',
                risk_score=5.0 - (i * 0.5),
                revenue=1000000,
                ebitda=200000,
                net_income=100000,
                total_assets=5000000 + (i * 1000000),
                total_liabilities=2000000,
                equity=3000000,
                current_ratio=1.5,
                debt_ratio=0.4,
                roe=0.15,
                profitability_margin=0.1
            )
            for i in range(1, 4)
        ]
        
        for company in companies:
            analyzer.add_company_metrics(company)
        
        metrics = analyzer.create_portfolio('port1', [1, 2, 3])
        assert metrics.total_value > 0
        assert 0 <= metrics.diversification_index <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
