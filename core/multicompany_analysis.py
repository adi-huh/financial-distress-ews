"""
Day 14: Multi-Company Analysis Tools
Peer comparisons, ranking systems, portfolio analysis, and group analysis
"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field, asdict
from collections import defaultdict


@dataclass
class CompanyProfile:
    """Company profile with financial metrics"""
    company_id: int
    name: str
    ticker: str
    industry: str
    country: str
    risk_score: float
    revenue: float
    ebitda: float
    net_income: float
    total_assets: float
    total_liabilities: float
    equity: float
    current_ratio: float
    debt_ratio: float
    roe: float
    profitability_margin: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class PeerComparison:
    """Peer comparison metrics"""
    company_id: int
    peer_id: int
    metric_name: str
    company_value: float
    peer_value: float
    industry_median: float
    percentile: float
    difference: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class CompanyRanking:
    """Company ranking in peer group"""
    company_id: int
    name: str
    ticker: str
    rank: int
    metric: str
    value: float
    industry: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics"""
    portfolio_id: str
    companies: List[int]
    total_value: float
    average_risk_score: float
    risk_concentration: float
    diversification_index: float
    portfolio_performance: float
    volatility: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ComparativeReport:
    """Comparative analysis report"""
    report_id: str
    title: str
    companies: List[Dict]
    metrics: Dict[str, List[float]]
    rankings: List[Dict]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self):
        return {
            'report_id': self.report_id,
            'title': self.title,
            'companies': self.companies,
            'metrics': self.metrics,
            'rankings': self.rankings,
            'recommendations': self.recommendations,
            'created_at': self.created_at.isoformat()
        }


class PeerComparisonEngine:
    """Engine for comparing companies with industry peers"""
    
    def __init__(self):
        self.companies: Dict[int, CompanyProfile] = {}
        self.industry_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.comparisons: List[PeerComparison] = []
    
    def add_company(self, profile: CompanyProfile) -> None:
        """Add company to comparison pool"""
        self.companies[profile.company_id] = profile
        
        # Update industry metrics
        industry = profile.industry
        self.industry_metrics[industry]['risk_score'].append(profile.risk_score)
        self.industry_metrics[industry]['revenue'].append(profile.revenue)
        self.industry_metrics[industry]['current_ratio'].append(profile.current_ratio)
        self.industry_metrics[industry]['debt_ratio'].append(profile.debt_ratio)
        self.industry_metrics[industry]['roe'].append(profile.roe)
        self.industry_metrics[industry]['profit_margin'].append(profile.profitability_margin)
    
    def compare_with_peers(self, company_id: int, metric: str = 'risk_score') -> List[PeerComparison]:
        """Compare company with industry peers on specified metric"""
        if company_id not in self.companies:
            raise ValueError(f"Company {company_id} not found")
        
        company = self.companies[company_id]
        peers = [c for c in self.companies.values() if c.industry == company.industry and c.company_id != company_id]
        
        comparisons = []
        metric_values = getattr(company, metric, None)
        
        if metric_values is None:
            raise ValueError(f"Metric {metric} not found")
        
        peer_values = [getattr(p, metric, 0) for p in peers]
        industry_median = float(np.median(peer_values)) if peer_values else 0
        
        for peer in peers:
            peer_value = getattr(peer, metric, 0)
            percentile = self._calculate_percentile(metric_values, peer_values)
            difference = metric_values - peer_value
            
            comparison = PeerComparison(
                company_id=company_id,
                peer_id=peer.company_id,
                metric_name=metric,
                company_value=metric_values,
                peer_value=peer_value,
                industry_median=industry_median,
                percentile=percentile,
                difference=difference
            )
            comparisons.append(comparison)
        
        self.comparisons.extend(comparisons)
        return comparisons
    
    def get_industry_benchmark(self, industry: str, metric: str) -> Dict[str, float]:
        """Get industry benchmark for metric"""
        values = self.industry_metrics[industry][metric]
        
        if not values:
            return {}
        
        return {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std_dev': float(np.std(values))
        }
    
    def _calculate_percentile(self, value: float, peer_values: List[float]) -> float:
        """Calculate percentile rank"""
        if not peer_values:
            return 50.0
        
        count = sum(1 for v in peer_values if v <= value)
        return (count / len(peer_values)) * 100


class RankingSystem:
    """System for ranking companies across multiple metrics"""
    
    def __init__(self):
        self.companies: Dict[int, CompanyProfile] = {}
        self.rankings: Dict[str, List[CompanyRanking]] = defaultdict(list)
    
    def add_company(self, profile: CompanyProfile) -> None:
        """Add company to ranking pool"""
        self.companies[profile.company_id] = profile
    
    def rank_by_metric(self, metric: str, industry: Optional[str] = None) -> List[CompanyRanking]:
        """Rank companies by specified metric"""
        companies = self.companies.values()
        
        if industry:
            companies = [c for c in companies if c.industry == industry]
        
        # Sort by metric (descending for positive metrics like ROE, ascending for risk)
        is_risk_metric = 'risk' in metric.lower()
        sorted_companies = sorted(
            companies,
            key=lambda c: getattr(c, metric, 0),
            reverse=not is_risk_metric
        )
        
        rankings = []
        for rank, company in enumerate(sorted_companies, 1):
            metric_value = getattr(company, metric, 0)
            ranking = CompanyRanking(
                company_id=company.company_id,
                name=company.name,
                ticker=company.ticker,
                rank=rank,
                metric=metric,
                value=metric_value,
                industry=company.industry
            )
            rankings.append(ranking)
        
        self.rankings[metric] = rankings
        return rankings
    
    def get_top_companies(self, metric: str, n: int = 10, industry: Optional[str] = None) -> List[CompanyRanking]:
        """Get top N companies by metric"""
        rankings = self.rank_by_metric(metric, industry)
        return rankings[:n]
    
    def get_worst_companies(self, metric: str, n: int = 10, industry: Optional[str] = None) -> List[CompanyRanking]:
        """Get worst N companies by metric"""
        rankings = self.rank_by_metric(metric, industry)
        return rankings[-n:]


class PortfolioAnalyzer:
    """Portfolio analysis tools"""
    
    def __init__(self):
        self.portfolios: Dict[str, List[int]] = {}
        self.company_metrics: Dict[int, CompanyProfile] = {}
    
    def create_portfolio(self, portfolio_id: str, company_ids: List[int]) -> PortfolioMetrics:
        """Create and analyze portfolio"""
        self.portfolios[portfolio_id] = company_ids
        
        companies = [self.company_metrics[cid] for cid in company_ids if cid in self.company_metrics]
        
        if not companies:
            raise ValueError("No valid companies in portfolio")
        
        # Calculate portfolio metrics
        total_value = sum(c.total_assets for c in companies)
        average_risk = np.mean([c.risk_score for c in companies])
        
        # Risk concentration (Herfindahl index)
        weights = [c.total_assets / total_value for c in companies]
        risk_concentration = sum(w**2 for w in weights)
        
        # Diversification index (1 - concentration)
        diversification = 1 - risk_concentration
        
        # Portfolio performance (weighted average return proxy)
        performance = np.average(
            [c.roe for c in companies],
            weights=[c.total_assets for c in companies]
        )
        
        # Volatility (std dev of risk scores)
        volatility = float(np.std([c.risk_score for c in companies]))
        
        metrics = PortfolioMetrics(
            portfolio_id=portfolio_id,
            companies=company_ids,
            total_value=total_value,
            average_risk_score=average_risk,
            risk_concentration=risk_concentration,
            diversification_index=diversification,
            portfolio_performance=performance,
            volatility=volatility
        )
        
        return metrics
    
    def optimize_portfolio(self, industry: str, target_risk: float, max_companies: int = 10) -> List[int]:
        """Optimize portfolio for target risk level"""
        industry_companies = [
            (cid, c) for cid, c in self.company_metrics.items()
            if c.industry == industry
        ]
        
        if not industry_companies:
            return []
        
        # Sort by distance from target risk
        sorted_companies = sorted(
            industry_companies,
            key=lambda x: abs(x[1].risk_score - target_risk)
        )
        
        # Select top N companies
        optimal_ids = [cid for cid, _ in sorted_companies[:max_companies]]
        
        return optimal_ids
    
    def get_portfolio_metrics(self, portfolio_id: str) -> Optional[PortfolioMetrics]:
        """Get metrics for existing portfolio"""
        if portfolio_id not in self.portfolios:
            return None
        
        return self.create_portfolio(portfolio_id, self.portfolios[portfolio_id])
    
    def add_company_metrics(self, profile: CompanyProfile) -> None:
        """Add company metrics for portfolio analysis"""
        self.company_metrics[profile.company_id] = profile


class GroupAnalysis:
    """Group-level analysis across multiple companies"""
    
    def __init__(self):
        self.groups: Dict[str, List[int]] = {}
        self.companies: Dict[int, CompanyProfile] = {}
    
    def create_group(self, group_id: str, name: str, company_ids: List[int]) -> None:
        """Create analysis group"""
        self.groups[group_id] = company_ids
    
    def add_company(self, profile: CompanyProfile) -> None:
        """Add company to analysis pool"""
        self.companies[profile.company_id] = profile
    
    def analyze_group(self, group_id: str) -> Dict:
        """Analyze group metrics"""
        if group_id not in self.groups:
            raise ValueError(f"Group {group_id} not found")
        
        company_ids = self.groups[group_id]
        companies = [self.companies[cid] for cid in company_ids if cid in self.companies]
        
        if not companies:
            return {}
        
        # Aggregate metrics
        metrics = {
            'group_id': group_id,
            'company_count': len(companies),
            'total_revenue': sum(c.revenue for c in companies),
            'total_assets': sum(c.total_assets for c in companies),
            'total_liabilities': sum(c.total_liabilities for c in companies),
            'average_risk_score': float(np.mean([c.risk_score for c in companies])),
            'risk_std_dev': float(np.std([c.risk_score for c in companies])),
            'average_roe': float(np.mean([c.roe for c in companies])),
            'average_profit_margin': float(np.mean([c.profitability_margin for c in companies])),
            'companies': [
                {
                    'id': c.company_id,
                    'name': c.name,
                    'risk_score': c.risk_score
                }
                for c in companies
            ]
        }
        
        return metrics
    
    def get_industry_summary(self, industry: str) -> Dict:
        """Get summary statistics for industry"""
        industry_companies = [c for c in self.companies.values() if c.industry == industry]
        
        if not industry_companies:
            return {}
        
        return {
            'industry': industry,
            'company_count': len(industry_companies),
            'average_risk_score': float(np.mean([c.risk_score for c in industry_companies])),
            'risk_range': (
                float(np.min([c.risk_score for c in industry_companies])),
                float(np.max([c.risk_score for c in industry_companies]))
            ),
            'total_market_value': sum(c.total_assets for c in industry_companies),
            'healthiest_company': max(industry_companies, key=lambda c: c.roe).name,
            'riskiest_company': max(industry_companies, key=lambda c: c.risk_score).name
        }


class ComparativeReportBuilder:
    """Build comparative analysis reports"""
    
    def __init__(self):
        self.peer_engine = PeerComparisonEngine()
        self.ranking_system = RankingSystem()
        self.portfolio_analyzer = PortfolioAnalyzer()
        self.group_analysis = GroupAnalysis()
    
    def build_peer_comparison_report(self, company_id: int, metrics: List[str]) -> ComparativeReport:
        """Build peer comparison report"""
        comparisons = []
        for metric in metrics:
            comparisons.extend(self.peer_engine.compare_with_peers(company_id, metric))
        
        # Get rankings
        rankings = []
        for metric in metrics:
            metric_rankings = self.ranking_system.rank_by_metric(metric)
            rankings.extend([r.to_dict() for r in metric_rankings])
        
        # Get company info
        companies = [c.to_dict() for c in self.peer_engine.companies.values()]
        
        report = ComparativeReport(
            report_id=f"peer_comp_{company_id}_{datetime.now().timestamp()}",
            title=f"Peer Comparison Report - Company {company_id}",
            companies=companies,
            metrics={metric: [c.company_value for c in comparisons if c.metric_name == metric] 
                    for metric in metrics},
            rankings=rankings,
            recommendations=self._generate_recommendations(comparisons)
        )
        
        return report
    
    def build_portfolio_report(self, portfolio_id: str) -> ComparativeReport:
        """Build portfolio analysis report"""
        metrics = self.portfolio_analyzer.get_portfolio_metrics(portfolio_id)
        
        if not metrics:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        companies = [self.portfolio_analyzer.company_metrics[cid].to_dict() 
                    for cid in metrics.companies]
        
        report = ComparativeReport(
            report_id=f"portfolio_{portfolio_id}_{datetime.now().timestamp()}",
            title=f"Portfolio Analysis Report - {portfolio_id}",
            companies=companies,
            metrics={
                'risk': [metrics.average_risk_score],
                'performance': [metrics.portfolio_performance],
                'diversification': [metrics.diversification_index]
            },
            rankings=[],
            recommendations=self._generate_portfolio_recommendations(metrics)
        )
        
        return report
    
    def build_industry_comparison_report(self, industry: str) -> ComparativeReport:
        """Build industry comparison report"""
        industry_companies = [c for c in self.peer_engine.companies.values() 
                             if c.industry == industry]
        
        if not industry_companies:
            raise ValueError(f"No companies found for industry {industry}")
        
        # Get multiple metrics
        metrics_list = ['risk_score', 'roe', 'profitability_margin']
        rankings = []
        
        for metric in metrics_list:
            metric_rankings = self.ranking_system.rank_by_metric(metric, industry)
            rankings.extend([r.to_dict() for r in metric_rankings])
        
        report = ComparativeReport(
            report_id=f"industry_{industry}_{datetime.now().timestamp()}",
            title=f"Industry Analysis Report - {industry}",
            companies=[c.to_dict() for c in industry_companies],
            metrics={metric: [getattr(c, metric, 0) for c in industry_companies] 
                    for metric in metrics_list},
            rankings=rankings,
            recommendations=self._generate_industry_recommendations(industry_companies)
        )
        
        return report
    
    def _generate_recommendations(self, comparisons: List[PeerComparison]) -> List[str]:
        """Generate recommendations from comparisons"""
        if not comparisons:
            return []
        
        recommendations = []
        
        # Analyze high risk companies
        high_risk = [c for c in comparisons if c.metric_name == 'risk_score' and c.company_value > 7]
        if high_risk:
            recommendations.append(f"Monitor {len(high_risk)} high-risk peer companies closely")
        
        # Analyze outperformers
        outperformers = [c for c in comparisons if c.difference > 0]
        if outperformers:
            recommendations.append(f"Consider best practices from {len(set(c.peer_id for c in outperformers))} outperforming peers")
        
        return recommendations
    
    def _generate_portfolio_recommendations(self, metrics: PortfolioMetrics) -> List[str]:
        """Generate portfolio recommendations"""
        recommendations = []
        
        if metrics.risk_concentration > 0.5:
            recommendations.append("Portfolio is concentrated; consider diversifying across more companies")
        
        if metrics.volatility > 2.0:
            recommendations.append("Portfolio volatility is high; rebalance for stability")
        
        if metrics.diversification_index < 0.3:
            recommendations.append("Low diversification index; add more uncorrelated assets")
        
        return recommendations
    
    def _generate_industry_recommendations(self, companies: List[CompanyProfile]) -> List[str]:
        """Generate industry recommendations"""
        recommendations = []
        
        avg_risk = np.mean([c.risk_score for c in companies])
        high_risk_count = len([c for c in companies if c.risk_score > avg_risk + 1.5])
        
        if high_risk_count / len(companies) > 0.3:
            recommendations.append(f"{high_risk_count} companies ({high_risk_count/len(companies)*100:.1f}%) above industry risk")
        
        avg_margin = np.mean([c.profitability_margin for c in companies])
        if avg_margin < 0.05:
            recommendations.append("Industry profitability margins are below 5%; monitor cost structures")
        
        return recommendations
