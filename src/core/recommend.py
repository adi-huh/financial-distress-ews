"""
Consulting Recommendations Engine
Generates strategic, actionable recommendations based on financial analysis
and risk assessment.
"""

import logging
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)


class ConsultingEngine:
    """
    Generate consulting-grade strategic recommendations.
    
    Provides specific, actionable advice based on:
    - Risk classification
    - Category scores (liquidity, solvency, profitability, etc.)
    - Trend analysis
    - Anomalies detected
    """
    
    def __init__(self):
        """Initialize Consulting Engine."""
        logger.info("ConsultingEngine initialized")
    
    def generate_recommendations(self,
                                data: pd.DataFrame,
                                risk_results: Dict,
                                anomalies: pd.DataFrame = None) -> Dict:
        """
        Generate comprehensive recommendations for all companies.
        
        Args:
            data: DataFrame with financial ratios
            risk_results: Risk scoring results
            anomalies: DataFrame with detected anomalies (optional)
            
        Returns:
            dict: Recommendations by company
        """
        logger.info("Generating strategic recommendations...")
        
        recommendations = {}
        
        for company, risk_data in risk_results.items():
            company_recs = self._generate_company_recommendations(
                company,
                risk_data,
                anomalies
            )
            recommendations[company] = company_recs
        
        logger.info(f"✓ Generated recommendations for {len(recommendations)} companies")
        return recommendations
    
    def _generate_company_recommendations(self,
                                         company: str,
                                         risk_data: Dict,
                                         anomalies: pd.DataFrame) -> Dict:
        """
        Generate recommendations for a single company.
        
        Args:
            company: Company name
            risk_data: Risk assessment results
            anomalies: Anomalies DataFrame
            
        Returns:
            dict: Structured recommendations
        """
        classification = risk_data['classification']
        category_scores = risk_data['category_scores']
        trend = risk_data.get('trend_factor', 'Unknown')
        
        recommendations = {
            'company': company,
            'classification': classification,
            'priority': self._determine_priority(classification),
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': [],
            'key_focus_areas': [],
            'estimated_timeline': ''
        }
        
        # Add recommendations based on classification
        if classification == 'Distress':
            recommendations.update(self._distress_recommendations(category_scores))
        elif classification == 'Caution':
            recommendations.update(self._caution_recommendations(category_scores))
        else:  # Stable
            recommendations.update(self._stable_recommendations(category_scores, trend))
        
        # Add category-specific recommendations
        self._add_category_recommendations(recommendations, category_scores)
        
        # Add anomaly-specific recommendations
        if anomalies is not None:
            company_anomalies = anomalies[anomalies['company'] == company]
            if len(company_anomalies) > 0:
                self._add_anomaly_recommendations(recommendations, company_anomalies)
        
        return recommendations
    
    def _distress_recommendations(self, category_scores: Dict) -> Dict:
        """Generate recommendations for distressed companies."""
        return {
            'immediate_actions': [
                "⚠️ URGENT: Convene crisis management team immediately",
                "⚠️ Freeze non-essential expenditures and capital projects",
                "⚠️ Initiate emergency cash flow analysis and daily monitoring",
                "⚠️ Contact creditors to negotiate payment extensions",
                "⚠️ Explore asset sales or divestiture of non-core operations"
            ],
            'short_term_actions': [
                "Implement aggressive cost reduction program (target 15-20%)",
                "Restructure debt agreements with lenders",
                "Accelerate accounts receivable collection",
                "Reduce inventory levels through clearance sales",
                "Consider equity injection or strategic partnership"
            ],
            'long_term_actions': [
                "Complete comprehensive business model review",
                "Develop turnaround strategy with external advisors",
                "Implement new governance and risk management framework",
                "Rebuild stakeholder confidence through transparent communication"
            ],
            'key_focus_areas': ['Cash Flow Management', 'Debt Restructuring', 'Cost Reduction'],
            'estimated_timeline': '6-18 months for stabilization'
        }
    
    def _caution_recommendations(self, category_scores: Dict) -> Dict:
        """Generate recommendations for companies in caution zone."""
        return {
            'immediate_actions': [
                "Conduct detailed financial health assessment",
                "Implement enhanced financial monitoring and reporting",
                "Review and optimize working capital management",
                "Analyze cost structure for improvement opportunities"
            ],
            'short_term_actions': [
                "Develop action plan to address weak financial areas",
                "Strengthen cash reserves and credit lines",
                "Review pricing strategy and profit margins",
                "Evaluate operational efficiency improvements",
                "Consider refinancing high-interest debt"
            ],
            'long_term_actions': [
                "Build financial resilience through diversification",
                "Invest in process improvements and automation",
                "Develop strategic plan for sustainable growth",
                "Enhance risk management capabilities"
            ],
            'key_focus_areas': self._identify_weak_areas(category_scores),
            'estimated_timeline': '3-12 months for improvement'
        }
    
    def _stable_recommendations(self, category_scores: Dict, trend: str) -> Dict:
        """Generate recommendations for stable companies."""
        if trend == 'Declining':
            actions = [
                "⚠️ Warning: Declining trend detected despite stable classification",
                "Investigate root causes of declining metrics",
                "Implement preventive measures to maintain stability"
            ]
        else:
            actions = [
                "Maintain current financial discipline and controls",
                "Continue monitoring key financial indicators",
                "Review quarterly for early warning signs"
            ]
        
        return {
            'immediate_actions': actions,
            'short_term_actions': [
                "Optimize capital structure for lower cost of capital",
                "Explore growth opportunities and market expansion",
                "Enhance operational efficiency further",
                "Consider strategic investments in innovation"
            ],
            'long_term_actions': [
                "Build competitive moat through differentiation",
                "Develop succession planning and talent management",
                "Invest in digital transformation initiatives",
                "Expand market share through strategic acquisitions"
            ],
            'key_focus_areas': ['Growth', 'Innovation', 'Market Leadership'],
            'estimated_timeline': 'Ongoing optimization'
        }
    
    def _add_category_recommendations(self, recommendations: Dict, category_scores: Dict):
        """Add specific recommendations based on category scores."""
        
        # Liquidity recommendations
        if category_scores.get('liquidity', 100) < 60:
            recommendations['immediate_actions'].append(
                "💧 LIQUIDITY: Improve short-term cash position immediately"
            )
            recommendations['short_term_actions'].extend([
                "Accelerate accounts receivable collections (offer early payment discounts)",
                "Negotiate extended payment terms with suppliers",
                "Reduce inventory through promotions or bulk sales",
                "Establish or expand lines of credit",
                "Consider sale-leaseback of assets"
            ])
        
        # Solvency recommendations
        if category_scores.get('solvency', 100) < 60:
            recommendations['immediate_actions'].append(
                "⚖️ SOLVENCY: Address excessive debt levels urgently"
            )
            recommendations['short_term_actions'].extend([
                "Develop debt reduction roadmap with specific targets",
                "Refinance high-interest debt with lower-cost alternatives",
                "Consider debt-to-equity conversion with creditors",
                "Suspend dividend payments to preserve cash",
                "Explore asset monetization opportunities"
            ])
        
        # Profitability recommendations
        if category_scores.get('profitability', 100) < 60:
            recommendations['immediate_actions'].append(
                "💰 PROFITABILITY: Improve margins and bottom line"
            )
            recommendations['short_term_actions'].extend([
                "Conduct comprehensive cost analysis by product/service line",
                "Review pricing strategy (consider selective price increases)",
                "Eliminate or restructure unprofitable products/services",
                "Implement zero-based budgeting approach",
                "Negotiate better terms with key suppliers"
            ])
        
        # Efficiency recommendations
        if category_scores.get('efficiency', 100) < 60:
            recommendations['immediate_actions'].append(
                "⚙️ EFFICIENCY: Optimize asset utilization"
            )
            recommendations['short_term_actions'].extend([
                "Conduct time-motion study of key operations",
                "Implement lean manufacturing/operations principles",
                "Upgrade to modern ERP/management systems",
                "Reduce cycle times in production and delivery",
                "Optimize inventory management (JIT where possible)"
            ])
        
        # Growth recommendations
        if category_scores.get('growth', 100) < 50:
            recommendations['long_term_actions'].extend([
                "Develop new product/service offerings",
                "Explore new market segments and geographies",
                "Invest in marketing and customer acquisition",
                "Consider strategic partnerships or acquisitions"
            ])
    
    def _add_anomaly_recommendations(self, recommendations: Dict, anomalies: pd.DataFrame):
        """Add recommendations specific to detected anomalies."""
        
        critical_anomalies = anomalies[anomalies['severity'].isin(['Critical', 'High'])]
        
        if len(critical_anomalies) > 0:
            recommendations['immediate_actions'].insert(0,
                f"🚨 ANOMALY ALERT: {len(critical_anomalies)} critical anomalies detected - investigate immediately"
            )
            
            for _, anomaly in critical_anomalies.iterrows():
                metric = anomaly.get('metric', 'Unknown')
                recommendations['immediate_actions'].append(
                    f"Investigate unusual {metric}: {anomaly.get('deviation', 'significant deviation')}"
                )
    
    def _identify_weak_areas(self, category_scores: Dict) -> List[str]:
        """Identify weakest scoring areas."""
        weak_areas = []
        
        area_names = {
            'liquidity': 'Liquidity Management',
            'solvency': 'Debt & Solvency',
            'profitability': 'Profitability',
            'efficiency': 'Operational Efficiency',
            'growth': 'Growth & Expansion'
        }
        
        for category, score in category_scores.items():
            if score < 60:
                weak_areas.append(area_names.get(category, category))
        
        return weak_areas if weak_areas else ['General Financial Health']
    
    def _determine_priority(self, classification: str) -> str:
        """Determine priority level based on classification."""
        if classification == 'Distress':
            return 'CRITICAL'
        elif classification == 'Caution':
            return 'HIGH'
        else:
            return 'NORMAL'
    
    def generate_executive_summary(self,
                                  company: str,
                                  recommendations: Dict) -> str:
        """
        Generate executive summary of recommendations.
        
        Args:
            company: Company name
            recommendations: Recommendation dictionary
            
        Returns:
            str: Executive summary text
        """
        priority = recommendations['priority']
        classification = recommendations['classification']
        timeline = recommendations['estimated_timeline']
        
        summary = f"""
EXECUTIVE SUMMARY - {company}
{'=' * 60}

RISK CLASSIFICATION: {classification} ({priority} Priority)
ESTIMATED TIMELINE: {timeline}

KEY FOCUS AREAS:
{chr(10).join('  • ' + area for area in recommendations['key_focus_areas'])}

IMMEDIATE ACTIONS REQUIRED:
{chr(10).join('  ' + str(i+1) + '. ' + action for i, action in enumerate(recommendations['immediate_actions'][:5]))}

For detailed recommendations, see full report.
"""
        return summary
    
    def export_recommendations_to_dict(self, recommendations: Dict) -> Dict:
        """
        Export recommendations in a structured format for API/export.
        
        Args:
            recommendations: Recommendations dictionary
            
        Returns:
            dict: Structured recommendations
        """
        return {
            company: {
                'priority': recs['priority'],
                'classification': recs['classification'],
                'focus_areas': recs['key_focus_areas'],
                'immediate_actions': recs['immediate_actions'],
                'short_term_actions': recs['short_term_actions'],
                'long_term_actions': recs['long_term_actions'],
                'timeline': recs['estimated_timeline']
            }
            for company, recs in recommendations.items()
        }


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example risk results
    risk_results = {
        'TechCorp': {
            'overall_score': 78,
            'classification': 'Stable',
            'trend_factor': 'Improving',
            'category_scores': {
                'liquidity': 85,
                'solvency': 80,
                'profitability': 75,
                'efficiency': 70,
                'growth': 80
            }
        },
        'DistressCo': {
            'overall_score': 32,
            'classification': 'Distress',
            'trend_factor': 'Declining',
            'category_scores': {
                'liquidity': 35,
                'solvency': 25,
                'profitability': 30,
                'efficiency': 40,
                'growth': 20
            }
        }
    }
    
    # Generate recommendations
    engine = ConsultingEngine()
    recommendations = engine.generate_recommendations(
        pd.DataFrame(),  # Would contain actual data
        risk_results
    )
    
    # Display summary
    for company, recs in recommendations.items():
        print(f"\n{'=' * 60}")
        print(f"{company} - {recs['classification']} ({recs['priority']})")
        print(f"{'=' * 60}")
        print("\nImmediate Actions:")
        for action in recs['immediate_actions'][:3]:
            print(f"  • {action}")
