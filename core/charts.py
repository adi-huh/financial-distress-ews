"""
Visualization Module
Creates charts, graphs, and visual representations of financial analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ChartGenerator:
    """
    Generate visualizations for financial analysis.
    
    Creates trend charts, comparison plots, risk gauges, and dashboards.
    """
    
    def __init__(self, output_dir: str = "charts"):
        """
        Initialize Chart Generator.
        
        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ChartGenerator initialized, output: {self.output_dir}")
    
    def create_dashboard(self,
                        data: pd.DataFrame,
                        risk_results: Dict,
                        output_dir: Path = None):
        """
        Create complete dashboard with multiple charts.
        
        Args:
            data: DataFrame with financial ratios
            risk_results: Risk scoring results
            output_dir: Output directory (optional)
        """
        logger.info("Creating dashboard visualizations...")
        
        save_dir = output_dir if output_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Trend charts for key ratios
        self.plot_ratio_trends(data, save_dir / "ratio_trends.png")
        
        # 2. Risk score comparison
        self.plot_risk_comparison(risk_results, save_dir / "risk_comparison.png")
        
        # 3. Category score radar chart
        self.plot_category_scores(risk_results, save_dir / "category_scores.png")
        
        # 4. Liquidity analysis
        self.plot_liquidity_analysis(data, save_dir / "liquidity.png")
        
        # 5. Profitability analysis
        self.plot_profitability_analysis(data, save_dir / "profitability.png")
        
        logger.info(f"✓ Dashboard charts saved to {save_dir}")
    
    def plot_ratio_trends(self, data: pd.DataFrame, save_path: Path = None):
        """
        Plot trends for key financial ratios over time.
        
        Args:
            data: DataFrame with ratios and year column
            save_path: Path to save the chart
        """
        if 'year' not in data.columns:
            logger.warning("Year column required for trend plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Key Financial Ratio Trends', fontsize=16, fontweight='bold')
        
        # Plot 1: Liquidity Trends
        if 'current_ratio' in data.columns:
            for company in data['company'].unique():
                company_data = data[data['company'] == company]
                axes[0, 0].plot(company_data['year'], company_data['current_ratio'],
                              marker='o', label=company, linewidth=2)
            axes[0, 0].set_title('Current Ratio Trend')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Current Ratio')
            axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Threshold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Profitability Trends (ROE)
        if 'roe' in data.columns:
            for company in data['company'].unique():
                company_data = data[data['company'] == company]
                axes[0, 1].plot(company_data['year'], company_data['roe'] * 100,
                              marker='o', label=company, linewidth=2)
            axes[0, 1].set_title('Return on Equity (ROE) Trend')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('ROE (%)')
            axes[0, 1].axhline(y=15, color='g', linestyle='--', alpha=0.5, label='Target')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Solvency Trends (Debt-to-Equity)
        if 'debt_to_equity' in data.columns:
            for company in data['company'].unique():
                company_data = data[data['company'] == company]
                axes[1, 0].plot(company_data['year'], company_data['debt_to_equity'],
                              marker='o', label=company, linewidth=2)
            axes[1, 0].set_title('Debt-to-Equity Ratio Trend')
            axes[1, 0].set_xlabel('Year')
            axes[1, 0].set_ylabel('Debt-to-Equity')
            axes[1, 0].axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Caution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Profitability Margin
        if 'net_profit_margin' in data.columns:
            for company in data['company'].unique():
                company_data = data[data['company'] == company]
                axes[1, 1].plot(company_data['year'], company_data['net_profit_margin'] * 100,
                              marker='o', label=company, linewidth=2)
            axes[1, 1].set_title('Net Profit Margin Trend')
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Profit Margin (%)')
            axes[1, 1].axhline(y=10, color='g', linestyle='--', alpha=0.5, label='Target')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.debug(f"Trend chart saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_risk_comparison(self, risk_results: Dict, save_path: Path = None):
        """
        Plot risk score comparison across companies.
        
        Args:
            risk_results: Risk scoring results
            save_path: Path to save the chart
        """
        companies = list(risk_results.keys())
        scores = [risk_results[c]['overall_score'] for c in companies]
        classifications = [risk_results[c]['classification'] for c in companies]
        
        # Create color map based on classification
        colors = []
        for classification in classifications:
            if classification == 'Stable':
                colors.append('green')
            elif classification == 'Caution':
                colors.append('orange')
            else:
                colors.append('red')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(companies, scores, color=colors, alpha=0.7, edgecolor='black')
        
        # Add score labels
        for i, (bar, score, classification) in enumerate(zip(bars, scores, classifications)):
            ax.text(score + 2, bar.get_y() + bar.get_height()/2,
                   f'{score:.1f} - {classification}',
                   va='center', fontweight='bold')
        
        # Add threshold lines
        ax.axvline(x=70, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Stable Threshold')
        ax.axvline(x=40, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Distress Threshold')
        
        # Add colored zones
        ax.axvspan(70, 100, alpha=0.1, color='green', label='Stable Zone')
        ax.axvspan(40, 70, alpha=0.1, color='orange', label='Caution Zone')
        ax.axvspan(0, 40, alpha=0.1, color='red', label='Distress Zone')
        
        ax.set_xlabel('Risk Score (0-100)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Company', fontsize=12, fontweight='bold')
        ax.set_title('Financial Risk Score Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 105)
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.debug(f"Risk comparison chart saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_category_scores(self, risk_results: Dict, save_path: Path = None):
        """
        Plot radar chart of category scores.
        
        Args:
            risk_results: Risk scoring results
            save_path: Path to save the chart
        """
        categories = ['Liquidity', 'Solvency', 'Profitability', 'Efficiency', 'Growth']
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for company, results in risk_results.items():
            scores = results['category_scores']
            values = [
                scores.get('liquidity', 0),
                scores.get('solvency', 0),
                scores.get('profitability', 0),
                scores.get('efficiency', 0),
                scores.get('growth', 0)
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=company)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=10)
        ax.set_title('Category Score Comparison', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.debug(f"Category scores chart saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_liquidity_analysis(self, data: pd.DataFrame, save_path: Path = None):
        """
        Plot liquidity metrics analysis.
        
        Args:
            data: DataFrame with liquidity ratios
            save_path: Path to save the chart
        """
        if 'year' not in data.columns:
            return
        
        latest_data = data.sort_values('year').groupby('company').last().reset_index()
        
        if not all(col in latest_data.columns for col in ['current_ratio', 'quick_ratio', 'cash_ratio']):
            logger.warning("Missing liquidity ratio columns")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(latest_data))
        width = 0.25
        
        ax.bar(x - width, latest_data['current_ratio'], width, label='Current Ratio', color='skyblue')
        ax.bar(x, latest_data['quick_ratio'], width, label='Quick Ratio', color='lightgreen')
        ax.bar(x + width, latest_data['cash_ratio'], width, label='Cash Ratio', color='lightcoral')
        
        ax.set_xlabel('Company', fontweight='bold')
        ax.set_ylabel('Ratio Value', fontweight='bold')
        ax.set_title('Liquidity Ratios Comparison (Latest Year)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(latest_data['company'], rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Minimum Threshold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.debug(f"Liquidity analysis chart saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_profitability_analysis(self, data: pd.DataFrame, save_path: Path = None):
        """
        Plot profitability metrics analysis.
        
        Args:
            data: DataFrame with profitability ratios
            save_path: Path to save the chart
        """
        if 'year' not in data.columns or 'company' not in data.columns:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Profitability Analysis', fontsize=16, fontweight='bold')
        
        # ROE comparison
        if 'roe' in data.columns:
            latest_data = data.sort_values('year').groupby('company').last().reset_index()
            axes[0].bar(latest_data['company'], latest_data['roe'] * 100, color='steelblue', alpha=0.7)
            axes[0].set_title('Return on Equity (ROE)')
            axes[0].set_ylabel('ROE (%)')
            axes[0].set_xlabel('Company')
            axes[0].axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Target (15%)')
            axes[0].legend()
            axes[0].grid(axis='y', alpha=0.3)
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Profit margins trend
        if all(col in data.columns for col in ['net_profit_margin', 'operating_margin']):
            for company in data['company'].unique():
                company_data = data[data['company'] == company].sort_values('year')
                axes[1].plot(company_data['year'], company_data['net_profit_margin'] * 100,
                           marker='o', label=f'{company} - Net', linewidth=2)
            axes[1].set_title('Profit Margin Trends')
            axes[1].set_ylabel('Margin (%)')
            axes[1].set_xlabel('Year')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.debug(f"Profitability analysis chart saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_risk_gauge(self, score: float, save_path: Path = None):
        """
        Create a gauge chart showing risk score.
        
        Args:
            score: Risk score (0-100)
            save_path: Path to save the chart
        """
        fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Background zones
        ax.fill_between(theta[:33], 0, 1, color='red', alpha=0.3, label='Distress')
        ax.fill_between(theta[33:70], 0, 1, color='orange', alpha=0.3, label='Caution')
        ax.fill_between(theta[70:], 0, 1, color='green', alpha=0.3, label='Stable')
        
        # Score needle
        score_angle = (score / 100) * np.pi
        ax.plot([score_angle, score_angle], [0, 1], color='black', linewidth=3)
        ax.scatter([score_angle], [1], color='black', s=100, zorder=5)
        
        # Labels
        ax.set_ylim(0, 1)
        ax.set_xticks([0, np.pi/2, np.pi])
        ax.set_xticklabels(['0', '50', '100'])
        ax.set_yticks([])
        ax.set_title(f'Risk Score: {score:.1f}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.debug(f"Risk gauge saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("ChartGenerator module ready")
    print("Use create_dashboard() to generate complete visualization suite")
