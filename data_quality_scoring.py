"""
Data Quality Scoring System

Comprehensive system for assessing and scoring data quality:
- Completeness scoring
- Consistency scoring
- Validity scoring
- Uniqueness scoring
- Accuracy scoring
- Timeliness scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Dimensions of data quality."""
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    INTEGRITY = "integrity"


@dataclass
class DimensionScore:
    """Score for a single quality dimension."""
    dimension: QualityDimension
    score: float  # 0-100
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    timestamp: str
    overall_score: float  # 0-100
    dimension_scores: Dict[str, float]
    total_issues: int
    data_shape: Tuple[int, int]
    dimension_details: Dict[str, DimensionScore] = field(default_factory=dict)
    critical_issues: List[str] = field(default_factory=list)
    summary: str = ""


class CompletenessScorer:
    """Scores data completeness (missing values)."""
    
    @staticmethod
    def score(df: pd.DataFrame, by_column: bool = False) -> Dict[str, Any]:
        """
        Score completeness of data.
        
        Args:
            df: Input dataframe
            by_column: Return scores per column if True
            
        Returns:
            Completeness score and details
        """
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        if by_column:
            column_completeness = {}
            for col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                column_completeness[col] = 100 - missing_pct
            
            overall = 100 - (missing_cells / total_cells * 100)
            
            return {
                'score': overall,
                'percentage_missing': (missing_cells / total_cells) * 100,
                'missing_cells': int(missing_cells),
                'by_column': column_completeness
            }
        
        else:
            completeness = 100 - (missing_cells / total_cells * 100)
            
            return {
                'score': max(0, min(100, completeness)),
                'percentage_missing': (missing_cells / total_cells) * 100,
                'missing_cells': int(missing_cells),
                'total_cells': total_cells
            }


class ConsistencyScorer:
    """Scores data consistency (format, type, and relationship consistency)."""
    
    @staticmethod
    def score(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Score data consistency.
        
        Args:
            df: Input dataframe
            
        Returns:
            Consistency score and details
        """
        issues = []
        consistency_score = 100
        
        # Check data type consistency
        for col in df.columns:
            # Check for mixed types
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='raise')
                except:
                    # Mixed types detected
                    issues.append(f"Column '{col}' has mixed data types")
                    consistency_score -= 5
        
        # Check for inconsistent column names
        standardized = df.columns.str.lower().str.replace(' ', '_')
        if not (df.columns == standardized).all():
            issues.append("Column names are not standardized")
            consistency_score -= 10
        
        # Check for duplicate column names
        if df.columns.duplicated().any():
            issues.append("Duplicate column names detected")
            consistency_score -= 20
        
        # Check for rows with all same value (potential copy-paste error)
        for idx, row in df.iterrows():
            if row.nunique() == 1:
                issues.append(f"Row {idx} has all identical values")
                consistency_score -= 2
                if consistency_score < 0:
                    consistency_score = 0
                    break
        
        # Check for impossible combinations
        if 'total_assets' in df.columns and 'current_assets' in df.columns:
            impossible = (df['current_assets'] > df['total_assets']).sum()
            if impossible > 0:
                issues.append(f"Current assets > Total assets in {impossible} rows")
                consistency_score -= 10
        
        return {
            'score': max(0, consistency_score),
            'issues': issues,
            'issues_count': len(issues)
        }


class ValidityScorer:
    """Scores data validity (correct values and formats)."""
    
    @staticmethod
    def score(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Score data validity.
        
        Args:
            df: Input dataframe
            
        Returns:
            Validity score and details
        """
        issues = []
        validity_score = 100
        
        # Check for numeric columns with negative values where they shouldn't be
        invalid_negative = []
        for col in df.select_dtypes(include=[np.number]).columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['revenue', 'sales', 'quantity', 'count', 'amount']):
                if (df[col] < 0).any():
                    invalid_negative.append(col)
                    issues.append(f"Column '{col}' has negative values where positive expected")
                    validity_score -= 10
        
        # Check for impossible percentage values
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['percentage', 'rate', 'pct']):
                if df[col].dtype in [np.float64, np.int64]:
                    invalid_pct = ((df[col] < 0) | (df[col] > 100)).sum()
                    if invalid_pct > 0:
                        issues.append(f"Column '{col}' has values outside 0-100 range")
                        validity_score -= 10
        
        # Check for outliers using IQR method
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = ((df[col] < Q1 - 3 * IQR) | (df[col] > Q3 + 3 * IQR)).sum()
            if outliers > 0:
                outlier_pct = (outliers / len(df)) * 100
                if outlier_pct > 5:
                    issues.append(f"Column '{col}' has {outlier_pct:.1f}% extreme outliers")
                    validity_score -= 5
        
        return {
            'score': max(0, validity_score),
            'issues': issues,
            'issues_count': len(issues),
            'negative_value_columns': invalid_negative
        }


class UniquenessScorer:
    """Scores data uniqueness (duplicate detection)."""
    
    @staticmethod
    def score(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Score data uniqueness.
        
        Args:
            df: Input dataframe
            
        Returns:
            Uniqueness score and details
        """
        # Check for complete duplicates
        complete_duplicates = df.duplicated().sum()
        complete_dup_pct = (complete_duplicates / len(df)) * 100 if len(df) > 0 else 0
        
        # Check for partial duplicates (same key columns)
        key_columns = ['company', 'company_id', 'ticker', 'symbol']
        key_cols_present = [col for col in key_columns if col in df.columns]
        
        partial_duplicates = 0
        if key_cols_present:
            partial_duplicates = df.duplicated(subset=key_cols_present).sum()
        
        # Calculate uniqueness score
        if complete_dup_pct > 10:
            uniqueness_score = 50
        elif complete_dup_pct > 5:
            uniqueness_score = 70
        elif complete_dup_pct > 1:
            uniqueness_score = 85
        else:
            uniqueness_score = 95
        
        issues = []
        if complete_duplicates > 0:
            issues.append(f"{complete_duplicates} complete duplicate rows ({complete_dup_pct:.1f}%)")
        if partial_duplicates > 0:
            issues.append(f"{partial_duplicates} partial duplicate rows")
        
        return {
            'score': uniqueness_score,
            'complete_duplicates': int(complete_duplicates),
            'partial_duplicates': int(partial_duplicates),
            'percentage': complete_dup_pct,
            'issues': issues
        }


class AccuracyScorer:
    """Scores data accuracy (based on known constraints and relationships)."""
    
    @staticmethod
    def score(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Score data accuracy based on financial relationships.
        
        Args:
            df: Input dataframe
            
        Returns:
            Accuracy score and details
        """
        issues = []
        accuracy_score = 100
        
        # Check balance sheet equation: Assets = Liabilities + Equity
        if all(col in df.columns for col in ['total_assets', 'total_liabilities', 'total_equity']):
            expected = df['total_liabilities'] + df['total_equity']
            error = (df['total_assets'] - expected).abs()
            error_pct = (error / df['total_assets']).mean()
            
            if error_pct > 0.05:  # 5% tolerance
                issues.append(f"Balance sheet equation error: {error_pct*100:.1f}%")
                accuracy_score -= 20
        
        # Check profit equation: Income = Revenue - Expenses
        if all(col in df.columns for col in ['revenue', 'expenses', 'net_income']):
            expected = df['revenue'] - df['expenses']
            error = (df['net_income'] - expected).abs()
            error_pct = (error / df['revenue']).mean()
            
            if error_pct > 0.05:
                issues.append(f"Profit equation error: {error_pct*100:.1f}%")
                accuracy_score -= 20
        
        # Check ratio reasonableness
        if all(col in df.columns for col in ['revenue', 'net_income']):
            margin = df['net_income'] / df['revenue']
            unreasonable = (margin > 1.0).sum() or (margin < -1.0).sum()
            if unreasonable > 0:
                issues.append(f"Profit margin values are unreasonable")
                accuracy_score -= 15
        
        return {
            'score': max(0, accuracy_score),
            'issues': issues,
            'issues_count': len(issues)
        }


class TimelinessScorer:
    """Scores data timeliness (how current the data is)."""
    
    @staticmethod
    def score(df: pd.DataFrame, date_column: str = 'date', 
             reference_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Score data timeliness.
        
        Args:
            df: Input dataframe
            date_column: Name of date column
            reference_date: Reference date to measure against
            
        Returns:
            Timeliness score and details
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        if date_column not in df.columns:
            return {'score': 50, 'reason': 'Date column not found'}
        
        try:
            df_dates = pd.to_datetime(df[date_column])
            latest_date = df_dates.max()
            days_old = (reference_date - latest_date).days
            
            if days_old == 0:
                timeliness_score = 100
            elif days_old < 7:  # Less than a week old
                timeliness_score = 95
            elif days_old < 30:  # Less than a month old
                timeliness_score = 85
            elif days_old < 90:  # Less than 3 months old
                timeliness_score = 70
            elif days_old < 180:  # Less than 6 months old
                timeliness_score = 50
            else:
                timeliness_score = 20
            
            return {
                'score': timeliness_score,
                'latest_date': latest_date.strftime('%Y-%m-%d'),
                'days_old': days_old,
                'date_range': f"{df_dates.min().strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}"
            }
        
        except Exception as e:
            logger.warning(f"Error scoring timeliness: {e}")
            return {'score': 50, 'reason': 'Error parsing dates'}


class IntegrityScorer:
    """Scores referential integrity and relationships."""
    
    @staticmethod
    def score(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Score data integrity based on logical relationships.
        
        Args:
            df: Input dataframe
            
        Returns:
            Integrity score and details
        """
        issues = []
        integrity_score = 100
        
        # Check for required columns
        financial_cols = ['revenue', 'expenses', 'profit', 'assets', 'liabilities']
        present_cols = [col for col in financial_cols if col in df.columns]
        
        if len(present_cols) < 2:
            issues.append("Missing required financial columns")
            integrity_score -= 20
        
        # Check for logical relationships between columns
        if 'revenue' in df.columns and 'expenses' in df.columns:
            if 'profit' in df.columns:
                # Verify relationship
                for idx, row in df.iterrows():
                    if pd.notna(row['revenue']) and pd.notna(row['expenses']) and pd.notna(row['profit']):
                        expected = row['revenue'] - row['expenses']
                        if abs(expected - row['profit']) > 0.01:
                            integrity_score -= 1
                            if integrity_score < 0:
                                break
        
        return {
            'score': max(0, integrity_score),
            'issues': issues,
            'present_columns': present_cols,
            'issues_count': len(issues)
        }


class DataQualityScorer:
    """Main data quality scoring system."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.dimension_scores: Dict[QualityDimension, DimensionScore] = {}
    
    def score_all(self, date_column: str = 'date') -> DataQualityReport:
        """
        Score all dimensions of data quality.
        
        Args:
            date_column: Name of date column for timeliness scoring
            
        Returns:
            Comprehensive data quality report
        """
        # Score each dimension
        completeness = CompletenessScorer.score(self.df)
        consistency = ConsistencyScorer.score(self.df)
        validity = ValidityScorer.score(self.df)
        uniqueness = UniquenessScorer.score(self.df)
        accuracy = AccuracyScorer.score(self.df)
        timeliness = TimelinessScorer.score(self.df, date_column)
        integrity = IntegrityScorer.score(self.df)
        
        # Calculate overall score (weighted average)
        scores = {
            'completeness': completeness['score'],
            'consistency': consistency['score'],
            'validity': validity['score'],
            'uniqueness': uniqueness['score'],
            'accuracy': accuracy['score'],
            'timeliness': timeliness['score'],
            'integrity': integrity['score']
        }
        
        # Weights (completeness and consistency are critical)
        weights = {
            'completeness': 0.25,
            'consistency': 0.25,
            'validity': 0.15,
            'uniqueness': 0.10,
            'accuracy': 0.15,
            'timeliness': 0.05,
            'integrity': 0.05
        }
        
        overall_score = sum(scores[k] * weights[k] for k in scores.keys())
        
        # Collect issues
        all_issues = []
        all_issues.extend(consistency.get('issues', []))
        all_issues.extend(validity.get('issues', []))
        all_issues.extend(uniqueness.get('issues', []))
        all_issues.extend(accuracy.get('issues', []))
        all_issues.extend(integrity.get('issues', []))
        
        critical_issues = [issue for issue in all_issues 
                         if any(term in issue.lower() for term in ['balance', 'equation', 'duplicate', 'missing'])]
        
        # Generate summary
        if overall_score >= 90:
            summary = "Excellent data quality. Minimal issues detected."
        elif overall_score >= 80:
            summary = "Good data quality. Some minor issues found."
        elif overall_score >= 70:
            summary = "Fair data quality. Several issues require attention."
        elif overall_score >= 60:
            summary = "Poor data quality. Multiple issues found."
        else:
            summary = "Critical data quality issues. Significant problems detected."
        
        report = DataQualityReport(
            timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            dimension_scores=scores,
            total_issues=len(all_issues),
            data_shape=self.df.shape,
            critical_issues=critical_issues,
            summary=summary
        )
        
        return report


# Example usage
if __name__ == "__main__":
    # Create sample data with some quality issues
    sample_df = pd.DataFrame({
        'company': ['Apple', 'Microsoft', 'Google', 'Apple', 'Amazon'],
        'revenue': [100000, 120000, 90000, 100000, 110000],  # Duplicate Apple
        'expenses': [60000, 70000, 50000, 60000, 65000],
        'profit': [40000, 50000, 40000, 40000, 45000],
        'total_assets': [500000, 600000, 550000, 500000, 700000],
        'total_liabilities': [200000, np.nan, 250000, 200000, 300000],  # Missing
        'total_equity': [300000, 350000, 300000, 300000, 400000],
        'date': pd.date_range('2023-01-01', periods=5, freq='M')
    })
    
    # Score quality
    scorer = DataQualityScorer(sample_df)
    report = scorer.score_all()
    
    print("Data Quality Report")
    print("=" * 50)
    print(f"Overall Score: {report.overall_score:.1f}/100")
    print(f"\nDimension Scores:")
    for dim, score in report.dimension_scores.items():
        print(f"  {dim}: {score:.1f}")
    print(f"\nTotal Issues Found: {report.total_issues}")
    print(f"\nSummary: {report.summary}")
