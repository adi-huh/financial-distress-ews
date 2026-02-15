"""
Data Validation Framework

Comprehensive framework for validating financial data:
- Type validation
- Range validation
- Relationship validation
- Business rule validation
- Temporal validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class ValidationType(Enum):
    """Types of validation rules."""
    TYPE = "type_validation"
    RANGE = "range_validation"
    FORMAT = "format_validation"
    RELATIONSHIP = "relationship_validation"
    BUSINESS_RULE = "business_rule_validation"
    TEMPORAL = "temporal_validation"
    CONSISTENCY = "consistency_validation"
    UNIQUENESS = "uniqueness_validation"


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationRule:
    """Single validation rule."""
    name: str
    column: str
    rule_type: ValidationType
    condition: Callable  # Callable or condition check
    error_level: ValidationLevel
    error_message: str
    auto_fix: Optional[Callable[[Any], Any]] = None
    enabled: bool = True


@dataclass
class ValidationResult:
    """Result of validation check."""
    rule_name: str
    column: str
    validation_type: ValidationType
    passed: bool
    failed_rows: List[int] = field(default_factory=list)
    failed_values: List[Any] = field(default_factory=list)
    error_count: int = 0
    error_message: str = ""
    error_level: ValidationLevel = ValidationLevel.WARNING
    recommendation: str = ""


class ColumnValidator:
    """Validator for individual columns."""
    
    def __init__(self, column_name: str, column_type: str = 'numeric'):
        self.column_name = column_name
        self.column_type = column_type
        self.rules: List[ValidationRule] = []
        self.results: List[ValidationResult] = []
    
    def add_type_validation(self, expected_type: str) -> 'ColumnValidator':
        """Add type validation rule."""
        def check_type(val):
            if pd.isna(val):
                return True
            if expected_type == 'numeric':
                return isinstance(val, (int, float, np.number))
            elif expected_type == 'string':
                return isinstance(val, str)
            elif expected_type == 'datetime':
                return isinstance(val, (pd.Timestamp, datetime))
            return True
        
        rule = ValidationRule(
            name=f"{self.column_name}_type_check",
            column=self.column_name,
            rule_type=ValidationType.TYPE,
            condition=check_type,
            error_level=ValidationLevel.ERROR,
            error_message=f"Expected {expected_type} type"
        )
        self.rules.append(rule)
        return self
    
    def add_range_validation(self, min_val: float = None, max_val: float = None) -> 'ColumnValidator':
        """Add range validation rule."""
        def check_range(val):
            if pd.isna(val):
                return True
            if min_val is not None and val < min_val:
                return False
            if max_val is not None and val > max_val:
                return False
            return True
        
        rule = ValidationRule(
            name=f"{self.column_name}_range_check",
            column=self.column_name,
            rule_type=ValidationType.RANGE,
            condition=check_range,
            error_level=ValidationLevel.WARNING,
            error_message=f"Value outside range [{min_val}, {max_val}]"
        )
        self.rules.append(rule)
        return self
    
    def add_format_validation(self, pattern: str) -> 'ColumnValidator':
        """Add format validation using regex."""
        def check_format(val):
            if pd.isna(val):
                return True
            return bool(re.match(pattern, str(val)))
        
        rule = ValidationRule(
            name=f"{self.column_name}_format_check",
            column=self.column_name,
            rule_type=ValidationType.FORMAT,
            condition=check_format,
            error_level=ValidationLevel.WARNING,
            error_message=f"Value doesn't match pattern {pattern}"
        )
        self.rules.append(rule)
        return self
    
    def add_not_null_validation(self) -> 'ColumnValidator':
        """Add not-null validation."""
        def check_not_null(val):
            return not pd.isna(val)
        
        rule = ValidationRule(
            name=f"{self.column_name}_not_null_check",
            column=self.column_name,
            rule_type=ValidationType.BUSINESS_RULE,
            condition=check_not_null,
            error_level=ValidationLevel.ERROR,
            error_message="Value cannot be null"
        )
        self.rules.append(rule)
        return self
    
    def add_positive_validation(self) -> 'ColumnValidator':
        """Add positive value validation."""
        def check_positive(val):
            if pd.isna(val):
                return True
            return val > 0
        
        rule = ValidationRule(
            name=f"{self.column_name}_positive_check",
            column=self.column_name,
            rule_type=ValidationType.BUSINESS_RULE,
            condition=check_positive,
            error_level=ValidationLevel.WARNING,
            error_message="Value should be positive"
        )
        self.rules.append(rule)
        return self
    
    def add_custom_validation(self, 
                             name: str,
                             condition: Callable,
                             error_message: str,
                             error_level: ValidationLevel = ValidationLevel.WARNING) -> 'ColumnValidator':
        """Add custom validation rule."""
        rule = ValidationRule(
            name=f"{self.column_name}_{name}",
            column=self.column_name,
            rule_type=ValidationType.BUSINESS_RULE,
            condition=condition,
            error_level=error_level,
            error_message=error_message
        )
        self.rules.append(rule)
        return self
    
    def validate(self, series: pd.Series) -> List[ValidationResult]:
        """Validate series against all rules."""
        self.results = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Apply rule to each value
            valid_mask = series.apply(lambda x: rule.condition(x) if callable(rule.condition) else True)
            failed_indices = np.where(~valid_mask)[0].tolist()
            
            result = ValidationResult(
                rule_name=rule.name,
                column=rule.column,
                validation_type=rule.rule_type,
                passed=len(failed_indices) == 0,
                failed_rows=failed_indices,
                failed_values=[series.iloc[i] for i in failed_indices],
                error_count=len(failed_indices),
                error_message=rule.error_message,
                error_level=rule.error_level
            )
            
            self.results.append(result)
        
        return self.results


class RelationshipValidator:
    """Validator for relationships between columns."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.rules: List[Callable] = []
        self.results: List[ValidationResult] = []
    
    def add_relationship_rule(self,
                             name: str,
                             columns: List[str],
                             condition: Callable,
                             error_level: ValidationLevel = ValidationLevel.WARNING) -> 'RelationshipValidator':
        """Add relationship validation rule."""
        def check_relationship(row):
            values = {col: row[col] for col in columns if col in row.index}
            return condition(values)
        
        self.rules.append({
            'name': name,
            'columns': columns,
            'check': check_relationship,
            'error_level': error_level
        })
        return self
    
    def add_sum_rule(self, 
                    name: str,
                    components: List[str],
                    total_column: str,
                    tolerance: float = 0.01) -> 'RelationshipValidator':
        """Add sum validation rule (e.g., assets = liabilities + equity)."""
        def check_sum(row):
            if any(pd.isna(row[col]) for col in components + [total_column]):
                return True
            component_sum = sum(row[col] for col in components)
            return abs(component_sum - row[total_column]) <= tolerance * row[total_column]
        
        return self.add_relationship_rule(
            name,
            components + [total_column],
            check_sum,
            ValidationLevel.ERROR
        )
    
    def add_ratio_bounds_rule(self,
                             name: str,
                             numerator: str,
                             denominator: str,
                             min_ratio: float = 0,
                             max_ratio: float = float('inf')) -> 'RelationshipValidator':
        """Add ratio bounds validation rule."""
        def check_ratio(row):
            if pd.isna(row[numerator]) or pd.isna(row[denominator]) or row[denominator] == 0:
                return True
            ratio = row[numerator] / row[denominator]
            return min_ratio <= ratio <= max_ratio
        
        return self.add_relationship_rule(
            name,
            [numerator, denominator],
            check_ratio,
            ValidationLevel.WARNING
        )
    
    def add_hierarchy_rule(self,
                          name: str,
                          parent: str,
                          child: str,
                          operator: str = '<=') -> 'RelationshipValidator':
        """Add hierarchical relationship rule (e.g., current_assets <= total_assets)."""
        def check_hierarchy(row):
            if pd.isna(row[parent]) or pd.isna(row[child]):
                return True
            
            if operator == '<=':
                return row[child] <= row[parent]
            elif operator == '>=':
                return row[child] >= row[parent]
            elif operator == '<':
                return row[child] < row[parent]
            elif operator == '>':
                return row[child] > row[parent]
            return True
        
        return self.add_relationship_rule(
            name,
            [parent, child],
            check_hierarchy,
            ValidationLevel.WARNING
        )
    
    def validate(self) -> List[ValidationResult]:
        """Validate all relationship rules."""
        self.results = []
        
        for rule in self.rules:
            failed_indices = []
            
            for idx, row in self.df.iterrows():
                try:
                    if not rule['check'](row):
                        failed_indices.append(idx)
                except Exception as e:
                    logger.warning(f"Error checking rule {rule['name']}: {e}")
            
            result = ValidationResult(
                rule_name=rule['name'],
                column=','.join(rule['columns']),
                validation_type=ValidationType.RELATIONSHIP,
                passed=len(failed_indices) == 0,
                failed_rows=failed_indices,
                error_count=len(failed_indices),
                error_level=rule['error_level']
            )
            
            self.results.append(result)
        
        return self.results


class TemporalValidator:
    """Validator for temporal/time-series data."""
    
    def __init__(self, df: pd.DataFrame, date_column: str = 'date'):
        self.df = df
        self.date_column = date_column
        self.results: List[ValidationResult] = []
    
    def validate_sequence(self) -> ValidationResult:
        """Check if dates are in chronological order."""
        if self.date_column not in self.df.columns:
            return ValidationResult(
                rule_name="temporal_sequence",
                column=self.date_column,
                validation_type=ValidationType.TEMPORAL,
                passed=True,
                error_message="Date column not found"
            )
        
        dates = pd.to_datetime(self.df[self.date_column])
        is_sorted = dates.is_monotonic_increasing
        
        result = ValidationResult(
            rule_name="temporal_sequence",
            column=self.date_column,
            validation_type=ValidationType.TEMPORAL,
            passed=is_sorted,
            error_message="Dates are not in chronological order"
        )
        
        self.results.append(result)
        return result
    
    def validate_frequency(self, expected_freq: str = 'D') -> ValidationResult:
        """Check if temporal data has consistent frequency."""
        if self.date_column not in self.df.columns:
            return ValidationResult(
                rule_name="temporal_frequency",
                column=self.date_column,
                validation_type=ValidationType.TEMPORAL,
                passed=True
            )
        
        dates = pd.to_datetime(self.df[self.date_column])
        
        if len(dates) < 2:
            return ValidationResult(
                rule_name="temporal_frequency",
                column=self.date_column,
                validation_type=ValidationType.TEMPORAL,
                passed=True,
                error_message="Not enough data points"
            )
        
        # Check frequency consistency
        diffs = dates.diff()[1:]  # Skip first NaN
        expected_diff = pd.Timedelta(expected_freq)
        
        # Allow small tolerance
        consistent = (diffs == expected_diff).sum() / len(diffs) > 0.95
        
        result = ValidationResult(
            rule_name="temporal_frequency",
            column=self.date_column,
            validation_type=ValidationType.TEMPORAL,
            passed=consistent,
            error_message=f"Dates don't have consistent {expected_freq} frequency"
        )
        
        self.results.append(result)
        return result


class ValidationFramework:
    """Main validation framework combining all validators."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_validators: Dict[str, ColumnValidator] = {}
        self.relationship_validator: Optional[RelationshipValidator] = None
        self.temporal_validator: Optional[TemporalValidator] = None
        self.all_results: List[ValidationResult] = []
    
    def add_column_validator(self, column: str, col_type: str = 'numeric') -> ColumnValidator:
        """Add column validator."""
        validator = ColumnValidator(column, col_type)
        self.column_validators[column] = validator
        return validator
    
    def get_relationship_validator(self) -> RelationshipValidator:
        """Get relationship validator."""
        if self.relationship_validator is None:
            self.relationship_validator = RelationshipValidator(self.df)
        return self.relationship_validator
    
    def get_temporal_validator(self, date_column: str = 'date') -> TemporalValidator:
        """Get temporal validator."""
        if self.temporal_validator is None:
            self.temporal_validator = TemporalValidator(self.df, date_column)
        return self.temporal_validator
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validations."""
        self.all_results = []
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_checks': 0,
            'passed': 0,
            'failed': 0,
            'by_level': {},
            'details': []
        }
        
        # Validate columns
        for col_name, validator in self.column_validators.items():
            if col_name in self.df.columns:
                results = validator.validate(self.df[col_name])
                self.all_results.extend(results)
        
        # Validate relationships
        if self.relationship_validator:
            results = self.relationship_validator.validate()
            self.all_results.extend(results)
        
        # Validate temporal
        if self.temporal_validator:
            # Results already added during validation
            pass
        
        # Compile report
        for result in self.all_results:
            report['total_checks'] += 1
            
            if result.passed:
                report['passed'] += 1
            else:
                report['failed'] += 1
            
            level = result.error_level.value
            report['by_level'][level] = report['by_level'].get(level, 0) + 1
            
            if not result.passed:
                report['details'].append({
                    'rule': result.rule_name,
                    'column': result.column,
                    'type': result.validation_type.value,
                    'level': level,
                    'errors': result.error_count,
                    'message': result.error_message
                })
        
        return report


# Example usage
if __name__ == "__main__":
    # Create sample financial data
    sample_df = pd.DataFrame({
        'company': ['A', 'B', 'C', 'D', 'E'],
        'revenue': [1000, 2000, 3000, -500, 1500],  # D has negative
        'expenses': [500, 1000, 1200, 300, 800],
        'profit': [500, 1000, 1800, -800, 700],
        'total_assets': [5000, 6000, 7000, 8000, 9000],
        'current_assets': [2000, 2500, 3000, 3500, 4000],
        'date': pd.date_range('2020-01-01', periods=5, freq='M')
    })
    
    # Create validation framework
    framework = ValidationFramework(sample_df)
    
    # Add column validators
    framework.add_column_validator('revenue').add_positive_validation()
    framework.add_column_validator('profit').add_not_null_validation()
    
    # Add relationship validators
    rel_val = framework.get_relationship_validator()
    rel_val.add_ratio_bounds_rule('profit_margin', 'profit', 'revenue', 0, 1)
    rel_val.add_hierarchy_rule('hierarchy', 'total_assets', 'current_assets', '>=')
    
    # Run validation
    report = framework.validate_all()
    
    print(f"Validation Report:")
    print(f"Total Checks: {report['total_checks']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")
    print(f"By Level: {report['by_level']}")
