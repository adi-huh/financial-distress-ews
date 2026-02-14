# Developer Guide

## Development Setup

### 1. Clone and Setup Environment

```bash
# Clone repository
git clone https://github.com/adi-huh/financial-distress-ews.git
cd financial-distress-ews

# Create virtual environment
python3.8+ -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Project Structure Walkthrough

```
financial-distress-ews/
├── app.py                          # Streamlit dashboard entry point
├── main.py                         # CLI entry point
├── loader.py                       # Data loading module
├── cleaner.py                      # Data preprocessing module
├── ratios.py                       # Financial ratio calculations
├── timeseries.py                   # Time-series analysis
├── zscore.py                       # Anomaly detection
├── score.py                        # Risk scoring engine
├── recommend.py                    # Consulting recommendations
├── charts.py                       # Visualization module
│
├── data/
│   ├── raw/                        # Original financial data
│   ├── processed/                  # Cleaned data (generated)
│   └── sample_data.csv             # Sample dataset
│
├── results/                        # Generated analysis results
│   ├── charts/                     # Generated charts
│   └── *.csv                       # Exported results
│
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
│
├── README.md                       # Project overview
├── ARCHITECTURE.md                 # System architecture
├── CONTRIBUTING.md                 # Contribution guidelines
├── SETUP_GUIDE.md                  # Installation instructions
├── QUICK_START.md                  # Quick start guide
├── LICENSE                         # MIT License
└── .gitignore                      # Git ignore patterns
```

## Code Style Guidelines

### 1. Python Style (PEP 8)

```python
# Good: Clear, descriptive names
def calculate_current_ratio(current_assets, current_liabilities):
    """Calculate current ratio (liquidity measure)."""
    if current_liabilities == 0:
        return None
    return current_assets / current_liabilities

# Bad: Ambiguous names
def calc_cr(ca, cl):
    if cl == 0:
        return None
    return ca / cl
```

### 2. Docstring Format

All functions must have docstrings following Google/NumPy style:

```python
def process_financial_data(input_file: str, output_dir: str = 'results') -> pd.DataFrame:
    """
    Process financial data through the complete pipeline.
    
    Loads raw financial data, applies cleaning, calculates ratios,
    performs analysis, and generates visualizations.
    
    Args:
        input_file (str): Path to CSV or Excel file containing financial data
        output_dir (str): Directory to save results (default: 'results')
    
    Returns:
        pd.DataFrame: Processed data with all calculated ratios
    
    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If data format is invalid
    
    Example:
        >>> data = process_financial_data('data/companies.csv')
        >>> print(data.head())
    """
```

### 3. Type Hints

Use type hints for all function signatures:

```python
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

def calculate_ratios(data: pd.DataFrame, 
                    company: str,
                    years: Optional[List[int]] = None) -> Dict[str, float]:
    """Calculate ratios for specific company and years."""
    pass
```

### 4. Logging

Use the logging module consistently:

```python
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def load_file(self, filepath: str) -> pd.DataFrame:
        logger.info(f"Loading data from: {filepath}")
        try:
            data = pd.read_csv(filepath)
            logger.debug(f"Loaded {len(data)} rows")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
```

### 5. Error Handling

Provide meaningful error messages:

```python
# Good: Specific exception with context
try:
    ratio = current_assets / current_liabilities
except ZeroDivisionError as e:
    logger.warning(f"Cannot calculate ratio for company {company}: zero liabilities")
    return None

# Bad: Silent failure
try:
    ratio = current_assets / current_liabilities
except:
    pass
```

## Module Development Guide

### 1. Adding a New Financial Ratio

Location: `ratios.py`

```python
def _calculate_custom_ratio(self, row: pd.Series) -> float:
    """
    Calculate custom financial ratio.
    
    Formula: (Value A - Value B) / Value C
    Interpretation: Higher is better
    """
    numerator = row.get('value_a', 0) - row.get('value_b', 0)
    denominator = row.get('value_c', 0)
    
    if denominator == 0:
        logger.warning("Denominator is zero, returning NaN")
        return np.nan
    
    return numerator / denominator

# Register in calculate_all_ratios()
result['custom_ratio'] = result.apply(
    lambda row: self._calculate_custom_ratio(row), 
    axis=1
)
```

### 2. Adding New Anomaly Detection Algorithm

Location: `zscore.py`

```python
class CustomDetector:
    """Detect anomalies using custom algorithm."""
    
    def __init__(self, sensitivity: float = 0.95):
        """Initialize detector with sensitivity parameter."""
        self.sensitivity = sensitivity
        logger.info(f"CustomDetector initialized with sensitivity={sensitivity}")
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using custom method."""
        # Implementation here
        pass
```

### 3. Adding Visualization Chart

Location: `charts.py`

```python
def create_custom_chart(self, data: pd.DataFrame, title: str):
    """Create custom visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Your plotting code here
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Category')
    ax.set_ylabel('Value')
    
    # Save figure
    output_file = self.output_dir / f"{title.lower().replace(' ', '_')}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Chart saved: {output_file}")
    plt.close(fig)
```

## Testing Guidelines

### 1. Unit Tests

Location: `tests/test_ratios.py`

```python
import pytest
import pandas as pd
from ratios import FinancialRatioEngine

class TestFinancialRatioEngine:
    
    @pytest.fixture
    def engine(self):
        """Fixture to initialize engine."""
        return FinancialRatioEngine()
    
    @pytest.fixture
    def sample_data(self):
        """Fixture with sample financial data."""
        return pd.DataFrame({
            'company': ['TechCorp'],
            'year': [2024],
            'revenue': [1000000],
            'net_income': [100000],
            'total_assets': [2000000],
            'current_assets': [500000],
            'current_liabilities': [300000],
            'total_debt': [800000],
            'equity': [1200000]
        })
    
    def test_current_ratio_calculation(self, engine, sample_data):
        """Test current ratio calculation."""
        result = engine.calculate_all_ratios(sample_data)
        
        # Current ratio = 500000 / 300000 = 1.667
        assert abs(result['current_ratio'].iloc[0] - 1.667) < 0.01
    
    def test_zero_division_handling(self, engine):
        """Test handling of zero values."""
        data = pd.DataFrame({
            'current_assets': [100],
            'current_liabilities': [0]  # This would cause division by zero
        })
        
        result = engine._calculate_liquidity_ratios(data)
        assert pd.isna(result['current_ratio'].iloc[0])
```

### 2. Integration Tests

Location: `tests/test_pipeline.py`

```python
def test_complete_pipeline():
    """Test complete analysis pipeline."""
    # Load sample data
    loader = DataLoader()
    data = loader.load_file('data/sample_data.csv')
    
    # Clean data
    cleaner = DataCleaner()
    clean_data = cleaner.clean(data)
    
    # Calculate ratios
    engine = FinancialRatioEngine()
    ratios = engine.calculate_all_ratios(clean_data)
    
    # Detect anomalies
    detector = AnomalyDetectionEngine()
    anomalies = detector.detect_all_anomalies(ratios)
    
    # Score risk
    scorer = RiskScoreEngine()
    scores = scorer.calculate_risk_score(ratios, anomalies)
    
    # Assert results
    assert len(scores) > 0
    assert all('overall_score' in v for v in scores.values())
    assert all(0 <= v['overall_score'] <= 100 for v in scores.values())
```

### 3. Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_ratios.py

# Run with verbose output
pytest -v tests/

# Run specific test function
pytest tests/test_ratios.py::TestFinancialRatioEngine::test_current_ratio_calculation
```

## Common Development Tasks

### 1. Adding Debug Output

```python
import logging

# Set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add debug statements
logger.debug(f"Processing company: {company}")
logger.debug(f"Data shape: {data.shape}")
logger.debug(f"Calculated ratio: {ratio:.4f}")
```

### 2. Profiling Performance

```python
import cProfile
import pstats

# Profile a function
profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = calculate_all_ratios(data)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

### 3. Handling Large Datasets

```python
# Process in chunks
def process_large_file(filepath: str, chunk_size: int = 10000):
    """Process large CSV file in chunks."""
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        processed = process_chunk(chunk)
        yield processed
```

## Git Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/add-new-ratio

# Make changes, commit regularly
git add .
git commit -m "Add new financial ratio calculation"

# Keep branch updated
git fetch origin
git rebase origin/main

# Push to remote
git push origin feature/add-new-ratio

# Create Pull Request on GitHub
# Include description of changes and testing performed
```

### 2. Commit Message Format

```
[TYPE] Short description (50 chars max)

Longer description explaining the change.
Can span multiple lines.

- List specific changes
- Or improvements made
- And any breaking changes

Fixes #123  # Reference issues if applicable
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### 3. Code Review Checklist

- [ ] Code follows PEP 8 style guide
- [ ] Docstrings present and accurate
- [ ] Type hints included
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] No logging of sensitive data
- [ ] Performance impact analyzed
- [ ] Documentation updated

## Debugging Techniques

### 1. Using Python Debugger

```python
import pdb

def calculate_ratio(data):
    pdb.set_trace()  # Execution stops here
    ratio = data['value_a'] / data['value_b']
    return ratio
```

Commands: `n` (next), `c` (continue), `p variable` (print), `l` (list code)

### 2. Logging Strategy

```python
# Development: Detailed logging
logging.basicConfig(level=logging.DEBUG)

# Production: Only important messages
logging.basicConfig(level=logging.INFO)

# Emergency: Minimal logging
logging.basicConfig(level=logging.ERROR)
```

### 3. Assertion-Driven Development

```python
def validate_ratio(ratio: float) -> bool:
    """Validate ratio is within expected bounds."""
    assert not pd.isna(ratio), "Ratio is NaN"
    assert ratio >= 0, "Ratio cannot be negative"
    assert ratio <= 100, "Ratio exceeds maximum"
    return True
```

## Performance Optimization Tips

1. **Use Vectorized Operations**: Prefer pandas/numpy over loops
2. **Cache Results**: Store expensive computations
3. **Profile Regularly**: Identify bottlenecks
4. **Use Generators**: For large datasets
5. **Parallel Processing**: Use multiprocessing for I/O-bound tasks

## Documentation Standards

Every module should have:
- Module-level docstring
- Class docstrings with responsibility
- Method docstrings with Args/Returns
- Inline comments for complex logic
- Usage examples where applicable

## Release Checklist

- [ ] All tests passing
- [ ] Code reviewed and approved
- [ ] Dependencies updated
- [ ] Version number bumped
- [ ] CHANGELOG.md updated
- [ ] README.md reflects new features
- [ ] Performance benchmarks acceptable
- [ ] No breaking changes documented
- [ ] Release notes prepared
- [ ] Tagged in git

## Getting Help

- Check existing issues on GitHub
- Review documentation in docs/
- Run tests to identify problems
- Check logs for error messages
- Ask in team discussions/Slack
