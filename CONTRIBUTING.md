# Contributing to Financial Distress EWS

Thank you for considering contributing to the Financial Distress Early Warning System! We welcome contributions from the community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Guidelines](#coding-guidelines)
- [Branching Strategy](#branching-strategy)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)

---

## 📜 Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other community members

---

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Screenshots** (if applicable)
- **Environment details** (OS, Python version, etc.)

**Example:**
```markdown
**Bug**: Risk score calculation returns NaN for missing data

**Steps to Reproduce:**
1. Upload CSV with missing revenue values
2. Click "Calculate Risk Score"
3. Observe NaN result

**Expected**: System should handle missing data gracefully
**Actual**: Calculation fails with NaN

**Environment**: Windows 11, Python 3.10, pandas 2.0.3
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title**
- **Detailed description** of the proposed feature
- **Use cases** and examples
- **Potential implementation approach** (if you have ideas)

### Pull Requests

We actively welcome your pull requests! Areas where contributions are particularly welcome:

- 🐛 Bug fixes
- ✨ New financial ratio calculations
- 📊 Additional visualization types
- 🧪 Test coverage improvements
- 📚 Documentation enhancements
- 🌍 Internationalization
- ⚡ Performance optimizations

---

## 🛠️ Development Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/financial-distress-ews.git
cd financial-distress-ews
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### 4. Install Pre-commit Hooks (Optional)
```bash
pre-commit install
```

### 5. Verify Setup
```bash
pytest tests/
```

---

## 💻 Coding Guidelines

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

#### Imports
```python
# Standard library imports
import os
import sys
from typing import List, Dict, Optional

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Local imports
from src.ratio_engine import ratios
from src.preprocessing import cleaner
```

#### Naming Conventions
- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

#### Documentation
Every function must have a docstring:

```python
def calculate_current_ratio(current_assets: float, current_liabilities: float) -> float:
    """
    Calculate the current ratio (liquidity metric).
    
    Args:
        current_assets (float): Total current assets
        current_liabilities (float): Total current liabilities
    
    Returns:
        float: Current ratio (current_assets / current_liabilities)
    
    Raises:
        ValueError: If current_liabilities is zero
        
    Example:
        >>> calculate_current_ratio(500000, 300000)
        1.67
    """
    if current_liabilities == 0:
        raise ValueError("Current liabilities cannot be zero")
    
    return current_assets / current_liabilities
```

#### Type Hints
Use type hints for all function signatures:

```python
from typing import List, Dict, Optional, Tuple

def analyze_ratios(
    data: pd.DataFrame, 
    years: List[int]
) -> Tuple[Dict[str, float], List[str]]:
    """Analyze financial ratios over multiple years."""
    pass
```

#### Error Handling
```python
# Good: Specific exception handling
try:
    ratio = calculate_ratio(assets, liabilities)
except ZeroDivisionError:
    logger.error("Cannot divide by zero in ratio calculation")
    ratio = None
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise

# Bad: Bare except
try:
    ratio = calculate_ratio(assets, liabilities)
except:  # Don't do this!
    pass
```

#### Logging
```python
import logging

logger = logging.getLogger(__name__)

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process financial data."""
    logger.info("Starting data processing")
    logger.debug(f"Input shape: {data.shape}")
    
    try:
        cleaned = clean_data(data)
        logger.info("Data processing completed successfully")
        return cleaned
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise
```

### Code Quality Standards

- **Line length**: Max 100 characters (use Black formatter)
- **Complexity**: Keep functions under 50 lines when possible
- **Comments**: Explain *why*, not *what*
- **Magic numbers**: Use named constants

```python
# Good
DISTRESS_THRESHOLD = 40
CAUTION_THRESHOLD = 70

if risk_score < DISTRESS_THRESHOLD:
    classification = "Distress"

# Bad
if risk_score < 40:  # What does 40 mean?
    classification = "Distress"
```

---

## 🌿 Branching Strategy

We use **Git Flow** branching model:

### Branch Types

1. **main**: Production-ready code
2. **develop**: Integration branch for features
3. **feature/**: New features (`feature/add-altman-zscore`)
4. **bugfix/**: Bug fixes (`bugfix/fix-ratio-calculation`)
5. **hotfix/**: Urgent production fixes (`hotfix/security-patch`)
6. **release/**: Release preparation (`release/v2.0.0`)

### Naming Convention

```
feature/short-description
bugfix/issue-number-description
hotfix/critical-issue
```

### Workflow

```bash
# Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/new-ratio-calculations

# Make changes, commit frequently
git add .
git commit -m "Add Altman Z-Score calculation"

# Push to your fork
git push origin feature/new-ratio-calculations

# Create Pull Request on GitHub
```

---

## 🔀 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass (`pytest tests/`)
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] No merge conflicts with `develop`

### PR Title Format

```
[Type] Short description

Types: Feature, Bugfix, Hotfix, Docs, Test, Refactor
```

**Examples:**
- `[Feature] Add Isolation Forest anomaly detection`
- `[Bugfix] Fix division by zero in ROE calculation`
- `[Docs] Update installation instructions`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature that breaks existing functionality)
- [ ] Documentation update

## Testing
Describe the tests you ran:
- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective
- [ ] New and existing unit tests pass locally

## Screenshots (if applicable)
```

### Review Process

1. At least **one approving review** required
2. All automated checks must pass (CI/CD)
3. No unresolved conversations
4. Branch must be up-to-date with base branch

---

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_ratios.py          # Ratio calculation tests
├── test_preprocessing.py   # Data cleaning tests
├── test_anomaly.py         # Anomaly detection tests
├── test_api.py             # API endpoint tests
└── fixtures/               # Test data
    └── sample_data.csv
```

### Writing Tests

```python
import pytest
import pandas as pd
from src.ratio_engine.ratios import calculate_current_ratio

class TestCurrentRatio:
    """Test suite for current ratio calculation."""
    
    def test_normal_calculation(self):
        """Test current ratio with valid inputs."""
        result = calculate_current_ratio(
            current_assets=500000,
            current_liabilities=300000
        )
        assert result == pytest.approx(1.67, rel=0.01)
    
    def test_zero_liabilities(self):
        """Test handling of zero liabilities."""
        with pytest.raises(ValueError):
            calculate_current_ratio(500000, 0)
    
    def test_negative_values(self):
        """Test handling of negative values."""
        with pytest.raises(ValueError):
            calculate_current_ratio(-500000, 300000)
    
    @pytest.mark.parametrize("assets,liabilities,expected", [
        (1000, 500, 2.0),
        (750, 250, 3.0),
        (100, 100, 1.0),
    ])
    def test_multiple_scenarios(self, assets, liabilities, expected):
        """Test multiple calculation scenarios."""
        result = calculate_current_ratio(assets, liabilities)
        assert result == pytest.approx(expected, rel=0.01)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_ratios.py

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_current"
```

### Test Coverage Requirements

- Minimum **80% code coverage** for new features
- **100% coverage** for critical financial calculations
- All edge cases must be tested

---

## 📝 Commit Message Guidelines

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting)
- **refactor**: Code refactoring
- **test**: Adding/updating tests
- **chore**: Maintenance tasks

### Examples

```bash
feat(ratios): add Altman Z-Score calculation

Implement the Altman Z-Score model for bankruptcy prediction.
Includes all five ratio components with proper weighting.

Closes #123

---

fix(preprocessing): handle missing values in revenue column

Previously caused NaN propagation in ratio calculations.
Now uses forward-fill interpolation for missing revenue.

Fixes #456
```

---

## 🏆 Recognition

Contributors will be added to the README.md acknowledgments section. Significant contributions may result in co-authorship credit.

---

## ❓ Questions?

- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Issues**: Report bugs via GitHub Issues
- 📧 **Email**: maintainer@yourproject.com

---

Thank you for contributing to Financial Distress EWS! 🎉

