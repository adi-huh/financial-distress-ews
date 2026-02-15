# System Architecture

## Overview

The Financial Distress Early Warning System is built with a modular, layered architecture following clean code principles.

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                          │
│  ┌──────────────────┐              ┌──────────────────────────┐ │
│  │ Streamlit        │              │ FastAPI / REST API       │ │
│  │ Dashboard        │              │ (Optional)               │ │
│  └──────────────────┘              └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │  Risk    │ │ Anomaly  │ │Consulting│ │  Visualization  │  │
│  │  Scoring │ │Detection │ │ Engine   │ │  (Charts/Viz)   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────────────┐   │
│  │ Financial│ │ Time     │ │ Financial Ratio Engine       │   │
│  │ Ratios   │ │ Series   │ │ (20+ ratios computed)        │   │
│  └──────────┘ └──────────┘ └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING LAYER                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   DataCleaner (Validation, Outlier Removal, Imputation) │  │
│  │   - Missing value handling                              │  │
│  │   - Outlier detection (IQR/Z-score)                     │  │
│  │   - Data normalization                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   DataLoader (CSV/Excel Support, Validation)            │  │
│  │   - Format detection                                    │  │
│  │   - Schema validation                                  │  │
│  │   - Error handling                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA STORAGE LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   CSV/Excel Files in /data folder                       │  │
│  │   - /data/raw/      (original data)                     │  │
│  │   - /data/processed/ (cleaned data)                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
1. INPUT: CSV/Excel Financial Data
                    ↓
2. LOAD: DataLoader validates structure
                    ↓
3. CLEAN: DataCleaner removes anomalies & imputes missing values
                    ↓
4. COMPUTE: FinancialRatioEngine calculates 20+ financial ratios
                    ↓
5. ANALYZE: TimeSeriesAnalyzer trends, volatility, correlations
                    ↓
6. DETECT: AnomalyDetectionEngine flags unusual patterns
                    ↓
7. SCORE: RiskScoreEngine computes composite risk (0-100)
                    ↓
8. RECOMMEND: ConsultingEngine generates strategic recommendations
                    ↓
9. VISUALIZE: ChartGenerator creates dashboards & reports
                    ↓
10. OUTPUT: Results exported (CSV/Excel/JSON) + Dashboard
```

## Module Descriptions

### 1. Data Ingestion (`loader.py`)
- **Responsibility**: Load and validate financial data
- **Inputs**: CSV/Excel files
- **Outputs**: Validated pandas DataFrame
- **Key Methods**:
  - `load_file()`: Auto-detect format and load
  - `validate_schema()`: Check required columns
  - `validate_data()`: Type checking, range validation

### 2. Data Preprocessing (`cleaner.py`)
- **Responsibility**: Clean and prepare data for analysis
- **Methods**:
  - `clean()`: Main cleaning pipeline
  - `handle_missing_values()`: Imputation strategies
  - `detect_outliers()`: IQR or Z-score detection
  - `normalize()`: Scaling to standard range

### 3. Financial Ratio Engine (`ratios.py`)
- **Responsibility**: Calculate financial ratios
- **Ratio Categories**:
  - **Liquidity**: Current ratio, Quick ratio, Cash ratio
  - **Solvency**: Debt-to-equity, Interest coverage, Debt ratio
  - **Profitability**: Net profit margin, ROA, ROE, ROIC
  - **Efficiency**: Asset turnover, Inventory turnover, Days sales outstanding
  - **Growth**: Revenue growth, Net income growth, Asset growth

### 4. Time-Series Analysis (`timeseries.py`)
- **Responsibility**: Analyze trends and patterns over time
- **Methods**:
  - `analyze_trends()`: Linear regression, momentum
  - `calculate_moving_averages()`: SMA, EMA
  - `calculate_volatility()`: Standard deviation, coefficient of variation
  - `calculate_correlation()`: Ratio correlations

### 5. Anomaly Detection (`zscore.py`)
- **Responsibility**: Detect unusual values
- **Methods**:
  - `ZScoreDetector`: Statistical z-score method
  - `IsolationForestDetector`: Machine learning method
  - `AnomalyDetectionEngine`: Combined approach

### 6. Risk Scoring (`score.py`)
- **Responsibility**: Compute composite risk score
- **Approach**: Weighted combination of ratio categories
- **Output**: 0-100 score + classification (Stable/Caution/Distress)
- **Formula**: 
  ```
  Score = Σ(category_score × weight)
  Where weights: liquidity(0.25) + solvency(0.30) + profitability(0.25) 
                 + efficiency(0.15) + growth(0.05)
  ```

### 7. Consulting Engine (`recommend.py`)
- **Responsibility**: Generate strategic recommendations
- **Inputs**: Risk scores, category scores, anomalies, trends
- **Outputs**: Actionable recommendations by category

### 8. Visualization (`charts.py`)
- **Responsibility**: Create visual dashboards
- **Charts**:
  - Risk scores comparison
  - Category score breakdowns
  - Ratio time-series trends
  - Anomaly heatmaps
  - Correlation matrices

## Design Patterns Used

### 1. **Pipeline Pattern**
Each module represents a stage in the data processing pipeline, with clear inputs and outputs.

### 2. **Strategy Pattern**
Multiple anomaly detection algorithms (Z-score, Isolation Forest) can be swapped.

### 3. **Factory Pattern**
`DataLoader` auto-detects file format and returns appropriate reader.

### 4. **Logging Pattern**
Centralized logging for debugging and audit trails.

## Technology Stack

- **Data Processing**: pandas, numpy
- **Statistics & ML**: scikit-learn, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: Streamlit (dashboard)
- **API Framework**: FastAPI (optional)
- **File I/O**: openpyxl, xlrd
- **Testing**: pytest, pytest-cov

## Scalability Considerations

1. **Data Volume**: Chunked processing for large datasets
2. **Multiple Companies**: Vectorized operations for efficiency
3. **Time Periods**: Sliding window analysis
4. **Real-time Updates**: API endpoints for continuous monitoring
5. **Distributed Processing**: Can be extended with Dask/Spark

## Security Considerations

1. Input validation at all entry points
2. Logging without exposing sensitive data
3. Error handling without stack trace exposure in production
4. Optional API authentication (FastAPI)
5. CORS configuration for web endpoints

## Performance Optimization

1. **Caching**: Store computed ratios between runs
2. **Vectorization**: NumPy operations over loops
3. **Lazy Evaluation**: Compute on-demand
4. **Parallel Processing**: Analyze multiple companies concurrently
5. **Memory Efficiency**: Stream large files instead of loading entirely

## Testing Strategy

- **Unit Tests**: Individual module functionality
- **Integration Tests**: End-to-end pipelines
- **Data Tests**: Input validation and schema checking
- **Regression Tests**: Known results comparison
- **Performance Tests**: Execution time benchmarks

## Deployment Options

1. **Standalone CLI**: `python main.py -i data.csv`
2. **Streamlit Dashboard**: `streamlit run app.py`
3. **FastAPI Server**: `uvicorn api.server:app --reload` (optional)
4. **Docker Container**: Containerized deployment
5. **Cloud Services**: AWS Lambda, Google Cloud Functions

## Future Enhancements

1. Machine learning models for distress prediction
2. Portfolio-level analysis
3. Peer comparison benchmarking
4. Industry-specific weighting
5. Real-time data integration (Yahoo Finance, etc.)
6. Multi-currency support
7. Regulatory compliance reporting
8. Stress testing scenarios
