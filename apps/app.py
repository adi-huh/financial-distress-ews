"""
Streamlit Dashboard for Financial Distress Early Warning System
Interactive web interface for financial analysis.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_ingestion.loader import DataLoader
from src.preprocessing.cleaner import DataCleaner
from src.ratio_engine.ratios import FinancialRatioEngine
from src.analytics.timeseries import TimeSeriesAnalyzer
from src.anomaly_detection.zscore import AnomalyDetectionEngine
from src.risk_score.score import RiskScoreEngine
from src.consulting.recommend import ConsultingEngine

# Page configuration
st.set_page_config(
    page_title="Financial Distress EWS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stable {
        color: green;
        font-weight: bold;
    }
    .caution {
        color: orange;
        font-weight: bold;
    }
    .distress {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🚨 Financial Distress Early Warning System</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Financial Data (CSV/Excel)",
    type=['csv', 'xlsx', 'xls'],
    help="Upload a file with financial statement data"
)

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Analysis Settings")

# Analysis options
run_anomaly_detection = st.sidebar.checkbox("Anomaly Detection", value=True)
anomaly_method = st.sidebar.selectbox(
    "Anomaly Method",
    ['Z-score', 'Isolation Forest', 'Both']
)
zscore_threshold = st.sidebar.slider(
    "Z-score Threshold",
    min_value=2.0,
    max_value=4.0,
    value=3.0,
    step=0.1
)

st.sidebar.markdown("---")
st.sidebar.info("""
**How to use:**
1. Upload your financial data CSV/Excel
2. Configure analysis settings
3. View results and recommendations
""")

# Main content
if uploaded_file is None:
    # Welcome screen
    st.info("👈 Please upload a financial data file to begin analysis")
    
    st.subheader("📋 Expected Data Format")
    st.markdown("""
    Your file should contain the following columns:
    - **company**: Company name
    - **year**: Financial year
    - **revenue**: Total revenue
    - **net_income**: Net income/profit
    - **total_assets**: Total assets
    - **current_assets**: Current assets
    - **current_liabilities**: Current liabilities
    - **total_debt**: Total debt
    - **equity**: Shareholders' equity
    - **inventory** (optional): Inventory value
    - **cogs** (optional): Cost of goods sold
    - **operating_income** (optional): Operating income
    - **interest_expense** (optional): Interest expense
    - **accounts_receivable** (optional): Accounts receivable
    - **cash** (optional): Cash and cash equivalents
    """)
    
    st.subheader("📊 Sample Data")
    sample_df = pd.DataFrame({
        'company': ['TechCorp', 'TechCorp'],
        'year': [2023, 2024],
        'revenue': [1000000, 1100000],
        'net_income': [100000, 110000],
        'total_assets': [2000000, 2200000],
        'current_assets': [500000, 550000],
        'current_liabilities': [300000, 320000],
        'total_debt': [800000, 850000],
        'equity': [1200000, 1350000]
    })
    st.dataframe(sample_df)
    
else:
    # Process uploaded file
    try:
        with st.spinner("Loading data..."):
            # Load data
            loader = DataLoader()
            
            # Save uploaded file temporarily
            temp_path = Path(f"temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            data = loader.load_file(temp_path)
            temp_path.unlink()  # Delete temp file
            
            st.success(f"✅ Loaded {len(data)} records from {len(data['company'].unique())} companies")
        
        # Display data summary
        with st.expander("📊 Data Summary", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", len(data))
            col2.metric("Companies", data['company'].nunique())
            col3.metric("Years", f"{data['year'].min()}-{data['year'].max()}")
            col4.metric("Columns", len(data.columns))
            
            st.dataframe(data.head(10))
        
        # Data Cleaning
        with st.spinner("Preprocessing data..."):
            cleaner = DataCleaner()
            clean_data = cleaner.clean(data)
            
            if len(clean_data) < len(data):
                st.warning(f"⚠️ Removed {len(data) - len(clean_data)} invalid records during cleaning")
        
        # Calculate Financial Ratios
        with st.spinner("Calculating financial ratios..."):
            ratio_engine = FinancialRatioEngine()
            ratios_df = ratio_engine.calculate_all_ratios(clean_data)
            
            st.success(f"✅ Calculated {len(ratio_engine.get_calculated_ratios(ratios_df))} financial ratios")
        
        # Display Ratios
        with st.expander("📈 Financial Ratios", expanded=False):
            # Company selector
            selected_company = st.selectbox("Select Company", ratios_df['company'].unique())
            company_ratios = ratios_df[ratios_df['company'] == selected_company]
            
            # Display ratio table
            display_cols = ['year', 'current_ratio', 'quick_ratio', 'debt_to_equity',
                          'roe', 'roa', 'net_profit_margin', 'asset_turnover']
            available_cols = ['year'] + [col for col in display_cols[1:] if col in company_ratios.columns]
            
            st.dataframe(
                company_ratios[available_cols].sort_values('year'),
                use_container_width=True
            )
        
        # Time-Series Analysis
        with st.spinner("Analyzing trends..."):
            ts_analyzer = TimeSeriesAnalyzer()
            trends = ts_analyzer.analyze_trends(ratios_df)
        
        # Anomaly Detection
        anomalies = pd.DataFrame()
        if run_anomaly_detection:
            with st.spinner("Detecting anomalies..."):
                detector = AnomalyDetectionEngine(
                    use_zscore=anomaly_method in ['Z-score', 'Both'],
                    use_isolation_forest=anomaly_method in ['Isolation Forest', 'Both'],
                    zscore_threshold=zscore_threshold
                )
                anomaly_results = detector.detect_all_anomalies(ratios_df)
                
                if 'zscore' in anomaly_results and len(anomaly_results['zscore']) > 0:
                    anomalies = anomaly_results['zscore']
                
                if len(anomalies) > 0:
                    st.warning(f"⚠️ Detected {len(anomalies)} anomalies")
                else:
                    st.success("✅ No anomalies detected")
        
        # Display Anomalies
        if len(anomalies) > 0:
            with st.expander("🚨 Detected Anomalies", expanded=True):
                # Filter by severity
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    ['Critical', 'High', 'Medium', 'Low'],
                    default=['Critical', 'High']
                )
                
                filtered_anomalies = anomalies[anomalies['severity'].isin(severity_filter)]
                
                if len(filtered_anomalies) > 0:
                    st.dataframe(
                        filtered_anomalies[['company', 'year', 'metric', 'value', 'z_score', 'severity']],
                        use_container_width=True
                    )
                else:
                    st.info("No anomalies match the selected severity levels")
        
        # Risk Score Calculation
        with st.spinner("Calculating risk scores..."):
            risk_engine = RiskScoreEngine()
            risk_results = risk_engine.calculate_risk_score(ratios_df, anomalies)
        
        # Display Risk Scores
        st.markdown("---")
        st.subheader("🎯 Risk Assessment")
        
        # Create columns for each company
        companies = list(risk_results.keys())
        num_companies = len(companies)
        
        if num_companies <= 3:
            cols = st.columns(num_companies)
        else:
            cols = st.columns(3)
        
        for idx, (company, results) in enumerate(risk_results.items()):
            col_idx = idx % len(cols)
            
            with cols[col_idx]:
                score = results['overall_score']
                classification = results['classification']
                
                # Determine color class
                if classification == 'Stable':
                    class_style = 'stable'
                elif classification == 'Caution':
                    class_style = 'caution'
                else:
                    class_style = 'distress'
                
                st.markdown(f"### {company}")
                st.metric("Risk Score", f"{score:.1f}/100")
                st.markdown(f'<p class="{class_style}">Classification: {classification}</p>', unsafe_allow_html=True)
                st.caption(f"Trend: {results['trend_factor']}")
                
                # Progress bar
                if classification == 'Stable':
                    bar_color = 'green'
                elif classification == 'Caution':
                    bar_color = 'orange'
                else:
                    bar_color = 'red'
                
                st.progress(score / 100)
                
                # Category breakdown
                with st.expander("Category Breakdown"):
                    for category, category_score in results['category_scores'].items():
                        st.write(f"**{category.title()}**: {category_score:.1f}")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Strategic Recommendations")
        
        with st.spinner("Generating recommendations..."):
            consultant = ConsultingEngine()
            recommendations = consultant.generate_recommendations(ratios_df, risk_results, anomalies)
        
        # Display recommendations
        rec_company = st.selectbox("Select Company for Recommendations", companies)
        company_rec = recommendations[rec_company]
        
        # Priority banner
        priority = company_rec['priority']
        if priority == 'CRITICAL':
            st.error(f"🚨 {priority} PRIORITY")
        elif priority == 'HIGH':
            st.warning(f"⚠️ {priority} PRIORITY")
        else:
            st.info(f"ℹ️ {priority} PRIORITY")
        
        st.markdown(f"**Timeline:** {company_rec['estimated_timeline']}")
        
        # Action items
        tab1, tab2, tab3 = st.tabs(["Immediate Actions", "Short-Term Actions", "Long-Term Actions"])
        
        with tab1:
            st.markdown("#### Immediate Actions Required")
            for action in company_rec['immediate_actions']:
                st.markdown(f"- {action}")
        
        with tab2:
            st.markdown("#### Short-Term Actions (3-6 months)")
            for action in company_rec['short_term_actions']:
                st.markdown(f"- {action}")
        
        with tab3:
            st.markdown("#### Long-Term Actions (6-18 months)")
            for action in company_rec['long_term_actions']:
                st.markdown(f"- {action}")
        
        # Download Results
        st.markdown("---")
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export ratios
            csv = ratios_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Financial Ratios (CSV)",
                data=csv,
                file_name="financial_ratios.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export risk report
            risk_report = risk_engine.generate_risk_report(ratios_df, anomalies)
            csv = risk_report.to_csv(index=False)
            st.download_button(
                label="🎯 Download Risk Report (CSV)",
                data=csv,
                file_name="risk_report.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.caption("Financial Distress Early Warning System v1.0 | Built with Streamlit")
