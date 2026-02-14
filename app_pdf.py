"""
Integrated Streamlit App for PDF Extraction and Financial Analysis

Combines PDF extraction with comprehensive financial analysis:
- Extract metrics from PDF → CSV
- Calculate 40+ financial ratios
- Perform time-series analysis
- Detect anomalies
- Calculate risk scores
- Generate recommendations
- Create visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Financial Distress EWS - PDF Extraction & Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
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
st.markdown("# 🚨 Financial Distress Early Warning System")
st.markdown("### 📊 PDF Extraction + Comprehensive Financial Analysis")
st.markdown("---")

# Mode selection
mode = st.radio(
    "Select Mode:",
    ["📄 PDF → CSV → Analysis", "📊 CSV Direct Analysis"],
    horizontal=True
)

# Import modules
try:
    from loader import DataLoader
    from cleaner import DataCleaner
    from ratios import FinancialRatioEngine
    from timeseries import TimeSeriesAnalyzer
    from zscore import AnomalyDetectionEngine
    from score import RiskScoreEngine
    from recommend import ConsultingEngine
    from charts import ChartGenerator
except ImportError as e:
    st.error(f"❌ Error importing modules: {e}")
    st.info("Make sure all required modules are available in the project directory")
    sys.exit(1)

# Try to import PDF extraction
try:
    from orchestrator import FinancialExtractionOrchestrator
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    PDF_EXTRACTION_AVAILABLE = False
    logger.warning("PDF extraction module not available")


def analyze_financial_data(df):
    """Run complete financial analysis pipeline."""
    
    results = {}
    
    try:
        # Step 1: Load data
        st.info("📥 Loading data...")
        loader = DataLoader()
        data = df.copy()
        
        # Step 2: Clean data
        st.info("🧹 Cleaning data...")
        cleaner = DataCleaner()
        clean_data = cleaner.clean(data)
        results['clean_data'] = clean_data
        
        if len(clean_data) < len(data):
            st.warning(f"⚠️ Removed {len(data) - len(clean_data)} invalid records")
        
        # Step 3: Calculate ratios
        st.info("📈 Calculating financial ratios...")
        ratio_engine = FinancialRatioEngine()
        ratios_df = ratio_engine.calculate_all_ratios(clean_data)
        results['ratios'] = ratios_df
        
        ratio_count = len(ratio_engine.get_calculated_ratios(ratios_df))
        st.success(f"✅ Calculated {ratio_count} financial ratios")
        
        # Step 4: Time-series analysis
        if 'year' in clean_data.columns:
            st.info("📊 Analyzing trends...")
            ts_analyzer = TimeSeriesAnalyzer()
            trends = ts_analyzer.analyze_trends(ratios_df)
            results['trends'] = trends
        
        # Step 5: Anomaly detection
        st.info("🔍 Detecting anomalies...")
        detector = AnomalyDetectionEngine(
            use_zscore=True,
            use_isolation_forest=True
        )
        anomaly_results = detector.detect_all_anomalies(ratios_df)
        results['anomalies'] = anomaly_results
        
        if 'zscore' in anomaly_results and len(anomaly_results['zscore']) > 0:
            st.warning(f"⚠️ Detected {len(anomaly_results['zscore'])} anomalies")
        else:
            st.success("✅ No critical anomalies detected")
        
        # Step 6: Risk scoring
        st.info("🎯 Computing risk scores...")
        risk_engine = RiskScoreEngine()
        risk_results = risk_engine.calculate_risk_score(
            ratios_df,
            anomaly_results.get('zscore', pd.DataFrame())
        )
        results['risk_scores'] = risk_results
        
        # Step 7: Recommendations
        st.info("💡 Generating recommendations...")
        consultant = ConsultingEngine()
        recommendations = consultant.generate_recommendations(
            ratios_df,
            risk_results,
            anomaly_results.get('zscore', pd.DataFrame())
        )
        results['recommendations'] = recommendations
        
        st.success("✅ Analysis Complete!")
        
        return results
    
    except Exception as e:
        st.error(f"❌ Error during analysis: {e}")
        logger.exception("Analysis error")
        return None


def display_analysis_results(results):
    """Display comprehensive analysis results."""
    
    if not results:
        return
    
    ratios_df = results.get('ratios', pd.DataFrame())
    risk_results = results.get('risk_scores', {})
    anomalies = results.get('anomalies', {}).get('zscore', pd.DataFrame())
    recommendations = results.get('recommendations', {})
    
    # ===== TAB 1: Financial Ratios =====
    with st.expander("📈 Financial Ratios", expanded=True):
        st.subheader("Financial Ratios Calculated")
        
        if not ratios_df.empty:
            # Display by category
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Liquidity Ratios:**")
                liquidity_cols = [c for c in ratios_df.columns if 'ratio' in c and any(x in c for x in ['current', 'quick', 'cash'])]
                if liquidity_cols:
                    st.dataframe(ratios_df[['company', 'year'] + liquidity_cols].head(10))
            
            with col2:
                st.markdown("**Profitability Ratios:**")
                profit_cols = [c for c in ratios_df.columns if any(x in c for x in ['margin', 'roe', 'roa'])]
                if profit_cols:
                    st.dataframe(ratios_df[['company', 'year'] + profit_cols].head(10))
            
            # Download ratios
            csv_ratios = ratios_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Ratios (CSV)",
                data=csv_ratios,
                file_name="financial_ratios.csv",
                mime="text/csv"
            )
    
    # ===== TAB 2: Risk Assessment =====
    with st.expander("🎯 Risk Assessment", expanded=True):
        st.subheader("Company Risk Scores")
        
        if risk_results:
            risk_data = []
            for company, scores in risk_results.items():
                risk_data.append({
                    'Company': company,
                    'Risk Score': scores.get('overall_score', 0),
                    'Classification': scores.get('classification', 'Unknown'),
                    'Trend': scores.get('trend_factor', 'N/A')
                })
            
            risk_df = pd.DataFrame(risk_data)
            
            # Display as metrics
            cols = st.columns(len(risk_df))
            for idx, (col, row) in enumerate(zip(cols, risk_df.itertuples())):
                with col:
                    score = row[2]  # Risk Score
                    classification = row[3]  # Classification
                    
                    # Color based on classification
                    if classification == 'Stable':
                        color = '🟢'
                    elif classification == 'Caution':
                        color = '🟡'
                    else:
                        color = '🔴'
                    
                    st.metric(
                        f"{color} {row[1]}",
                        f"{score:.1f}/100",
                        delta=classification
                    )
            
            # Download risk scores
            csv_risk = risk_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Risk Scores (CSV)",
                data=csv_risk,
                file_name="risk_scores.csv",
                mime="text/csv"
            )
    
    # ===== TAB 3: Anomalies =====
    if not anomalies.empty:
        with st.expander("🚨 Detected Anomalies", expanded=False):
            st.subheader("Financial Anomalies Detected")
            
            # Filter by severity
            severity_filter = st.multiselect(
                "Filter by Severity:",
                ['Critical', 'High', 'Medium', 'Low'],
                default=['Critical', 'High']
            )
            
            filtered = anomalies[anomalies['severity'].isin(severity_filter)]
            
            if not filtered.empty:
                st.dataframe(
                    filtered[['company', 'year', 'metric', 'value', 'z_score', 'severity']],
                    use_container_width=True
                )
                
                # Download anomalies
                csv_anomalies = filtered.to_csv(index=False)
                st.download_button(
                    label="📥 Download Anomalies (CSV)",
                    data=csv_anomalies,
                    file_name="anomalies.csv",
                    mime="text/csv"
                )
            else:
                st.info("No anomalies found in selected severity levels")
    
    # ===== TAB 4: Recommendations =====
    if recommendations:
        with st.expander("💡 Strategic Recommendations", expanded=False):
            st.subheader("AI-Generated Recommendations")
            
            selected_company = st.selectbox(
                "Select Company:",
                list(recommendations.keys())
            )
            
            if selected_company in recommendations:
                rec = recommendations[selected_company]
                
                # Priority
                priority = rec.get('priority', 'NORMAL')
                if priority == 'CRITICAL':
                    st.error(f"🚨 {priority} PRIORITY")
                elif priority == 'HIGH':
                    st.warning(f"⚠️ {priority} PRIORITY")
                else:
                    st.info(f"ℹ️ {priority} PRIORITY")
                
                # Timeline
                st.markdown(f"**Timeline:** {rec.get('estimated_timeline', 'N/A')}")
                
                # Action items
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Immediate:**")
                    for action in rec.get('immediate_actions', [])[:3]:
                        st.markdown(f"- {action}")
                
                with col2:
                    st.markdown("**Short-term (3-6 mo):**")
                    for action in rec.get('short_term_actions', [])[:3]:
                        st.markdown(f"- {action}")
                
                with col3:
                    st.markdown("**Long-term (6-18 mo):**")
                    for action in rec.get('long_term_actions', [])[:3]:
                        st.markdown(f"- {action}")


# ===== MODE 1: PDF EXTRACTION =====
if mode == "📄 PDF → CSV → Analysis":
    if not PDF_EXTRACTION_AVAILABLE:
        st.error("❌ PDF extraction module not available")
        st.info("The orchestrator module is required for PDF extraction")
    else:
        st.subheader("Step 1: Upload PDF Annual Report")
        
        uploaded_pdf = st.file_uploader("📄 Upload PDF", type=['pdf'])
        
        if uploaded_pdf:
            st.info("Processing PDF...")
            
            # Save temporarily
            temp_pdf = f"temp_{uploaded_pdf.name}"
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            
            try:
                # Extract metrics
                sample_dir = '/Users/adi/Documents/financial-distress-ews/annual_reports_2024'
                orchestrator = FinancialExtractionOrchestrator(sample_pdf_dir=sample_dir)
                
                with st.spinner("🔍 Extracting financial metrics from PDF..."):
                    extraction_result = orchestrator.extract_and_analyze_single(
                        temp_pdf,
                        output_dir='extraction_output'
                    )
                
                # Display extraction results
                st.subheader("Step 2: Extracted Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Company", extraction_result['company'])
                col2.metric("Fiscal Year", extraction_result['fiscal_year'])
                col3.metric("Quality Score", f"{extraction_result['quality_score']:.1f}/100")
                col4.metric("Metrics Extracted", extraction_result['metrics_extracted'])
                
                # Convert extracted metrics to DataFrame
                metrics_data = {
                    'company': [extraction_result['company']],
                    'year': [extraction_result['fiscal_year']],
                }
                metrics_data.update(extraction_result['cleaned_metrics'])
                df = pd.DataFrame(metrics_data)
                
                st.success("✅ Extraction Complete!")
                
                # Download extracted CSV
                csv_extracted = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Extracted Metrics (CSV)",
                    data=csv_extracted,
                    file_name=f"{extraction_result['company']}_extracted.csv",
                    mime="text/csv"
                )
                
                # Run analysis
                st.subheader("Step 3: Financial Analysis")
                
                with st.spinner("⏳ Running comprehensive analysis..."):
                    analysis_results = analyze_financial_data(df)
                
                if analysis_results:
                    display_analysis_results(analysis_results)
            
            except Exception as e:
                st.error(f"❌ Error: {e}")
                logger.exception("PDF processing error")
            
            finally:
                Path(temp_pdf).unlink(missing_ok=True)


# ===== MODE 2: CSV ANALYSIS =====
elif mode == "📊 CSV Direct Analysis":
    st.subheader("Upload CSV with Financial Data")
    
    uploaded_csv = st.file_uploader("📊 Upload CSV", type=['csv', 'xlsx'])
    
    if uploaded_csv:
        try:
            # Load data
            if uploaded_csv.name.endswith('.csv'):
                df = pd.read_csv(uploaded_csv)
            else:
                df = pd.read_excel(uploaded_csv)
            
            st.success(f"✅ Loaded {len(df)} records")
            
            # Preview data
            with st.expander("📋 Data Preview", expanded=False):
                st.dataframe(df.head(10))
            
            # Run analysis
            st.subheader("Financial Analysis")
            
            with st.spinner("⏳ Running comprehensive analysis..."):
                analysis_results = analyze_financial_data(df)
            
            if analysis_results:
                display_analysis_results(analysis_results)
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            logger.exception("CSV analysis error")
    
    else:
        st.info("👈 Upload a CSV file to begin analysis")
        
        st.subheader("Expected Columns:")
        st.markdown("""
        - **company**: Company name
        - **year**: Financial year
        - **revenue**: Total revenue
        - **net_income**: Net profit
        - **total_assets**: Total assets
        - **current_assets**: Current assets
        - **current_liabilities**: Current liabilities
        - **total_debt**: Total debt
        - **equity**: Shareholders' equity
        """)


# Footer
st.markdown("---")
st.markdown("""
### 🚀 Features:
- 📄 **PDF Extraction** - Intelligent metric extraction from annual reports
- 📊 **CSV Analysis** - Direct financial data analysis
- 📈 **40+ Ratios** - Comprehensive financial ratio calculations
- 🔍 **Anomaly Detection** - Z-score and Isolation Forest detection
- 🎯 **Risk Scoring** - Composite risk assessment (0-100)
- 💡 **AI Recommendations** - Strategic consulting recommendations
- 📥 **Export CSV** - Download all analysis results

**Financial Distress Early Warning System | v1.0**
""")
