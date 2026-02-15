"""
Financial Distress Early Warning System - Unified Dashboard
Combines CSV analysis and PDF extraction in one interface.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.loader import DataLoader
from src.core.cleaner import DataCleaner
from src.core.ratios import FinancialRatioEngine
from src.core.score import RiskScoreEngine
from src.core.recommend import ConsultingEngine
from src.core.charts import ChartGenerator

# Page config
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
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stable { color: #28a745; font-weight: bold; }
    .caution { color: #ffc107; font-weight: bold; }
    .distress { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚨 Financial Distress Early Warning System</h1>', unsafe_allow_html=True)

# Sidebar - Mode Selection
st.sidebar.title("📋 Analysis Mode")
mode = st.sidebar.radio(
    "Choose Input Method:",
    ["📊 Upload CSV/Excel", "📄 Upload PDF Report", "🎓 Learn How It Works"],
    help="Select how you want to input financial data"
)

st.sidebar.markdown("---")

# MAIN CONTENT

if mode == "📊 Upload CSV/Excel":
    st.header("📊 CSV/Excel Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Financial Data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file with company financial statements"
    )
    
    if uploaded_file:
        try:
            # Load data
            with st.spinner("Loading data..."):
                loader = DataLoader()
                # Save temp file
                temp_path = Path(f"temp_{uploaded_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                data = loader.load_file(str(temp_path))
                temp_path.unlink()
            
            st.success(f"✅ Loaded {len(data)} records from {data['company'].nunique()} companies")
            
            # Data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(data.head(10), use_container_width=True)
            
            # Clean data
            with st.spinner("Cleaning data..."):
                cleaner = DataCleaner()
                clean_data = cleaner.clean(data)
            
            # Calculate ratios
            with st.spinner("Calculating financial ratios..."):
                ratio_engine = FinancialRatioEngine()
                ratios_df = ratio_engine.calculate_all_ratios(clean_data)
            
            # Calculate risk scores
            with st.spinner("Calculating risk scores..."):
                risk_engine = RiskScoreEngine()
                risk_results = risk_engine.calculate_risk_score(ratios_df, pd.DataFrame())
            
            # Display Results
            st.markdown("---")
            st.header("🎯 Risk Assessment")
            
            # Create company cards
            companies = list(risk_results.keys())
            cols_per_row = 3
            
            for i in range(0, len(companies), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, company in enumerate(companies[i:i+cols_per_row]):
                    with cols[j]:
                        results = risk_results[company]
                        score = results['overall_score']
                        classification = results['classification']
                        
                        # Determine color
                        if classification == 'Stable':
                            class_style = 'stable'
                            emoji = '🟢'
                        elif classification == 'Caution':
                            class_style = 'caution'
                            emoji = '🟡'
                        else:
                            class_style = 'distress'
                            emoji = '🔴'
                        
                        st.markdown(f"### {emoji} {company}")
                        st.metric("Risk Score", f"{score:.1f}/100")
                        st.markdown(f'<p class="{class_style}">{classification}</p>', unsafe_allow_html=True)
                        
                        # Progress bar
                        st.progress(score / 100)
                        
                        # Category breakdown
                        with st.expander("Details"):
                            for cat, cat_score in results['category_scores'].items():
                                st.write(f"**{cat.title()}**: {cat_score:.1f}")
            
            # Generate recommendations
            st.markdown("---")
            st.header("💡 Recommendations")
            
            consultant = ConsultingEngine()
            recommendations = consultant.generate_recommendations(ratios_df, risk_results)
            
            selected_company = st.selectbox("Select Company", companies)
            company_rec = recommendations[selected_company]
            
            # Priority indicator
            priority = company_rec['priority']
            if priority == 'CRITICAL':
                st.error(f"🚨 {priority} PRIORITY")
            elif priority == 'HIGH':
                st.warning(f"⚠️ {priority} PRIORITY")
            else:
                st.info(f"ℹ️ {priority} PRIORITY")
            
            # Recommendations in tabs
            tab1, tab2, tab3 = st.tabs(["🎯 Immediate", "📅 Short-Term", "🎯 Long-Term"])
            
            with tab1:
                for action in company_rec['immediate_actions']:
                    st.markdown(f"- {action}")
            
            with tab2:
                for action in company_rec['short_term_actions']:
                    st.markdown(f"- {action}")
            
            with tab3:
                for action in company_rec['long_term_actions']:
                    st.markdown(f"- {action}")
            
            # Download results
            st.markdown("---")
            st.header("📥 Download Results")
            
            csv_data = ratios_df.to_csv(index=False)
            st.download_button(
                "📊 Download Financial Ratios (CSV)",
                csv_data,
                "financial_ratios.csv",
                "text/csv"
            )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("Error Details"):
                st.exception(e)
    
    else:
        # Instructions
        st.info("👈 Upload a CSV or Excel file to begin analysis")
        
        st.markdown("### Expected File Format")
        st.markdown("""
        Your file should have these columns:
        - **company** - Company name
        - **year** - Fiscal year
        - **revenue** - Total revenue
        - **net_income** - Net profit
        - **total_assets** - Total assets
        - **current_assets** - Current assets
        - **current_liabilities** - Current liabilities
        - **total_debt** - Total debt
        - **equity** - Shareholders' equity
        """)
        
        # Sample data
        st.markdown("### Sample Data Format")
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
        st.dataframe(sample_df, use_container_width=True)

elif mode == "📄 Upload PDF Report":
    st.header("📄 PDF Annual Report Analysis")
    
    try:
        from src.pdf_extraction.orchestrator import FinancialExtractionOrchestrator
        
        uploaded_pdf = st.file_uploader("Upload Annual Report (PDF)", type=['pdf'])
        
        if uploaded_pdf:
            # Save temporarily
            temp_pdf = f"temp_{uploaded_pdf.name}"
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            
            with st.spinner("Extracting financial metrics from PDF..."):
                try:
                    orchestrator = FinancialExtractionOrchestrator()
                    result = orchestrator.extract_and_analyze_single(
                        temp_pdf,
                        output_dir='extraction_output'
                    )
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Company", result['company'])
                    col2.metric("Fiscal Year", result['fiscal_year'])
                    col3.metric("Quality Score", f"{result['quality_score']:.1f}/100")
                    
                    if result['analysis']:
                        st.markdown("---")
                        st.header("Financial Health Analysis")
                        
                        analysis = result['analysis']
                        col1, col2 = st.columns(2)
                        col1.metric("Health Score", f"{analysis['financial_health_score']:.1f}/100")
                        col2.metric("Risk Level", analysis['distress_risk_level'])
                        
                        tab1, tab2 = st.tabs(["✅ Strengths", "⚠️ Weaknesses"])
                        
                        with tab1:
                            for strength in analysis['key_strengths']:
                                st.success(f"✅ {strength}")
                        
                        with tab2:
                            for weakness in analysis['key_weaknesses']:
                                st.warning(f"⚠️ {weakness}")
                    
                    # Show extracted metrics
                    if result['cleaned_metrics']:
                        st.markdown("---")
                        st.header("📊 Extracted Metrics")
                        metrics_df = pd.DataFrame([result['cleaned_metrics']])
                        st.dataframe(metrics_df, use_container_width=True)
                    
                    st.success("✅ Extraction Complete!")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    Path(temp_pdf).unlink(missing_ok=True)
        
        else:
            st.info("👈 Upload a PDF annual report to extract financial data")
    
    except ImportError:
        st.error("❌ PDF extraction module not available")
        st.info("Install required dependencies or use CSV analysis mode")

else:  # Learn How It Works
    st.header("🎓 How It Works")
    
    st.markdown("""
    ## 📊 Financial Distress Early Warning System
    
    This system analyzes company financial data to predict potential financial distress before it becomes critical.
    
    ### 🔄 Analysis Pipeline
    
    1. **Data Input** 📥
       - Upload CSV/Excel with financial statements
       - OR upload PDF annual reports
    
    2. **Data Cleaning** 🧹
       - Handle missing values
       - Remove outliers
       - Validate data quality
    
    3. **Calculate Ratios** 📈
       - **Liquidity**: Current ratio, Quick ratio, Cash ratio
       - **Solvency**: Debt-to-equity, Interest coverage
       - **Profitability**: ROE, ROA, Net margin
       - **Efficiency**: Asset turnover, Inventory turnover
       - **Growth**: Revenue growth, Income growth
    
    4. **Risk Scoring** ⚖️
       - Composite score (0-100)
       - Weighted by category importance
       - Classification: Stable / Caution / Distress
    
    5. **Recommendations** 💡
       - Immediate actions
       - Short-term strategies
       - Long-term improvements
    
    ### 🎯 Risk Score Interpretation
    
    - **70-100 (Stable)** 🟢: Healthy financial position
    - **40-69 (Caution)** 🟡: Warning signs detected
    - **0-39 (Distress)** 🔴: Critical situation
    
    ### 📚 Financial Ratios Explained
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Current Ratio**
                