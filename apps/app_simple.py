"""
Streamlit Dashboard - Financial Distress Early Warning System
Simplified version with CSV and PDF extraction modes.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Financial Distress EWS",
    page_icon="📊",
    layout="wide"
)

# Title
st.markdown("# 🚨 Financial Distress Early Warning System")
st.markdown("---")

# Mode selection
mode = st.radio("Select Mode:", ["📊 CSV Analysis", "📄 PDF Extraction"], horizontal=True)

if mode == "📊 CSV Analysis":
    st.subheader("📊 CSV Analysis Mode")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Financial Data (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} records")
            
            # Display data
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Summary statistics
            st.subheader("Summary Statistics")
            numeric_cols = df.select_dtypes(include=['number']).columns
            st.dataframe(df[numeric_cols].describe())
            
            # Download option
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download Processed Data",
                data=csv_data,
                file_name="processed_data.csv",
                mime="text/csv"
            )
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    else:
        st.info("👈 Upload a CSV or Excel file to begin analysis")
        
        st.subheader("Expected Columns:")
        st.markdown("""
        - **company**: Company name
        - **year**: Financial year
        - **revenue**: Total revenue
        - **net_income**: Net profit
        - **total_assets**: Total assets
        - **total_liabilities**: Total liabilities
        - **equity**: Shareholders' equity
        """)


elif mode == "📄 PDF Extraction":
    st.subheader("📄 PDF Extraction Mode")
    
    try:
        from orchestrator import FinancialExtractionOrchestrator
        
        # Initialize orchestrator
        orchestrator = FinancialExtractionOrchestrator(
            sample_pdf_dir='/Users/adi/Documents/financial-distress-ews/annual_reports_2024'
        )
        
        extraction_mode = st.selectbox("Select Extraction Mode:", [
            "Extract Single PDF",
            "Batch Extract",
            "Train on Samples"
        ])
        
        if extraction_mode == "Extract Single PDF":
            st.markdown("#### Extract Financial Metrics from Single PDF")
            
            uploaded_pdf = st.file_uploader("Upload PDF Annual Report", type=['pdf'])
            
            if uploaded_pdf:
                # Save temporarily
                temp_pdf = f"temp_{uploaded_pdf.name}"
                with open(temp_pdf, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                
                with st.spinner("Extracting metrics..."):
                    try:
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
                            st.subheader("Financial Analysis")
                            col1, col2 = st.columns(2)
                            col1.metric("Health Score", f"{result['analysis']['financial_health_score']:.1f}")
                            col2.metric("Risk Level", result['analysis']['distress_risk_level'])
                            
                            if result['analysis']['key_strengths']:
                                st.markdown("**Strengths:**")
                                for strength in result['analysis']['key_strengths']:
                                    st.markdown(f"✅ {strength}")
                            
                            if result['analysis']['key_weaknesses']:
                                st.markdown("**Weaknesses:**")
                                for weakness in result['analysis']['key_weaknesses']:
                                    st.markdown(f"⚠️ {weakness}")
                        
                        # Display metrics
                        if result['cleaned_metrics']:
                            st.subheader("Extracted Metrics")
                            metrics_df = pd.DataFrame([result['cleaned_metrics']])
                            st.dataframe(metrics_df.iloc[:, :5])
                        
                        st.success("✅ Extraction Complete!")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                    
                    finally:
                        # Clean up temp file
                        Path(temp_pdf).unlink(missing_ok=True)
            
            else:
                st.info("👈 Upload a PDF annual report")
        
        elif extraction_mode == "Batch Extract":
            st.markdown("#### Batch Extract from PDF Directory")
            
            pdf_dir = st.text_input(
                "Enter PDF Directory Path:",
                value="/Users/adi/Documents/financial-distress-ews/annual_reports_2024"
            )
            
            if st.button("Start Batch Processing"):
                with st.spinner("Processing PDFs..."):
                    try:
                        result = orchestrator.extract_and_analyze_batch(
                            pdf_dir,
                            output_dir='batch_output'
                        )
                        
                        summary = result['summary']
                        
                        # Display summary
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total PDFs", summary['total_pdfs_processed'])
                        col2.metric("Successful", summary['successful_extractions'])
                        col3.metric("Avg Quality", f"{summary['avg_quality_score']:.1f}")
                        
                        # Risk distribution
                        st.subheader("Risk Distribution")
                        risk_dist = pd.DataFrame(
                            list(summary['risk_distribution'].items()),
                            columns=['Risk Level', 'Count']
                        )
                        st.bar_chart(risk_dist.set_index('Risk Level'))
                        
                        # Top performers
                        if summary['top_performers']:
                            st.subheader("🏆 Top Performers")
                            top_df = pd.DataFrame(summary['top_performers'])
                            st.dataframe(top_df)
                        
                        # At-risk companies
                        if summary['at_risk_companies']:
                            st.subheader("⚠️ At-Risk Companies")
                            risk_df = pd.DataFrame(summary['at_risk_companies'])
                            st.dataframe(risk_df)
                        
                        st.success("✅ Batch Processing Complete!")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        elif extraction_mode == "Train on Samples":
            st.markdown("#### Train Pattern Learner on Sample PDFs")
            
            sample_dir = st.text_input(
                "Sample PDFs Directory:",
                value="/Users/adi/Documents/financial-distress-ews/annual_reports_2024"
            )
            
            if st.button("Train Pattern Learner"):
                with st.spinner("Training..."):
                    try:
                        result = orchestrator.train_on_samples(sample_dir)
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Patterns Learned", result['patterns_learned'])
                        col2.metric("Training Samples", result['training_samples'])
                        
                        st.success(f"✅ Training Complete! Patterns saved to {result['patterns_file']}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    except ImportError:
        st.error("❌ PDF extraction module not available. Install required dependencies.")
        st.info("The orchestrator module is not properly installed.")


# Footer
st.markdown("---")
st.markdown("📊 Financial Distress Early Warning System | Version 1.0")
