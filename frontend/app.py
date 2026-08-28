import base64
import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    layout="wide",
)

# Clinical styling
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        padding: 10px 14px;
        border-radius: 4px;
        color: #78350F;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("System Configuration")
api_url = st.sidebar.text_input("FastAPI Endpoint URL", value=os.getenv("API_URL", "http://localhost:8000"))

# Check API health
try:
    health_resp = requests.get(f"{api_url}/health", timeout=2)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        if health_data.get("model_loaded"):
            st.sidebar.success("Endpoint Online — Model Loaded")
        else:
            st.sidebar.warning("Endpoint Online — Model Weights Pending")
    else:
        st.sidebar.error("Endpoint Error")
except Exception:
    st.sidebar.error("Endpoint Offline (localhost:8000)")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Target Classes**
- Glioma
- Meningioma
- Pituitary
- No Tumor (Control)

**Evaluation Protocol**
Stratified 5-fold cross-validation with inverse frequency loss weighting and Monte Carlo Dropout uncertainty estimation.
""")

# Main UI Header
st.markdown('<div class="main-header">Brain Tumor MRI Classification & Explainability Console</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-class transfer learning evaluation with Grad-CAM feature mapping and Bayesian uncertainty estimation.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>Notice:</strong> This software is a research prototype and is not approved as a medical diagnostic device. Scans must be reviewed by certified clinical personnel.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Brain MRI Scan (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"],
    help="Upload an axial, coronal, or sagittal T1/T2 weighted brain MRI slice.",
)

if uploaded_file is not None:
    col_input, col_action = st.columns([3, 1])
    with col_input:
        st.info(f"Loaded: `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")
    with col_action:
        classify_btn = st.button("Run Inference", use_container_width=True, type="primary")

    if classify_btn:
        with st.spinner("Executing inference and generating Grad-CAM overlay..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                resp = requests.post(f"{api_url}/predict", files=files, timeout=10)

                if resp.status_code == 200:
                    data = resp.json()
                    predicted_class = data["predicted_class"]
                    confidence = data["confidence"]
                    probabilities = data["probabilities"]
                    overlay_b64 = data["gradcam_overlay"]

                    st.markdown("---")
                    st.subheader("Diagnostic Classification & Uncertainty")

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Predicted Class", predicted_class.upper())
                    with m2:
                        st.metric("Mean Confidence", f"{confidence:.1%}")
                    with m3:
                        uncertainty_val = data.get("epistemic_uncertainty", 0.0)
                        st.metric("Epistemic Std (±)", f"{uncertainty_val:.3f}")
                    with m4:
                        entropy_val = data.get("predictive_entropy", 0.0)
                        st.metric("Entropy Index", f"{entropy_val:.2f} / 1.0")

                    # Clinical Safety Alert Badge
                    clinical_status = data.get("clinical_status", "LOW_RISK_CONFIDENT")
                    status_desc = data.get("status_description", "")
                    if "HIGH" in clinical_status:
                        st.error(f"**Clinical Alert: {clinical_status}**\n\n{status_desc}")
                    elif "MODERATE" in clinical_status:
                        st.warning(f"**Clinical Notice: {clinical_status}**\n\n{status_desc}")
                    else:
                        st.success(f"**Clinical Status: {clinical_status}**\n\n{status_desc}")

                    st.markdown("---")
                    st.subheader("Explainability (Grad-CAM Attention Mapping)")
                    img_col1, img_col2 = st.columns(2)

                    with img_col1:
                        st.image(uploaded_file, caption="Original MRI Scan", use_container_width=True)
                    with img_col2:
                        overlay_bytes = base64.b64decode(overlay_b64)
                        st.image(overlay_bytes, caption="Grad-CAM Convolutional Heatmap", use_container_width=True)

                    st.markdown("---")
                    st.subheader("Multi-Class Probability & Variance Distribution")
                    std_probs = data.get("std_probabilities", {})
                    df_probs = pd.DataFrame([
                        {
                            "Class": cls_name.replace("_", " ").title(),
                            "Probability (%)": prob * 100,
                            "Epistemic Std (±%)": std_probs.get(cls_name, 0.0) * 100,
                        }
                        for cls_name, prob in probabilities.items()
                    ])
                    st.bar_chart(df_probs.set_index("Class")["Probability (%)"])
                    st.dataframe(df_probs, use_container_width=True)

                    # Automated Clinical PDF Report Generation
                    st.markdown("---")
                    st.subheader("Export Case Report")
                    with st.spinner("Compiling PDF summary..."):
                        report_files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        report_resp = requests.post(f"{api_url}/report", files=report_files, timeout=15)
                        if report_resp.status_code == 200:
                            st.download_button(
                                label="Download Clinical Diagnostic Report (PDF)",
                                data=report_resp.content,
                                file_name=f"case_report_{uploaded_file.name}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                type="primary",
                            )
                        else:
                            st.warning("PDF report generation failed.")

                elif resp.status_code == 503:
                    st.warning("Model weights are not initialized. Please train the model (`python scripts/train_final.py`) first.")
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to API at `{api_url}`. Ensure FastAPI server is running (`uvicorn src.api.main:app`).")
            except Exception as e:
                st.error(f"An error occurred: {e}")



