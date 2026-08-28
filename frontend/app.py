import base64
import io
import os
import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# Page Configuration & Rich Clinical Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Diagnostic Console",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .app-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0369A1 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
        color: #F8FAFC;
    }
    .app-subtitle {
        font-size: 1.0rem;
        color: #94A3B8;
        font-weight: 500;
    }
    
    .telemetry-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .telemetry-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .status-badge-confident {
        background-color: #ECFDF5;
        border: 1.5px solid #10B981;
        color: #065F46;
        padding: 12px 16px;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 16px;
    }
    .status-badge-moderate {
        background-color: #FFFBEB;
        border: 1.5px solid #F59E0B;
        color: #92400E;
        padding: 12px 16px;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 16px;
    }
    .status-badge-review {
        background-color: #FEF2F2;
        border: 1.5px solid #EF4444;
        color: #991B1B;
        padding: 12px 16px;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 16px;
    }
    
    .disclaimer-banner {
        background: #FEF3C7;
        border-left: 4px solid #D97706;
        color: #78350F;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-bottom: 18px;
    }
    
    .sample-btn-box {
        background: #F8FAFC;
        border: 1px dashed #CBD5E1;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration & Health Telemetry
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ System Telemetry")
api_url = st.sidebar.text_input("FastAPI Endpoint URL", value=os.getenv("API_URL", "http://localhost:8000"))

try:
    health_resp = requests.get(f"{api_url}/health", timeout=2)
    if health_resp.status_code == 200:
        hdata = health_resp.json()
        if hdata.get("model_loaded"):
            st.sidebar.success("🟢 API Online — ResNet50 Loaded")
        else:
            st.sidebar.warning("🟡 API Online — Model Weights Pending")
    else:
        st.sidebar.error("🔴 API Health Check Failed")
except Exception:
    st.sidebar.error("🔴 API Offline (Ensure uvicorn is running on :8000)")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**🎯 Clinical Benchmark (7,200 Scans)**
- **Test Accuracy**: `96.19%`
- **False Negative Rate**: `0.44%` (99.56% Sensitivity)
- **Mean ROC-AUC**: `0.998`

**🔬 Explainability & Inference**
- **Grad-CAM Layer**: `conv5_block3_out`
- **Bayesian Uncertainty**: Monte Carlo Dropout ($N=20$)
- **Report Engine**: ReportLab Clinical PDF Generator
""")

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">🧠 NeuroScan AI — Clinical Diagnostic & Explainability Console</div>
    <div class="app-subtitle">Deep learning multi-class brain MRI classification with Grad-CAM activation mapping and Bayesian uncertainty quantification.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    <strong>⚠️ Research & Educational Platform:</strong> This diagnostic interface is an automated AI benchmarking suite and is not approved as an in-vitro medical diagnostic device. All interpretations require validation by certified radiological personnel.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab_console, tab_benchmarks, tab_modelcard = st.tabs([
    "🏥 Live MRI Diagnostic Console",
    "📊 Model Benchmarks & Confusion Matrix",
    "🛡️ Clinical Model Card & Safety Scope"
])

# =========================================================
# TAB 1: Live Diagnostic Console
# =========================================================
with tab_console:
    SAMPLE_FILES = {
        "glioma": "data/raw/glioma/Testing_Te-gl_1.jpg",
        "meningioma": "data/raw/meningioma/Testing_Te-aug-me_1.jpg",
        "pituitary": "data/raw/pituitary/Testing_Te-pi_1.jpg",
        "no_tumor": "data/raw/no_tumor/Testing_Te-no_1.jpg",
    }

    st.markdown('<div class="sample-btn-box"><b>⚡ Quick Test Gallery (1-Click Sample Scans):</b><br>Select an authentic clinical MRI scan to evaluate instantly:</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    if "active_bytes" not in st.session_state:
        st.session_state.active_bytes = None
        st.session_state.active_name = None

    with c1:
        if st.button("🧪 Sample Glioma", use_container_width=True):
            if os.path.exists(SAMPLE_FILES["glioma"]):
                with open(SAMPLE_FILES["glioma"], "rb") as f:
                    st.session_state.active_bytes = f.read()
                st.session_state.active_name = "Testing_Te-gl_1.jpg"

    with c2:
        if st.button("🧪 Sample Meningioma", use_container_width=True):
            if os.path.exists(SAMPLE_FILES["meningioma"]):
                with open(SAMPLE_FILES["meningioma"], "rb") as f:
                    st.session_state.active_bytes = f.read()
                st.session_state.active_name = "Testing_Te-aug-me_1.jpg"

    with c3:
        if st.button("🧪 Sample Pituitary", use_container_width=True):
            if os.path.exists(SAMPLE_FILES["pituitary"]):
                with open(SAMPLE_FILES["pituitary"], "rb") as f:
                    st.session_state.active_bytes = f.read()
                st.session_state.active_name = "Testing_Te-pi_1.jpg"

    with c4:
        if st.button("🧪 Sample Healthy Control", use_container_width=True):
            if os.path.exists(SAMPLE_FILES["no_tumor"]):
                with open(SAMPLE_FILES["no_tumor"], "rb") as f:
                    st.session_state.active_bytes = f.read()
                st.session_state.active_name = "Testing_Te-no_1.jpg"

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Or Upload Your Own Brain MRI Scan (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload an axial or coronal T1/T2 weighted brain MRI slice.",
    )

    if uploaded_file is not None:
        st.session_state.active_bytes = uploaded_file.getvalue()
        st.session_state.active_name = uploaded_file.name

    if st.session_state.active_bytes is not None:
        st.info(f"📁 Loaded Scan: **`{st.session_state.active_name}`** ({len(st.session_state.active_bytes) / 1024:.1f} KB)")
        
        col_exec, _ = st.columns([1, 3])
        with col_exec:
            run_btn = st.button("🚀 Run Full Diagnostic & Explainability Inference", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Analyzing MRI scan with ResNet50, computing Monte Carlo uncertainty, and resolving Grad-CAM..."):
                try:
                    files = {"file": (st.session_state.active_name, st.session_state.active_bytes, "image/jpeg")}
                    resp = requests.post(f"{api_url}/predict", files=files, timeout=12)

                    if resp.status_code == 200:
                        st.session_state.diag_data = resp.json()
                    else:
                        st.error(f"API Inference Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Could not connect to FastAPI endpoint: {e}")

        # Render diagnostic results if available in session
        if "diag_data" in st.session_state and st.session_state.diag_data is not None:
            data = st.session_state.diag_data
            pred_class = data["predicted_class"]
            conf = data["confidence"]
            epistemic_std = data.get("epistemic_uncertainty", 0.0)
            entropy = data.get("predictive_entropy", 0.0)
            clinical_status = data.get("clinical_status", "LOW_RISK_CONFIDENT")
            status_desc = data.get("status_description", "")
            overlay_b64 = data["gradcam_overlay"]

            st.markdown("### 📋 Primary Diagnostic Telemetry")
            
            # Clinical status banner
            if "HIGH" in clinical_status:
                st.markdown(f'<div class="status-badge-review">🚨 <b>CLINICAL STATUS: {clinical_status}</b><br>{status_desc}</div>', unsafe_allow_html=True)
            elif "MODERATE" in clinical_status:
                st.markdown(f'<div class="status-badge-moderate">⚠️ <b>CLINICAL STATUS: {clinical_status}</b><br>{status_desc}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-badge-confident">✅ <b>CLINICAL STATUS: {clinical_status}</b><br>{status_desc}</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Predicted Pathology", pred_class.replace("_", " ").upper())
            with m2:
                st.metric("Model Confidence", f"{conf * 100:.2f}%")
            with m3:
                st.metric("Epistemic Variance (±)", f"{epistemic_std:.4f}")
            with m4:
                st.metric("Predictive Shannon Entropy", f"{entropy:.4f} / 1.0")

            st.markdown("---")
            st.markdown("### 🔬 Visual Explainability (Interactive Grad-CAM Heatmap)")

            # Interactive Alpha / Opacity Slider for Dynamic Blending
            alpha = st.slider("Heatmap Overlay Intensity (Alpha Blend)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            
            # Reconstruct original image and blend
            orig_pil = Image.open(io.BytesIO(st.session_state.active_bytes)).convert("RGB").resize((224, 224))
            orig_np = np.array(orig_pil)

            overlay_raw_bytes = base64.b64decode(overlay_b64)
            overlay_pil = Image.open(io.BytesIO(overlay_raw_bytes)).convert("RGB").resize((224, 224))
            overlay_np = np.array(overlay_pil)

            blended_np = cv2.addWeighted(orig_np, 1.0 - alpha, overlay_np, alpha, 0)

            v1, v2, v3 = st.columns(3)
            with v1:
                st.image(orig_pil, caption="1. Original MRI Input", use_container_width=True)
            with v2:
                st.image(overlay_pil, caption="2. Pure Grad-CAM Attention Map", use_container_width=True)
            with v3:
                st.image(blended_np, caption=f"3. Interactive Dynamic Blend (Alpha = {alpha:.2f})", use_container_width=True)

            st.markdown("---")
            st.markdown("### 📊 Multi-Class Probability & Bayesian Epistemic Distribution")
            
            probs = data.get("probabilities", {})
            std_probs = data.get("std_probabilities", {})
            
            prob_df = pd.DataFrame([
                {
                    "Pathology Class": cname.replace("_", " ").title(),
                    "Probability (%)": p * 100,
                    "Epistemic Std Dev (±%)": std_probs.get(cname, 0.0) * 100,
                    "Classification Category": "Healthy Anatomical Control" if cname == "no_tumor" else "Intracranial Tumor Neoplasm"
                }
                for cname, p in probs.items()
            ])
            
            pcol1, pcol2 = st.columns([3, 2])
            with pcol1:
                st.bar_chart(prob_df.set_index("Pathology Class")["Probability (%)"])
            with pcol2:
                st.dataframe(prob_df, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📄 Export Official Clinical Diagnostic Report (PDF)")
            with st.spinner("Compiling structured ReportLab PDF document..."):
                report_files = {"file": (st.session_state.active_name, st.session_state.active_bytes, "image/jpeg")}
                pdf_resp = requests.post(f"{api_url}/report", files=report_files, timeout=15)
                if pdf_resp.status_code == 200:
                    st.download_button(
                        label="📥 Download Clinical Diagnostic PDF Report",
                        data=pdf_resp.content,
                        file_name=f"clinical_report_{st.session_state.active_name}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                    )
                else:
                    st.warning("Could not pre-compile PDF report.")

# =========================================================
# TAB 2: Model Benchmarks & Confusion Matrix
# =========================================================
with tab_benchmarks:
    st.markdown("### 🏆 Comprehensive Model Architecture Comparison (7,200 Scans)")
    
    benchmark_df = pd.DataFrame([
        {"Model Architecture": "Baseline CNN (From Scratch)", "Parameters": "~2.1M", "Test Accuracy": "88.40%", "Macro F1": "87.90%", "False Negative Rate (FNR)": "3.80%", "Mean ROC-AUC": "0.954"},
        {"Model Architecture": "ResNet50 (Fine-Tuned Champion)", "Parameters": "~24.1M", "Test Accuracy": "96.19%", "Macro F1": "96.18%", "False Negative Rate (FNR)": "0.44%", "Mean ROC-AUC": "0.998"},
        {"Model Architecture": "EfficientNetB0 (Fine-Tuned)", "Parameters": "~4.3M", "Test Accuracy": "91.75%", "Macro F1": "91.50%", "False Negative Rate (FNR)": "1.85%", "Mean ROC-AUC": "0.976"},
    ])
    st.table(benchmark_df)

    st.markdown("### 🔢 Confusion Matrix (ResNet50 Champion on 7,200 Clinical Scans)")
    cm_df = pd.DataFrame(
        [
            [1745, 25, 22, 8],
            [29, 1606, 158, 7],
            [1, 4, 1786, 9],
            [7, 1, 3, 1789],
        ],
        index=["Actual Glioma", "Actual Meningioma", "Actual Pituitary", "Actual No Tumor"],
        columns=["Predicted Glioma", "Predicted Meningioma", "Predicted Pituitary", "Predicted No Tumor"]
    )
    st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=True)

    st.markdown("### 🎯 Key Clinical Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Tumor Sensitivity", "99.56%", help="Only 24 missed tumor cases out of 5,400 pathological scans.")
    with k2:
        st.metric("Healthy Control Specificity", "99.39%", help="1,789 correct healthy scans out of 1,800.")
    with k3:
        st.metric("Clinical False Negative Rate", "0.44%", help="Prioritized clinical triage metric.")
    with k4:
        st.metric("Mean Multi-Class ROC-AUC", "0.998", help="Area under receiver operating characteristic curve.")

# =========================================================
# TAB 3: Model Card & Safety Scope
# =========================================================
with tab_modelcard:
    st.markdown("""
    ### 🛡️ Clinical Model Card & Ethical AI Disclaimers
    
    #### 1. Intended Purpose & Operational Scope
    - **Intended Use**: Educational, algorithmic explainability research, and comparative deep learning benchmarking across multi-class brain MRI cohorts.
    - **Prohibited / Out-of-Scope Use**: Strictly not approved for primary clinical diagnosis, surgical navigation, treatment triage, or patient management without prospective multi-scanner clinical trials and FDA/CE-MDR regulatory clearance.
    
    #### 2. Clinical Decision Hierarchy (False Negative Minimization)
    In neurological imaging, a **False Negative** (predicting `no_tumor` when pathology is present) carries catastrophic consequences compared to a False Positive (which triggers secondary radiologist confirmation). The model was trained with **inverse-frequency class weighting** to penalize false negatives severely.
    
    $$\\text{FNR} = \\frac{\\text{Pathological scans predicted as No Tumor}}{\\text{Total Pathological Scans}} = \\frac{24}{5400} = 0.44\\%$$
    
    #### 3. Epistemic Uncertainty & Safety Thresholds
    Monte Carlo Dropout is restricted to the dense classification head to preserve batch normalization statistics. Predictions are flagged based on three criteria:
    - **`HIGH_RISK_RADIOLOGIST_REVIEW`**: Predictive entropy $H \\ge 0.50$, confidence $< 65\\%$, or stochastic variance $\\sigma_{\\max} \\ge 0.35$.
    - **`MODERATE_AMBIGUITY`**: Predictive entropy $H \\ge 0.25$, confidence $< 85\\%$, or stochastic variance $\\sigma_{\\max} \\ge 0.28$.
    - **`LOW_RISK_CONFIDENT`**: High confidence with stable convergence across stochastic dropout iterations.
    """)
