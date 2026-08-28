import base64
import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("⚙️ System Configuration")
api_url = st.sidebar.text_input("FastAPI Endpoint URL", value=os.getenv("API_URL", "http://localhost:8000"))

# Check API health
try:
    health_resp = requests.get(f"{api_url}/health", timeout=2)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        if health_data.get("model_loaded"):
            st.sidebar.success("● API Online & Model Loaded")
        else:
            st.sidebar.warning("▲ API Online (Model Weights Pending)")
    else:
        st.sidebar.error("✖ API Error")
except Exception:
    st.sidebar.error("✖ API Offline (Check localhost:8000)")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📋 Classes
- **Glioma** (Tumor)
- **Meningioma** (Tumor)
- **Pituitary** (Tumor)
- **No Tumor** (Healthy)

### 🔬 Primary Metric
Prioritizes **False Negative Rate (FNR)** to minimize missed tumor cases over raw accuracy.
""")

# Main UI Header
st.markdown('<div class="main-header">🧠 Brain Tumor MRI Classifier & Explainability</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-class transfer learning evaluation with automated Grad-CAM visual attention mapping.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Medical & Research Disclaimer:</strong> This system is a research and educational prototype. It is NOT a clinical diagnostic device and must not be used for medical decisions or patient diagnosis.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a Brain MRI Scan (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"],
    help="Upload an axial/sagittal/coronal T1/T2 weighted brain MRI scan.",
)

if uploaded_file is not None:
    col_input, col_action = st.columns([3, 1])
    with col_input:
        st.info(f"Loaded: `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")
    with col_action:
        classify_btn = st.button("🚀 Analyze Scan", use_container_width=True, type="primary")

    if classify_btn:
        with st.spinner("Running inference and generating Grad-CAM explainability overlay..."):
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
                    st.subheader("📊 Diagnostic Summary")

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Predicted Classification", predicted_class.upper())
                    with m2:
                        st.metric("Model Confidence", f"{confidence:.1%}")
                    with m3:
                        is_tumor = predicted_class != "no_tumor"
                        status_label = "TUMOR DETECTED" if is_tumor else "NO TUMOR DETECTED"
                        st.metric("Clinical Category", status_label)

                    st.markdown("---")
                    st.subheader("🔍 Visual Explainability (Grad-CAM)")
                    img_col1, img_col2 = st.columns(2)

                    with img_col1:
                        st.image(uploaded_file, caption="Original MRI Scan", use_container_width=True)
                    with img_col2:
                        overlay_bytes = base64.b64decode(overlay_b64)
                        st.image(overlay_bytes, caption="Grad-CAM Attention Heatmap (Where the model focused)", use_container_width=True)

                    st.markdown("---")
                    st.subheader("📈 Class Probability Distribution")
                    df_probs = pd.DataFrame(
                        list(probabilities.items()),
                        columns=["Class", "Probability"]
                    )
                    df_probs["Probability (%)"] = df_probs["Probability"] * 100
                    st.bar_chart(df_probs.set_index("Class")["Probability (%)"])

                elif resp.status_code == 503:
                    st.warning("⚠️ Model weights are not loaded on the backend. Please train the model (`python scripts/train_final.py`) first.")
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to API at `{api_url}`. Ensure FastAPI server is running (`uvicorn src.api.main:app`).")
            except Exception as e:
                st.error(f"An error occurred: {e}")

