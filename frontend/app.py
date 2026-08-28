import base64

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.title("Brain Tumor MRI Classifier")
st.caption("Research and educational project — not a diagnostic tool. See docs/MODEL_CARD.md.")

uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file and st.button("Classify"):
    with st.spinner("Running inference..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        resp = requests.post(f"{API_URL}/predict", files=files)
        data = resp.json()

    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Original scan")
    with col2:
        overlay_bytes = base64.b64decode(data["gradcam_overlay"])
        st.image(overlay_bytes, caption="Grad-CAM: where the model looked")

    st.markdown(f"**Predicted class:** {data['predicted_class']}")
    st.markdown(f"**Confidence:** {data['confidence']:.2%}")
