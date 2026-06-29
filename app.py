import os
import streamlit as st
import download_models
from PIL import Image

from caremd import CAREMD2

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="CARE-MD",
    page_icon="🩺",
    layout="wide"
)

st.title("CARE-MD")
st.subheader("Clinical Reasoning-Aligned Framework for Self-eXplainable Medical Diagnosis")

st.markdown("---")

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return CAREMD2()

care = load_model()

# ==========================================================
# IMAGE UPLOAD
# ==========================================================

uploaded = st.file_uploader(
    "Upload a Dermoscopic Image",
    type=["jpg","jpeg","png"]
)

if uploaded is not None:

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    image_path = os.path.join(
        upload_dir,
        uploaded.name
    )

    with open(image_path,"wb") as f:
        f.write(uploaded.getbuffer())

    st.image(
        image_path,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Generate CARE-MD Report"):

        with st.spinner("Running CARE-MD..."):

            report_path = care.run(image_path)

        st.success("Analysis Completed Successfully")

        st.markdown("---")

        st.subheader("Diagnosis")

        col1,col2,col3 = st.columns(3)

        with col1:
            st.metric(
                "Phase-2",
                care.prediction.upper()
            )

        with col2:
            st.metric(
                "Phase-3",
                care.phase3_prediction.upper()
            )

        with col3:
            st.metric(
                "Final",
                care.final_prediction.upper()
            )

        st.write(
            f"**Confidence :** {care.confidence*100:.2f}%"
        )

        st.write(
            f"**Evidence Support :** {care.phase3_support}/10"
        )

        st.write(
            f"**Validation :** {care.validation}"
        )

        st.markdown("---")

        st.subheader("Generated Report")

        st.image(
            report_path,
            use_container_width=True
        )

        with open(report_path,"rb") as f:

            st.download_button(

                "Download Report",

                data=f,

                file_name=os.path.basename(report_path),

                mime="image/png"
            )
