import streamlit as st
import cv2
import numpy as np
from colorize import colorize

st.set_page_config(page_title="Image Colorizer", layout="wide")

st.title("🎨 Image Colorization")
st.caption("Compare Local vs Pretrained Models")

# UPLOAD

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

image = None

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

# MODEL SELECT

if image is not None:

    mode_option = st.selectbox(
        "Choose Mode", ["Use Both (Compare)", "Local Only", "Pretrained Only"]
    )

    mode_map = {
        "Use Both (Compare)": "both",
        "Local Only": "local",
        "Pretrained Only": "pretrained",
    }

    mode = mode_map[mode_option]

    # UPLOAD PREVIEW

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=200
    )

    if st.button("🚀 Colorize"):

        with st.spinner("Processing..."):
            result = colorize(image, mode=mode)

        st.markdown("---")

        # DUAL MODE

        if mode == "both":

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Original")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")

            with col2:
                st.subheader("Local Model")
                st.image(
                    cv2.cvtColor(result["local"], cv2.COLOR_BGR2RGB), width="stretch"
                )

            with col3:
                st.subheader("Pretrained")
                st.image(
                    cv2.cvtColor(result["pretrained"], cv2.COLOR_BGR2RGB),
                    width="stretch",
                )

        # INDIVIDUAL MODES

        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")

            with col2:
                st.subheader("Colorized")
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), width="stretch")

        # DOWNLOAD (PRETRAINED BEST)

        if mode == "both":
            final_img = result["pretrained"]
        else:
            final_img = result

        _, buffer = cv2.imencode(".png", final_img)

        st.download_button(
            "⬇️ Download Best Output",
            buffer.tobytes(),
            file_name="colorized.png",
            mime="image/png",
        )
