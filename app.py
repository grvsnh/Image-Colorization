import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

MODEL_PATH = "colorization_model.keras"


@st.cache_resource
def load_colorization_model():
    return load_model(MODEL_PATH, compile=False)


def rgb_to_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0] / 255.0
    return L


def lab_to_rgb(L, AB):
    lab = np.zeros((L.shape[0], L.shape[1], 3))
    lab[:, :, 0] = L * 255
    lab[:, :, 1:] = AB * 255
    rgb = cv2.cvtColor(lab.astype("uint8"), cv2.COLOR_LAB2RGB)
    return rgb


def colorize_image(image, model):

    original_h, original_w = image.shape[:2]

    L = rgb_to_lab(image)

    L_resized = cv2.resize(L, (128, 128))
    L_input = np.expand_dims(L_resized, axis=(0, -1))

    pred_AB = model.predict(L_input)[0]

    pred_AB = cv2.resize(pred_AB, (original_w, original_h))

    colorized = lab_to_rgb(L, pred_AB)

    return colorized


# ---------------- UI ---------------- #

st.set_page_config(
    page_title="Image Colorizer",
    page_icon="🎨",
    layout="wide"
)

# Animated title
st.markdown("""
<h1 style="text-align:center;">
Image Colorization using Deep Learning<span id="dots">...</span>
</h1>

<script>
let dots = document.getElementById("dots");
let count = 0;

setInterval(function() {
    count = (count + 1) % 4;
    dots.innerHTML = ".".repeat(count);
}, 500);
</script>
""", unsafe_allow_html=True)


st.markdown(
    "<p style='text-align:center;'>Upload a grayscale image and the AI will bring it to life with color.</p>",
    unsafe_allow_html=True
)

model = load_colorization_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    if st.button("Colorize Image"):

        with st.spinner("Colorizing image..."):
            result = colorize_image(image, model)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Colorized")
            st.image(result, use_container_width=True)

        center_col = st.columns([1,2,1])[1]

        result_pil = Image.fromarray(result)

        with center_col:
            st.download_button(
                label="Download Colorized Image",
                data=result_pil.tobytes(),
                file_name="colorized_image.png",
                mime="image/png",
                use_container_width=True
            )


# Footer
st.markdown("""
---
<p style="text-align:center; font-size:14px;">
Made with ❤️ using Deep Learning<br>
<a href="https://github.com/grvsnh/Image-Colorization" target="_blank">
View on GitHub
</a>
</p>
""", unsafe_allow_html=True)
