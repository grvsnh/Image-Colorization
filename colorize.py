import cv2
import numpy as np
import os
import urllib.request
import time
import streamlit as st
from tensorflow.keras.models import load_model

# =============================
# PATHS
# =============================
CORE_MODEL_PATH = "models/local-trained/colorization_model.keras"
MODEL_DIR = "models/pre-trained"

# =============================
# YOUR VERIFIED MIRRORS ✅
# =============================
PROTO_URL = "https://huggingface.co/spaces/BilalSardar/Black-N-White-To-Color/resolve/main/colorization_deploy_v2.prototxt"
MODEL_URL = "https://huggingface.co/spaces/BilalSardar/Black-N-White-To-Color/resolve/main/colorization_release_v2.caffemodel"
POINTS_URL = "https://huggingface.co/spaces/BilalSardar/Black-N-White-To-Color/resolve/main/pts_in_hull.npy"


# =============================
# DOWNLOAD WITH PROGRESS + SPEED
# =============================
def download_file(url, path, desc):

    if os.path.exists(path):
        return

    progress_bar = st.progress(0)
    status = st.empty()

    start_time = time.time()

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        progress = min(downloaded / total_size, 1.0)

        elapsed = time.time() - start_time
        speed = downloaded / (1024 * 1024 * elapsed + 1e-5)

        total_mb = total_size / (1024 * 1024)
        done_mb = downloaded / (1024 * 1024)

        progress_bar.progress(progress)
        status.text(
            f"{desc}: {done_mb:.1f}/{total_mb:.1f} MB "
            f"({progress*100:.0f}%) | {speed:.2f} MB/s"
        )

    try:
        urllib.request.urlretrieve(url, path, reporthook)

        status.text(f"{desc} completed ✅")
        time.sleep(0.5)

    except Exception:
        progress_bar.empty()
        status.empty()
        st.warning(f"⚠️ Failed to download {desc}. Pretrained model disabled.")
        return False

    progress_bar.empty()
    status.empty()
    return True


# =============================
# ENSURE FILES
# =============================
def ensure_pretrained_files():
    os.makedirs(MODEL_DIR, exist_ok=True)

    proto_path = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
    model_path = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
    points_path = os.path.join(MODEL_DIR, "pts_in_hull.npy")

    ok1 = download_file(PROTO_URL, proto_path, "Config")
    ok2 = download_file(MODEL_URL, model_path, "Model (~100MB)")
    ok3 = download_file(POINTS_URL, points_path, "Color data")

    if not (ok1 and ok2 and ok3):
        return None, None, None

    return proto_path, model_path, points_path


# =============================
# LOAD LOCAL MODEL
# =============================
@st.cache_resource
def load_local_model():
    return load_model(CORE_MODEL_PATH, compile=False)


# =============================
# LOAD PRETRAINED MODEL
# =============================
@st.cache_resource
def load_pretrained():

    proto_path, model_path, points_path = ensure_pretrained_files()

    if proto_path is None:
        return None

    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    pts = np.load(points_path)
    pts = pts.transpose().reshape(2, 313, 1, 1)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")

    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net


# =============================
# COLOR BOOST
# =============================
def boost_colors(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# =============================
# LOCAL MODEL
# =============================
def colorize_local(image):

    model = load_local_model()

    img = cv2.resize(image, (128, 128))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0] / 255.0
    L = L.reshape(1, 128, 128, 1)

    pred_ab = model.predict(L, verbose=0)[0]

    pred_ab = (pred_ab + 1) * 128
    pred_ab = cv2.resize(pred_ab, (128, 128))

    lab_out = np.zeros((128, 128, 3))
    lab_out[:, :, 0] = L[0, :, :, 0] * 255
    lab_out[:, :, 1:] = pred_ab

    result = cv2.cvtColor(lab_out.astype("uint8"), cv2.COLOR_LAB2BGR)
    result = cv2.resize(result, (image.shape[1], image.shape[0]))

    return boost_colors(result)


# =============================
# PRETRAINED MODEL
# =============================
def colorize_pretrained(image):

    net = load_pretrained()

    if net is None:
        return image  # fallback

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype("float32") / 255.0

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    L = lab[:, :, 0]

    L_resized = cv2.resize(L, (224, 224))
    L_resized -= 50

    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    lab_out = np.zeros((image.shape[0], image.shape[1], 3))
    lab_out[:, :, 0] = L
    lab_out[:, :, 1:] = ab

    colorized = cv2.cvtColor(lab_out.astype("float32"), cv2.COLOR_LAB2BGR)

    colorized = np.clip(colorized, 0, 1)
    colorized = (colorized * 255).astype("uint8")

    return colorized


# =============================
# MAIN
# =============================
def colorize(image, mode="pretrained"):

    if mode == "local":
        return colorize_local(image)

    if mode == "both":
        return {
            "local": colorize_local(image),
            "pretrained": colorize_pretrained(image),
        }

    return colorize_pretrained(image)
