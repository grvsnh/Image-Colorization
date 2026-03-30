import cv2
import numpy as np
from tensorflow.keras.models import load_model

# PATHS

CORE_MODEL_PATH = "models/local-trained/colorization_model.keras"

PROTO_PATH = "models/pre-trained/colorization_deploy_v2.prototxt"
MODEL_PATH = "models/pre-trained/colorization_release_v2.caffemodel"
POINTS_PATH = "models/pre-trained/pts_in_hull.npy"

core_model = None
net = None


# LOAD MODELS


def load_local_model():
    global core_model
    if core_model is None:
        core_model = load_model(CORE_MODEL_PATH, compile=False)
    return core_model


def load_pretrained():
    global net
    if net is None:
        net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

        pts = np.load(POINTS_PATH)
        pts = pts.transpose().reshape(2, 313, 1, 1)

        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")

        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net


# COLOR BOOST


def boost_colors(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)  # saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)  # brightness

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# LOCAL MODEL


def colorize_local(image):

    model = load_local_model()

    img = cv2.resize(image, (128, 128))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0] / 255.0
    L = L.reshape(1, 128, 128, 1)

    pred_ab = model.predict(L, verbose=0)[0]

    # [-1,1] → LAB scale
    pred_ab = (pred_ab + 1) * 128
    pred_ab = cv2.resize(pred_ab, (128, 128))

    lab_out = np.zeros((128, 128, 3))
    lab_out[:, :, 0] = L[0, :, :, 0] * 255
    lab_out[:, :, 1:] = pred_ab

    result = cv2.cvtColor(lab_out.astype("uint8"), cv2.COLOR_LAB2BGR)

    result = cv2.resize(result, (image.shape[1], image.shape[0]))

    result = boost_colors(result)

    return result


# PRETRAINED MODEL


def colorize_pretrained(image):

    net = load_pretrained()

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


# MAIN


def colorize(image, mode="pretrained"):

    if mode == "local":
        return colorize_local(image)

    if mode == "both":
        return {
            "local": colorize_local(image),
            "pretrained": colorize_pretrained(image),
        }

    return colorize_pretrained(image)
