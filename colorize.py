import sys
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


MODEL_PATH = "colorization_model.keras"


def load_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Error: could not read image:", image_path)
        sys.exit(1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def rgb_to_lab(image):

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    L = lab[:, :, 0] / 255.0
    AB = lab[:, :, 1:] / 255.0

    return L, AB


def lab_to_rgb(L, AB):

    lab = np.zeros((L.shape[0], L.shape[1], 3))

    lab[:, :, 0] = L * 255
    lab[:, :, 1:] = AB * 255

    rgb = cv2.cvtColor(lab.astype("uint8"), cv2.COLOR_LAB2RGB)

    return rgb


def colorize_image(image_path):

    print("Loading model...")
    model = load_model(MODEL_PATH, compile=False)

    print("Loading image...")
    img = load_image(image_path)

    original_h, original_w = img.shape[:2]

    # Convert to LAB
    L, _ = rgb_to_lab(img)

    # Resize grayscale to model input size
    L_resized = cv2.resize(L, (128, 128))

    # Prepare model input
    L_input = np.expand_dims(L_resized, axis=(0, -1))

    print("Predicting colors...")
    pred_AB = model.predict(L_input)[0]

    # Upscale predicted colors to original resolution
    pred_AB = cv2.resize(pred_AB, (original_w, original_h))

    # Reconstruct RGB image
    colorized = lab_to_rgb(L, pred_AB)

    # Ensure output directory exists
    os.makedirs("colorized_images", exist_ok=True)

    filename = os.path.basename(image_path)
    output_path = os.path.join("colorized_images", "colorized_" + filename)

    # Save result
    cv2.imwrite(output_path, cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))

    print("Saved:", output_path)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python colorize.py image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    colorize_image(image_path)
