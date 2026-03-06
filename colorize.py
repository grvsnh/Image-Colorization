import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam


# =============================
# Load Dataset
# =============================

(X_train, _), (X_test, _) = cifar10.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# =============================
# Convert to Grayscale
# =============================

def rgb_to_gray(images):

    gray_images = []

    for img in images:
        img_uint8 = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        gray_images.append(gray)

    gray_images = np.array(gray_images)
    gray_images = np.expand_dims(gray_images, axis=-1)

    return gray_images


X_train_gray = rgb_to_gray(X_train)
X_test_gray = rgb_to_gray(X_test)


# =============================
# Generator Model
# =============================

def build_generator():

    input_layer = Input(shape=(32,32,1))

    x = Conv2D(64,3,activation="relu",padding="same")(input_layer)
    x = BatchNormalization()(x)

    x = Conv2D(128,3,activation="relu",padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(128,3,activation="relu",padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(64,3,activation="relu",padding="same")(x)

    output = Conv2D(3,3,activation="sigmoid",padding="same")(x)

    model = Model(inputs=input_layer, outputs=output)

    return model


generator = build_generator()

generator.compile(
    optimizer=Adam(0.001),
    loss="mse"
)

generator.summary()


# =============================
# Train Model
# =============================

print("\nTraining colorization model...\n")

generator.fit(
    X_train_gray,
    X_train,
    validation_data=(X_test_gray, X_test),
    epochs=10,
    batch_size=64
)


# =============================
# Generate Color Images
# =============================

num_images = 5

test_gray = X_test_gray[:num_images]

predicted_color = generator.predict(test_gray)


# =============================
# Save Result
# =============================

plt.figure(figsize=(10,4))

for i in range(num_images):

    plt.subplot(3,num_images,i+1)
    plt.imshow(X_test[i])
    plt.title("Original")
    plt.axis("off")

    plt.subplot(3,num_images,i+1+num_images)
    plt.imshow(test_gray[i].squeeze(),cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(3,num_images,i+1+2*num_images)
    plt.imshow(predicted_color[i])
    plt.title("Predicted")
    plt.axis("off")

plt.tight_layout()
plt.savefig("result.png")

print("\nSaved colorization results → result.png")
