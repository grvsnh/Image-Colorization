import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, Input, UpSampling2D, Reshape, LeakyReLU
from tensorflow.keras.optimizers import Adam


# =============================
# Load Dataset
# =============================

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)


# =============================
# Data Augmentation
# =============================

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)


# =============================
# Generator
# =============================

def build_generator():

    model = Sequential()

    model.add(Dense(256 * 8 * 8, activation="relu", input_dim=100))
    model.add(Reshape((8, 8, 256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256, 3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(128, 3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, 3, padding="same", activation="sigmoid"))

    return model


# =============================
# Discriminator
# =============================

def build_discriminator():

    model = Sequential()

    model.add(Conv2D(32, 3, strides=2, input_shape=(32, 32, 3), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, 3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    return model


# =============================
# Build GAN
# =============================

def build_gan(generator, discriminator):

    discriminator.trainable = False

    gan_input = Input(shape=(100,))
    img = generator(gan_input)
    gan_output = discriminator(img)

    gan = Model(gan_input, gan_output)

    gan.compile(
        loss="binary_crossentropy",
        optimizer=Adam(0.0002, 0.5)
    )

    return gan


# =============================
# Initialize Models
# =============================

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(
    loss="binary_crossentropy",
    optimizer=Adam(0.0002, 0.5),
    metrics=["accuracy"]
)

gan = build_gan(generator, discriminator)

print("\nGAN Summary\n")
gan.summary()


# =============================
# Training Function
# =============================

def train_gan(generator, discriminator, gan, epochs, batch_size):

    half_batch = batch_size // 2

    for epoch in range(epochs):

        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (half_batch, 100))
        fake_imgs = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((half_batch, 1)))

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch(noise, valid_y)

        print(f"{epoch+1} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")


# =============================
# Train Model
# =============================

epochs = 50
batch_size = 64

train_gan(generator, discriminator, gan, epochs, batch_size)


# =============================
# Convert Images to Grayscale
# =============================

def convert_to_grayscale(images):

    gray_images = []

    for img in images:

        img_uint8 = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0

        gray_images.append(gray)

    gray_images = np.array(gray_images)
    gray_images = np.expand_dims(gray_images, axis=-1)

    return gray_images


# =============================
# Post Processing
# =============================

def post_process(image):

    image = (image * 255).astype(np.uint8)

    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    image = image.astype(np.float32) / 255.0

    return image


# =============================
# Generate and Save Result
# =============================

num_images = 5

sample_images = X_test[:num_images]

grayscale_images = convert_to_grayscale(sample_images)

noise = np.random.normal(0, 1, (num_images, 100))

generated_images = generator.predict(noise)

processed_images = np.zeros_like(generated_images)

for i in range(num_images):
    processed_images[i] = post_process(generated_images[i])


# =============================
# Save Output Image
# =============================

plt.figure(figsize=(10,4))

for i in range(num_images):

    plt.subplot(2, num_images, i+1)
    plt.imshow(grayscale_images[i].squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(2, num_images, i+1+num_images)
    plt.imshow(processed_images[i])
    plt.axis("off")

plt.tight_layout()
plt.savefig("result.png")

print("\nSaved generated output as result.png in current folder")
