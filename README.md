# 🎨 Image Colorization using Deep Learning

This project implements **automatic image colorization** using a deep learning model trained on the CIFAR-10 dataset.
The model learns to convert **grayscale images into color images** using a convolutional neural network.

The goal of this project is to explore how neural networks can **predict plausible colors from grayscale inputs**.

---

## 📌 Features

* Converts grayscale images into color images
* Uses a convolutional neural network for color prediction
* Trained using the CIFAR-10 dataset
* Saves generated outputs as `result.png`
* Supports testing on grayscale images

---

## 🧠 Model Overview

The model takes a grayscale image as input and predicts the color version.

Pipeline:

```
Grayscale Image (32x32x1)
        ↓
Convolutional Neural Network
        ↓
Predicted Color Image (32x32x3)
```

The network learns patterns such as:

* edges
* shapes
* textures

to infer possible colors.

---

## 📂 Project Structure

```
Image-Colorization
│
├── colorize.py              # Main training and colorization script
├── image_colourisation.py  # Initial GAN experiment
├── requirements.txt
├── .gitignore
└── result.png               # Generated output image
```

---

## 📊 Dataset

This project uses the **CIFAR-10 dataset**.

Dataset properties:

* 60,000 images
* 32×32 resolution
* 10 classes
* RGB color images

During training, the images are converted to grayscale and the model learns to reconstruct the original colors.

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/grvsnh/Image-Colorization.git
cd Image-Colorization
```

Create a virtual environment:

```
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🚀 Running the Project

Run the training script:

```
python colorize.py
```

After training completes, the model generates colorized images and saves them as:

```
result.png
```

---

## 🖼 Example Output

The output image contains:

```
Original Image
Grayscale Image
Predicted Color Image
```

Example layout:

```
Original | Grayscale | Predicted
```

---

## 🔬 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* scikit-learn

---

## 🚀 Future Improvements

Possible improvements include:

* Using a **U-Net architecture** for better color prediction
* Training on higher-resolution datasets
* Implementing a **GAN-based colorization model**
* Creating a **web interface for image uploads**
* Supporting real-world grayscale photographs

---

## 📜 License

This project is open-source and available under the MIT License.

---
