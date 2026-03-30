# 🎨 Image Colorization using Deep Learning

A dual-model image colorization system that converts black & white images into realistic color images using deep learning.

This project combines:

- A **custom-trained lightweight model**
- A **state-of-the-art pretrained model (Zhang et al., ECCV 2016)**

---

## 🚀 Features

- 🧠 Dual Model System
    - Local trained model (fast, lightweight)
    - Pretrained model (high-quality results)

- 🔄 Comparison Mode  
  View **Original vs Local vs Pretrained** side-by-side

- 🎨 Color Boosting  
  Enhances saturation for better visual output in the custom model

- 📱 Mobile-Friendly UI  
  Clean Streamlit interface optimized for all devices

- ⚡ Fast Processing  
  Real-time colorization

---

## 🧠 Models Used

### 🔹 1. Local Model (Custom Trained)

- Trained on Oxford-IIIT dataset
- Input: Grayscale (L channel)
- Output: Color channels (AB)
- Architecture: CNN-based encoder-decoder
- Loss: MAE
- Activation: Tanh (full color range)

> Designed to demonstrate training pipeline under hardware constraints

---

### 🔹 2. Pretrained Model

Based on:

**"Colorful Image Colorization" — Zhang et al. (ECCV 2016)**

- Trained on large-scale datasets
- Implemented using OpenCV DNN (Caffe)
- Produces high-quality realistic results

---

## 🧪 How It Works

1. Convert image → LAB color space
2. Extract L channel (grayscale)
3. Predict AB channels using model
4. Combine L + AB
5. Convert back to RGB

---

## 🖼️ Pipeline Visualization

![Pipeline](assets/image_process.jpg)

---

## 🧬 LAB Color Space

![LAB](assets/LAB-color.webp)

---

## 🏗️ Model Architecture

![Architecture](assets/network-architecture.webp)

---

## ⚙️ Installation

```bash
git clone https://github.com/grvsnh/Image-Colorization.git
```

```bash
cd Image-Colorization
```

```bash
python -m venv venv
```

```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

```bash
pip install -r requirements.txt
```

---

▶️ Run the App

```bash
streamlit run app.py
```

---

🖥️ Usage

1. Upload a grayscale image

2. Select mode:

    Use Both (Compare)

    Local Model

    Pretrained Model

3. Click Colorize

4. Download result

---

📁 Project Structure

```bash
Image-Colorization/
│
├── app.py
├── colorize.py
├── requirements.txt
│
├── models/
│ ├── local-trained/
│ │ └── colorization_model.keras
│ │
│ ├── pre-trained/
│ │ ├── colorization_release_v2.caffemodel
│ │ ├── colorization_deploy_v2.prototxt
│ │ └── pts_in_hull.npy
│ │
│ └── model_training/
│ ├── core_model.ipynb
│ ├── colorization_training.ipynb
│ └── colorization_example.ipynb
│
├── assets/
└── README.md
```

---

🧠 Limitations

Local model produces less realistic colors

Training limited due to hardware constraints

Colorization is inherently ambiguous

---

🔮 Future Improvements

GAN-based colorization

Better dataset (COCO / ImageNet subset)

Real-time video colorization

Web deployment with GPU

---

📚 References

Zhang, R., Isola, P., & Efros, A. A. (2016). Colorful Image Colorization

OpenCV DNN Module

TensorFlow / Keras

---
