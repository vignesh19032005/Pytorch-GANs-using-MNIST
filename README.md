``# Generative Adversarial Network (GAN) for MNIST using PyTorch

This project implements a **Generative Adversarial Network (GAN)** using **PyTorch** to generate handwritten digits similar to the MNIST dataset.

## 🎯 Project Overview

- Utilizes **PyTorch** to build and train a **GAN**.
- The model consists of a **Generator** and **Discriminator**, trained in an adversarial setup.
- The Generator learns to create realistic-looking digits from noise.
- The Discriminator learns to distinguish real MNIST digits from generated ones.

## 📌 Dataset

- **MNIST Dataset**: A collection of 70,000 grayscale images of handwritten digits (0-9).
- Downloaded and preprocessed using `torchvision.datasets.MNIST`.

## 🛠️ Installation & Setup

### 1️⃣ Install Dependencies
```bash
pip install torch torchvision matplotlib numpy`` 

### 2️⃣ Run the Training Script

bash

CopyEdit

`python train.py` 

🏗️ Model Architecture
----------------------

### Generator:

*   Uses transposed convolution layers (`ConvTranspose2d`) to upsample noise into a 28x28 image.
*   Activation functions: ReLU & Tanh.

### Discriminator:

*   Uses convolutional layers (`Conv2d`) to classify whether an image is real or fake.
*   Activation functions: LeakyReLU & Sigmoid.

📈 Training Details
-------------------

*   **Loss Function:** Binary Cross-Entropy (BCE)
*   **Optimizer:** Adam
*   **Training Steps:**
    1.  Train the Discriminator to differentiate real and generated digits.
    2.  Train the Generator to create more realistic digits.

🎥 Training Progress
--------------------

Here’s a video showing how the generated digits improve over training:

🚀 Future Improvements
----------------------

*   Experimenting with **DCGAN (Deep Convolutional GAN)**
*   Adding **Conditional GAN (cGAN)** for class-specific digit generation
*   Hyperparameter tuning for better stability

📌 References
-------------

*   **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
*   **GAN Paper (Goodfellow et al., 2014)**: [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)

* * *

Feel free to modify the post and README to fit your style! Let me know if you need any refinements. 🚀
