📌 Overview

This repository contains the implementation of a Deepfake Detection model leveraging Convolutional Neural Networks (CNNs), Vision Transformers (ViT), ResNet, and Logistic Regression. The goal is to detect manipulated facial images and videos using deep learning and traditional machine learning techniques.

🔍 Problem Statement

Deepfakes pose a significant threat to digital security, misinformation, and privacy. This project aims to develop a robust model that can efficiently differentiate between real and fake images/videos.

🏗️ Model Architectures

The project explores multiple architectures:

CNN: Captures spatial features and local patterns.
ResNet: Handles deeper networks using residual connections to prevent vanishing gradients.
Vision Transformer (ViT): Extracts global dependencies using self-attention mechanisms for effective feature representation.
Logistic Regression: Acts as a simple baseline model for classification.
📂 Dataset

We have used publicly available deepfake datasets such as:

FaceForensics++
Celeb-DF
⚙️ Methodology

Data Preprocessing
✔️ Face detection and alignment
✔️ Data augmentation for better generalization

Model Training
✔️ Implemented CNN, ResNet, ViT, and Logistic Regression architectures
✔️ Used Transfer Learning for better performance
✔️ Fine-tuned hyperparameters for optimization
![output](https://github.com/user-attachments/assets/72550fed-c50e-4453-9ad2-368e343ccd67)

Evaluation Metrics
✔️ Accuracy
✔️ Precision, Recall, and F1-Score
✔️ AUC-ROC Curve

🛠️ Technologies Used

Python, TensorFlow/Keras, PyTorch
OpenCV for image processing
Matplotlib & Seaborn for visualization
Scikit-learn for evaluation metrics & Logistic Regression
🚀 Results

Our best-performing model achieved:
✅ CNN: 86.19% Accuracy
✅ ResNet: 92.68% Accuracy
✅ ViT: 98.11% Accuracy
✅ Logistic Regression: 51.05% Accuracy (Baseline Model)

📜 Research Paper

The research paper detailing this work is available in the repository under DeepFake_Final.pdf.
