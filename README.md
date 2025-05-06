# Brain Tumor Classification and Segmentation using Deep Learning

## 🔬 Overview

Brain tumor detection is a critical challenge in medical imaging, where early and accurate diagnosis is essential for effective treatment planning. This project leverages Deep Learning (DL) and Machine Learning (ML) techniques to build an automated system for both classification and segmentation of brain tumors from MRI scans. By utilizing large datasets and state-of-the-art neural networks, we aim to support medical professionals in making faster and more reliable decisions.

## 🧠 Problem Statement

Brain tumors are among the most dangerous and fatal diseases. Manual diagnosis is time-consuming and prone to human error due to fatigue and variability in interpretation. Therefore, there's a pressing need for a computerized, accurate, and dependable system to assist doctors with automatic tumor detection and classification.

## 🎯 Purpose

This project is designed to:
- Automatically **segment** and **classify** brain tumors from MRI scans.
- Improve diagnostic **accuracy** and reduce **human error**.
- Provide a **web interface** to make the system accessible and easy to use.
- Reduce the time and effort required in reviewing neuro-imaging manually.

## 💡 Approach

- **Classification Models**:  
  - `ResNet50`  
  - `Hybrid Vision Transformer (ViT-CNN)`

- **Segmentation Models**:  
  - `Standard 3D U-Net`  
  - `Attention 3D U-Net`

- **Web Interface**:  
  - Built using Flask to allow users to upload MRI scans and view classification and segmentation results.

## 🧾 Datasets

- **Classification Dataset**: 7000+ MRI scans from [Kaggle]([https://www.kaggle.com](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset))
- **Segmentation Dataset**: BraTS 2020 Dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Flask (Web Framework)
- OpenCV & NumPy
- Scikit-learn
- Jupyter Notebooks

## 📊 Evaluation

We evaluated the models using various performance metrics such as:
- Accuracy
- Dice Similarity Coefficient
- Precision & Recall
- F1 Score

## 🔍 Background

MRI imaging is commonly used to diagnose brain tumors. Manual review of MRI scans is difficult due to overlapping tumor features. DL models like CNNs and Transformers have demonstrated high performance in classifying and segmenting brain anomalies, making them suitable for this task.

## 🤖 Role of Deep Learning

Deep Learning models are powerful in learning complex patterns directly from imaging data. CNNs and Transformer-based architectures like ViT are capable of extracting spatial and contextual features, significantly enhancing diagnostic capabilities.

## 🚀 Motivation

This project was inspired by a strong interest in AI and its potential in healthcare. It combines technical curiosity with a meaningful real-world impact by contributing toward the improvement of early detection and diagnosis of brain tumors.

## 📁 Project Structure

