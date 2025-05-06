# Visualization on Flask

To create an interactive and user-friendly experience, a Flask-based web interface was developed to visualize outputs of both classification and segmentation models. Users can upload MRI images through the interface and view real-time output predictions, such as tumor type classification and segmented tumor regions. The integration of trained deep learning models with Flask ensures smooth deployment and allows end-users, such as medical professionals or researchers, to access results without requiring technical knowledge. This web application effectively bridges the gap between model development and practical usage, providing clear visual feedback for each prediction.

---

## 🖥️ Flask Interface Overview

The Flask-based web interface provides an intuitive platform for real-time brain tumor detection. It supports multiple deep learning models, including **ResNet50** and **Hybrid Vision Transformer (ViT)** for tumor classification. Users can easily select a model, upload MRI scans, and receive instant predictions with high-confidence scores.

- **Glioma Detection** (99.98% confidence using ResNet50)
- **Meningioma Classification** (99.81% confidence via Hybrid ViT)

In addition to classification, the interface also provides segmented tumor regions, allowing users to see spatial localization and demarcation of tumor boundaries for improved diagnostic clarity.

The system was designed to be **user-friendly**, removing entry barriers for non-technical users and embedding the trained models in an accessible web-based environment. This facilitates user interaction and provides real-time feedback, making it easier for medical professionals to leverage the power of AI in their diagnostic process.

---

## 📊 Flask Interface Features

- **Model Selection**: Users can choose between different deep learning models (e.g., ResNet50, Hybrid ViT, 3D UNet, Attention 3D UNet).
- **Real-Time Predictions**: After uploading an MRI scan, users immediately receive tumor type classification and segmentation predictions.
- **Segmentation Visualization**: Segmented tumor areas are displayed, including precise tumor boundaries and abnormal regions.

By combining **advanced classification** and **segmentation** frameworks, the web application connects cutting-edge AI research with clinical workflows, empowering stakeholders to make informed decisions efficiently.

---

## 🖼️ Visualizations

### ResNet50 Prediction Visualization on Web Interface

![ResNet50 Prediction Visualization on Web Interface](https://github.com/user-attachments/assets/809c8bc4-c33a-42a1-ba1a-d613abc2eccf)


### Hybrid Vision Transformer Prediction on Web-Page

![Hybrid ViT Prediction Visualization on Web Interface](https://github.com/user-attachments/assets/ef2c60b4-dc28-4840-84c6-85d19cf8c26e)


The Flask-based web interface demonstrates versatility in real-time brain tumor detection. Predictions are made using models like ResNet50 and Hybrid Vision Transformer (ViT) for classification. The interface provides high-confidence predictions (e.g., Glioma at 99.98% confidence) and also visualizes segmented tumor areas.

---

## 🧠 Segmentation Models on Flask

### Standard 3D U-Net Segmentation Visualization on Web Interface

![Standard 3D U-Net Segmentation Visualization on Web Interface](https://github.com/user-attachments/assets/8f2c4e1c-ba79-4ff3-96b7-f45a7cd9af89)


### Attention 3D U-Net Segmentation Visualization on Web Interface

![Attention 3D U-Net Segmentation Visualization on Web Interface](https://github.com/user-attachments/assets/cb4e9030-6d2e-4f84-b9b1-169c51bb6528)


The Flask-based web interface allows users to upload NIfTI-format MRI files (e.g., **BraTS20_Training_001_fair.nii**) and obtain real-time predictions. The interface displays both the **input MRI slices** and the corresponding **segmentation results**, highlighting tumor boundaries and regions.

The integration of both **Standard 3D U-Net** and **Attention 3D U-Net** segmentation models offers clear and precise visualizations, making the interface a powerful tool for medical professionals to validate model outputs. The design emphasizes accessibility, ensuring that the interface is intuitive and does not require technical expertise.

---

## ✅ Conclusion

The Flask-based web interface for the Brain Tumor MRI Detection project provides an **interactive and user-friendly platform** for analyzing MRI scans. By integrating advanced deep learning models, it offers **real-time tumor classification** and **precise segmentation** results. This deployment bridges the gap between cutting-edge AI technologies and practical healthcare applications, enabling faster and more accurate tumor detection, which could play a crucial role in clinical decision-making and research.
