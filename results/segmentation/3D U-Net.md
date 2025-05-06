# Segmentation Results – Standard 3D U-Net

## 📊 Standard 3D U-Net Performance Metrics

**Standard 3D U-Net Metrics**

| Metric                      | Value  |
|-----------------------------|--------|
| Accuracy                   | 0.9924 |
| Dice Coefficient           | 0.6294 |
| Dice Coefficient (Edema)  | 0.5821 |
| Dice Coefficient (Enhancing) | 0.5468 |
| Dice Coefficient (Necrotic) | 0.3912 |
| Loss                      | 0.0231 |
| Mean IoU (1)              | 0.6439 |
| Precision                 | 0.9930 |
| Sensitivity               | 0.9910 |
| Specificity               | 0.9976 |

---

### 🧠 Model Analysis

Standard 3D U-Net achieved an impressive **accuracy of 0.9924**, showcasing its reliability in identifying tumor regions from volumetric MRI scans.

- The **overall Dice coefficient** of 0.6294 indicates a good overlap between predicted and ground truth segmentations.
- **Edema (0.5821)** and **enhancing regions (0.5468)** had moderate Dice scores, showing the model's reasonable performance in identifying these sub-regions.
- **Necrotic core (0.3912)** showed lower performance, which is expected due to its often subtle and irregular appearance in MRI scans.

Further metrics highlight the model's strength:
- **Precision: 0.9930**, **Sensitivity: 0.9910**, and **Specificity: 0.9976** indicate low false positives and negatives, which are vital for clinical reliability.
- A **mean Intersection over Union (IoU) of 0.6439** reflects consistent performance across all segmented classes.

---

## 📈 Training History

![Standard 3D U-Net Model History](https://github.com/user-attachments/assets/c9b77e4c-fe2e-404e-841e-0c8e2e669fe2)

---

## 🧩 Segmentation Predictions

![Standard 3D U-Net Model Prediction 1](https://github.com/user-attachments/assets/1557b28c-2260-481c-9d06-bb2057703be8)
![Standard 3D U-Net Model Prediction 2](https://github.com/user-attachments/assets/e41bde97-a20e-43bc-9060-4ef7526c873d)


The prediction image from the Standard 3D U-Net model visually demonstrates its capacity to delineate tumor boundaries across different sub-regions:
- The segmented regions include **edema**, **enhancing tumor**, and **necrotic core**.
- The model's predictions align closely with the ground truth labels, particularly for large, well-formed tumor regions.

These visual results validate the model’s ability to understand spatial continuity in 3D MRI volumes and produce clinically meaningful segmentation masks.

---

## ✅ Summary

The Standard 3D U-Net model exhibits **robust segmentation capabilities** on brain MRI scans. While the model showed slightly reduced performance in detecting necrotic cores, its overall precision, sensitivity, and specificity metrics strongly support its use in aiding clinical diagnosis and treatment planning. This architecture remains a **reliable and interpretable baseline** for volumetric medical image segmentation tasks.

