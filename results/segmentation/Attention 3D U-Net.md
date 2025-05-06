# Segmentation Results – Attention 3D U-Net

The Attention 3D U-Net model, an extension of the traditional 3D U-Net, incorporates attention mechanisms that help the network focus on relevant features in 3D MRI scans. This results in enhanced tumor region segmentation, particularly for complex and subtle areas that standard models might miss.

---

## 📊 Attention 3D U-Net Performance Metrics

**Attention 3D U-Net Metrics**

| Metric                      | Value  |
|-----------------------------|--------|
| Accuracy                   | 0.9940 |
| Dice Coefficient           | 0.6280 |
| Dice Coefficient (Edema)   | 0.6290 |
| Dice Coefficient (Enhancing) | 0.5231 |
| Dice Coefficient (Necrotic) | 0.3616 |
| Loss                        | 0.0176 |
| Mean IoU (1)               | 0.6455 |
| Precision                  | 0.9942 |
| Sensitivity                | 0.9927 |
| Specificity                | 0.9981 |

---

### 🧠 Model Analysis

The Attention 3D U-Net achieved an impressive **accuracy of 0.9940**, with a very low **loss** of 0.0176, indicating minimal prediction error. The **Dice coefficient of 0.6280** suggests good overlap between predicted and actual tumor regions.

- **Edema segmentation** performed the best, with a Dice coefficient of **0.6290**, showing the model’s effective ability to detect swelling.
- The model showed slightly lower performance for the **enhancing tumor (0.5231)** and **necrotic tissue (0.3616)**, indicating some challenges in segmenting these more variable or subtle regions.
  
Notable performance metrics include:
- **Precision: 0.9942**, **Sensitivity: 0.9927**, and **Specificity: 0.9981**, demonstrating the model's high accuracy in detecting true tumor regions and minimizing false positives and false negatives.
- **Mean IoU of 0.6455** confirms strong spatial segmentation capability.

These results demonstrate that the Attention 3D U-Net performs well in classifying and segmenting tumor components, with some room for improvement in detecting smaller or more complex structures like necrotic tissue.

---

## 📈 Training History

![Attention 3D U-Net Model History](https://github.com/user-attachments/assets/f6f382b3-3dd5-4043-88ff-9e5c0516f003)


---

## 🧩 Segmentation Predictions

![Attention 3D U-Net Model Prediction](https://github.com/user-attachments/assets/a462470a-682e-49a5-9922-705f1e770c8a)


The prediction image from the Attention 3D U-Net model highlights its enhanced ability to focus on relevant tumor regions using attention gates. This model shows a **superior ability to delineate tumor boundaries**, especially in complex areas such as edema, enhancing tumors, and necrotic regions.

The attention mechanism suppresses irrelevant background features and emphasizes critical tumor areas, resulting in more accurate segmentation. The predicted mask aligns closely with the ground truth, suggesting that the Attention 3D U-Net can capture spatial information more effectively than standard models.

---

## ✅ Summary

The Attention 3D U-Net demonstrates **exceptional tumor segmentation performance**, with high accuracy and strong precision, sensitivity, and specificity. While the model performs well overall, the **segmentation of necrotic tissue** remains a challenge, and future improvements could focus on addressing this. The attention mechanism greatly enhances the model's focus on relevant features, making it a promising tool for clinical tumor analysis and diagnosis.

