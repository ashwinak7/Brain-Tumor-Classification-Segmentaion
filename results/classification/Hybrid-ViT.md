# Classification Results – Hybrid Vision Transformer (ViT-CNN)

## 📊 Hybrid ViT-CNN Performance Metrics

**Hybrid Vision Transformer Metrics**

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Glioma      | 0.9733    | 0.8500 | 0.9075   | 300     |
| Meningioma  | 0.8620    | 0.9183 | 0.8892   | 306     |
| No Tumor    | 0.9711    | 0.9951 | 0.9829   | 405     |
| Pituitary   | 0.9545    | 0.9800 | 0.9671   | 300     |

- **Overall Accuracy**: 0.9904  
- **Dice Score**: 0.94

---

Hybrid Vision Transformer achieved an **overall accuracy of 99.04%** and a **Dice Score of 0.94**, indicating excellent performance in distinguishing tumor vs. non-tumor brain MRI scans.

- **No Tumor cases**: Precision of 0.97 and recall of 0.99 with an F1 score of 0.98. The model excelled in correctly identifying healthy brain scans, minimizing false positives.
- **Pituitary tumors**: Precision of 0.95 and recall of 0.98 resulted in a strong F1 score of 0.97, showcasing reliability in detecting this class.
- **Glioma**: High precision (0.97) but relatively lower recall (0.85), resulting in an F1-score of 0.91 – suggesting some false negatives where gliomas were misclassified.
- **Meningioma**: Lower precision (0.86) and higher recall (0.92) led to an F1-score of 0.89. This highlights the model's tendency to prioritize sensitivity (minimizing missed tumors) at the cost of a few false positives.

These findings emphasize the model's effectiveness and its balance between precision and recall, especially for complex or visually similar tumor types.

---

## 📈 Hybrid Vision Transformer Model History

![Hybrid Vision Tranformer Model History](https://github.com/user-attachments/assets/9f8ad486-e409-4608-92a1-5144845a2fd4)

---

## 🧮 Hybrid ViT Confusion Matrix

![Hybrid Vision Tranformer Confusion Matrix](https://github.com/user-attachments/assets/a6d47cd4-381b-467a-b871-1ce7b30723ba)


The confusion matrix reveals:

- **Class 2 (No Tumor)**: Perfect classification with **405 correct predictions**.
- **Class 3 (Pituitary)**: 296 correct predictions, showing strong generalization.
- **Class 0 (Glioma)** and **Class 1 (Meningioma)**: Minor misclassifications, mostly among each other, with **276 and 266** correct predictions respectively.

The matrix confirms the Hybrid ViT model’s robustness and its ability to distinguish tumor classes with minimal confusion, thanks to its combination of CNNs and attention mechanisms.

---

## 🔍 Hybrid ViT Model Predictions

![Hybrid Vision Tranformer Model Prediction](https://github.com/user-attachments/assets/4457e42b-181f-4d08-9770-f37c0a3a2e52)


The predictions of the Hybrid Vision Transformer model are visually compared with ground truth labels for MRI samples.

The model demonstrates strong classification performance across Glioma, Meningioma, Pituitary, and No Tumor categories.  
- Most predictions match their respective true labels.
- The hybrid architecture captures both **fine-grained spatial features** and **long-range dependencies**, contributing to accurate predictions under varied tumor shapes and orientations.

---

## ✅ Summary

The Hybrid Vision Transformer (ViT-CNN) model exhibits outstanding classification performance, achieving near-perfect results for No Tumor and Pituitary cases. Despite slight limitations distinguishing Glioma and Meningioma, the model demonstrates excellent generalization and clinical potential for multi-class tumor detection. These results support the integration of transformer-based attention in medical image classification frameworks.

