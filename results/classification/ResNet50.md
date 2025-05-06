# Classification Results – ResNet50

## 📊 ResNet50 Performance Metrics

**ResNet50 Metrics**

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Glioma      | 0.97      | 0.99   | 0.98     | 310     |
| Meningioma  | 0.97      | 0.94   | 0.96     | 326     |
| No Tumor    | 1.00      | 0.99   | 0.99     | 419     |
| Pituitary   | 0.96      | 0.99   | 0.98     | 350     |

- **Overall Accuracy**: 0.97  
- **Dice Score**: 0.96

The model achieved an overall accuracy of **97%** and a Dice Score of **0.96**, indicating a high level of confidence in distinguishing between tumor and non-tumor cases. 

- Glioma and Pituitary classes showed high precision (0.97 and 0.96) and recall (0.99 for both), producing an F1-score of 0.98.
- No Tumor cases achieved **perfect precision (1.00)** and **recall (0.99)**, representing the model’s excellent ability to identify healthy scans.
- Meningioma had slightly lower recall (0.94), leading to an F1-score of 0.96, possibly due to similarity with other tumor types.

---

## 📈 ResNet50 Model History

![ResNet50 Model History](https://github.com/user-attachments/assets/a5bcdaa8-3104-4422-b1f4-b10b3cd45749)

---

## 🧮 ResNet50 Confusion Matrix

![ResNet50 Confusion Matrix](../figures/resnet50_confusion_matrix.png)

The confusion matrix shows strong predictive accuracy, with most predictions aligned diagonally (correct predictions).  
- **Class 0 (Glioma)** and **Class 2 (Pituitary)** achieved particularly high correct classifications: 306 and 415 respectively.
- **Class 1 (Meningioma)** showed slight confusion due to MRI feature similarity with other classes.
- **Class 3 (No Tumor)** showed high accuracy with 346 correct predictions.

---

## 🔍 ResNet50 Model Predictions

![ResNet50 Predictions](../figures/resnet50_model_predictions.png)

The prediction figure visually demonstrates the model’s performance on various MRI samples. Ground truth labels and predictions are compared per image. Most predictions align well with actual diagnoses, highlighting the model’s practical diagnostic capability.

---

## ✅ Summary

The ResNet50 model provides balanced and high classification performance across brain tumor types, particularly for Glioma, Pituitary, and No Tumor classes. Slight inconsistencies for Meningioma classification may be improved with further data augmentation or attention-based refinements. These results support ResNet50’s strength in MRI-based tumor classification tasks.
