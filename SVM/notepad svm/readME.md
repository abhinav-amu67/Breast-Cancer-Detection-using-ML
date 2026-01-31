# Breast Cancer Detection using SVM

This project uses a Support Vector Machine (SVM) with RBF kernel to classify
breast cancer tumors as Malignant or Benign using the Wisconsin Breast Cancer Dataset.

Dataset:
- Source: sklearn.datasets.load_breast_cancer
- Samples: 569
- Features: 30
- Classes:
  - 0 → Malignant
  - 1 → Benign

Model Configuration:
- Algorithm: Support Vector Machine (SVM)
- Kernel: RBF
- C: 1
- Gamma: scale
- Probability: True
- Feature Scaling: StandardScaler

Results (Default Threshold = 0.5):

Confusion Matrix:
    [[46  4]
     [ 1 92]]

Classification Report:

| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| Malignant (0) | 0.98 | 0.92 | 0.95 |
| Benign (1) | 0.96 | 0.99 | 0.97 |

Accuracy: 97%

Confusion Matrix Visualization:
![Confusion Matrix](svm_screenshots/confusion_matrix.png)

ROC Curve:
- ROC-AUC ≈ 0.99

![ROC Curve](svm_screenshots/roc_curve.png)

How to Run:
    python svm_model.py

Dependencies:
    pip install numpy pandas matplotlib scikit-learn

Project Structure:
    Breast-Cancer-Detection-using-ML/
    ├── svm_model.py
    ├── svm_screenshots/
    │   ├── confusion_matrix.png
    │   └── roc_curve.png
    └── README.md

Author:
Abhinav Kumar  
B.Tech (Electronics Engineering)  
Machine Learning Enthusiast
