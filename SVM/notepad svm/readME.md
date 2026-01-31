# Breast Cancer Detection using SVM (Threshold Optimized)

This project implements a Support Vector Machine (SVM) classifier for breast
cancer detection using the Wisconsin Breast Cancer Dataset.  
In addition to the default decision threshold (0.5), the model is further
improved by selecting an optimal threshold based on ROC analysis.

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
- C: 1 (we must search for better result not GridSearchCV)
- Gamma: scale 
- Probability Enabled: True
- Feature Scaling: StandardScaler

--------------------------------------------------
Results with Default Threshold (0.5)
--------------------------------------------------

Confusion Matrix:     <br>
    [[46  4]      <br>
     [ 1 92]]

Classification Report:

| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| Malignant (0) | 0.98 | 0.92 | 0.95 |
| Benign (1) | 0.96 | 0.99 | 0.97 |

Accuracy: 97%

--------------------------------------------------
ROC Curve & Threshold Optimization
--------------------------------------------------

- ROC-AUC Score: 0.9965
- Best Threshold (Youden’s J statistic): 0.7140

--------------------------------------------------
Results with Best Threshold (0.714)
--------------------------------------------------

Confusion Matrix:     <br>
    [[49  1]       <br>
     [ 1 92]]

Classification Report:

| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| Malignant (0) | 0.98 | 0.98 | 0.98 |
| Benign (1) | 0.99 | 0.99 | 0.99 |

Accuracy: 99%

Key Insight:
Optimizing the decision threshold significantly improves recall for the
Malignant class, which is critical in medical diagnosis where false negatives
are more dangerous than false positives.


How to Run:
    python svm_model.py

Dependencies:
    pip install numpy pandas matplotlib scikit-learn

Project Structure:
    Breast-Cancer-Detection-using-ML/
    ├── SVM/
    │   └── svm_model.py
    ├── svm_screenshots/
    │   ├── roc_curve.png
    │   └── confusion_matrix.png
    └── README.md

Author:
Abhinav Kumar  
B.Tech (Electronics Engineering)  
Machine Learning Enthusiast




