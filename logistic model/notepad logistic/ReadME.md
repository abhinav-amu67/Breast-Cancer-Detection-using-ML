## Logistic Regression Model

This module applies Logistic Regression to classify breast tumors as malignant or benign using standardized clinical features from the Wisconsin Breast Cancer dataset.

---

### Model Setup
- Dataset: sklearn Breast Cancer
- Train–test split: 75% / 25%
- Feature scaling: StandardScaler
- Max iterations: 100

---

### Performance (Default Threshold = 0.5)

- Accuracy: **97.20%**

Confusion Matrix:<br>
[[46 4] <br>
[ 0 93]]


Key observations:
- No false negatives (important for medical diagnosis)
- Few false positives present

---

### Threshold Optimization (ROC-Based)

ROC analysis was used to identify an optimal decision threshold.

- Best threshold: **0.6527**
- ROC–AUC: **0.998**

---

### Performance (Optimized Threshold)

- Accuracy: **98.60%**

Confusion Matrix:<br>
[[48 2]<br>
[ 0 93]]

Improvements after threshold tuning:
- Reduced false positives
- Maintained zero false negatives
- Better precision–recall balance

---



