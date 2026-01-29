#OVERVIEW
This module applies the K-Nearest Neighbors (KNN) algorithm to classify breast tumors as malignant or benign using standardized clinical features.
**What i actually did**
1.Loaded Breast Cancer dataset from sklearn
2.Performed train–test split (75/25)
3.Standardized features using StandardScaler
    Why scaling was necessary ?
    Feature scaling was essential because KNN relies on distance calculations, and
    unscaled features would dominate neighbor selection.
4.Trained a KNN classifier
5.Evaluated using confusion matrix, classification report, and ROC curve
**Results summary**
Accuracy ≈ 95%
Confusion Matrix:
    False Positives: 5
    False Negatives: 2
ROC-AUC ≈ 0.97
    ROC analysis was performed to explore threshold tuning. However, changing the
    threshold did not significantly alter predictions due to KNN’s discrete
    probability outputs,but even after that the output is not impressive logistic 
    model is better than this model.
**Limitations**
KNN performance is sensitive to the choice of k and does not scale well for large datasets.
