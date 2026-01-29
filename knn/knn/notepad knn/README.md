#OVERVIEW
<br>
This module applies the K-Nearest Neighbors (KNN) algorithm to classify breast tumors as malignant or benign using standardized clinical features.
<br>
**What i actually did**
<br>
1.Loaded Breast Cancer dataset from sklearn<br>
2.Performed train–test split (75/25)<br>
3.Standardized features using StandardScaler<br>
    Why scaling was necessary ?<br>
    Feature scaling was essential because KNN relies on distance calculations, and
    unscaled features would dominate neighbor selection.<br>
4.Trained a KNN classifier<br>
5.Evaluated using confusion matrix, classification report, and ROC curve<br>
**Results summary**<br>
Accuracy ≈ 95%<br>
Confusion Matrix:<br>
    False Positives: 5<br>
    False Negatives: 2<br>
ROC-AUC ≈ 0.97<br>
    ROC analysis was performed to explore threshold tuning. However, changing the
    threshold did not significantly alter predictions due to KNN’s discrete
    probability outputs,but even after that the output is not impressive logistic 
    model is better than this model.<br>
**Limitations**<br>
KNN performance is sensitive to the choice of k and does not scale well for large datasets.<br>
