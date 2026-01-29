# Overview
This module implements the **K-Nearest Neighbors (KNN)** algorithm to classify breast tumors as **malignant or benign** using standardized clinical features from the Breast Cancer Wisconsin dataset.

## What I Actually Did
1. Loaded the Breast Cancer dataset from `sklearn`
2. Performed a **75/25 train–test split**
3. Standardized features using **StandardScaler**  
   **Why scaling was necessary:**  
   KNN is a distance-based algorithm. Without scaling, features with larger numeric ranges dominate distance calculations, leading to biased neighbor selection. Standardization ensures all features contribute equally.
4. Trained a **KNN classifier**
5. Evaluated the model using:
   - Confusion Matrix  
   - Classification Report  
   - ROC Curve

## Results Summary
- **Accuracy:** ~95%  
- **False Positives:** 5  
- **False Negatives:** 2  
- **ROC–AUC:** ~0.97  

ROC analysis was used to explore threshold tuning. However, changing the threshold did **not significantly affect predictions** due to KNN’s discrete probability outputs.

## Model Insight
Although KNN achieved strong performance, **Logistic Regression produced more stable and interpretable results** on this dataset, making it a better choice for this medical classification task.

## Limitations
- Sensitive to the choice of **k**
- Computationally expensive for large datasets
- Poor scalability as dataset size increases
