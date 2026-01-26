import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
X=df.drop('target',axis=1)
y=df['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=67)
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X_train_scale=std.fit_transform(X_train)
X_test_scale=std.transform(X_test)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=100)
model.fit(X_train_scale,y_train)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
y_pred=model.predict(X_test_scale)
print("=== Default threshold (0.5) ===")
print("accuracy:",accuracy_score(y_test,y_pred))
print("classificantion:\n",classification_report(y_test,y_pred,zero_division=0))
print("confusion matrix:\n",confusion_matrix(y_test,y_pred))

from sklearn.metrics import roc_curve, roc_auc_score
y_prob = model.predict_proba(X_test_scale)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

J = tpr - fpr
best_idx = np.argmax(J)
best_threshold = thresholds[best_idx]
print("Best threshold:", best_threshold)
y_pred_best = (y_prob >= best_threshold).astype(int)

# 🔹 Metrics with best threshold
print("\n=== Using Best Threshold ===")
print("AUC:", roc_auc_score(y_test, y_prob))
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best, zero_division=0))
