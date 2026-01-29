import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
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
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train_scale,y_train)
y_pred=knn.predict(X_test_scale)
#print("prediction",y_pred)
print("\n__for Default Threshold(0.5)__ ")
from sklearn.metrics import classification_report,confusion_matrix
print("confusion matrix:\n",confusion_matrix(y_test,y_pred))
print("classification\n",classification_report(y_test,y_pred,zero_division=0))
print("\n__with Best Threshold__" )
from sklearn.metrics import roc_curve,roc_auc_score
from numpy import argmax
y_prob=knn.predict_proba(X_test_scale)[:,1]
fpr,tpr,thresholds=roc_curve(y_test,y_prob)



Best_idx=argmax(tpr-fpr)
Best_threshold = thresholds[Best_idx]
y_pred_threshold=(y_prob>=Best_threshold).astype(int)
print("confusion matrix:\n",confusion_matrix(y_test,y_pred_threshold))
print("classification\n",classification_report(y_test,y_pred_threshold,zero_division=0))
print("Best threshold:",Best_threshold)
print("AUC", roc_auc_score(y_test,y_prob))

plt.plot(fpr,tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

