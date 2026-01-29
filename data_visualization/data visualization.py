from sklearn.decomposition import PCA
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
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scale)

plt.figure(figsize=(7,5))
plt.scatter(X_pca[y_train==0, 0], X_pca[y_train==0, 1],
            alpha=0.4, label="Malignant", color="red")
plt.scatter(X_pca[y_train==1, 0], X_pca[y_train==1, 1],
            alpha=0.4, label="Benign", color="green")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.title("PCA Visualization of Breast Cancer Data")
plt.show()
