import numpy as np
import pandas as pd


# 1. Load dataset

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target   # 0 = malignant, 1 = benign

from sklearn.model_selection import train_test_split
# 2. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=67
)
# 3. Pipeline (Scaler + SVM)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf'))
])
# 4. Parameter grid
param_grid = {
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__gamma': [0.01, 0.1, 1, 'scale']
}
# 5. GridSearchCV
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
)
grid.fit(X_train, y_train)

# 6. Best parameters

print("Best Parameters:", grid.best_params_)
print("Best CV F1 Score:", grid.best_score_)

# 7. Final evaluation on test set

from sklearn.metrics import confusion_matrix, classification_report
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

