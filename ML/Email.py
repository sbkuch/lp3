import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("emails (1).csv")

df.head()

df.isnull().sum()

X = df.iloc[:,1:3001]
X

Y = df.iloc[:,-1].values
Y

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

from sklearn.metrics import classification_report, confusion_matrix

# -------- Support Vector Machine --------
svc = SVC(C=1.0, kernel='rbf', gamma='auto')
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, svc_pred))
print("SVM Classification Report:\n", classification_report(y_test, svc_pred))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, svc_pred))

import numpy as np
# make sure everything is numeric and contiguous
X_train = np.ascontiguousarray(X_train.values, dtype=np.float64)
X_test  = np.ascontiguousarray(X_test.values,  dtype=np.float64)
y_train = np.array(y_train)
y_test  = np.array(y_test)

# -------- K-Nearest Neighbors --------
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print("KNN Accuracy:", knn.score(X_test, y_test))
print("KNN Classification Report:\n", classification_report(y_test, knn_pred))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
