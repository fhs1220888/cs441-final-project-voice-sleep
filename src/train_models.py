import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import joblib
import os
import json

# Load extracted features
df = pd.read_csv("data/processed/features.csv")

X = df.drop(columns=["label", "file"])
y = df["label"]


# REGRESSION (Sleep Score)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print("\n===== Linear Regression =====")
reg = LinearRegression().fit(X_train_reg_scaled, y_train_reg)
pred_reg = reg.predict(X_test_reg_scaled)
rmse = np.sqrt(mean_squared_error(y_test_reg, pred_reg))
print("RMSE:", rmse)


# CLASSIFICATION (Good / Bad Sleep)
y_binary = (y >= 3).astype(int)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

print("\n===== Logistic Regression =====")
clf = LogisticRegression(max_iter=200).fit(X_train_clf_scaled, y_train_clf)
pred_clf = clf.predict(X_test_clf_scaled)
acc_clf = accuracy_score(y_test_clf, pred_clf)
print("Accuracy:", acc_clf)

print("\n===== kNN =====")
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_clf_scaled, y_train_clf)
pred_knn = knn.predict(X_test_clf_scaled)
acc_knn = accuracy_score(y_test_clf, pred_knn)
print("Accuracy:", acc_knn)

# SAVE MODELS
os.makedirs("models", exist_ok=True)

# Save regression pipeline
joblib.dump(scaler_reg, "models/scaler_reg.pkl")
joblib.dump(reg, "models/regression_model.pkl")

# Save classification pipeline
joblib.dump(scaler_clf, "models/scaler_clf.pkl")
joblib.dump(clf, "models/classifier_model.pkl")

print("\nSaved models to /models/")

feature_names = list(X.columns)
json.dump(feature_names, open("models/feature_names.json", "w"))