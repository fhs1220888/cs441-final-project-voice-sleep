import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

df = pd.read_csv("data/processed/features.csv")

X = df.drop(columns=["label", "file"])
y = df["label"]

# ----- Regression -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n===== Linear Regression =====")
reg = LinearRegression().fit(X_train_scaled, y_train)
pred_reg = reg.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, pred_reg))
print("RMSE:", rmse)

# ----- Classification -----
y_binary = (y >= 3).astype(int)  # 3–5 good sleep, 1–2 bad sleep

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n===== Logistic Regression =====")
clf = LogisticRegression().fit(X_train_scaled, y_train)
pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

print("\n===== kNN =====")
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)
pred_knn = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, pred_knn)
print("Accuracy:", acc_knn)
