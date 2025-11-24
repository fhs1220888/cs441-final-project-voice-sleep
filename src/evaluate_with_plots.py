import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score



# Load Data
df = pd.read_csv("../data/processed/features.csv")

X = df.drop(columns=["label", "file"])
y_reg = df["label"]
y_clf = (df["label"] >= 3).astype(int)


# Define Models
models_reg = {
    "LinearRegression": LinearRegression(),
    "kNN_Regressor": KNeighborsRegressor(n_neighbors=5)
}

models_clf = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "kNN_Classifier": KNeighborsClassifier(n_neighbors=5)
}


# Helper: K-Fold Regression
def evaluate_regression(models, X, y, k=5):
    results = []

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for name, model in models.items():
        rmse_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, pred))
            rmse_scores.append(rmse)

        results.append({
            "Model": name,
            "RMSE_Mean": np.mean(rmse_scores),
            "RMSE_Std": np.std(rmse_scores)
        })

    return pd.DataFrame(results)


# Helper: K-Fold Classification
def evaluate_classification(models, X, y, k=5):
    results = []

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for name, model in models.items():
        acc_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)
            acc_scores.append(acc)

        results.append({
            "Model": name,
            "Acc_Mean": np.mean(acc_scores),
            "Acc_Std": np.std(acc_scores)
        })

    return pd.DataFrame(results)


# Run Evaluation
reg_df = evaluate_regression(models_reg, X, y_reg)
clf_df = evaluate_classification(models_clf, X, y_clf)

print("\n===== Regression (RMSE) Results =====\n")
print(reg_df)

print("\n===== Classification (Accuracy) Results =====\n")
print(clf_df)


# Plot RMSE
plt.figure(figsize=(8, 5))
plt.bar(reg_df["Model"], reg_df["RMSE_Mean"], yerr=reg_df["RMSE_Std"], capsize=5)
plt.ylabel("RMSE")
plt.title("Model Comparison (Regression RMSE)")
plt.savefig("../data/processed/rmse_plot.png")
plt.close()


# Plot Accuracy
plt.figure(figsize=(8, 5))
plt.bar(clf_df["Model"], clf_df["Acc_Mean"], yerr=clf_df["Acc_Std"], capsize=5)
plt.ylabel("Accuracy")
plt.title("Model Comparison (Classification Accuracy)")
plt.savefig("../data/processed/accuracy_plot.png")
plt.close()

print("\nSaved: rmse_plot.png & accuracy_plot.png in data/processed/")
