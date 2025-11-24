import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score


# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv("../data/processed/features.csv")

X = df.drop(columns=["label", "file"])
y_reg = df["label"]                       # regression labels
y_clf = (df["label"] >= 3).astype(int)    # classification labels (good vs bad sleep)

models_reg = {
    "LinearRegression": LinearRegression(),
    "kNN_Regressor": KNeighborsRegressor(n_neighbors=5)
}

models_clf = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "kNN_Classifier": KNeighborsClassifier(n_neighbors=5)
}


# ---------------------------
# Helper: Run K-Fold Evaluation
# ---------------------------
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
            "Metric": "RMSE",
            "Mean": np.mean(rmse_scores),
            "Std": np.std(rmse_scores)
        })

    return results


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
            "Metric": "Accuracy",
            "Mean": np.mean(acc_scores),
            "Std": np.std(acc_scores)
        })

    return results


# ---------------------------
# Run Evaluation
# ---------------------------
reg_results = evaluate_regression(models_reg, X, y_reg)
clf_results = evaluate_classification(models_clf, X, y_clf)

all_results = pd.DataFrame(reg_results + clf_results)

print("\n===== Evaluation Results (5-fold CV) =====\n")
print(all_results)

# Optionally save results
all_results.to_csv("../data/processed/eval_results.csv", index=False)
print("\nSaved results to eval_results.csv")
