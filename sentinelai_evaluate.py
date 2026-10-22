import time
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("Loading UNSW-NB15 datasets...")

train = pd.read_csv("UNSW_NB15_training-set.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv")

# Keep the real labels for evaluation, not as model inputs.
excluded_columns = ["id", "label", "attack_cat"]

X_train_all = train.drop(columns=excluded_columns)
X_test = test.drop(columns=excluded_columns)
y_test = test["label"].astype(int)

# Train the unsupervised model on normal traffic only.
normal_mask = train["label"].astype(int) == 0
X_train = X_train_all.loc[normal_mask].copy()

# Detect categorical and numeric input features.
categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_features = [
    col for col in X_train.columns
    if col not in categorical_features
]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Numeric features:", len(numeric_features))
print("Categorical features:", categorical_features)

# Encode categories and scale numeric features.
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        ), categorical_features)
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("detector", IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Isolation Forest...")
start_time = time.time()
model.fit(X_train)

print("Predicting test records...")
raw_predictions = model.predict(X_test)

# Isolation Forest: -1 = anomaly, 1 = normal.
y_pred = (raw_predictions == -1).astype(int)

# Larger scores indicate more anomalous records.
anomaly_scores = -model.decision_function(X_test)

# Calculate evaluation metrics.
tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)
false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

print("\n===== SENTINELAI EVALUATION =====")
print(f"Precision:          {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:             {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-score:           {f1:.4f} ({f1 * 100:.2f}%)")
print(
    f"False-positive rate: {false_positive_rate:.4f} "
    f"({false_positive_rate * 100:.2f}%)"
)

print("\n===== CONFUSION MATRIX COUNTS =====")
print("True negatives (normal correctly identified):", tn)
print("False positives (normal flagged as attack):", fp)
print("False negatives (attack missed):", fn)
print("True positives (attack correctly detected):", tp)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Normal", "Attack"],
    zero_division=0
))

# Save test predictions for the database and dashboard steps.
results = pd.DataFrame({
    "id": test["id"],
    "actual_label": y_test,
    "predicted_label": y_pred,
    "anomaly_score": anomaly_scores,
    "attack_cat": test["attack_cat"]
})

results["status"] = results["predicted_label"].map({
    0: "Normal",
    1: "Suspicious"
})

output_file = "sentinelai_evaluation_results.csv"
results.to_csv(output_file, index=False)

print("Results saved to:", output_file)
print("Suspicious records:", int(y_pred.sum()))
print(f"Elapsed time: {time.time() - start_time:.1f} seconds")
print("\nEvaluation completed.")
