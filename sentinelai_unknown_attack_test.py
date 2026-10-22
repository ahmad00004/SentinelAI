
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE = "UNSW_NB15_testing-set.csv"
HELD_OUT_CATEGORY = "Worms"

# Load the original datasets.
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

# All model inputs, excluding identifiers and target/label columns.
feature_columns = [
    c for c in train.columns
    if c not in ["id", "label", "attack_cat"]
]

# Check that the testing dataset has the same input features.
missing = set(feature_columns) - set(test.columns)
if missing:
    raise ValueError(f"Missing test features: {sorted(missing)}")

# Use ONLY normal training records to fit the model.
normal_train = train[
    (train["label"] == 0) &
    (train["attack_cat"].str.strip().str.lower() == "normal")
].copy()

if normal_train.empty:
    raise ValueError("No normal training records found.")

# Evaluation contains normal traffic and the held-out attack category.
test_categories = test["attack_cat"].astype(str).str.strip()
evaluation_mask = (
    test_categories.str.lower().eq("normal") |
    test_categories.str.lower().eq(HELD_OUT_CATEGORY.lower())
)
evaluation = test.loc[evaluation_mask].copy()

if evaluation.empty:
    raise ValueError(
        f"No Normal or {HELD_OUT_CATEGORY} records found in test data."
    )

# Ground truth: 0 = Normal, 1 = held-out attack.
actual = (
    evaluation["attack_cat"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq(HELD_OUT_CATEGORY.lower())
    .astype(int)
)

# Identify categorical and numeric columns.
categorical_columns = [
    c for c in ["proto", "service", "state"]
    if c in feature_columns
]
numeric_columns = [
    c for c in feature_columns
    if c not in categorical_columns
]

# Fit preprocessing ONLY on normal training records.
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_columns),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
    ],
    remainder="drop",
)

X_train = preprocessor.fit_transform(normal_train[feature_columns])
X_test = preprocessor.transform(evaluation[feature_columns])

# Train a fresh model; do not load or overwrite the production model.
model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train)

# Isolation Forest: -1 = anomaly, +1 = inlier.
raw_predictions = model.predict(X_test)
predicted = (raw_predictions == -1).astype(int)

# Higher score means more anomalous in this report.
anomaly_scores = -model.decision_function(X_test)

tn, fp, fn, tp = confusion_matrix(
    actual, predicted, labels=[0, 1]
).ravel()

precision = precision_score(
    actual, predicted, zero_division=0
)
recall = recall_score(
    actual, predicted, zero_division=0
)
f1 = f1_score(
    actual, predicted, zero_division=0
)
false_positive_rate = (
    fp / (fp + tn) if (fp + tn) else 0.0
)

results = evaluation[["attack_cat"]].copy()
results["actual_unknown_attack"] = actual.to_numpy()
results["prediction"] = predicted
results["status"] = [
    "Suspicious" if value == 1 else "Normal"
    for value in predicted
]
results["anomaly_score"] = anomaly_scores
results.to_csv(
    "sentinelai_unknown_worms_results.csv",
    index=False,
)

metrics = {
    "held_out_attack_category": HELD_OUT_CATEGORY,
    "training_method": "Isolation Forest trained on normal records only",
    "training_normal_records": int(len(normal_train)),
    "evaluation_normal_records": int((actual == 0).sum()),
    "evaluation_held_out_attack_records": int((actual == 1).sum()),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "precision": float(precision),
    "detection_rate_recall": float(recall),
    "f1_score": float(f1),
    "false_positive_rate": float(false_positive_rate),
    "threshold_tuned_on_held_out_attacks": False,
}

with open(
    "sentinelai_unknown_worms_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=2)

print("\n=== SENTINELAI UNKNOWN-ATTACK EXPERIMENT ===")
print(f"Held-out category: {HELD_OUT_CATEGORY}")
print(f"Normal training records: {len(normal_train)}")
print(f"Normal test records: {(actual == 0).sum()}")
print(f"Held-out attack test records: {(actual == 1).sum()}")

print("\nConfusion matrix (actual rows, predicted columns):")
print("                 Predicted Normal  Predicted Suspicious")
print(f"Actual Normal    {tn:16d}  {fp:20d}")
print(f"Actual Attack    {fn:16d}  {tp:20d}")

print("\nMetrics:")
print(f"Precision:          {precision:.4f}")
print(f"Detection rate:     {recall:.4f}")
print(f"F1-score:           {f1:.4f}")
print(f"False-positive rate:{false_positive_rate:.4f}")

print("\nSaved files:")
print("- sentinelai_unknown_worms_results.csv")
print("- sentinelai_unknown_worms_metrics.json")
print("\nThis experiment did not modify the API model or MySQL.")

