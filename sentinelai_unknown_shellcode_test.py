import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE = "UNSW_NB15_testing-set.csv"
HELD_OUT_CATEGORY = "Shellcode"
RANDOM_STATE = 42

# All output filenames are unique to this experiment.
RESULTS_FILE = "sentinelai_unknown_shellcode_results.csv"
METRICS_FILE = "sentinelai_unknown_shellcode_metrics.json"
CHART_FILE = "sentinelai_shellcode_score_distribution.png"

# 1. Load the original datasets.
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

required_columns = {"label", "attack_cat"}
if not required_columns.issubset(train.columns):
    raise ValueError("Training data is missing label or attack_cat.")
if not required_columns.issubset(test.columns):
    raise ValueError("Testing data is missing label or attack_cat.")

# Exclude identifiers and all target/label columns from model inputs.
excluded = {"id", "label", "attack_cat"}
feature_columns = [c for c in train.columns if c not in excluded]

missing = set(feature_columns) - set(test.columns)
if missing:
    raise ValueError(f"Missing test features: {sorted(missing)}")

# Verify this category exists in the original test set.
train_categories = train["attack_cat"].astype(str).str.strip().str.lower()
test_categories = test["attack_cat"].astype(str).str.strip().str.lower()

if not test_categories.eq(HELD_OUT_CATEGORY.lower()).any():
    raise ValueError(
        f"No {HELD_OUT_CATEGORY} records found in test data."
    )

# 2. Fit using NORMAL training traffic only.
normal_train = train.loc[
    (train["label"] == 0) &
    train_categories.eq("normal")
].copy()

if normal_train.empty:
    raise ValueError("No normal training records found.")

# 3. Evaluation contains only normal traffic and held-out Shellcode.
evaluation_mask = (
    test_categories.eq("normal") |
    test_categories.eq(HELD_OUT_CATEGORY.lower())
)
evaluation = test.loc[evaluation_mask].copy()

evaluation_categories = (
    evaluation["attack_cat"].astype(str).str.strip().str.lower()
)

# Ground truth: 0 = Normal, 1 = Shellcode.
actual = evaluation_categories.eq(
    HELD_OUT_CATEGORY.lower()
).astype(int)

if actual.nunique() != 2:
    raise ValueError("Evaluation must contain both Normal and Shellcode.")

# 4. Define preprocessing using training feature names.
categorical_columns = [
    c for c in ["proto", "service", "state"]
    if c in feature_columns
]
numeric_columns = [
    c for c in feature_columns
    if c not in categorical_columns
]

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

# IMPORTANT: fit preprocessing ONLY on normal training records.
X_train = preprocessor.fit_transform(
    normal_train[feature_columns]
)

# Transform the held-out test records without fitting on them.
X_test = preprocessor.transform(
    evaluation[feature_columns]
)

# 5. Fit a fresh model. Do not load or overwrite the production model.
model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train)

# Higher score means more anomalous.
raw_predictions = model.predict(X_test)
predicted = (raw_predictions == -1).astype(int)
scores = -model.decision_function(X_test)

# 6. Evaluate with the model's unchanged default threshold.
tn, fp, fn, tp = confusion_matrix(
    actual, predicted, labels=[0, 1]
).ravel()

precision = precision_score(actual, predicted, zero_division=0)
recall = recall_score(actual, predicted, zero_division=0)
f1 = f1_score(actual, predicted, zero_division=0)
fpr = fp / (fp + tn) if (fp + tn) else 0.0

roc_auc = roc_auc_score(actual, scores)
avg_precision = average_precision_score(actual, scores)

# 7. Save per-record predictions.
results = evaluation[["attack_cat"]].copy()
results["actual_unknown_attack"] = actual.to_numpy()
results["prediction"] = predicted
results["status"] = [
    "Suspicious" if p == 1 else "Normal"
    for p in predicted
]
results["anomaly_score"] = scores
results.to_csv(RESULTS_FILE, index=False)

# 8. Save metrics for the project report.
metrics = {
    "held_out_attack_category": HELD_OUT_CATEGORY,
    "training_method": "Isolation Forest fitted on normal training records only",
    "preprocessing_fit_on": "Normal training records only",
    "threshold": "Isolation Forest default; not tuned on held-out test labels",
    "random_state": RANDOM_STATE,
    "training_normal_records": int(len(normal_train)),
    "evaluation_normal_records": int((actual == 0).sum()),
    "evaluation_held_out_attack_records": int((actual == 1).sum()),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "precision_at_default_threshold": float(precision),
    "detection_rate_recall": float(recall),
    "f1_score": float(f1),
    "false_positive_rate": float(fpr),
    "roc_auc": float(roc_auc),
    "average_precision": float(avg_precision),
    "attack_prevalence": float(actual.mean()),
    "threshold_tuned_on_held_out_attacks": False,
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

# 9. Save a density chart. This does not change the evaluation.
plt.figure(figsize=(10, 6))
plt.hist(
    results.loc[results["attack_cat"].str.strip().str.lower().eq("normal"),
                "anomaly_score"],
    bins=80,
    density=True,
    alpha=0.6,
    label="Normal",
)
plt.hist(
    results.loc[
        results["attack_cat"].str.strip().str.lower().eq(
            HELD_OUT_CATEGORY.lower()
        ),
        "anomaly_score",
    ],
    bins=30,
    density=True,
    alpha=0.6,
    label=f"Unseen {HELD_OUT_CATEGORY}",
)
plt.xlabel("Anomaly score (higher = more anomalous)")
plt.ylabel("Density")
plt.title("SentinelAI: Normal vs Unseen Shellcode Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(CHART_FILE, dpi=160)
plt.close()

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
print(f"False-positive rate:{fpr:.4f}")
print(f"ROC-AUC:            {roc_auc:.4f}")
print(f"Average precision:  {avg_precision:.4f}")
print(f"Attack prevalence:  {actual.mean():.4f}")

print("\nSaved files:")
print(f"- {RESULTS_FILE}")
print(f"- {METRICS_FILE}")
print(f"- {CHART_FILE}")
print("\nProduction API, production model, and MySQL were not modified.")
