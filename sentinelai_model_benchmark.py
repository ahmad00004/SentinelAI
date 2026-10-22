
import json
import time
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ==================================================
# SentinelAI: Reproducible Isolation Forest Benchmark
# ==================================================

SEED = 42
TARGET_VALIDATION_FPR = 0.01

TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE = "UNSW_NB15_testing-set.csv"

RESULTS_FILE = "sentinelai_benchmark_predictions.csv"
METRICS_FILE = "sentinelai_benchmark_metrics.json"
CATEGORY_FILE = "sentinelai_benchmark_by_category.csv"

start_time = time.time()

print("=" * 55)
print("SENTINELAI MODEL BENCHMARK")
print("=" * 55)

print("\nLoading datasets...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

required = {"label", "attack_cat", "proto", "service", "state"}

for filename, frame in [
    (TRAIN_FILE, train_df),
    (TEST_FILE, test_df),
]:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {sorted(missing)}"
        )

# Identify normal and attack records.
normal_train = train_df[train_df["label"] == 0].copy()
attack_train = train_df[train_df["label"] == 1].copy()

normal_test = test_df[test_df["label"] == 0].copy()

print(f"Training file rows: {len(train_df)}")
print(f"Testing file rows: {len(test_df)}")
print(f"Normal training rows: {len(normal_train)}")
print(f"Attack training rows (validation only): {len(attack_train)}")
print(f"Normal test rows: {len(normal_test)}")

# Hold out normal records for validation.
fit_df, normal_val_df = train_test_split(
    normal_train,
    test_size=0.20,
    random_state=SEED,
)

# Model fitting uses normal records only.
# Attack records from the training file are used for validation,
# not for fitting the Isolation Forest.
validation_df = pd.concat(
    [normal_val_df, attack_train],
    ignore_index=True,
)

y_val = validation_df["label"].astype(int).to_numpy()

# Exclude labels, categories and identifiers from model features.
excluded = {"id", "label", "attack_cat"}
feature_columns = [
    col for col in train_df.columns if col not in excluded
]

categorical_columns = [
    col for col in ["proto", "service", "state"]
    if col in feature_columns
]
numeric_columns = [
    col for col in feature_columns
    if col not in categorical_columns
]

print(f"Feature count: {len(feature_columns)}")
print(f"Numeric features: {len(numeric_columns)}")
print(f"Categorical features: {len(categorical_columns)}")

# Learn preprocessing from the fitting subset only.
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

print("\nPreparing features...")
X_fit = preprocessor.fit_transform(fit_df[feature_columns])
X_val = preprocessor.transform(validation_df[feature_columns])
X_test = preprocessor.transform(test_df[feature_columns])

# Fixed, documented model settings.
print("Training Isolation Forest...")
model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=SEED,
    n_jobs=-1,
)

model.fit(X_fit)

# Higher anomaly score means more anomalous.
# Isolation Forest decision_function is lower for anomalies.
val_scores = -model.decision_function(X_val)
test_scores = -model.decision_function(X_test)

# --------------------------------------------------
# Benchmark A: Isolation Forest's default predictions
# --------------------------------------------------

default_val_predictions = (
    model.predict(X_val) == -1
).astype(int)

default_test_predictions = (
    model.predict(X_test) == -1
).astype(int)


def calculate_metrics(y_true, y_pred, scores):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "false_positive_rate": float(
            fp / (fp + tn)
        ) if (fp + tn) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
    }


# --------------------------------------------------
# Benchmark B: Threshold chosen using validation data
# --------------------------------------------------

fpr_values, tpr_values, thresholds = roc_curve(
    y_val, val_scores
)

normal_validation_count = int((y_val == 0).sum())
attack_validation_count = int((y_val == 1).sum())

# Calculate precision and F1 for each ROC threshold.
tp_values = tpr_values * attack_validation_count
fp_values = fpr_values * normal_validation_count

precision_values = np.divide(
    tp_values,
    tp_values + fp_values,
    out=np.zeros_like(tp_values),
    where=(tp_values + fp_values) != 0,
)

f1_values = np.divide(
    2 * precision_values * tpr_values,
    precision_values + tpr_values,
    out=np.zeros_like(tpr_values),
    where=(precision_values + tpr_values) != 0,
)

eligible = np.flatnonzero(
    fpr_values <= TARGET_VALIDATION_FPR
)

if len(eligible) == 0:
    raise RuntimeError(
        "No validation threshold satisfies the false-positive target."
    )

best_index = max(
    eligible,
    key=lambda i: (
        f1_values[i],
        precision_values[i],
        thresholds[i],
    ),
)

selected_threshold = float(thresholds[best_index])

calibrated_val_predictions = (
    val_scores >= selected_threshold
).astype(int)

calibrated_test_predictions = (
    test_scores >= selected_threshold
).astype(int)

# Test labels are used only for evaluation, never threshold selection.
y_test = test_df["label"].astype(int).to_numpy()

default_test_metrics = calculate_metrics(
    y_test, default_test_predictions, test_scores
)

calibrated_test_metrics = calculate_metrics(
    y_test, calibrated_test_predictions, test_scores
)

validation_metrics = calculate_metrics(
    y_val, calibrated_val_predictions, val_scores
)

# --------------------------------------------------
# Per-attack-category results for calibrated model
# --------------------------------------------------

category_results = []

for category, group in test_df.groupby(
    "attack_cat", dropna=False
):
    indexes = group.index.to_numpy()
    category_y = group["label"].astype(int).to_numpy()
    category_predictions = calibrated_test_predictions[indexes]

    category_results.append({
        "attack_category": str(category),
        "records": int(len(group)),
        "actual_attacks": int((category_y == 1).sum()),
        "detected_attacks": int(
            ((category_y == 1) & (category_predictions == 1)).sum()
        ),
        "missed_attacks": int(
            ((category_y == 1) & (category_predictions == 0)).sum()
        ),
        "false_alarms": int(
            ((category_y == 0) & (category_predictions == 1)).sum()
        ),
        "attack_recall": float(
            recall_score(
                category_y, category_predictions, zero_division=0
            )
        ) if (category_y == 1).any() else None,
    })

category_df = pd.DataFrame(category_results)
category_df.to_csv(CATEGORY_FILE, index=False)

# --------------------------------------------------
# Save individual predictions
# --------------------------------------------------

predictions_df = pd.DataFrame({
    "true_label": y_test,
    "attack_category": test_df["attack_cat"].astype(str).to_numpy(),
    "anomaly_score": test_scores,
    "default_prediction": default_test_predictions,
    "calibrated_prediction": calibrated_test_predictions,
    "default_status": np.where(
        default_test_predictions == 1, "Suspicious", "Normal"
    ),
    "calibrated_status": np.where(
        calibrated_test_predictions == 1, "Suspicious", "Normal"
    ),
})

predictions_df.to_csv(RESULTS_FILE, index=False)

metrics = {
    "benchmark": "SentinelAI Isolation Forest",
    "random_seed": SEED,
    "n_estimators": 200,
    "contamination": "auto",
    "fit_data": "Normal training records only",
    "normal_fit_records": int(len(fit_df)),
    "normal_validation_records": int(len(normal_val_df)),
    "attack_validation_records": int(len(attack_train)),
    "test_records": int(len(test_df)),
    "feature_count": int(len(feature_columns)),
    "target_validation_fpr": TARGET_VALIDATION_FPR,
    "selected_threshold": selected_threshold,
    "validation_calibrated_metrics": validation_metrics,
    "test_default_metrics": default_test_metrics,
    "test_calibrated_metrics": calibrated_test_metrics,
    "threshold_selected_using_test_labels": False,
    "test_set_caveat": (
        "The official test set was examined in previous experiments. "
        "These results are reproducible but exploratory, not a pristine "
        "independent final evaluation."
    ),
    "elapsed_seconds": round(time.time() - start_time, 2),
    "output_files": [
        RESULTS_FILE,
        METRICS_FILE,
        CATEGORY_FILE,
    ],
}

with open(METRICS_FILE, "w") as file:
    json.dump(metrics, file, indent=4)

# --------------------------------------------------
# Print summary
# --------------------------------------------------

def print_metrics(title, values):
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"TN={values['TN']:,} | FP={values['FP']:,} | "
        f"FN={values['FN']:,} | TP={values['TP']:,}"
    )
    print(f"Precision:           {values['precision']:.4f}")
    print(f"Recall:              {values['recall']:.4f}")
    print(f"F1-score:            {values['f1']:.4f}")
    print(
        f"False-positive rate: "
        f"{values['false_positive_rate']:.4f}"
    )
    print(f"ROC-AUC:             {values['roc_auc']:.4f}")
    print(
        f"Average precision:   {values['average_precision']:.4f}"
    )


print_metrics(
    "TEST A: DEFAULT ISOLATION FOREST",
    default_test_metrics,
)

print(f"\nValidation-selected threshold: {selected_threshold:.8f}")
print(f"Validation FPR: {validation_metrics['false_positive_rate']:.4f}")

print_metrics(
    "TEST B: VALIDATION-SELECTED THRESHOLD",
    calibrated_test_metrics,
)

print("\nATTACK CATEGORY RESULTS")
print(category_df.to_string(index=False))

print("\nSaved benchmark files:")
print(f"- {RESULTS_FILE}")
print(f"- {METRICS_FILE}")
print(f"- {CATEGORY_FILE}")
print(f"\nElapsed time: {metrics['elapsed_seconds']} seconds")
print("\nBenchmark completed.")

