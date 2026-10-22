
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE = "UNSW_NB15_testing-set.csv"
CATEGORY = "Shellcode"
RANDOM_STATE = 42
MAX_VALIDATION_FPR = 0.01

PREFIX = "sentinelai_shellcode_candidate"

# ---------- 1. Load datasets ----------
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

for name, df in [("training", train), ("testing", test)]:
    if not {"label", "attack_cat"}.issubset(df.columns):
        raise ValueError(f"{name} dataset lacks label or attack_cat.")

train_cat = train["attack_cat"].astype(str).str.strip().str.lower()
test_cat = test["attack_cat"].astype(str).str.strip().str.lower()

excluded = {"id", "label", "attack_cat"}
features = [c for c in train.columns if c not in excluded]

missing = set(features) - set(test.columns)
if missing:
    raise ValueError(f"Test data missing features: {sorted(missing)}")

# ---------- 2. Create a validation split ----------
# Model fitting uses NORMAL records only.
normal = train.loc[
    (train["label"] == 0) & train_cat.eq("normal")
].copy()

# Attack examples are validation-only; never fit the model on them.
shellcode_train = train.loc[
    train_cat.eq(CATEGORY.lower())
].copy()

if len(normal) < 2 or shellcode_train.empty:
    raise ValueError("Insufficient normal or Shellcode training records.")

normal_fit, normal_val = train_test_split(
    normal,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

# Use Shellcode from the original training file for validation.
# This does NOT mean these attack records are used to fit the model.
validation = pd.concat(
    [normal_val, shellcode_train],
    ignore_index=True,
)

y_val = (
    validation["attack_cat"]
    .astype(str).str.strip().str.lower()
    .eq(CATEGORY.lower())
    .astype(int)
    .to_numpy()
)

if len(np.unique(y_val)) != 2:
    raise ValueError("Validation must contain normal and Shellcode records.")

# ---------- 3. Fit preprocessing on normal fitting data only ----------
categorical = [
    c for c in ["proto", "service", "state"] if c in features
]
numeric = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ],
    remainder="drop",
)

X_fit = preprocessor.fit_transform(normal_fit[features])
X_val = preprocessor.transform(validation[features])

# ---------- 4. Fit a separate candidate model ----------
model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_fit)

# Base score: higher means more anomalous according to Isolation Forest.
val_base_score = -model.decision_function(X_val)

# ---------- 5. Select direction using validation only ----------
# Both choices are evaluated on validation labels, not test labels.
directions = {
    "normal_direction": val_base_score,
    "inverted_direction": -val_base_score,
}

direction_auc = {
    name: float(roc_auc_score(y_val, score))
    for name, score in directions.items()
}

selected_direction = max(direction_auc, key=direction_auc.get)
val_score = directions[selected_direction]

# ---------- 6. Choose threshold on validation only ----------
# Predict positive when score >= threshold.
# Among thresholds meeting the FPR constraint, choose maximum F1.
# If tied, prefer higher precision, then higher threshold.
unique_scores = np.unique(val_score)
threshold_candidates = np.concatenate([
    [np.nextafter(unique_scores.max(), np.inf)],
    unique_scores,
])

best = None

for threshold in threshold_candidates:
    pred = (val_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_val, pred, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = precision_score(y_val, pred, zero_division=0)
    recall = recall_score(y_val, pred, zero_division=0)
    f1 = f1_score(y_val, pred, zero_division=0)

    if fpr <= MAX_VALIDATION_FPR:
        candidate = (f1, precision, threshold, fpr, recall)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

if best is None:
    raise RuntimeError(
        "Could not find a threshold satisfying the validation FPR limit."
    )

_, val_precision, selected_threshold, val_fpr, val_recall = best

# ---------- 7. Freeze direction and threshold; evaluate test once ----------
# The official test file is used only after validation choices are frozen.
test_eval_mask = (
    test_cat.eq("normal") | test_cat.eq(CATEGORY.lower())
)
evaluation = test.loc[test_eval_mask].copy()

test_categories = (
    evaluation["attack_cat"].astype(str).str.strip().str.lower()
)
y_test = test_categories.eq(CATEGORY.lower()).astype(int).to_numpy()

if len(np.unique(y_test)) != 2:
    raise ValueError("Test evaluation must contain normal and Shellcode.")

X_test = preprocessor.transform(evaluation[features])
test_base_score = -model.decision_function(X_test)

if selected_direction == "inverted_direction":
    test_score = -test_base_score
else:
    test_score = test_base_score

# No threshold tuning or direction selection on test labels.
y_pred = (test_score >= selected_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

test_precision = precision_score(y_test, y_pred, zero_division=0)
test_recall = recall_score(y_test, y_pred, zero_division=0)
test_f1 = f1_score(y_test, y_pred, zero_division=0)
test_fpr = fp / (fp + tn) if (fp + tn) else 0.0
test_auc = roc_auc_score(y_test, test_score)
test_ap = average_precision_score(y_test, test_score)

# ---------- 8. Save independent outputs ----------
results = evaluation[["attack_cat"]].copy()
results["actual_shellcode"] = y_test
results["predicted_attack"] = y_pred
results["status"] = np.where(y_pred == 1, "Suspicious", "Normal")
results["candidate_score"] = test_score
results.to_csv(f"{PREFIX}_results.csv", index=False)

metrics = {
    "experiment": "Validation-selected Shellcode score direction and threshold",
    "held_out_category": CATEGORY,
    "training_file": TRAIN_FILE,
    "test_file": TEST_FILE,
    "normal_model_fit_records": int(len(normal_fit)),
    "normal_validation_records": int(len(normal_val)),
    "shellcode_validation_records": int(len(shellcode_train)),
    "validation_auc_normal_direction": direction_auc["normal_direction"],
    "validation_auc_inverted_direction": direction_auc["inverted_direction"],
    "selected_direction": selected_direction,
    "selected_threshold": float(selected_threshold),
    "validation_fpr": float(val_fpr),
    "validation_recall": float(val_recall),
    "validation_precision": float(val_precision),
    "validation_fpr_target": MAX_VALIDATION_FPR,
    "test_normal_records": int((y_test == 0).sum()),
    "test_shellcode_records": int((y_test == 1).sum()),
    "test_true_negatives": int(tn),
    "test_false_positives": int(fp),
    "test_false_negatives": int(fn),
    "test_true_positives": int(tp),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "test_f1": float(test_f1),
    "test_false_positive_rate": float(test_fpr),
    "test_roc_auc": float(test_auc),
    "test_average_precision": float(test_ap),
    "threshold_selected_using_test_labels": False,
    "direction_selected_using_test_labels": False,
    "important_note": (
        "The official test set was previously inspected in earlier "
        "experiments, so this is not a pristine first-use test evaluation."
    ),
}

with open(f"{PREFIX}_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# Score distribution after validation-selected direction.
plt.figure(figsize=(10, 6))
plt.hist(
    test_score[y_test == 0],
    bins=80,
    density=True,
    alpha=0.6,
    label="Normal",
)
plt.hist(
    test_score[y_test == 1],
    bins=40,
    density=True,
    alpha=0.6,
    label="Shellcode",
)
plt.axvline(
    selected_threshold,
    linestyle="--",
    label="Validation-selected threshold",
)
plt.xlabel("Candidate score (higher = more suspicious)")
plt.ylabel("Density")
plt.title("SentinelAI validation-selected Shellcode candidate")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PREFIX}_score_distribution.png", dpi=160)
plt.close()

print("\n=== VALIDATION-SELECTED SHELLCODE CANDIDATE ===")
print("Normal records used for fitting:", len(normal_fit))
print("Normal records used for validation:", len(normal_val))
print("Shellcode records used for validation:", len(shellcode_train))
print("\nValidation AUC by direction:")
for name, auc in direction_auc.items():
    print(f"{name}: {auc:.4f}")

print("\nFrozen direction:", selected_direction)
print("Frozen threshold:", f"{selected_threshold:.8f}")
print("Validation FPR:", f"{val_fpr:.4f}")

print("\nFinal test confusion matrix:")
print("                 Predicted Normal  Predicted Suspicious")
print(f"Actual Normal    {tn:16d}  {fp:20d}")
print(f"Actual Shellcode {fn:16d}  {tp:20d}")

print("\nFinal test metrics:")
print(f"Precision:          {test_precision:.4f}")
print(f"Recall:             {test_recall:.4f}")
print(f"F1-score:           {test_f1:.4f}")
print(f"False-positive rate:{test_fpr:.4f}")
print(f"ROC-AUC:            {test_auc:.4f}")
print(f"Average precision:  {test_ap:.4f}")

print("\nSaved:")
print(f"- {PREFIX}_results.csv")
print(f"- {PREFIX}_metrics.json")
print(f"- {PREFIX}_score_distribution.png")
print("\nProduction model, API, and MySQL were not modified.")

