import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

MODEL_FILE = "sentinelai_supervised_rf.joblib"
TRAIN_FILE = "UNSW_NB15_training-set.csv"

artifact = joblib.load(MODEL_FILE)
model = artifact["model"]
threshold = float(artifact["threshold"])
feature_columns = artifact["feature_columns"]

print("Loading training data and reproducing the original validation split...")
df = pd.read_csv(TRAIN_FILE, low_memory=False)
df["attack_cat"] = df["attack_cat"].astype(str).str.strip()

X_all = df[feature_columns].copy()
y_all = df["attack_cat"].copy()

X_fit, X_val, y_fit, y_val = train_test_split(
    X_all,
    y_all,
    test_size=0.20,
    random_state=42,
    stratify=y_all,
)

# Use the saved model; do not retrain.
classes = list(model.named_steps["classifier"].classes_)
normal_index = classes.index("Normal")
attack_indices = [i for i, c in enumerate(classes) if c != "Normal"]

probabilities = model.predict_proba(X_val)
scores = 1.0 - probabilities[:, normal_index]

actual_normal = (y_val.to_numpy() == "Normal")
predicted_suspicious = scores >= threshold

false_positive_mask = actual_normal & predicted_suspicious
false_negative_mask = (~actual_normal) & (~predicted_suspicious)

# Most likely attack category, even for a missed attack.
attack_probabilities = probabilities[:, attack_indices]
best_attack_indices = np.argmax(attack_probabilities, axis=1)
best_attack_categories = [
    classes[attack_indices[i]] for i in best_attack_indices
]

report = X_val.reset_index(drop=True).copy()
report.insert(0, "source_index", X_val.index.to_numpy())
report.insert(1, "actual_category", y_val.to_numpy())
report["suspicion_score"] = scores
report["threshold"] = threshold
report["predicted_status"] = np.where(
    predicted_suspicious, "Suspicious", "Normal"
)
report["most_likely_attack_category"] = best_attack_categories
report["error_type"] = np.select(
    [false_positive_mask, false_negative_mask],
    ["False Positive", "False Negative"],
    default="Correct",
)

false_positives = report[false_positive_mask].copy()
false_negatives = report[false_negative_mask].copy()

report.to_csv("sentinelai_validation_all_predictions.csv", index=False)
false_positives.to_csv("sentinelai_validation_false_positives.csv", index=False)
false_negatives.to_csv("sentinelai_validation_false_negatives.csv", index=False)

tn = int((actual_normal & ~predicted_suspicious).sum())
fp = int(false_positive_mask.sum())
fn = int(false_negative_mask.sum())
tp = int((~actual_normal & predicted_suspicious).sum())

print("\n=== VALIDATION RESULTS ===")
print(f"Records: {len(report)}")
print(f"Threshold: {threshold:.6f}")
print(f"True negatives: {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives: {tp}")
print(f"Validation false-positive rate: {fp / max(tn + fp, 1):.4%}")
print(f"Validation attack recall: {tp / max(tp + fn, 1):.4%}")

print("\n=== FALSE POSITIVES BY PROTOCOL ===")
if fp:
    print(false_positives["proto"].value_counts(dropna=False).to_string())
else:
    print("No false positives found.")

print("\n=== FALSE NEGATIVES BY ACTUAL ATTACK CATEGORY ===")
if fn:
    print(false_negatives["actual_category"].value_counts().to_string())
else:
    print("No false negatives found.")

print("\nSaved reports:")
print("sentinelai_validation_all_predictions.csv")
print("sentinelai_validation_false_positives.csv")
print("sentinelai_validation_false_negatives.csv")
print("\nNo model was retrained and no threshold was changed.")
