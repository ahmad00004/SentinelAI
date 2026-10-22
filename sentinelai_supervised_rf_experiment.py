import json
import joblib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
TARGET_FPR = 0.01

TRAIN_FILE = Path("UNSW_NB15_training-set.csv")
TEST_FILE = Path("UNSW_NB15_testing-set.csv")

OUTPUT_PREDICTIONS = Path("sentinelai_supervised_rf_predictions.csv")
OUTPUT_METRICS = Path("sentinelai_supervised_rf_metrics.json")
OUTPUT_CATEGORIES = Path("sentinelai_supervised_rf_by_category.csv")


def binary_metrics(y_true, y_pred, scores):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = np.asarray(scores, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": float(precision_score(
            y_true, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, y_pred, zero_division=0
        )),
        "false_positive_rate": float(
            fp / (fp + tn) if fp + tn else 0
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
    }


def main():
    start = time.time()

    print("Loading UNSW-NB15 datasets...")
    train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
    test_df = pd.read_csv(TEST_FILE, low_memory=False)

    required = {"label", "attack_cat"}
    if not required.issubset(train_df.columns):
        raise ValueError("Training CSV must contain label and attack_cat.")
    if not required.issubset(test_df.columns):
        raise ValueError("Testing CSV must contain label and attack_cat.")

    train_df["attack_cat"] = train_df["attack_cat"].astype(str).str.strip()
    test_df["attack_cat"] = test_df["attack_cat"].astype(str).str.strip()

    # Remove identifiers and target columns to prevent target leakage.
    excluded = {"id", "label", "attack_cat"}
    feature_columns = [
        c for c in train_df.columns if c not in excluded
    ]

    missing = set(feature_columns) - set(test_df.columns)
    if missing:
        raise ValueError(f"Test CSV is missing columns: {sorted(missing)}")

    X_all = train_df[feature_columns].copy()
    y_all = train_df["attack_cat"].copy()

    X_test = test_df[feature_columns].copy()
    y_test_category = test_df["attack_cat"].copy()

    if "Normal" not in set(y_all):
        raise ValueError("Could not find the 'Normal' attack category.")

    # Hold out validation data for threshold selection.
    # The validation data is not used to fit the classifier.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.20,
        random_state=SEED,
        stratify=y_all,
    )

    numeric_features = X_fit.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_features = [
        c for c in feature_columns if c not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )
    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    # Balanced class weights give rare categories more influence.
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])

    print(f"Training records: {len(X_fit):,}")
    print(f"Validation records: {len(X_val):,}")
    print(f"Test records: {len(X_test):,}")
    print(f"Features: {len(feature_columns)}")
    print("Training class-weighted Random Forest...")

    model.fit(X_fit, y_fit)

    classes = list(model.named_steps["classifier"].classes_)
    if "Normal" not in classes:
        raise ValueError("'Normal' class is missing from model classes.")

    normal_index = classes.index("Normal")

    print("Scoring validation data...")
    val_probabilities = model.predict_proba(X_val)
    val_scores = 1.0 - val_probabilities[:, normal_index]
    y_val_binary = (y_val != "Normal").astype(int).to_numpy()

    # Select a threshold using validation data only.
    fpr, tpr, thresholds = roc_curve(y_val_binary, val_scores)
    negatives = int((y_val_binary == 0).sum())
    positives = int((y_val_binary == 1).sum())

    valid = (
        (fpr <= TARGET_FPR)
        & np.isfinite(thresholds)
    )
    candidates = np.flatnonzero(valid)

    if len(candidates) == 0:
        raise RuntimeError("No finite validation threshold meets FPR target.")

    candidate_precision = (
        tpr[candidates] * positives
        / np.maximum(
            tpr[candidates] * positives
            + fpr[candidates] * negatives,
            1e-12,
        )
    )
    candidate_f1 = (
        2 * candidate_precision * tpr[candidates]
        / np.maximum(candidate_precision + tpr[candidates], 1e-12)
    )

    best_position = max(
        range(len(candidates)),
        key=lambda i: (
            candidate_f1[i],
            candidate_precision[i],
            tpr[candidates[i]],
        ),
    )
    best_index = candidates[best_position]
    threshold = float(thresholds[best_index])

    # Save a separate experimental model; do not overwrite production.
    artifact = {
        "model": model,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "classes": classes,
        "normal_class": "Normal",
        "model_name": "SentinelAI Supervised Random Forest",
        "version": "experimental-1.0",
    }
    joblib.dump(artifact, "sentinelai_supervised_rf.joblib", compress=3)
    print("Saved separate model: sentinelai_supervised_rf.joblib")

    val_predictions = (val_scores >= threshold).astype(int)
    validation_metrics = binary_metrics(
        y_val_binary, val_predictions, val_scores
    )

    print(f"Validation-selected threshold: {threshold:.6f}")
    print(
        "Validation FPR: "
        f"{validation_metrics['false_positive_rate']:.4%}"
    )

    print("Scoring official test data...")
    test_probabilities = model.predict_proba(X_test)
    test_scores = 1.0 - test_probabilities[:, normal_index]

    # Default multiclass prediction.
    default_category = model.predict(X_test)
    default_binary = (default_category != "Normal").astype(int)

    # Threshold-calibrated suspicious/normal decision.
    calibrated_binary = (test_scores >= threshold).astype(int)

    # For events flagged suspicious, report the most likely attack class.
    non_normal_indices = [
        i for i, category in enumerate(classes)
        if category != "Normal"
    ]
    attack_class_probabilities = test_probabilities[:, non_normal_indices]
    attack_class_names = np.asarray(classes)[non_normal_indices]
    best_attack_indices = attack_class_probabilities.argmax(axis=1)
    calibrated_category = attack_class_names[best_attack_indices]
    calibrated_category = np.where(
        calibrated_binary == 1, calibrated_category, "Normal"
    )

    y_test_binary = (y_test_category != "Normal").astype(int).to_numpy()

    default_metrics = binary_metrics(
        y_test_binary, default_binary, test_scores
    )
    calibrated_metrics = binary_metrics(
        y_test_binary, calibrated_binary, test_scores
    )

    predictions = pd.DataFrame({
        "actual_category": y_test_category,
        "actual_label": y_test_binary,
        "default_predicted_category": default_category,
        "default_suspicious": default_binary,
        "suspicion_score": test_scores,
        "calibrated_predicted_category": calibrated_category,
        "calibrated_suspicious": calibrated_binary,
    })
    predictions.to_csv(OUTPUT_PREDICTIONS, index=False)

    # Recall by true category, including the three rare categories of interest.
    category_rows = []
    for category in sorted(y_test_category.unique()):
        mask = (y_test_category == category).to_numpy()
        count = int(mask.sum())
        is_attack = category != "Normal"

        category_rows.append({
            "category": category,
            "records": count,
            "default_detected": int(default_binary[mask].sum()),
            "default_recall": (
                float(default_binary[mask].mean()) if is_attack else None
            ),
            "calibrated_detected": int(calibrated_binary[mask].sum()),
            "calibrated_missed": (
                int(count - calibrated_binary[mask].sum())
                if is_attack else None
            ),
            "calibrated_recall": (
                float(calibrated_binary[mask].mean())
                if is_attack else None
            ),
        })

    category_df = pd.DataFrame(category_rows)
    category_df.to_csv(OUTPUT_CATEGORIES, index=False)

    metrics = {
        "experiment": "Class-weighted multiclass Random Forest",
        "random_seed": SEED,
        "target_validation_fpr": TARGET_FPR,
        "selected_threshold": threshold,
        "training_records_used_for_fit": int(len(X_fit)),
        "validation_records": int(len(X_val)),
        "test_records": int(len(X_test)),
        "features": feature_columns,
        "classes": classes,
        "validation_metrics_at_selected_threshold": validation_metrics,
        "test_default_argmax": default_metrics,
        "test_threshold_calibrated": calibrated_metrics,
        "caveat": (
            "This is an exploratory benchmark. The official test set has "
            "been examined in earlier experiments, so results are not a "
            "pristine independent final evaluation. The model learns known "
            "attack categories and does not prove detection of unseen attacks."
        ),
    }

    with OUTPUT_METRICS.open("w") as file:
        json.dump(metrics, file, indent=2)

    print("\n=== TEST RESULTS: DEFAULT MULTICLASS PREDICTION ===")
    print(json.dumps(default_metrics, indent=2))

    print("\n=== TEST RESULTS: VALIDATION-CALIBRATED THRESHOLD ===")
    print(json.dumps(calibrated_metrics, indent=2))

    print("\n=== RESULTS FOR RARE ATTACK CATEGORIES ===")
    selected = category_df[
        category_df["category"].isin(
            ["Worms", "Shellcode", "Backdoor", "Normal"]
        )
    ]
    print(selected.to_string(index=False))

    print("\nSaved experiment outputs:")
    print(f"  {OUTPUT_PREDICTIONS}")
    print(f"  {OUTPUT_METRICS}")
    print(f"  {OUTPUT_CATEGORIES}")
    print(f"Elapsed time: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
