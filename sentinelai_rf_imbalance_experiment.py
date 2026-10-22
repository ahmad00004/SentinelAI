"""Validation-only experiments for the Random Forest's Fuzzer/Shellcode misses.

This script reads only UNSW_NB15_training-set.csv. It reproduces the existing
80/20 stratified split (seed 42), splits the 80% fit portion again so model
weights and alert thresholds are calibrated without using the held-out
validation portion, and never reads or writes the official test set.

It does not replace the production Isolation Forest or the current RF artifact.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
CALIBRATION_SEED = 43
CALIBRATION_FPR_TARGETS = [0.01, 0.0075, 0.005]
TRAIN_FILE = Path("UNSW_NB15_training-set.csv")
METRICS_FILE = Path("sentinelai_rf_imbalance_experiment_metrics.csv")
CATEGORY_FILE = Path("sentinelai_rf_imbalance_experiment_by_category.csv")


def make_model(numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ))

    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def suspicion_scores(model, X):
    classes = list(model.named_steps["classifier"].classes_)
    normal_index = classes.index("Normal")
    return 1.0 - model.predict_proba(X)[:, normal_index]


def choose_threshold(y_true, scores, target_fpr):
    y_binary = (np.asarray(y_true) != "Normal").astype(int)
    fpr, tpr, thresholds = roc_curve(y_binary, scores)
    negatives = int((y_binary == 0).sum())
    positives = int((y_binary == 1).sum())
    valid = np.flatnonzero((fpr <= target_fpr) & np.isfinite(thresholds))
    if not len(valid):
        raise RuntimeError(
            f"No finite calibration threshold meets FPR target {target_fpr:.3%}."
        )

    precision = (
        tpr[valid] * positives
        / np.maximum(tpr[valid] * positives + fpr[valid] * negatives, 1e-12)
    )
    f1 = 2 * precision * tpr[valid] / np.maximum(precision + tpr[valid], 1e-12)
    best = max(
        range(len(valid)),
        key=lambda i: (f1[i], precision[i], tpr[valid[i]]),
    )
    return float(thresholds[valid[best]])


def evaluate(y_true, scores, threshold, target_fpr, experiment, fit_multiplier,
             calib_n, eval_n):
    y_binary = (np.asarray(y_true) != "Normal").astype(int)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_binary, predicted, labels=[0, 1]).ravel()
    row = {
        "experiment": experiment,
        "fuzzer_sample_weight": fit_multiplier,
        "calibration_fpr_target": target_fpr,
        "threshold": threshold,
        "calibration_records": calib_n,
        "validation_records": eval_n,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "precision": float(precision_score(y_binary, predicted, zero_division=0)),
        "recall": float(recall_score(y_binary, predicted, zero_division=0)),
        "f1": float(f1_score(y_binary, predicted, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn) if fp + tn else 0),
    }
    return row


def main():
    started = time.time()
    print("Loading the training CSV only; the official test CSV is not opened.")
    df = pd.read_csv(TRAIN_FILE, low_memory=False)
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()

    excluded = {"id", "label", "attack_cat"}
    feature_columns = [column for column in df.columns if column not in excluded]
    X_all = df[feature_columns].copy()
    y_all = df["attack_cat"].copy()

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_fit, y_fit, test_size=0.20, random_state=CALIBRATION_SEED,
        stratify=y_fit,
    )

    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in feature_columns if column not in numeric]
    print(
        f"Fit subset: {len(X_train):,}; calibration: {len(X_cal):,}; "
        f"held-out validation: {len(X_val):,}; features: {len(feature_columns)}"
    )

    # Sample weights multiply the existing balanced_subsample weights. This
    # isolates the effect of giving Fuzzer examples additional influence.
    configurations = [
        ("balanced_baseline", 1.0),
        ("fuzzer_weight_x2", 2.0),
        ("fuzzer_weight_x4", 4.0),
        ("fuzzer_weight_x8", 8.0),
    ]
    metric_rows = []
    category_rows = []

    for name, fuzzer_weight in configurations:
        print(f"\nTraining {name}...")
        model = make_model(numeric, categorical)
        sample_weight = np.where(y_train.to_numpy() == "Fuzzers", fuzzer_weight, 1.0)
        model.fit(
            X_train,
            y_train,
            classifier__sample_weight=sample_weight,
        )

        calibration_scores = suspicion_scores(model, X_cal)
        validation_scores = suspicion_scores(model, X_val)
        for target_fpr in CALIBRATION_FPR_TARGETS:
            threshold = choose_threshold(
                y_cal.to_numpy(), calibration_scores, target_fpr
            )
            metrics = evaluate(
                y_val.to_numpy(), validation_scores, threshold, target_fpr,
                name, fuzzer_weight, len(X_cal), len(X_val),
            )
            metric_rows.append(metrics)
            print(
                f"Calibration FPR target={target_fpr:.2%}; "
                f"threshold={threshold:.6f}; "
                f"validation FPR={metrics['false_positive_rate']:.3%}; "
                f"attack recall={metrics['recall']:.3%}; F1={metrics['f1']:.4f}"
            )

            validation_predicted = validation_scores >= threshold
            for category in sorted(pd.unique(y_val)):
                mask = (y_val.to_numpy() == category)
                count = int(mask.sum())
                detected = int(validation_predicted[mask].sum())
                category_rows.append({
                    "experiment": name,
                    "calibration_fpr_target": target_fpr,
                    "category": category,
                    "records": count,
                    "detected": detected,
                    "missed": count - detected,
                    "recall": float(detected / count) if count else None,
                })

    metrics_df = pd.DataFrame(metric_rows)
    categories_df = pd.DataFrame(category_rows)
    metrics_df.to_csv(METRICS_FILE, index=False)
    categories_df.to_csv(CATEGORY_FILE, index=False)

    print("\n=== VALIDATION COMPARISON ===")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== FUZZERS AND SHELLCODE RECALL ===")
    focus = categories_df[categories_df["category"].isin(["Fuzzers", "Shellcode"])]
    print(focus.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {METRICS_FILE} and {CATEGORY_FILE}")
    print("No production model was overwritten; the official test CSV was not used.")
    print(f"Elapsed: {time.time() - started:.1f} seconds")


if __name__ == "__main__":
    main()
