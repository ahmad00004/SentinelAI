"""Train and evaluate weighted binary RF candidates on the canonical train split.

The CSV filenames supplied in this project folder have their canonical UNSW
partitions reversed: the 175,341-row training partition is in
UNSW_NB15_testing-set.csv. The 82,332-row canonical test partition is not
opened. Thresholds are selected on a calibration slice, then measured on a
separate validation slice. The selected weighted RF artifact is also loaded by
the API, so running this script intentionally updates the active RF candidate.
"""

import time
from pathlib import Path

import joblib
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
FPR_TARGETS = [0.0075, 0.005]
# In the provided project folder, these two CSV filenames contain the
# opposite canonical UNSW-NB15 partitions. This file has 175,341 rows and is
# the actual training partition; do not train from the smaller 82,332-row file.
TRAIN_FILE = Path("UNSW_NB15_testing-set.csv")
METRICS_FILE = Path("sentinelai_rf_binary_attack_experiment_metrics.csv")
CATEGORY_FILE = Path("sentinelai_rf_binary_attack_experiment_by_category.csv")
CANDIDATE_FILE = Path("sentinelai_supervised_rf_binary_weighted_candidate.joblib")


def make_model(numeric, categorical):
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
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


def choose_threshold(y_binary, scores, target_fpr):
    fpr, tpr, thresholds = roc_curve(y_binary, scores)
    negatives = int((y_binary == 0).sum())
    positives = int((y_binary == 1).sum())
    valid = np.flatnonzero((fpr <= target_fpr) & np.isfinite(thresholds))
    if not len(valid):
        raise RuntimeError(f"No calibration threshold meets {target_fpr:.3%} FPR.")
    precision = tpr[valid] * positives / np.maximum(
        tpr[valid] * positives + fpr[valid] * negatives, 1e-12
    )
    f1 = 2 * precision * tpr[valid] / np.maximum(precision + tpr[valid], 1e-12)
    best = max(range(len(valid)), key=lambda i: (f1[i], precision[i], tpr[valid[i]]))
    return float(thresholds[valid[best]])


def main():
    started = time.time()
    print(
        "Loading the canonical 175,341-row UNSW-NB15 training partition "
        "from the project file; the 82,332-row holdout is not opened."
    )
    df = pd.read_csv(TRAIN_FILE, low_memory=False)
    if len(df) != 175_341:
        raise ValueError(
            f"Expected 175,341 canonical training rows in {TRAIN_FILE}; "
            f"found {len(df):,}. Verify the dataset files before training."
        )
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    features = [c for c in df.columns if c not in {"id", "label", "attack_cat"}]
    X, categories = df[features].copy(), df["attack_cat"].copy()
    y = (categories != "Normal").astype(int)

    X_fit, X_val, y_fit, y_val, cat_fit, cat_val = train_test_split(
        X, y, categories, test_size=0.20, random_state=SEED, stratify=categories
    )
    X_train, X_cal, y_train, y_cal, _, cat_cal = train_test_split(
        X_fit, y_fit, cat_fit, test_size=0.20, random_state=CALIBRATION_SEED,
        stratify=cat_fit,
    )
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in features if c not in numeric]

    configurations = [
        ("binary_baseline", 1.0, 1.0),
        ("binary_fuzzer_x4", 4.0, 1.0),
        ("binary_fuzzer_shellcode_x4", 4.0, 4.0),
        ("binary_fuzzer_shellcode_x8", 8.0, 8.0),
        ("binary_fuzzer_x16_shellcode_x8", 16.0, 8.0),
    ]
    metric_rows, category_rows = [], []

    print(
        f"Fit={len(X_train):,}; calibration={len(X_cal):,}; "
        f"validation={len(X_val):,}; features={len(features)}"
    )
    for name, fuzzer_weight, shellcode_weight in configurations:
        print(f"\nTraining {name}...")
        model = make_model(numeric, categorical)
        sample_weight = np.ones(len(y_train), dtype=float)
        sample_weight[cat_fit.loc[X_train.index].to_numpy() == "Fuzzers"] *= fuzzer_weight
        sample_weight[cat_fit.loc[X_train.index].to_numpy() == "Shellcode"] *= shellcode_weight
        model.fit(X_train, y_train, classifier__sample_weight=sample_weight)
        positive_index = list(model.named_steps["classifier"].classes_).index(1)
        cal_scores = model.predict_proba(X_cal)[:, positive_index]
        val_scores = model.predict_proba(X_val)[:, positive_index]

        for target in FPR_TARGETS:
            threshold = choose_threshold(y_cal.to_numpy(), cal_scores, target)
            if name == "binary_fuzzer_shellcode_x4" and target == 0.0075:
                joblib.dump({
                    "model": model,
                    "threshold": threshold,
                    "feature_columns": features,
                    "classes": [0, 1],
                    "normal_class": 0,
                    "positive_class": 1,
                    "model_name": "SentinelAI Binary RF Weighted Candidate",
                    "version": "experimental-fuzzer-shellcode-weighted-1.0",
                    "training_source_file": str(TRAIN_FILE),
                    "training_record_count": int(len(df)),
                    "training_split_seed": SEED,
                    "calibration_split_seed": CALIBRATION_SEED,
                    "calibration_fpr_target": target,
                    "class_sample_weights": {
                        "Fuzzers": fuzzer_weight,
                        "Shellcode": shellcode_weight,
                    },
                    "warning": (
                        "SentinelAI API primary classifier for the educational "
                        "prototype. Metrics come from a held-out UNSW-NB15 "
                        "validation split, not live-network validation."
                    ),
                }, CANDIDATE_FILE, compress=3)
                print(f"Saved separate experimental candidate: {CANDIDATE_FILE}")
            val_pred = (val_scores >= threshold).astype(int)
            y_val_np = y_val.to_numpy()
            tn, fp, fn, tp = confusion_matrix(y_val_np, val_pred, labels=[0, 1]).ravel()
            metric_rows.append({
                "model": name,
                "fuzzer_sample_weight": fuzzer_weight,
                "shellcode_sample_weight": shellcode_weight,
                "calibration_fpr_target": target,
                "threshold": threshold,
                "validation_TN": int(tn), "validation_FP": int(fp),
                "validation_FN": int(fn), "validation_TP": int(tp),
                "validation_precision": float(precision_score(y_val_np, val_pred, zero_division=0)),
                "validation_attack_recall": float(recall_score(y_val_np, val_pred, zero_division=0)),
                "validation_F1": float(f1_score(y_val_np, val_pred, zero_division=0)),
                "validation_FPR": float(fp / (fp + tn) if fp + tn else 0),
            })
            print(
                f"calibration FPR target={target:.2%}; threshold={threshold:.6f}; "
                f"validation FPR={fp / (fp + tn):.3%}; "
                f"attack recall={val_pred[y_val_np == 1].mean():.3%}"
            )
            cat_val_np = cat_val.to_numpy()
            for category in sorted(pd.unique(cat_val)):
                mask = cat_val_np == category
                count = int(mask.sum())
                detected = int(val_pred[mask].sum())
                category_rows.append({
                    "model": name,
                    "calibration_fpr_target": target,
                    "category": category,
                    "records": count,
                    "detected": detected,
                    "missed": count - detected,
                    "recall": detected / count if count else None,
                })

    metrics_df = pd.DataFrame(metric_rows)
    categories_df = pd.DataFrame(category_rows)
    metrics_df.to_csv(METRICS_FILE, index=False)
    categories_df.to_csv(CATEGORY_FILE, index=False)
    print("\n=== VALIDATION METRICS ===")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== FUZZERS AND SHELLCODE ===")
    focus = categories_df[categories_df.category.isin(["Fuzzers", "Shellcode"])]
    print(focus.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {METRICS_FILE}, {CATEGORY_FILE}, {CANDIDATE_FILE}")
    print(
        "The API-loaded weighted binary RF artifact was updated from the "
        "canonical training partition; the 82,332-row holdout was not used."
    )
    print(f"Elapsed: {time.time() - started:.1f} seconds")


if __name__ == "__main__":
    main()
