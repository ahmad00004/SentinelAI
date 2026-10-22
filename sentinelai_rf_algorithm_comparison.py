"""Compare a boosting alternative for low-recall UNSW-NB15 categories.

The script uses only the training CSV, reproduces the fixed 80/20 validation
split, selects thresholds on an inner calibration subset of the 80% fit data,
and reports results on the held-out validation subset. It never opens the
official test CSV and never modifies the production Isolation Forest.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SEED = 42
CALIBRATION_SEED = 43
FPR_TARGETS = [0.0075, 0.005]
TRAIN_FILE = Path("UNSW_NB15_training-set.csv")
METRICS_FILE = Path("sentinelai_rf_algorithm_comparison_metrics.csv")
CATEGORY_FILE = Path("sentinelai_rf_algorithm_comparison_by_category.csv")


def make_model(numeric, categorical):
    transformers = []
    if numeric:
        transformers.append((
            "numeric", SimpleImputer(strategy="median"), numeric
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                )),
            ]),
            categorical,
        ))
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("classifier", HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=180,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )),
    ])


def score(model, X):
    classes = list(model.named_steps["classifier"].classes_)
    return 1.0 - model.predict_proba(X)[:, classes.index("Normal")]


def choose_threshold(y, scores, target_fpr):
    y_binary = (np.asarray(y) != "Normal").astype(int)
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
    print("Loading training CSV only; official test CSV is not opened.")
    df = pd.read_csv(TRAIN_FILE, low_memory=False)
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    features = [c for c in df.columns if c not in {"id", "label", "attack_cat"}]
    X, y = df[features].copy(), df["attack_cat"].copy()
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_fit, y_fit, test_size=0.20, random_state=CALIBRATION_SEED,
        stratify=y_fit,
    )
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in features if c not in numeric]

    configurations = [
        ("hist_gradient_boosting_balanced", 1.0),
        ("hist_gradient_boosting_fuzzer_x2", 2.0),
    ]
    metric_rows, category_rows = [], []

    print(
        f"Fit={len(X_train):,}, calibration={len(X_cal):,}, "
        f"validation={len(X_val):,}, features={len(features)}"
    )
    for name, fuzzer_weight in configurations:
        print(f"\nTraining {name}...")
        model = make_model(numeric, categorical)
        sample_weight = np.where(y_train.to_numpy() == "Fuzzers", fuzzer_weight, 1.0)
        model.fit(X_train, y_train, classifier__sample_weight=sample_weight)
        cal_scores = score(model, X_cal)
        val_scores = score(model, X_val)

        for target in FPR_TARGETS:
            threshold = choose_threshold(y_cal.to_numpy(), cal_scores, target)
            val_pred = (val_scores >= threshold).astype(int)
            y_binary = (y_val.to_numpy() != "Normal").astype(int)
            tn, fp, fn, tp = confusion_matrix(y_binary, val_pred, labels=[0, 1]).ravel()
            metric_rows.append({
                "model": name,
                "fuzzer_sample_weight": fuzzer_weight,
                "calibration_fpr_target": target,
                "threshold": threshold,
                "validation_TN": int(tn), "validation_FP": int(fp),
                "validation_FN": int(fn), "validation_TP": int(tp),
                "validation_precision": float(precision_score(y_binary, val_pred, zero_division=0)),
                "validation_attack_recall": float(recall_score(y_binary, val_pred, zero_division=0)),
                "validation_F1": float(f1_score(y_binary, val_pred, zero_division=0)),
                "validation_FPR": float(fp / (fp + tn) if fp + tn else 0),
            })
            print(
                f"calibration FPR target={target:.2%}; threshold={threshold:.6f}; "
                f"validation FPR={fp / (fp + tn):.3%}; "
                f"attack recall={val_pred[y_binary == 1].mean():.3%}"
            )
            for category in sorted(pd.unique(y_val)):
                mask = y_val.to_numpy() == category
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
    print(f"\nSaved: {METRICS_FILE}, {CATEGORY_FILE}")
    print("No production model was overwritten; the official test CSV was not used.")
    print(f"Elapsed: {time.time() - started:.1f} seconds")


if __name__ == "__main__":
    main()
