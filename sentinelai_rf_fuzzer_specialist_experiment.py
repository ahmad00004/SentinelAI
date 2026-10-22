"""Evaluate a separate Fuzzer specialist alongside the multiclass RF.

Uses only UNSW_NB15_training-set.csv. The existing 80/20 split (seed 42) is
reproduced; an inner calibration split is drawn only from the 80% fit portion.
The held-out validation portion evaluates the policy. The official test CSV
is never opened, and no existing model artifact is replaced.
"""

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
TARGET_FPR = 0.0075
TRAIN_FILE = Path("UNSW_NB15_training-set.csv")
METRICS_FILE = Path("sentinelai_rf_fuzzer_specialist_metrics.csv")
CATEGORY_FILE = Path("sentinelai_rf_fuzzer_specialist_by_category.csv")


def make_pipeline(numeric, categorical, binary=False):
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


def general_score(model, X):
    classes = list(model.named_steps["classifier"].classes_)
    return 1.0 - model.predict_proba(X)[:, classes.index("Normal")]


def threshold_at_fpr(y, scores, max_fpr):
    y_binary = (np.asarray(y) != "Normal").astype(int)
    fpr, tpr, thresholds = roc_curve(y_binary, scores)
    negatives = int((y_binary == 0).sum())
    positives = int((y_binary == 1).sum())
    valid = np.flatnonzero((fpr <= max_fpr) & np.isfinite(thresholds))
    if not len(valid):
        raise RuntimeError("No calibration threshold meets the requested FPR.")
    precision = tpr[valid] * positives / np.maximum(
        tpr[valid] * positives + fpr[valid] * negatives, 1e-12
    )
    f1 = 2 * precision * tpr[valid] / np.maximum(precision + tpr[valid], 1e-12)
    best = max(range(len(valid)), key=lambda i: (f1[i], precision[i], tpr[valid[i]]))
    return float(thresholds[valid[best]])


def metrics(y, pred, name, params):
    actual = (np.asarray(y) != "Normal").astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[0, 1]).ravel()
    return {
        "policy": name,
        **params,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "precision": float(precision_score(actual, pred, zero_division=0)),
        "attack_recall": float(recall_score(actual, pred, zero_division=0)),
        "F1": float(f1_score(actual, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn) if fp + tn else 0),
    }


def main():
    started = time.time()
    print("Loading training CSV only; the official test CSV is not opened.")
    df = pd.read_csv(TRAIN_FILE, low_memory=False)
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    feature_columns = [c for c in df.columns if c not in {"id", "label", "attack_cat"}]
    X_all, y_all = df[feature_columns].copy(), df["attack_cat"].copy()

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_fit, y_fit, test_size=0.20, random_state=CALIBRATION_SEED,
        stratify=y_fit,
    )
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in feature_columns if c not in numeric]

    print(f"Fit: {len(X_train):,}; calibration: {len(X_cal):,}; validation: {len(X_val):,}")
    print("Training multiclass baseline...")
    baseline = make_pipeline(numeric, categorical)
    baseline.fit(X_train, y_train)
    cal_general = general_score(baseline, X_cal)
    val_general = general_score(baseline, X_val)
    baseline_threshold = threshold_at_fpr(y_cal, cal_general, TARGET_FPR)
    cal_baseline_pred = cal_general >= baseline_threshold
    baseline_cal_recall = float(
        cal_baseline_pred[(y_cal.to_numpy() != "Normal")].mean()
    )
    val_baseline_pred = val_general >= baseline_threshold

    print("Training one-vs-rest Fuzzer specialist...")
    specialist = make_pipeline(numeric, categorical, binary=True)
    y_train_fuzzer = (y_train == "Fuzzers").astype(int)
    y_cal_fuzzer = (y_cal == "Fuzzers").astype(int).to_numpy()
    specialist.fit(X_train, y_train_fuzzer)
    specialist_classes = list(specialist.named_steps["classifier"].classes_)
    fuzzer_index = specialist_classes.index(1)
    cal_fuzzer_score = specialist.predict_proba(X_cal)[:, fuzzer_index]
    val_fuzzer_score = specialist.predict_proba(X_val)[:, fuzzer_index]

    # Choose the combined policy on calibration data. Preserve at least 99% of
    # baseline attack recall there while maximizing Fuzzer recall, with global
    # false-positive rate capped at 1%.
    general_thresholds = np.unique(np.quantile(cal_general, np.linspace(0, 1, 121)))
    specialist_thresholds = np.unique(
        np.quantile(cal_fuzzer_score, np.linspace(0, 1, 121))
    )
    cal_attack = (y_cal.to_numpy() != "Normal")
    cal_normal = ~cal_attack
    cal_fuzzer = (y_cal.to_numpy() == "Fuzzers")
    max_fp = int(np.floor(TARGET_FPR * cal_normal.sum()))
    minimum_attack_recall = max(0.0, baseline_cal_recall - 0.01)
    best_policy = None

    for general_cutoff in general_thresholds:
        base_flags = cal_general >= general_cutoff
        for specialist_cutoff in specialist_thresholds:
            flags = base_flags | (cal_fuzzer_score >= specialist_cutoff)
            fp = int((flags & cal_normal).sum())
            if fp > max_fp:
                continue
            attack_recall = float(flags[cal_attack].mean())
            if attack_recall < minimum_attack_recall:
                continue
            fuzzer_recall = float(flags[cal_fuzzer].mean())
            precision = float(
                (flags & cal_attack).sum() / max(flags.sum(), 1)
            )
            objective = (fuzzer_recall, attack_recall, precision)
            if best_policy is None or objective > best_policy[0]:
                best_policy = (
                    objective, float(general_cutoff), float(specialist_cutoff)
                )

    if best_policy is None:
        raise RuntimeError("No combined thresholds meet calibration constraints.")
    _, general_cutoff, specialist_cutoff = best_policy

    combined_val_pred = (
        (val_general >= general_cutoff)
        | (val_fuzzer_score >= specialist_cutoff)
    ).astype(int)
    baseline_val_pred = val_baseline_pred.astype(int)

    common = {
        "calibration_fpr_cap": TARGET_FPR,
        "calibration_attack_recall_floor": minimum_attack_recall,
        "baseline_threshold": baseline_threshold,
        "general_threshold": general_cutoff,
        "fuzzer_specialist_threshold": specialist_cutoff,
    }
    metric_rows = [
        metrics(y_val, baseline_val_pred, "multiclass_baseline", common),
        metrics(y_val, combined_val_pred, "baseline_plus_fuzzer_specialist", common),
    ]

    y_val_array = y_val.to_numpy()
    category_rows = []
    for category in sorted(pd.unique(y_val)):
        mask = y_val_array == category
        for policy_name, pred in [
            ("multiclass_baseline", baseline_val_pred),
            ("baseline_plus_fuzzer_specialist", combined_val_pred),
        ]:
            detected = int(pred[mask].sum())
            count = int(mask.sum())
            category_rows.append({
                "policy": policy_name,
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

    print("\n=== HELD-OUT VALIDATION COMPARISON ===")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== FUZZERS AND SHELLCODE ===")
    focus = categories_df[categories_df.category.isin(["Fuzzers", "Shellcode"])]
    print(focus.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {METRICS_FILE} and {CATEGORY_FILE}")
    print("No production model was overwritten; the official test CSV was not used.")
    print(f"Elapsed: {time.time() - started:.1f} seconds")


if __name__ == "__main__":
    main()
