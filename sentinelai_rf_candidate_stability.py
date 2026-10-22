"""Check the weighted binary RF candidate across calibration split seeds.

Only the UNSW-NB15 training CSV is used. The official test CSV is never opened.
Each run retains the fixed seed-42 outer validation split and varies only the
inner calibration split from the fit portion.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from sentinelai_rf_binary_attack_experiment import (
    FPR_TARGETS,
    TRAIN_FILE,
    choose_threshold,
    make_model,
)


SEED = 42
CALIBRATION_SEEDS = [43, 44, 45]
TARGET_FPR = 0.0075
METRICS_FILE = Path("sentinelai_rf_candidate_stability_metrics.csv")
CATEGORY_FILE = Path("sentinelai_rf_candidate_stability_by_category.csv")


def main():
    started = time.time()
    print("Loading training CSV only; official test CSV is not opened.")
    df = pd.read_csv(TRAIN_FILE, low_memory=False)
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    feature_columns = [c for c in df.columns if c not in {"id", "label", "attack_cat"}]
    X, categories = df[feature_columns].copy(), df["attack_cat"].copy()
    y = (categories != "Normal").astype(int)
    X_fit, X_val, y_fit, y_val, cat_fit, cat_val = train_test_split(
        X, y, categories, test_size=0.20, random_state=SEED, stratify=categories
    )
    numeric = X_fit.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in feature_columns if c not in numeric]
    metric_rows, category_rows = [], []

    for calibration_seed in CALIBRATION_SEEDS:
        X_train, X_cal, y_train, y_cal, cat_train, _ = train_test_split(
            X_fit, y_fit, cat_fit, test_size=0.20,
            random_state=calibration_seed, stratify=cat_fit,
        )
        for name, fuzzer_weight, shellcode_weight in [
            ("binary_baseline", 1.0, 1.0),
            ("binary_weighted_candidate", 4.0, 4.0),
        ]:
            print(f"Seed {calibration_seed}: training {name}...")
            model = make_model(numeric, categorical)
            labels = cat_train.to_numpy()
            weights = np.ones(len(labels), dtype=float)
            weights[labels == "Fuzzers"] *= fuzzer_weight
            weights[labels == "Shellcode"] *= shellcode_weight
            model.fit(X_train, y_train, classifier__sample_weight=weights)
            positive_index = list(model.named_steps["classifier"].classes_).index(1)
            cal_scores = model.predict_proba(X_cal)[:, positive_index]
            val_scores = model.predict_proba(X_val)[:, positive_index]
            threshold = choose_threshold(
                y_cal.to_numpy(), cal_scores, TARGET_FPR
            )
            predicted = (val_scores >= threshold).astype(int)
            y_val_array = y_val.to_numpy()
            tn, fp, fn, tp = confusion_matrix(
                y_val_array, predicted, labels=[0, 1]
            ).ravel()
            metric_rows.append({
                "calibration_seed": calibration_seed,
                "model": name,
                "threshold": threshold,
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                "precision": float(precision_score(y_val_array, predicted, zero_division=0)),
                "attack_recall": float(recall_score(y_val_array, predicted, zero_division=0)),
                "F1": float(f1_score(y_val_array, predicted, zero_division=0)),
                "FPR": float(fp / (fp + tn) if fp + tn else 0),
            })
            print(
                f"  threshold={threshold:.6f}, validation FPR={fp / (fp + tn):.3%}, "
                f"attack recall={predicted[y_val_array == 1].mean():.3%}"
            )
            for category in sorted(pd.unique(cat_val)):
                mask = cat_val.to_numpy() == category
                count = int(mask.sum())
                detected = int(predicted[mask].sum())
                category_rows.append({
                    "calibration_seed": calibration_seed,
                    "model": name,
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
    print("\n=== VALIDATION METRICS BY CALIBRATION SEED ===")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== FUZZERS/SHELLCODE RECALL BY CALIBRATION SEED ===")
    focus = categories_df[categories_df.category.isin(["Fuzzers", "Shellcode"])]
    print(focus.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {METRICS_FILE}, {CATEGORY_FILE}")
    print("No production model was overwritten; official test CSV was not used.")
    print(f"Elapsed: {time.time() - started:.1f} seconds")


if __name__ == "__main__":
    main()
