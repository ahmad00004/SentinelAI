import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE = "UNSW_NB15_testing-set.csv"
MODEL_FILE = "sentinelai_supervised_rf.joblib"

OUTPUT_FILE = "sentinelai_threshold_comparison.csv"

SEED = 42
FPR_TARGETS = [0.01, 0.005, 0.0025, 0.001]


def calculate_metrics(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": precision_score(
            y_true, predicted, zero_division=0
        ),
        "recall": recall_score(
            y_true, predicted, zero_division=0
        ),
        "f1": f1_score(
            y_true, predicted, zero_division=0
        ),
        "false_positive_rate": (
            fp / (fp + tn) if (fp + tn) else 0
        ),
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(
            y_true, scores
        ),
    }


def choose_threshold(y_true, scores, target_fpr):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    negatives = int((y_true == 0).sum())
    positives = int((y_true == 1).sum())

    eligible = np.flatnonzero(
        (fpr <= target_fpr) & np.isfinite(thresholds)
    )

    if len(eligible) == 0:
        return None

    precision = (
        tpr[eligible] * positives
        / np.maximum(
            tpr[eligible] * positives
            + fpr[eligible] * negatives,
            1e-12,
        )
    )

    f1 = (
        2 * precision * tpr[eligible]
        / np.maximum(precision + tpr[eligible], 1e-12)
    )

    best = max(
        range(len(eligible)),
        key=lambda i: (
            f1[i],
            precision[i],
            tpr[eligible[i]],
        ),
    )

    return float(thresholds[eligible[best]])


def main():
    print("Loading existing experimental model...")
    artifact = joblib.load(MODEL_FILE)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    classes = list(model.named_steps["classifier"].classes_)
    normal_index = classes.index("Normal")
    original_threshold = float(artifact["threshold"])

    print("Loading UNSW-NB15 data...")
    train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
    test_df = pd.read_csv(TEST_FILE, low_memory=False)

    for df in (train_df, test_df):
        df["attack_cat"] = (
            df["attack_cat"].astype(str).str.strip()
        )

    # Reproduce the original validation split exactly.
    X_all = train_df[feature_columns].copy()
    y_all = train_df["attack_cat"].copy()

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.20,
        random_state=SEED,
        stratify=y_all,
    )

    # Sanity check: the loaded model must match this split's classes.
    if set(y_fit.unique()) != set(classes):
        raise ValueError(
            "Model classes do not match the reproduced training split."
        )

    X_test = test_df[feature_columns].copy()

    y_val_binary = (y_val != "Normal").astype(int).to_numpy()
    y_test_binary = (
        test_df["attack_cat"] != "Normal"
    ).astype(int).to_numpy()

    print("Scoring validation data...")
    val_prob = model.predict_proba(X_val)
    val_scores = 1.0 - val_prob[:, normal_index]

    print("Scoring test data for exploratory comparison...")
    test_prob = model.predict_proba(X_test)
    test_scores = 1.0 - test_prob[:, normal_index]

    # Include the existing threshold, then select alternatives
    # using validation data only.
    candidates = [
        ("Existing model threshold", original_threshold, None)
    ]

    for target in FPR_TARGETS:
        threshold = choose_threshold(
            y_val_binary, val_scores, target
        )
        if threshold is not None:
            candidates.append((
                f"Validation target FPR {target:.2%}",
                threshold,
                target,
            ))

    rows = []

    for name, threshold, target in candidates:
        val_metrics = calculate_metrics(
            y_val_binary, val_scores, threshold
        )
        test_metrics = calculate_metrics(
            y_test_binary, test_scores, threshold
        )

        rows.append({
            "experiment": name,
            "target_validation_fpr": target,
            "threshold": threshold,
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_fpr": val_metrics["false_positive_rate"],
            "test_precision_exploratory": test_metrics["precision"],
            "test_recall_exploratory": test_metrics["recall"],
            "test_f1_exploratory": test_metrics["f1"],
            "test_fpr_exploratory": test_metrics["false_positive_rate"],
            "test_TN": test_metrics["TN"],
            "test_FP": test_metrics["FP"],
            "test_FN": test_metrics["FN"],
            "test_TP": test_metrics["TP"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_average_precision": test_metrics["average_precision"],
        })

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_FILE, index=False)

    print("\n=== THRESHOLD COMPARISON ===")
    display_columns = [
        "experiment",
        "threshold",
        "validation_fpr",
        "validation_recall",
        "validation_f1",
        "test_fpr_exploratory",
        "test_recall_exploratory",
        "test_f1_exploratory",
    ]
    print(
        report[display_columns].to_string(
            index=False,
            formatters={
                "threshold": "{:.6f}".format,
                "validation_fpr": "{:.3%}".format,
                "validation_recall": "{:.3%}".format,
                "validation_f1": "{:.4f}".format,
                "test_fpr_exploratory": "{:.3%}".format,
                "test_recall_exploratory": "{:.3%}".format,
                "test_f1_exploratory": "{:.4f}".format,
            },
        )
    )

    print(f"\nSaved report: {OUTPUT_FILE}")
    print(
        "\nNOTE: Test metrics are exploratory because the test set "
        "was examined in earlier experiments."
    )
    print("No model, API, dashboard, or database was modified.")


if __name__ == "__main__":
    main()
