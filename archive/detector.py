
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

FEATURES = [
    "failed_logins",
    "bytes_sent",
    "connections",
    "hour",
]


def detect_threats(logs: pd.DataFrame) -> pd.DataFrame:
    # Handle empty input
    if logs.empty:
        result = logs.copy()
        result["anomaly_score"] = pd.Series(dtype=float)
        result["status"] = pd.Series(dtype=str)
        return result

    # Check required features
    missing = set(FEATURES) - set(logs.columns)

    if missing:
        raise ValueError(
            f"Missing features: {sorted(missing)}"
        )

    # Work on a copy
    result = logs.copy()

    # Convert features to numeric values
    for feature in FEATURES:
        result[feature] = pd.to_numeric(
            result[feature],
            errors="raise"
        )

    # Reject missing values
    if result[FEATURES].isna().any().any():
        raise ValueError(
            "Missing values found in security data."
        )

    # Reject infinite or non-finite values
    if not np.isfinite(
        result[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite or non-finite values found "
            "in security data."
        )

    # Range validation
    non_negative_features = [
        "failed_logins",
        "bytes_sent",
        "connections",
    ]

    for feature in non_negative_features:
        if (result[feature] < 0).any():
            raise ValueError(
                f"{feature} cannot contain negative values."
            )

    # Counts must be whole numbers
    count_features = [
        "failed_logins",
        "bytes_sent",
        "connections",
    ]

    for feature in count_features:
        if (result[feature] % 1 != 0).any():
            raise ValueError(
                f"{feature} must contain whole numbers."
            )

    # Hour must be between 0 and 23
    if not result["hour"].between(0, 23).all():
        raise ValueError(
            "hour must be between 0 and 23."
        )

    # Hour represents a whole number
    if (result["hour"] % 1 != 0).any():
        raise ValueError(
            "hour must contain whole numbers."
        )

    # Train the Isolation Forest model
    model = IsolationForest(
        n_estimators=100,
        contamination=0.15,
        random_state=42,
    )

    predictions = model.fit_predict(result[FEATURES])

    # Calculate anomaly scores
    result["anomaly_score"] = model.decision_function(
        result[FEATURES]
    )

    # Assign threat status
    result["status"] = [
        "Suspicious" if prediction == -1 else "Normal"
        for prediction in predictions
    ]

    return result


