import joblib
import numpy as np
import pandas as pd
import requests

MODEL_FILE = "sentinelai_supervised_rf.joblib"
TEST_FILE = "UNSW_NB15_testing-set.csv"
OUTPUT_FILE = "sentinelai_offline_vs_api_comparison.csv"
API_URL = "http://127.0.0.1:8001/rf-test/events"
SAMPLE_SIZE = 20


def json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main():
    artifact = joblib.load(MODEL_FILE)
    model = artifact["model"]
    columns = artifact["feature_columns"]
    threshold = float(artifact["threshold"])

    classes = list(model.named_steps["classifier"].classes_)
    normal_index = classes.index("Normal")
    attack_classes = [c for c in classes if c != "Normal"]

    df = pd.read_csv(TEST_FILE, low_memory=False)
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()

    normal = (
        df[df["attack_cat"] == "Normal"]
        .head(SAMPLE_SIZE)
        .reset_index(drop=True)
    )

    if len(normal) != SAMPLE_SIZE:
        raise ValueError("Could not find 20 Normal records.")

    X = normal[columns].copy()

    print("Calculating offline predictions...")
    probabilities = model.predict_proba(X)
    scores = 1.0 - probabilities[:, normal_index]
    attack_indices = [classes.index(c) for c in attack_classes]

    rows = []

    for i in range(len(X)):
        features = {
            col: json_value(X.iloc[i][col])
            for col in columns
        }

        offline_score = float(scores[i])
        offline_status = (
            "Suspicious" if offline_score >= threshold else "Normal"
        )

        if offline_status == "Suspicious":
            offline_category = attack_classes[
                int(np.argmax(probabilities[i, attack_indices]))
            ]
        else:
            offline_category = "Normal"

        print(f"Sending record {i + 1}/{len(X)}...")

        try:
            response = requests.post(
                API_URL,
                json={"features": features},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            api_score = float(result["suspicion_score"])
            api_status = result["status"]
            api_category = result.get("predicted_category")

            rows.append({
                "record_number": i + 1,
                "actual_category": "Normal",
                "proto": features.get("proto"),
                "service": features.get("service"),
                "state": features.get("state"),
                "offline_score": offline_score,
                "api_score": api_score,
                "score_difference": abs(offline_score - api_score),
                "offline_threshold": threshold,
                "api_threshold": result.get("threshold_used"),
                "offline_status": offline_status,
                "api_status": api_status,
                "offline_attack_category": offline_category,
                "api_attack_category": api_category,
                "scores_match": np.isclose(
                    offline_score, api_score, atol=1e-6, rtol=0
                ),
                "statuses_match": offline_status == api_status,
                "categories_match": offline_category == api_category,
                "api_event_id": result.get("event_id"),
                "error": "",
            })

        except Exception as exc:
            print(f"Record {i + 1} failed: {exc}")
            rows.append({
                "record_number": i + 1,
                "actual_category": "Normal",
                "offline_score": offline_score,
                "api_score": None,
                "score_difference": None,
                "offline_threshold": threshold,
                "api_threshold": None,
                "offline_status": offline_status,
                "api_status": None,
                "offline_attack_category": offline_category,
                "api_attack_category": None,
                "scores_match": False,
                "statuses_match": False,
                "categories_match": False,
                "api_event_id": None,
                "error": str(exc),
            })

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_FILE, index=False)

    successful = report[report["api_score"].notna()]

    print("\n=== OFFLINE VS API COMPARISON ===")
    if not successful.empty:
        print(successful[[
            "record_number",
            "offline_score",
            "api_score",
            "score_difference",
            "offline_status",
            "api_status",
            "offline_attack_category",
            "api_attack_category",
            "scores_match",
            "statuses_match",
        ]].to_string(
            index=False,
            formatters={
                "offline_score": "{:.6f}".format,
                "api_score": "{:.6f}".format,
                "score_difference": "{:.8f}".format,
            },
        ))

        print("\n=== SUMMARY ===")
        print(f"Successful API requests: {len(successful)}/{SAMPLE_SIZE}")
        print(
            f"Scores matching: {int(successful['scores_match'].sum())}"
            f"/{len(successful)}"
        )
        print(
            f"Statuses matching: {int(successful['statuses_match'].sum())}"
            f"/{len(successful)}"
        )
        print(
            f"Attack categories matching: "
            f"{int(successful['categories_match'].sum())}/{len(successful)}"
        )
        print(
            "Normal records flagged offline: "
            f"{int((successful['offline_status'] == 'Suspicious').sum())}"
        )
        print(
            "Normal records flagged by API: "
            f"{int((successful['api_status'] == 'Suspicious').sum())}"
        )
        print(
            "Maximum score difference: "
            f"{successful['score_difference'].max():.8f}"
        )

    print(f"\nReport saved to {OUTPUT_FILE}")
    print("API requests may add rows to the experimental rf_test_events table.")


if __name__ == "__main__":
    main()
