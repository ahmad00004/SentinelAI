
import os
import json
import math
import joblib
import pandas as pd
import mysql.connector

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime, timezone
from sentinelai_rules import evaluate_rules, load_rules

# 1. LOAD THE EXISTING TRAINED MODEL

MODEL_PATH = "sentinelai_model.joblib"
CLASSIFIER_PATH = "sentinelai_supervised_rf_binary_weighted_candidate.joblib"

artifact = joblib.load(MODEL_PATH)
classifier_artifact = joblib.load(CLASSIFIER_PATH)

model = artifact["model"]
FEATURE_COLUMNS = artifact["feature_columns"]
NUMERIC_FEATURES = artifact["numeric_features"]
CATEGORICAL_FEATURES = artifact["categorical_features"]
classifier = classifier_artifact["model"]
CLASSIFIER_THRESHOLD = float(classifier_artifact["threshold"])
CLASSIFIER_CLASSES = list(classifier.named_steps["classifier"].classes_)
if set(classifier_artifact["feature_columns"]) != set(FEATURE_COLUMNS):
    raise RuntimeError("Isolation Forest and binary RF feature sets do not match.")
CLASSIFIER_FEATURE_COLUMNS = classifier_artifact["feature_columns"]
SECURITY_RULES = load_rules()


# 2. CREATE FASTAPI APPLICATION

app = FastAPI(
    title="SentinelAI Real-Time Detection API",
    version="1.5.0"
)


# 3. REQUEST DATA MODEL

class SecurityEvent(BaseModel):
    features: dict[str, Any]
    source: str = "api"
    event_type: str = "network_flow"
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    raw_data: dict[str, Any] | None = None


def normalize_event_metadata(event: SecurityEvent) -> tuple[str, str, datetime]:
    """Validate source metadata and convert the event time to UTC for MySQL."""
    source = event.source.strip()
    event_type = event.event_type.strip()

    if not source or len(source) > 128:
        raise HTTPException(
            status_code=422,
            detail="source must contain 1 to 128 non-whitespace characters"
        )

    if not event_type or len(event_type) > 128:
        raise HTTPException(
            status_code=422,
            detail="event_type must contain 1 to 128 non-whitespace characters"
        )

    event_time = event.timestamp
    if event_time.tzinfo is None:
        # Naive timestamps are interpreted as UTC so the stored value is
        # deterministic across machines and deployment time zones.
        event_time = event_time.replace(tzinfo=timezone.utc)
    event_time = event_time.astimezone(timezone.utc).replace(tzinfo=None)

    return source, event_type, event_time

# 4. MYSQL CONNECTION

def get_db_connection():
    password = os.environ.get("SENTINELAI_DB_PASSWORD")

    if not password:
        raise RuntimeError(
            "Set the SENTINELAI_DB_PASSWORD environment variable first."
        )

    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password=password,
        database="sentinel_ai"
    )

# 5. HOME ENDPOINT
@app.get("/")
def home():
    return {
        "system": "SentinelAI",
        "status": "running",
        "expected_features": len(FEATURE_COLUMNS)
    }

# 6. DATABASE HEALTH CHECK
@app.get("/health/db")
def database_health():
    conn = None

    try:
        conn = get_db_connection()

        if not conn.is_connected():
            raise RuntimeError("MySQL connection is not active.")

        return {"database": "connected"}

    except Exception as exc:
        print(
            f"MySQL health check failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True
        )

        raise HTTPException(
            status_code=503,
            detail="Database connection failed. Check MySQL and credentials."
        )

    finally:
        if conn is not None and conn.is_connected():
            conn.close()

# 7. REAL-TIME EVENT DETECTION
@app.post("/events")
def detect_event(event: SecurityEvent):

    features = event.features
    source, event_type, event_timestamp = normalize_event_metadata(event)

    # Validate required and unexpected features
    missing = [
        col for col in FEATURE_COLUMNS
        if col not in features
    ]

    extra = [
        col for col in features
        if col not in FEATURE_COLUMNS
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing features",
                "features": missing
            }
        )

    if extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unexpected features",
                "features": extra
            }
        )

    # Preserve the exact feature order used by the model
    row = {
        col: features[col]
        for col in FEATURE_COLUMNS
    }

    # Validate numeric features
    for col in NUMERIC_FEATURES:
        try:
            value = float(row[col])

            if not math.isfinite(value):
                raise ValueError("Value must be finite")

            row[col] = value

        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{col}' must be a finite number"
            )

    # Validate categorical features
    for col in CATEGORICAL_FEATURES:
        if not isinstance(row[col], str):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{col}' must be a string"
            )

    # Create one-row DataFrame for the existing model
    frame = pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS
    )

    # Run anomaly detection
    try:
        prediction = int(model.predict(frame)[0])

        anomaly_score = float(
            -model.decision_function(frame)[0]
        )

        if not math.isfinite(anomaly_score):
            raise ValueError("Model returned a non-finite score")

    except Exception as exc:
        print(
            f"Model prediction failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True
        )

        raise HTTPException(
            status_code=422,
            detail="The model could not score this event."
        )

    isolation_forest_status = "Suspicious" if prediction == -1 else "Normal"

    # Use the supervised binary RF for the primary Normal/Attack decision.
    # Isolation Forest remains a separate context signal because its current
    # false-positive rate is too high to drive the primary alert decision.
    try:
        classifier_frame = frame[CLASSIFIER_FEATURE_COLUMNS]
        positive_index = CLASSIFIER_CLASSES.index(1)
        classifier_score = float(
            classifier.predict_proba(classifier_frame)[0][positive_index]
        )
        if not math.isfinite(classifier_score):
            raise ValueError("Classifier returned a non-finite score")
        classifier_prediction = int(classifier_score >= CLASSIFIER_THRESHOLD)
    except Exception as exc:
        print(
            f"Binary RF prediction failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise HTTPException(
            status_code=422,
            detail="The binary Random Forest could not score this event.",
        )

    classifier_status = "Suspicious" if classifier_prediction == 1 else "Normal"
    matched_rules = evaluate_rules(row, SECURITY_RULES)
    detection_methods = []
    detection_reasons = []
    severity_rank = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    alert_severity = None

    if classifier_status == "Suspicious":
        detection_methods.append("binary_random_forest")
        alert_severity = "Medium"
        detection_reasons.append(
            "Binary Random Forest flagged this event as a potential attack "
            f"(attack_score={classifier_score:.6f}, "
            f"threshold={CLASSIFIER_THRESHOLD:.6f}). "
            "The score is a model output, not a calibrated attack probability."
        )

    if matched_rules:
        detection_methods.append("security_rule")
        rule_severity = max(
            (match["severity"] for match in matched_rules),
            key=severity_rank.get,
        )
        if alert_severity is None or severity_rank[rule_severity] > severity_rank[alert_severity]:
            alert_severity = rule_severity
        operator_symbols = {
            "eq": "==", "neq": "!=", "gt": ">", "gte": ">=",
            "lt": "<", "lte": "<=",
        }
        for match in matched_rules:
            detection_reasons.append(
                f"{match['rule_id']} {match['name']}: {match['field']}="
                f"{match['observed_value']} "
                f"{operator_symbols[match['operator']]} {match['threshold']}. "
                f"{match['description']}"
            )

    status = "Suspicious" if detection_reasons else "Normal"
    alert_status = "New" if status == "Suspicious" else None
    if status == "Normal":
        detection_reasons.append(
            "Binary Random Forest did not flag this event "
            f"(attack_score={classifier_score:.6f}, "
            f"threshold={CLASSIFIER_THRESHOLD:.6f})."
        )
        detection_reasons.append(
            "No enabled security rules matched."
            if any(rule["enabled"] for rule in SECURITY_RULES)
            else "No security rules are enabled."
        )
    detection_reasons.append(
        "Isolation Forest anomaly signal: "
        f"{isolation_forest_status} (score={anomaly_score:.6f}); "
        "this secondary signal does not independently create an alert."
    )
    detection_reason = " ".join(detection_reasons)

    # Preserve the submitted payload when supplied. Legacy clients that send
    # only {"features": ...} still get a useful raw event record.
    raw_data = event.raw_data if event.raw_data is not None else {
        "features": features
    }

    try:
        raw_data_json = json.dumps(raw_data, allow_nan=False)
        normalized_fields_json = json.dumps(row, allow_nan=False)
        detection_methods_json = json.dumps(detection_methods, allow_nan=False)
        matched_rules_json = json.dumps(matched_rules, allow_nan=False)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="raw_data and normalized feature values must be JSON serializable"
        )

   # 8. SAVE DETECTION TO MYSQL
    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO live_security_events
                (source, event_type, event_timestamp, raw_data,
                 normalized_fields, raw_event, anomaly_score, status, prediction,
                 alert_severity, detection_reason, alert_status,
                 detection_methods, matched_rules, classifier_score,
                 classifier_threshold, classifier_prediction,
                 isolation_forest_flagged)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                source,
                event_type,
                event_timestamp,
                raw_data_json,
                normalized_fields_json,
                normalized_fields_json,
                anomaly_score,
                status,
                prediction,
                alert_severity,
                detection_reason,
                alert_status,
                detection_methods_json,
                matched_rules_json,
                classifier_score,
                CLASSIFIER_THRESHOLD,
                classifier_prediction,
                int(prediction == -1),
            )
        )

        conn.commit()
        event_id = cursor.lastrowid

    except Exception as exc:

        if conn is not None:
            try:
                conn.rollback()
            except Exception as rollback_exc:
                print(
                    f"MySQL rollback failed: "
                    f"{type(rollback_exc).__name__}: {rollback_exc}",
                    flush=True
                )

        # Print the real error in the API terminal
        print(
            f"MySQL save failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True
        )

        raise HTTPException(
            status_code=503,
            detail="Detection calculated, but saving to MySQL failed."
        )

    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception as close_exc:
                print(
                    f"MySQL cursor close failed: {close_exc}",
                    flush=True
                )

        if conn is not None:
            try:
                if conn.is_connected():
                    conn.close()
            except Exception as close_exc:
                print(
                    f"MySQL connection close failed: {close_exc}",
                    flush=True
                )

   # 9. RETURN DETECTION RESULT
    return {
        "system": "SentinelAI",
        "event_id": event_id,
        "source": source,
        "event_type": event_type,
        "timestamp": event_timestamp.replace(tzinfo=timezone.utc).isoformat(),
        "status": status,
        "model_status": classifier_status,
        "classifier_name": classifier_artifact.get(
            "model_name", "Binary Random Forest"
        ),
        "classifier_score": round(classifier_score, 6),
        "classifier_threshold": round(CLASSIFIER_THRESHOLD, 6),
        "isolation_forest_status": isolation_forest_status,
        "alert_severity": alert_severity,
        "detection_reason": detection_reason,
        "detection_methods": detection_methods,
        "matched_rules": matched_rules,
        "alert_status": alert_status,
        "anomaly_score": round(anomaly_score, 6),
        "prediction": prediction,
        "saved_to_mysql": True,
        "note": (
            "Binary RF is the primary known-attack classifier; Isolation Forest "
            "is a secondary anomaly signal. A detection is not proof of an attack."
        )
    }

