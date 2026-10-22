import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mysql.connector
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ARTIFACT_PATH = Path("sentinelai_supervised_rf.joblib")

if not ARTIFACT_PATH.exists():
    raise RuntimeError(
        "Model artifact missing. Run sentinelai_supervised_rf_experiment.py first."
    )

artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
threshold = float(artifact["threshold"])
feature_columns = artifact["feature_columns"]
classes = list(artifact["classes"])
normal_index = classes.index("Normal")

attack_indices = [
    i for i, name in enumerate(classes) if name != "Normal"
]
attack_names = [classes[i] for i in attack_indices]

app = FastAPI(
    title="SentinelAI Random Forest Test API",
    version="experimental-1.0",
    description="Isolated experimental endpoint; production API is unchanged.",
)


class SecurityEvent(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="UNSW-NB15 feature names and values.",
    )


def get_db():
    password = os.getenv("SENTINELAI_DB_PASSWORD")
    if not password:
        raise RuntimeError("SENTINELAI_DB_PASSWORD is not set.")

    return mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password=password,
        database="sentinel_ai",
        connection_timeout=5,
    )


def ensure_table():
    connection = get_db()
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rf_test_events (
                event_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                raw_event JSON NOT NULL,
                suspicion_score DOUBLE NOT NULL,
                threshold_used DOUBLE NOT NULL,
                status VARCHAR(20) NOT NULL,
                predicted_category VARCHAR(100) NOT NULL,
                model_version VARCHAR(50) NOT NULL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
        cursor.close()
    finally:
        connection.close()


@app.get("/")
def root():
    return {
        "service": "SentinelAI Random Forest Test API",
        "model_version": artifact["version"],
        "production_model_modified": False,
        "docs": "/docs",
    }


@app.get("/health/db")
def database_health():
    try:
        connection = get_db()
        connection.close()
        return {"database": "connected"}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {exc}",
        )


@app.post("/rf-test/events")
def detect_event(event: SecurityEvent):
    try:
        # Reject unknown fields to catch misspelled feature names.
        extras = set(event.features) - set(feature_columns)
        if extras:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Unknown feature names.",
                    "unknown_features": sorted(extras),
                },
            )

        # Missing fields become NaN and are handled by the saved imputers.
        frame = pd.DataFrame(
            [{name: event.features.get(name) for name in feature_columns}],
            columns=feature_columns,
        )

        probabilities = model.predict_proba(frame)[0]
        suspicion_score = float(1.0 - probabilities[normal_index])
        suspicious = suspicion_score >= threshold

        if suspicious:
            best_attack_index = max(
                attack_indices,
                key=lambda i: probabilities[i],
            )
            predicted_category = classes[best_attack_index]
            status = "Suspicious"
        else:
            predicted_category = "Normal"
            status = "Normal"

        raw_json = json.dumps(
            event.features,
            default=str,
            allow_nan=False,
        )

        connection = get_db()
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO rf_test_events
                (raw_event, suspicion_score, threshold_used, status,
                 predicted_category, model_version)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    raw_json,
                    suspicion_score,
                    threshold,
                    status,
                    predicted_category,
                    artifact["version"],
                ),
            )
            connection.commit()
            event_id = cursor.lastrowid
            cursor.close()
        finally:
            connection.close()

        return {
            "event_id": event_id,
            "status": status,
            "predicted_category": predicted_category,
            "suspicion_score": round(suspicion_score, 6),
            "threshold": round(threshold, 6),
            "saved_to_mysql": True,
            "table": "rf_test_events",
            "model_version": artifact["version"],
            "note": (
                "Experimental prediction, not proof of an attack."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.on_event("startup")
def startup():
    ensure_table()
