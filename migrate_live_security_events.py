"""Add canonical event fields to the existing live event table.

Run once before starting sentinelai_api.py:
    python migrate_live_security_events.py

The migration is idempotent and preserves the existing columns used by the
dashboard and older API clients.
"""

import os
import mysql.connector


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
        database="sentinel_ai",
        connection_timeout=5,
    )


def migrate():
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("SHOW COLUMNS FROM live_security_events")
        existing_columns = {row[0] for row in cursor.fetchall()}

        additions = {
            "source": "VARCHAR(128) NOT NULL DEFAULT 'api'",
            "event_type": "VARCHAR(128) NOT NULL DEFAULT 'network_flow'",
            "event_timestamp": "DATETIME(6) NULL",
            "raw_data": "JSON NULL",
            "normalized_fields": "JSON NULL",
            "alert_severity": "VARCHAR(16) NULL",
            "detection_reason": "TEXT NULL",
            "alert_status": "VARCHAR(32) NULL",
            "detection_methods": "JSON NULL",
            "matched_rules": "JSON NULL",
            "assigned_to": "VARCHAR(128) NULL",
            "investigation_notes": "TEXT NULL",
            "classifier_score": "DOUBLE NULL",
            "classifier_threshold": "DOUBLE NULL",
            "classifier_prediction": "TINYINT NULL",
            "isolation_forest_flagged": "TINYINT NULL",
        }

        for column, definition in additions.items():
            if column not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE live_security_events "
                    f"ADD COLUMN `{column}` {definition}"
                )

        # Backfill existing rows so the new columns are usable immediately.
        cursor.execute(
            """
            UPDATE live_security_events
            SET event_timestamp = COALESCE(event_timestamp, detected_at,
                                           UTC_TIMESTAMP(6)),
                raw_data = COALESCE(raw_data, raw_event, JSON_OBJECT()),
                normalized_fields = COALESCE(normalized_fields, raw_event,
                                             JSON_OBJECT()),
                alert_severity = COALESCE(
                    alert_severity,
                    CASE WHEN status = 'Suspicious' THEN 'Medium' ELSE NULL END
                ),
                detection_reason = COALESCE(
                    detection_reason,
                    CASE
                        WHEN status = 'Suspicious' THEN CONCAT(
                            'Isolation Forest flagged this event as anomalous ',
                            '(prediction=', prediction,
                            ', anomaly_score=', FORMAT(anomaly_score, 6),
                            '). The score is a model output, not an attack probability.'
                        )
                        ELSE CONCAT(
                            'Isolation Forest did not flag this event as anomalous ',
                            '(prediction=', prediction,
                            ', anomaly_score=', FORMAT(anomaly_score, 6), ').'
                        )
                    END
                ),
                alert_status = COALESCE(
                    alert_status,
                    CASE WHEN status = 'Suspicious' THEN 'New' ELSE NULL END
                )
            WHERE event_timestamp IS NULL
               OR raw_data IS NULL
               OR normalized_fields IS NULL
               OR detection_reason IS NULL
            """
        )
        cursor.execute(
            "ALTER TABLE live_security_events "
            "MODIFY COLUMN event_timestamp DATETIME(6) NOT NULL, "
            "MODIFY COLUMN raw_data JSON NOT NULL, "
            "MODIFY COLUMN normalized_fields JSON NOT NULL, "
            "MODIFY COLUMN detection_reason TEXT NOT NULL"
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sentinelai_response_simulations (
                simulation_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                event_id BIGINT NOT NULL,
                action_type VARCHAR(64) NOT NULL,
                reason TEXT NULL,
                performed_by VARCHAR(128) NULL,
                outcome TEXT NOT NULL,
                created_at DATETIME(6) NOT NULL,
                INDEX idx_simulation_event_created (event_id, created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        connection.commit()
        print(
            "Migration complete: live_security_events stores canonical events, "
            "alert metadata, matched rules, classifier and anomaly signals, "
            "ownership, notes, and simulated response records."
        )
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    migrate()
