
import getpass

import numpy as np
import pandas as pd
import mysql.connector

from detector import detect_threats


REQUIRED_COLUMNS = [
    "event_id",
    "failed_logins",
    "bytes_sent",
    "connections",
    "hour",
]

FEATURE_COLUMNS = [
    "failed_logins",
    "bytes_sent",
    "connections",
    "hour",
]


def validate_logs(logs):
    # Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in logs.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    logs = logs.copy()

    # Convert required columns to numeric values
    for column in REQUIRED_COLUMNS:
        logs[column] = pd.to_numeric(
            logs[column],
            errors="coerce",
        )

    # Reject rows with missing, invalid, or infinite values
    numeric_values = logs[REQUIRED_COLUMNS].to_numpy(
        dtype=float
    )

    valid_mask = (
        logs[REQUIRED_COLUMNS].notna().all(axis=1)
        & np.isfinite(numeric_values).all(axis=1)
    )

    invalid_count = int((~valid_mask).sum())

    if invalid_count:
        print(
            f"Warning: Skipping {invalid_count} rows "
            "with missing or invalid numeric values."
        )

    logs = logs.loc[valid_mask].copy()

    if logs.empty:
        return logs

    # Event IDs must be whole numbers
    if (logs["event_id"] % 1 != 0).any():
        raise ValueError("event_id must contain whole numbers.")

    if (logs["event_id"] < 0).any():
        raise ValueError("event_id cannot be negative.")

    logs["event_id"] = logs["event_id"].astype("int64")

    # Event IDs must be unique
    if logs["event_id"].duplicated().any():
        raise ValueError(
            "Duplicate event IDs found in CSV. "
            "Each event must have a unique event_id."
        )

    # Counts and bytes must be non-negative whole numbers
    for column in FEATURE_COLUMNS:
        if column == "hour":
            continue

        if (logs[column] < 0).any():
            raise ValueError(
                f"{column} cannot contain negative values."
            )

        if (logs[column] % 1 != 0).any():
            raise ValueError(
                f"{column} must contain whole numbers."
            )

    # Hour must be a whole number from 0 through 23
    if not logs["hour"].between(0, 23).all():
        raise ValueError("hour must be between 0 and 23.")

    if (logs["hour"] % 1 != 0).any():
        raise ValueError("hour must contain whole numbers.")

    return logs


def save_results_to_mysql(logs, results):
    connection = None
    cursor = None

    try:
        # Prompt securely instead of hardcoding the password
        password = getpass.getpass("MySQL root password: ")

        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password=password,
            database="sentinel_ai",
        )

        cursor = connection.cursor()

        # Save or update the original security events.
        # Existing event IDs are updated instead of duplicated.
        event_query = """
            INSERT INTO security_events
                (event_id, failed_logins, bytes_sent,
                 connections, hour)
            VALUES (%s, %s, %s, %s, %s) AS new
            ON DUPLICATE KEY UPDATE
                failed_logins = new.failed_logins,
                bytes_sent = new.bytes_sent,
                connections = new.connections,
                hour = new.hour
        """

        event_records = [
            (
                int(row.event_id),
                int(row.failed_logins),
                int(row.bytes_sent),
                int(row.connections),
                int(row.hour),
            )
            for row in logs.itertuples(index=False)
        ]

        if event_records:
            cursor.executemany(event_query, event_records)

        # Remove old detection results for these event IDs.
        # This keeps repeated runs from accumulating duplicates.
        delete_query = """
            DELETE FROM threat_detections
            WHERE event_id = %s
        """

        if event_records:
            cursor.executemany(
                delete_query,
                [(record[0],) for record in event_records],
            )

        # Save the latest anomaly scores and statuses
        detection_query = """
            INSERT INTO threat_detections
                (event_id, anomaly_score, status)
            VALUES (%s, %s, %s)
        """

        detection_records = [
            (
                int(row.event_id),
                float(row.anomaly_score),
                str(row.status),
            )
            for row in results.itertuples(index=False)
        ]

        if detection_records:
            cursor.executemany(
                detection_query,
                detection_records,
            )

        connection.commit()

        print("\n===== MySQL Storage =====")
        print(f"Security events saved: {len(event_records)}")
        print(f"Detection results saved: {len(detection_records)}")
        print("Database transaction committed successfully.")

    except Exception:
        if connection is not None and connection.is_connected():
            connection.rollback()

        raise

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()


def main():
    print("\n===== SentinelAI Threat Detection =====")

    # Read security events from CSV
    logs = pd.read_csv("security_logs.csv")

    print(f"Total events loaded: {len(logs)}")

    # Validate input data
    logs = validate_logs(logs)

    print(f"Valid events for analysis: {len(logs)}")

    if logs.empty:
        print("No valid events available for analysis.")
        return

    # Run the existing Isolation Forest detector
    results = detect_threats(logs)

    # Display detection results
    print(
        results[
            ["event_id", "failed_logins",
             "anomaly_score", "status"]
        ].to_string(index=False)
    )

    # Display suspicious events
    suspicious = results[
        results["status"] == "Suspicious"
    ]

    print("\n===== Suspicious Events =====")
    print(f"Suspicious events found: {len(suspicious)}")

    if suspicious.empty:
        print("No unusual events detected.")
    else:
        print(suspicious.to_string(index=False))

    # Store events and detections in MySQL
    save_results_to_mysql(logs, results)


if __name__ == "__main__":
    main()
