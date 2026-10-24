import hashlib
import json
import os
import time
from datetime import datetime

import mysql.connector
from mysql.connector import Error

# Initial, configurable detection settings.
WINDOW_SECONDS = int(os.getenv("SENTINELAI_SCAN_WINDOW_SECONDS", "60"))
PORT_THRESHOLD = int(os.getenv("SENTINELAI_SCAN_PORT_THRESHOLD", "25"))
POLL_SECONDS = int(os.getenv("SENTINELAI_DETECTOR_POLL_SECONDS", "10"))
RULE_ID = "LIVE-PORT-001"
RULE_NAME = "Possible TCP/UDP port scan"

def get_connection():
    password = os.getenv("SENTINELAI_DB_PASSWORD")
    if not password:
        raise RuntimeError(
            "SENTINELAI_DB_PASSWORD is not set in this terminal."
        )
    return mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password=password,
        database="sentinel_ai",
        connection_timeout=10,
    )

def find_candidates(connection):
    # Use recent flow activity, grouped by source and destination.
    # Port numbers are evidence for investigation, not proof of an attack.
    sql = """
        SELECT
            source_ip,
            destination_ip,
            COUNT(DISTINCT destination_port) AS port_count,
            JSON_ARRAYAGG(destination_port) AS ports_json,
            GROUP_CONCAT(DISTINCT protocol ORDER BY protocol) AS protocols,
            MIN(first_seen) AS first_seen,
            MAX(last_seen) AS last_seen
        FROM network_flows
        WHERE flow_key IS NOT NULL
          AND source_ip IS NOT NULL
          AND destination_ip IS NOT NULL
          AND destination_port BETWEEN 1 AND 65535
          AND UPPER(protocol) IN ('TCP', 'UDP')
          AND last_seen >= UTC_TIMESTAMP(6) - INTERVAL %s SECOND
        GROUP BY source_ip, destination_ip
        HAVING COUNT(DISTINCT destination_port) >= %s
    """
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(sql, (WINDOW_SECONDS, PORT_THRESHOLD))
        rows = cursor.fetchall()
    finally:
        cursor.close()

    # Produce sorted, unique port lists and stable fingerprints per minute.
    for row in rows:
        ports = json.loads(row["ports_json"]) if isinstance(
            row["ports_json"], str
        ) else row["ports_json"]
        row["ports"] = sorted({
            int(port) for port in (ports or []) if port is not None
        })
        row["port_count"] = len(row["ports"])

    return [row for row in rows if row["port_count"] >= PORT_THRESHOLD]

def save_candidate(connection, row):
    # One fingerprint per source/destination/rule/minute.
    now = datetime.utcnow()
    minute_bucket = now.strftime("%Y%m%d%H%M")
    fingerprint_text = "|".join((
        RULE_ID,
        row["source_ip"],
        row["destination_ip"],
        minute_bucket,
    ))
    fingerprint = hashlib.sha256(
        fingerprint_text.encode("utf-8")
    ).hexdigest()

    reason = (
        f"{row['port_count']} distinct destination ports were observed "
        f"from {row['source_ip']} to {row['destination_ip']} in the "
        f"last {WINDOW_SECONDS} seconds. This is a potential scan signal "
        "requiring investigation, not confirmation of malicious activity."
    )

    sql = """
        INSERT INTO network_threat_alerts (
            fingerprint, rule_id, rule_name, severity, status,
            source_ip, destination_ip, protocol_summary,
            distinct_destination_ports, window_seconds, observed_ports,
            first_seen, last_seen, detection_reason
        )
        VALUES (%s, %s, %s, 'Medium', 'New', %s, %s, %s, %s, %s,
                %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            distinct_destination_ports = GREATEST(
                distinct_destination_ports,
                VALUES(distinct_destination_ports)
            ),
            observed_ports = VALUES(observed_ports),
            last_seen = VALUES(last_seen),
            detection_reason = VALUES(detection_reason)
    """
    cursor = connection.cursor()
    try:
        cursor.execute(sql, (
            fingerprint,
            RULE_ID,
            RULE_NAME,
            row["source_ip"],
            row["destination_ip"],
            row["protocols"],
            row["port_count"],
            WINDOW_SECONDS,
            json.dumps(row["ports"]),
            row["first_seen"],
            row["last_seen"],
            reason,
        ))
        connection.commit()
        return cursor.rowcount
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()

def main():
    if WINDOW_SECONDS <= 0 or PORT_THRESHOLD <= 0 or POLL_SECONDS <= 0:
        raise ValueError("Window, threshold, and poll interval must be positive.")

    print("SentinelAI Live Threat Detector", flush=True)
    print(f"Rule: {RULE_ID} — {RULE_NAME}", flush=True)
    print(
        f"Threshold: {PORT_THRESHOLD} distinct ports / "
        f"{WINDOW_SECONDS} seconds; poll every {POLL_SECONDS} seconds.",
        flush=True,
    )
    print("Alerts are investigation signals, not proof of an attack.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    connection = get_connection()
    try:
        while True:
            try:
                if not connection.is_connected():
                    connection.reconnect(attempts=3, delay=2)

                candidates = find_candidates(connection)
                for row in candidates:
                    save_candidate(connection, row)
                    print(
                        f"{datetime.now().isoformat(timespec='seconds')} | "
                        f"Potential port scan: {row['source_ip']} -> "
                        f"{row['destination_ip']} | "
                        f"{row['port_count']} distinct ports | "
                        f"protocols={row['protocols']}",
                        flush=True,
                    )

                if not candidates:
                    print(
                        f"{datetime.now().isoformat(timespec='seconds')} | "
                        "No port-scan threshold match in the recent-flow window.",
                        flush=True,
                    )
            except (Error, ValueError, TypeError, json.JSONDecodeError) as exc:
                print(f"Detection cycle failed: {exc}", flush=True)
                try:
                    connection.rollback()
                except Error:
                    pass

            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopping SentinelAI Live Threat Detector.", flush=True)
    finally:
        connection.close()

if __name__ == "__main__":
    main()
