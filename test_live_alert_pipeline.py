import json
import os
import sys
from datetime import datetime, timedelta

import mysql.connector
import sentinelai_live_threat_detector as detector

TEST_SOURCE = "192.0.2.10"
TEST_DESTINATION = "192.0.2.20"
TEST_PREFIX = "SENTINELAI-TEST-"

if not os.getenv("SENTINELAI_DB_PASSWORD"):
    sys.exit("ERROR: SENTINELAI_DB_PASSWORD is not set.")

if detector.PORT_THRESHOLD != 25 or detector.WINDOW_SECONDS != 60:
    sys.exit(
        f"Unexpected detector settings: threshold={detector.PORT_THRESHOLD}, "
        f"window={detector.WINDOW_SECONDS}. No test data inserted."
    )

connection = detector.get_connection()
cursor = connection.cursor(dictionary=True)

try:
    cursor.execute(
        """
        SELECT COUNT(*) AS n
        FROM network_flows
        WHERE flow_key LIKE %s
        """,
        (TEST_PREFIX + "%",),
    )
    existing = cursor.fetchone()["n"]
    if existing:
        sys.exit(
            f"Safety stop: found {existing} existing test flows. "
            "No records inserted."
        )

    now = datetime.utcnow()
    first_seen = now - timedelta(seconds=20)

    insert_sql = """
        INSERT INTO network_flows (
            source_ip, destination_ip, protocol, source_port,
            destination_port, interface_name, packet_count,
            total_bytes, duration_seconds, first_seen, last_seen,
            flow_key
        )
        VALUES (
            %s, %s, 'TCP', %s, %s, 'TEST-ONLY', 1,
            60, 0.1, %s, %s, %s
        )
    """

    for port in range(1, detector.PORT_THRESHOLD + 1):
        flow_key = f"{TEST_PREFIX}{port:02d}".ljust(64, "X")
        cursor.execute(
            insert_sql,
            (
                TEST_SOURCE,
                TEST_DESTINATION,
                40000 + port,
                port,
                first_seen,
                now,
                flow_key,
            ),
        )

    connection.commit()
    print(f"Inserted {detector.PORT_THRESHOLD} marked test flows.")

    candidates = detector.find_candidates(connection)
    matching = [
        row for row in candidates
        if row["source_ip"] == TEST_SOURCE
        and row["destination_ip"] == TEST_DESTINATION
    ]

    if not matching:
        raise RuntimeError(
            "The detector did not return the synthetic test candidate."
        )

    candidate = matching[0]
    print(
        "Detector candidate:",
        candidate["source_ip"],
        "->",
        candidate["destination_ip"],
        "| distinct ports:",
        candidate["port_count"],
    )

    detector.save_candidate(connection, candidate)
    print("Called the production detector's save_candidate() function.")

    cursor.execute(
        """
        SELECT alert_id, rule_id, rule_name, severity, status,
               source_ip, destination_ip, distinct_destination_ports,
               window_seconds, detection_reason
        FROM network_threat_alerts
        WHERE rule_id = %s
          AND source_ip = %s
          AND destination_ip = %s
        ORDER BY alert_id DESC
        LIMIT 5
        """,
        (detector.RULE_ID, TEST_SOURCE, TEST_DESTINATION),
    )
    alerts = cursor.fetchall()
    if not alerts:
        raise RuntimeError(
            "No saved alert found for the synthetic test pair."
        )

    print("Saved alert record:")
    print(json.dumps(alerts[0], default=str, indent=2))
    print(
        "\nTEST FLOWS ARE STILL PRESENT for dashboard verification. "
        "Do not rerun this script until cleanup is complete."
    )

except Exception:
    connection.rollback()
    raise
finally:
    cursor.close()
    connection.close()
