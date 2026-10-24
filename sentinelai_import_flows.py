import json
import os
from datetime import datetime
from pathlib import Path

import mysql.connector

INPUT_FILE = Path("network_flows.jsonl")


def parse_datetime(value):
    # MySQL DATETIME does not store timezone information.
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=None)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    password = os.getenv("SENTINELAI_DB_PASSWORD")
    if not password:
        raise RuntimeError(
            "SENTINELAI_DB_PASSWORD is not set. "
            "Set it in your terminal before importing."
        )

    connection = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password=password,
        database="sentinel_ai",
    )

    insert_sql = """
        INSERT INTO network_flows (
            source_ip, destination_ip, protocol,
            source_port, destination_port, interface_name,
            packet_count, total_bytes, duration_seconds,
            first_seen, last_seen
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    imported = 0

    try:
        cursor = connection.cursor()

        with INPUT_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                flow = json.loads(line)

                values = (
                    flow["source_ip"],
                    flow["destination_ip"],
                    flow["protocol"],
                    flow["source_port"],
                    flow["destination_port"],
                    flow["interface"],
                    flow["packet_count"],
                    flow["total_bytes"],
                    flow["duration_seconds"],
                    parse_datetime(flow["first_seen"]),
                    parse_datetime(flow["last_seen"]),
                )

                cursor.execute(insert_sql, values)
                imported += 1

        connection.commit()
        print("Flows imported:", imported)
        print("Destination table: sentinel_ai.network_flows")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()
