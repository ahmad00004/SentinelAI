import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import mysql.connector
from mysql.connector import Error

INPUT_FILE = Path("network_telemetry.jsonl")
POLL_INTERVAL_SECONDS = 1
FLUSH_INTERVAL_SECONDS = 5
RETRY_INTERVAL_SECONDS = 3

# Aggregates directional flows in memory until committed to MySQL.
flows = {}


def parse_timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        # Store UTC-naive timestamps consistently for MySQL DATETIME.
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def flow_key(packet):
    fields = (
        packet.get("interface"),
        packet.get("source_ip"),
        packet.get("destination_ip"),
        packet.get("protocol"),
        packet.get("source_port"),
        packet.get("destination_port"),
    )
    canonical = json.dumps(fields, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_connection():
    password = os.getenv("SENTINELAI_DB_PASSWORD")
    if not password:
        raise RuntimeError(
            "SENTINELAI_DB_PASSWORD is not set. "
            "Set it in this terminal before starting the processor."
        )

    return mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password=password,
        database="sentinel_ai",
        connection_timeout=10,
    )


def update_flow(packet):
    key = flow_key(packet)
    timestamp = parse_timestamp(packet["timestamp"])
    size = int(packet.get("packet_size_bytes", 0))

    if size < 0:
        return

    if key not in flows:
        flows[key] = {
            "flow_key": key,
            "interface": packet.get("interface"),
            "source_ip": packet.get("source_ip"),
            "destination_ip": packet.get("destination_ip"),
            "protocol": packet.get("protocol"),
            "source_port": packet.get("source_port"),
            "destination_port": packet.get("destination_port"),
            "packet_count": 0,
            "total_bytes": 0,
            "first_seen": timestamp,
            "last_seen": timestamp,
        }

    flow = flows[key]
    flow["packet_count"] += 1
    flow["total_bytes"] += size
    flow["first_seen"] = min(flow["first_seen"], timestamp)
    flow["last_seen"] = max(flow["last_seen"], timestamp)
    flow["duration_seconds"] = (
        flow["last_seen"] - flow["first_seen"]
    ).total_seconds()


def flush_flows(connection):
    if not flows:
        return

    sql = """
        INSERT INTO network_flows (
            flow_key, source_ip, destination_ip, protocol,
            source_port, destination_port, interface_name,
            packet_count, total_bytes, duration_seconds,
            first_seen, last_seen
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            packet_count = packet_count + VALUES(packet_count),
            total_bytes = total_bytes + VALUES(total_bytes),
            first_seen = LEAST(first_seen, VALUES(first_seen)),
            last_seen = GREATEST(last_seen, VALUES(last_seen)),
            duration_seconds = TIMESTAMPDIFF(
                MICROSECOND,
                LEAST(first_seen, VALUES(first_seen)),
                GREATEST(last_seen, VALUES(last_seen))
            ) / 1000000
    """

    rows = [
        (
            flow["flow_key"],
            flow["source_ip"],
            flow["destination_ip"],
            flow["protocol"],
            flow["source_port"],
            flow["destination_port"],
            flow["interface"],
            flow["packet_count"],
            flow["total_bytes"],
            flow["duration_seconds"],
            flow["first_seen"],
            flow["last_seen"],
        )
        for flow in flows.values()
    ]

    cursor = connection.cursor()
    try:
        cursor.executemany(sql, rows)
        connection.commit()
        print(
            f"{datetime.now().isoformat(timespec='seconds')} | "
            f"Committed {len(rows)} directional flow updates to MySQL",
            flush=True,
        )
        flows.clear()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}. Start the network monitor first."
        )

    print("SentinelAI Live Flow Processor")
    print(f"Input: {INPUT_FILE.resolve()}")
    print("Reading only newly appended packet records.")
    print("Press Ctrl+C to stop.")

    connection = get_connection()

    # Begin at EOF to avoid importing the historical capture again.
    position = INPUT_FILE.stat().st_size
    file_signature = (INPUT_FILE.stat().st_dev, INPUT_FILE.stat().st_ino)
    pending_line = b""
    last_flush = time.monotonic()

    try:
        while True:
            try:
                stat = INPUT_FILE.stat()
                current_signature = (stat.st_dev, stat.st_ino)

                # Handle file replacement or truncation explicitly.
                if current_signature != file_signature or stat.st_size < position:
                    print(
                        "Capture file was replaced or truncated; "
                        "starting at its current end to avoid replaying old packets.",
                        flush=True,
                    )
                    file_signature = current_signature
                    position = stat.st_size
                    pending_line = b""

                with INPUT_FILE.open("rb") as file:
                    file.seek(position)

                    while True:
                        raw_line = file.readline()
                        if not raw_line:
                            break

                        position = file.tell()

                        # Do not parse a line until its newline arrives.
                        if not raw_line.endswith(b"\n"):
                            pending_line += raw_line
                            break

                        complete_line = pending_line + raw_line
                        pending_line = b""

                        if not complete_line.strip():
                            continue

                        try:
                            packet = json.loads(complete_line.decode("utf-8"))
                            update_flow(packet)
                        except (
                            json.JSONDecodeError,
                            UnicodeDecodeError,
                            KeyError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            print(
                                f"Skipping invalid packet record: {exc}",
                                flush=True,
                            )

                if time.monotonic() - last_flush >= FLUSH_INTERVAL_SECONDS:
                    if not connection.is_connected():
                        connection.reconnect(attempts=3, delay=2)

                    flush_flows(connection)
                    last_flush = time.monotonic()

            except (Error, OSError) as exc:
                print(
                    f"Temporary input/database error: {exc}. "
                    f"Retrying in {RETRY_INTERVAL_SECONDS} seconds; "
                    "buffered flows are retained.",
                    flush=True,
                )
                try:
                    if not connection.is_connected():
                        connection.reconnect(attempts=3, delay=2)
                except Error:
                    pass
                time.sleep(RETRY_INTERVAL_SECONDS)

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nStopping processor and attempting final flush...", flush=True)
        try:
            if not connection.is_connected():
                connection.reconnect(attempts=3, delay=2)
            flush_flows(connection)
        except Error as exc:
            print(
                f"Final flush failed: {exc}. Buffered data was not cleared "
                "from memory; keep the process output for troubleshooting.",
                flush=True,
            )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
