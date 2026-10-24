import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

INPUT_FILE = Path("network_telemetry.jsonl")
OUTPUT_FILE = Path("network_flows.jsonl")


def parse_timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def main():
    flows = defaultdict(
        lambda: {
            "packet_count": 0,
            "total_bytes": 0,
            "first_seen": None,
            "last_seen": None,
            "interface": None,
        }
    )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            packet = json.loads(line)

            key = (
                packet.get("source_ip"),
                packet.get("destination_ip"),
                packet.get("protocol"),
                packet.get("source_port"),
                packet.get("destination_port"),
            )

            timestamp = parse_timestamp(packet["timestamp"])
            flow = flows[key]

            flow["packet_count"] += 1
            flow["total_bytes"] += int(packet.get("packet_size_bytes", 0))
            flow["interface"] = packet.get("interface")

            if flow["first_seen"] is None or timestamp < flow["first_seen"]:
                flow["first_seen"] = timestamp

            if flow["last_seen"] is None or timestamp > flow["last_seen"]:
                flow["last_seen"] = timestamp

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for key, flow in flows.items():
            first_seen = flow["first_seen"]
            last_seen = flow["last_seen"]

            record = {
                "source_ip": key[0],
                "destination_ip": key[1],
                "protocol": key[2],
                "source_port": key[3],
                "destination_port": key[4],
                "interface": flow["interface"],
                "packet_count": flow["packet_count"],
                "total_bytes": flow["total_bytes"],
                "duration_seconds": round(last_seen - first_seen, 6),
                "first_seen": datetime.fromtimestamp(
                    first_seen
                ).astimezone().isoformat(),
                "last_seen": datetime.fromtimestamp(
                    last_seen
                ).astimezone().isoformat(),
            }

            file.write(json.dumps(record) + "\n")

    print("Input packets:", sum(f["packet_count"] for f in flows.values()))
    print("Unique directional flows:", len(flows))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
