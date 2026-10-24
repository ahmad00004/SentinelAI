import json
import os
from datetime import datetime, timezone
from pathlib import Path

from scapy.all import IP, IPv6, TCP, UDP, ICMP, sniff

INTERFACE = "en0"
OUTPUT_FILE = Path("network_telemetry.jsonl")


def extract_packet_metadata(packet):
    """Extract basic metadata without storing packet payloads."""
    timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        "timestamp": timestamp,
        "interface": INTERFACE,
        "source_ip": None,
        "destination_ip": None,
        "ip_version": None,
        "protocol": "OTHER",
        "source_port": None,
        "destination_port": None,
        "packet_size_bytes": len(packet),
    }

    if IP in packet:
        ip = packet[IP]
        record["source_ip"] = ip.src
        record["destination_ip"] = ip.dst
        record["ip_version"] = 4
        record["protocol"] = str(ip.proto)

    elif IPv6 in packet:
        ip = packet[IPv6]
        record["source_ip"] = ip.src
        record["destination_ip"] = ip.dst
        record["ip_version"] = 6
        record["protocol"] = str(ip.nh)

    if TCP in packet:
        record["protocol"] = "TCP"
        record["source_port"] = int(packet[TCP].sport)
        record["destination_port"] = int(packet[TCP].dport)

    elif UDP in packet:
        record["protocol"] = "UDP"
        record["source_port"] = int(packet[UDP].sport)
        record["destination_port"] = int(packet[UDP].dport)

    elif ICMP in packet:
        record["protocol"] = "ICMP"

    return record


def handle_packet(packet):
    record = extract_packet_metadata(packet)

    with OUTPUT_FILE.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record) + "\n")

    print(
        f'{record["timestamp"]} | '
        f'{record["protocol"]:5} | '
        f'{record["source_ip"]}:{record["source_port"]} -> '
        f'{record["destination_ip"]}:{record["destination_port"]} | '
        f'{record["packet_size_bytes"]} bytes',
        flush=True,
    )


def main():
    print("SentinelAI Network Monitor")
    print(f"Interface: {INTERFACE}")
    print(f"Output: {OUTPUT_FILE.resolve()}")
    print("Capturing packet metadata. Press Ctrl+C to stop.")
    print("Run with authorized packet-capture permissions if required.\n")

    try:
        sniff(
            iface=INTERFACE,
            prn=handle_packet,
            store=False,
        )
    except PermissionError:
        print(
            "Permission denied. Stop the monitor and restart it "
            "with the required macOS packet-capture permissions."
        )
    except OSError as exc:
        print(f"Capture error: {exc}")


if __name__ == "__main__":
    main()
