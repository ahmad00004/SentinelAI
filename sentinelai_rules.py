"""Configurable, explainable rules for normalized SentinelAI events."""

import json
import operator
from pathlib import Path
from typing import Any


RULES_PATH = Path(__file__).with_name("sentinelai_rules.json")
SEVERITIES = {"Low", "Medium", "High", "Critical"}
OPERATORS = {
    "eq": operator.eq,
    "neq": operator.ne,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
}


def load_rules(path: Path = RULES_PATH) -> list[dict[str, Any]]:
    """Load and validate the local rule configuration."""
    try:
        with path.open("r", encoding="utf-8") as rules_file:
            config = json.load(rules_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load SentinelAI rules from {path}: {exc}") from exc

    rules = config.get("rules") if isinstance(config, dict) else None
    if not isinstance(rules, list):
        raise RuntimeError("SentinelAI rules config must contain a 'rules' list.")

    seen_ids = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise RuntimeError("Each SentinelAI rule must be an object.")
        required = {"id", "name", "description", "field", "operator", "value", "severity", "enabled"}
        missing = required - rule.keys()
        if missing:
            raise RuntimeError(f"Rule is missing required fields: {sorted(missing)}")
        if not isinstance(rule["id"], str) or not rule["id"].strip():
            raise RuntimeError("Rule IDs must be non-empty strings.")
        if rule["id"] in seen_ids:
            raise RuntimeError(f"Duplicate SentinelAI rule ID: {rule['id']}")
        seen_ids.add(rule["id"])
        if rule["operator"] not in OPERATORS:
            raise RuntimeError(f"Unsupported operator in rule {rule['id']}.")
        if rule["severity"] not in SEVERITIES:
            raise RuntimeError(f"Unsupported severity in rule {rule['id']}.")
        if not isinstance(rule["enabled"], bool):
            raise RuntimeError(f"Rule {rule['id']} enabled must be true or false.")
    return rules


def evaluate_rules(features: dict[str, Any], rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return enabled rules that match, including evidence for alert triage."""
    matches = []
    for rule in rules:
        if not rule["enabled"]:
            continue

        field = rule["field"]
        if field not in features:
            continue

        observed = features[field]
        try:
            matched = OPERATORS[rule["operator"]](observed, rule["value"])
        except (TypeError, ValueError):
            matched = False

        if matched:
            matches.append({
                "rule_id": rule["id"],
                "name": rule["name"],
                "severity": rule["severity"],
                "field": field,
                "operator": rule["operator"],
                "observed_value": observed,
                "threshold": rule["value"],
                "description": rule["description"],
                "tuning_note": rule.get("tuning_note"),
            })
    return matches
