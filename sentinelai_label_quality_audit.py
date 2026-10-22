"""Bounded-memory audit for conflicting labels on identical UNSW features.

This is a review aid only. It never changes labels, trains a model, or edits
the source dataset. Rows are read in small chunks; a temporary SQLite database
stores per-feature signatures so duplicates spanning chunks are detected.

Example:
    python sentinelai_label_quality_audit.py
    python sentinelai_label_quality_audit.py --chunk-size 5000
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(__file__).with_name("UNSW_NB15_testing-set.csv")
DEFAULT_SUMMARY = Path(__file__).with_name("sentinelai_label_quality_summary.json")
DEFAULT_GROUPS = Path(__file__).with_name("sentinelai_label_conflict_groups.csv")
TARGET_COLUMNS = {"id", "label", "attack_cat"}
KNOWN_CATEGORICAL = {"proto", "service", "state"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--conflicts", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument(
        "--chunk-size", type=int, default=5000,
        help="Rows held in memory per input chunk (default: 5000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")
    if not args.input.is_file():
        raise SystemExit(f"Input CSV not found: {args.input}")

    probe = pd.read_csv(args.input, nrows=1000, low_memory=False)
    required = {"label", "attack_cat"}
    if not required.issubset(probe.columns):
        raise SystemExit("Input must contain 'label' and 'attack_cat' columns")
    feature_columns = [c for c in probe.columns if c not in TARGET_COLUMNS]
    if not feature_columns:
        raise SystemExit("No feature columns remain after excluding labels and ID")

    # Fix dtypes across chunks so an integer-valued feature has the same
    # signature whether a particular chunk contains missing values or not.
    dtype_map: dict[str, str] = {
        c: "string" for c in KNOWN_CATEGORICAL if c in probe.columns
    }
    dtype_map.update({c: "float64" for c in feature_columns if c not in dtype_map})
    dtype_map["attack_cat"] = "string"
    if "id" in probe.columns:
        dtype_map["id"] = "string"
    if "label" in probe.columns:
        dtype_map["label"] = "string"

    select_columns = feature_columns + ["attack_cat"]
    if "id" in probe.columns:
        select_columns.append("id")

    with tempfile.TemporaryDirectory(prefix="sentinelai-label-audit-") as temp_dir:
        database_path = Path(temp_dir) / "feature-label-groups.sqlite"
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute(
            """CREATE TABLE feature_labels (
                   signature TEXT NOT NULL,
                   category TEXT NOT NULL,
                   row_count INTEGER NOT NULL,
                   PRIMARY KEY (signature, category)
               )"""
        )

        records_scanned = 0
        for chunk in pd.read_csv(
            args.input,
            usecols=select_columns,
            dtype=dtype_map,
            chunksize=args.chunk_size,
            low_memory=False,
        ):
            categories = chunk["attack_cat"].fillna("Unknown").str.strip()
            signature_values = pd.util.hash_pandas_object(
                chunk[feature_columns], index=False
            ).map(lambda value: f"{int(value):016x}")
            grouped = (
                pd.DataFrame({"signature": signature_values, "category": categories})
                .groupby(["signature", "category"], dropna=False)
                .size()
                .rename("row_count")
                .reset_index()
            )
            connection.executemany(
                """INSERT INTO feature_labels(signature, category, row_count)
                   VALUES (?, ?, ?)
                   ON CONFLICT(signature, category) DO UPDATE SET
                       row_count = row_count + excluded.row_count""",
                (
                    (row.signature, row.category, int(row.row_count))
                    for row in grouped.itertuples(index=False)
                ),
            )
            records_scanned += len(chunk)
            connection.commit()

        summary = {
            # Store only the source filename; do not leak a machine-specific
            # absolute path into a report that may be shared or committed.
            "input_file": args.input.name,
            "records_scanned": records_scanned,
            "feature_count": len(feature_columns),
            "chunk_size": args.chunk_size,
            "distinct_feature_patterns": 0,
            "repeated_feature_patterns": 0,
            "rows_in_repeated_patterns": 0,
            "cross_category_pattern_groups": 0,
            "rows_in_cross_category_groups": 0,
            "normal_attack_conflict_groups": 0,
            "fuzzer_normal_conflict_groups": 0,
            "fuzzer_rows_sharing_pattern_with_normal": 0,
            "normal_rows_sharing_pattern_with_fuzzer": 0,
            "interpretation": (
                "Review-only label/feature ambiguity report. No labels are "
                "changed and no model is trained. Identical feature patterns "
                "with different labels cannot be separated using these inputs alone."
            ),
        }

        aggregate = connection.execute(
            """SELECT COUNT(*),
                      COALESCE(SUM(CASE WHEN total_rows > 1 THEN 1 ELSE 0 END), 0),
                      COALESCE(SUM(CASE WHEN total_rows > 1 THEN total_rows ELSE 0 END), 0),
                      COALESCE(SUM(CASE WHEN category_count > 1 THEN 1 ELSE 0 END), 0),
                      COALESCE(SUM(CASE WHEN category_count > 1 THEN total_rows ELSE 0 END), 0)
               FROM (
                   SELECT signature, SUM(row_count) AS total_rows,
                          COUNT(*) AS category_count
                   FROM feature_labels GROUP BY signature
               )"""
        ).fetchone()
        (
            summary["distinct_feature_patterns"],
            summary["repeated_feature_patterns"],
            summary["rows_in_repeated_patterns"],
            summary["cross_category_pattern_groups"],
            summary["rows_in_cross_category_groups"],
        ) = map(int, aggregate)

        conflict_query = """SELECT signature,
                   SUM(row_count) AS total_rows,
                   SUM(CASE WHEN category = 'Normal' THEN row_count ELSE 0 END) AS normal_rows,
                   SUM(CASE WHEN category = 'Fuzzers' THEN row_count ELSE 0 END) AS fuzzer_rows,
                   COUNT(*) AS category_count,
                   group_concat(category || ':' || row_count, '|') AS category_counts
            FROM feature_labels
            GROUP BY signature
            HAVING COUNT(*) > 1
            ORDER BY total_rows DESC, signature"""

        args.conflicts.parent.mkdir(parents=True, exist_ok=True)
        pd.read_sql_query(conflict_query, connection).to_csv(args.conflicts, index=False)

        normal_attack = connection.execute(
            """SELECT COUNT(*) FROM (
                   SELECT signature
                   FROM feature_labels
                   GROUP BY signature
                   HAVING SUM(CASE WHEN category = 'Normal' THEN row_count ELSE 0 END) > 0
                      AND SUM(CASE WHEN category <> 'Normal' THEN row_count ELSE 0 END) > 0
               )"""
        ).fetchone()[0]
        fuzzer_normal = connection.execute(
            """SELECT COUNT(*),
                      COALESCE(SUM(fuzzer_rows), 0),
                      COALESCE(SUM(normal_rows), 0)
               FROM (
                   SELECT signature,
                          SUM(CASE WHEN category = 'Fuzzers' THEN row_count ELSE 0 END) AS fuzzer_rows,
                          SUM(CASE WHEN category = 'Normal' THEN row_count ELSE 0 END) AS normal_rows
                   FROM feature_labels GROUP BY signature
                   HAVING SUM(CASE WHEN category = 'Fuzzers' THEN row_count ELSE 0 END) > 0
                      AND SUM(CASE WHEN category = 'Normal' THEN row_count ELSE 0 END) > 0
               )"""
        ).fetchone()
        summary["normal_attack_conflict_groups"] = int(normal_attack)
        summary["fuzzer_normal_conflict_groups"] = int(fuzzer_normal[0])
        summary["fuzzer_rows_sharing_pattern_with_normal"] = int(fuzzer_normal[1])
        summary["normal_rows_sharing_pattern_with_fuzzer"] = int(fuzzer_normal[2])
        connection.close()

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Conflicting feature groups: {args.conflicts}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
