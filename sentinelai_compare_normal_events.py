import pandas as pd

results = pd.read_csv("sentinelai_rf_api_batch_results.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv", low_memory=False)

# The batch API test used the first 20 Normal records.
normal = test[
    test["attack_cat"].astype(str).str.strip() == "Normal"
].head(20).copy().reset_index(drop=True)

normal["batch_row"] = range(1, len(normal) + 1)

# Match predictions to the same 20 Normal records.
normal_results = results[
    results["actual"].astype(str).str.strip() == "Normal"
].copy().reset_index(drop=True)

normal_results["batch_row"] = range(1, len(normal_results) + 1)

comparison = normal.merge(
    normal_results[["batch_row", "predicted", "score", "status"]],
    on="batch_row",
    how="inner"
)

comparison["result"] = comparison["status"].apply(
    lambda x: "False Positive" if x == "Suspicious"
    else "Correct Normal"
)

print("\n=== ALL 20 NORMAL EVENTS ===")
print(
    comparison[
        ["batch_row", "proto", "service", "state",
         "predicted", "score", "result"]
    ].to_string(index=False)
)

# Compare numerical features between the two groups.
exclude = {
    "id", "label", "attack_cat", "batch_row",
    "predicted", "score", "status", "result"
}

numeric_columns = [
    c for c in comparison.columns
    if c not in exclude
    and pd.api.types.is_numeric_dtype(comparison[c])
]

fp = comparison[comparison["result"] == "False Positive"]
correct = comparison[comparison["result"] == "Correct Normal"]

print("\n=== NUMERIC FEATURE COMPARISON ===")
print("Compare these values as clues, not proof of causation.")

rows = []
for col in numeric_columns:
    a = fp[col].dropna()
    b = correct[col].dropna()

    if len(a) and len(b):
        rows.append({
            "feature": col,
            "false_positive_median": round(a.median(), 4),
            "correct_normal_median": round(b.median(), 4),
            "false_positive_mean": round(a.mean(), 4),
            "correct_normal_mean": round(b.mean(), 4),
        })

summary = pd.DataFrame(rows)
if not summary.empty:
    summary["median_difference"] = (
        summary["false_positive_median"]
        - summary["correct_normal_median"]
    )
    summary["absolute_difference"] = (
        summary["median_difference"].abs()
    )
    print(
        summary.sort_values(
            "absolute_difference", ascending=False
        ).head(15).drop(
            columns=["absolute_difference"]
        ).to_string(index=False)
    )

comparison.to_csv(
    "sentinelai_normal_event_comparison.csv", index=False
)
print("\nSaved: sentinelai_normal_event_comparison.csv")
