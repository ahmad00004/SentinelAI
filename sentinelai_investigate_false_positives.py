import pandas as pd

results = pd.read_csv("sentinelai_rf_api_batch_results.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv", low_memory=False)

# The batch test used the first 20 Normal rows, followed by attack rows.
normal = test[
    test["attack_cat"].astype(str).str.strip() == "Normal"
].head(20).copy()

normal = normal.reset_index(drop=True)
normal["batch_row"] = range(1, len(normal) + 1)

normal_results = results[results["actual"] == "Normal"].copy()
normal_results = normal_results.reset_index(drop=True)
normal_results["batch_row"] = range(1, len(normal_results) + 1)

false_positives = normal_results[
    normal_results["status"] == "Suspicious"
].copy()

print("\n=== NORMAL EVENTS FLAGGED AS SUSPICIOUS ===")
print(
    false_positives[
        ["batch_row", "actual", "predicted", "score", "status"]
    ].to_string(index=False)
)

# Match these batch rows to the corresponding original dataset records.
matched = normal.merge(
    false_positives[["batch_row", "predicted", "score"]],
    on="batch_row",
    how="inner",
)

exclude = {"id", "label", "attack_cat"}
feature_columns = [
    c for c in test.columns if c not in exclude
]

for col in feature_columns:
    matched[col] = pd.to_numeric(matched[col], errors="coerce")

print("\n=== FALSE POSITIVES: FEATURE VALUES ===")
print("Only showing features that vary across these records.")
if len(matched):
    varying = [
        c for c in feature_columns
        if matched[c].nunique(dropna=False) > 1
    ]
    if varying:
        print(
            matched[
                ["batch_row", "predicted", "score"] + varying
            ].to_string(index=False)
        )
    else:
        print("No varying feature values found.")
else:
    print("No false-positive records were matched.")

matched.to_csv(
    "sentinelai_rf_false_positive_details.csv",
    index=False,
)
print("\nSaved: sentinelai_rf_false_positive_details.csv")
