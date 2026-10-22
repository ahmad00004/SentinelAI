import time
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8001/rf-test/events"

print("Loading UNSW-NB15 testing dataset...")
df = pd.read_csv("UNSW_NB15_testing-set.csv", low_memory=False)
df["attack_cat"] = df["attack_cat"].astype(str).str.strip()

# Test 20 Normal records and up to 20 attack records.
normal = df[df["attack_cat"] == "Normal"].head(20)

attacks = (
    df[df["attack_cat"] != "Normal"]
    .groupby("attack_cat", group_keys=False)
    .head(2)
    .head(20)
)

sample = pd.concat([normal, attacks], ignore_index=True)
results = []

print(f"Selected {len(normal)} Normal and {len(attacks)} attack records.")
print("Sending requests to the Random Forest Test API...\n")

for i, (_, row) in enumerate(sample.iterrows(), start=1):
    actual = row["attack_cat"]

    features = row.drop(
        labels=["id", "label", "attack_cat"],
        errors="ignore",
    ).to_dict()

    try:
        response = requests.post(
            API_URL,
            json={"features": features},
            timeout=30,
        )

        if response.status_code != 200:
            print(f"{i:02d}. HTTP {response.status_code}: {response.text}")
            continue

        result = response.json()
        predicted = result["predicted_category"]
        status = result["status"]

        results.append({
            "actual": actual,
            "predicted": predicted,
            "status": status,
            "score": result["suspicion_score"],
            "binary_correct": (
                (actual != "Normal") == (status == "Suspicious")
            ),
        })

        print(
            f"{i:02d}. Actual={actual:<18} "
            f"Predicted={predicted:<18} "
            f"Status={status:<10} "
            f"Score={result['suspicion_score']:.4f}"
        )

    except requests.RequestException as exc:
        print(f"{i:02d}. Request failed: {exc}")
        break

    time.sleep(0.05)

if not results:
    raise SystemExit(
        "\nNo successful API responses. Check that port 8001 is running."
    )

out = pd.DataFrame(results)
out.to_csv("sentinelai_rf_api_batch_results.csv", index=False)

normal_rows = out[out["actual"] == "Normal"]
attack_rows = out[out["actual"] != "Normal"]

print("\n========== BATCH TEST SUMMARY ==========")
print(f"Successful API requests: {len(out)}")
print(f"Normal records tested: {len(normal_rows)}")
print(
    "Normal records incorrectly flagged: "
    f"{int((normal_rows['status'] == 'Suspicious').sum())}"
)
print(f"Attack records tested: {len(attack_rows)}")
print(
    "Attack records detected: "
    f"{int((attack_rows['status'] == 'Suspicious').sum())}"
)

print("\nResults by actual category:")
print(
    out.groupby("actual")
    .agg(
        records=("predicted", "size"),
        flagged_suspicious=("status", lambda x: (x == "Suspicious").sum()),
    )
    .to_string()
)

print("\nSaved results: sentinelai_rf_api_batch_results.csv")
