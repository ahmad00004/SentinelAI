import json
import pandas as pd

df = pd.read_csv("UNSW_NB15_testing-set.csv", nrows=1)
features = df.drop(
    columns=["id", "label", "attack_cat"],
    errors="ignore",
).iloc[0].to_dict()

with open("sentinelai_rf_test_request.json", "w") as f:
    json.dump({"features": features}, f, default=str, allow_nan=False)

print("Created sentinelai_rf_test_request.json")
print("Actual dataset category:", df.iloc[0]["attack_cat"])
