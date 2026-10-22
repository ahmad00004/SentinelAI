import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

# In this project folder the train/test filenames contain the opposite
# canonical UNSW-NB15 partitions. The 175,341-row file is the true training set.
TRAIN_FILE = "UNSW_NB15_testing-set.csv"
print(f"Loading canonical UNSW-NB15 training dataset from {TRAIN_FILE}...")
train = pd.read_csv(TRAIN_FILE, low_memory=False)
if len(train) != 175_341:
    raise ValueError(
        f"Expected 175,341 canonical training rows in {TRAIN_FILE}; "
        f"found {len(train):,}. Verify the dataset files before training."
    )

excluded = ["id", "label", "attack_cat"]
X_all = train.drop(columns=excluded)

# Train the anomaly detector using normal traffic only.
normal_mask = train["label"].astype(int) == 0
X_train = X_all.loc[normal_mask].copy()

categorical_features = ["proto", "service", "state"]
numeric_features = [
    col for col in X_train.columns
    if col not in categorical_features
]

print(f"Normal training rows: {len(X_train)}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print("Training Isolation Forest...")

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("categorical",
     OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("detector", IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline.fit(X_train)

artifact = {
    "model": pipeline,
    "feature_columns": X_train.columns.tolist(),
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "threshold": 0.0,
    "training_source_file": TRAIN_FILE,
    "training_record_count": int(len(train)),
    "role": "Supplemental anomaly signal; does not independently create attack alerts.",
}

joblib.dump(artifact, "sentinelai_model.joblib")

print("Saved model: sentinelai_model.joblib")
print("Expected input features:", len(artifact["feature_columns"]))
print("Model is ready for real-time scoring.")
