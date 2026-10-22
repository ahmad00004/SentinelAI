# SentinelAI

SentinelAI is an educational SIEM-style prototype for network-flow event detection and investigation. It combines a weighted binary Random Forest as the primary classifier, an Isolation Forest anomaly signal, configurable security rules, FastAPI ingestion, MySQL storage, and a Streamlit dashboard.

## Current scope

- The API accepts normalized UNSW-NB15-style flow features and records detections in MySQL.
- The Random Forest is the primary known-attack classifier. Isolation Forest is supplemental; it does not independently create alerts.
- Configurable rules can create alerts with severity and an explanation.
- The dashboard supports live API-event review, search and filters, investigation notes and status, and simulated responses.
- The response workflow only records simulations. It does not block traffic or change external systems.

This is a research and demonstration project, not a production SIEM. It does not capture or monitor a real network. Its evaluation uses UNSW-NB15 data; Fuzzer and Exploit detection limitations remain, and results may not generalize to operational traffic.

## Requirements

- Python 3.13
- MySQL Server
- Dependencies listed in `requirements.txt`

## Setup

Create and activate a virtual environment, then install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create the `sentinel_ai` MySQL database and ensure the existing `live_security_events` table is present. Set the database password without putting it in a source file:

```zsh
mysql -u root -p < schema.sql
unset SENTINELAI_DB_PASSWORD
read -s "SENTINELAI_DB_PASSWORD?MySQL password: "
export SENTINELAI_DB_PASSWORD
python migrate_live_security_events.py
```

Run the API in one terminal:

```bash
uvicorn sentinelai_api:app --reload
```

Run the dashboard in another terminal:

```bash
source .venv/bin/activate
streamlit run dashboard.py
```

The dashboard asks for the MySQL password in its sidebar. The API health endpoints are `/` and `/health/db`; interactive API documentation is at `/docs`.

## Model and data files

The two `.joblib` files in the project are required by the API. The dashboard also reads the weighted Random Forest validation CSV reports. The UNSW-NB15 CSV files support the experiments and label-quality audit. Do not treat the dataset as live telemetry.

The UNSW-NB15 source files are not committed here. Download the training and testing partitions from the [official UNSW-NB15 dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset). The dataset page grants academic research use and requests citation of its associated publications; follow its citation guidance in reports and presentations.

## Label-quality audit

Run the review-only audit with bounded memory:

```bash
python sentinelai_label_quality_audit.py --chunk-size 5000
```

It writes a JSON summary and a CSV of identical feature patterns with conflicting attack-category labels. It does not alter labels or train a model.

## Secrets and local files

Keep database passwords and other credentials out of source control. `.gitignore` excludes the virtual environment, Python bytecode, local environment files, and temporary SQLite files.
