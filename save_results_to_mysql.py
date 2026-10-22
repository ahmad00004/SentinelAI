import getpass
import pandas as pd
import mysql.connector

CSV_FILE = "sentinelai_evaluation_results.csv"
TABLE_NAME = "unsw_nb15_detections"
BATCH_SIZE = 5000

print("Loading SentinelAI evaluation results...")
df = pd.read_csv(CSV_FILE)

required = [
    "id", "actual_label", "predicted_label",
    "anomaly_score", "attack_cat", "status"
]

if not set(required).issubset(df.columns):
    raise ValueError("The results CSV is missing required columns.")

print("Rows to import:", len(df))

password = getpass.getpass("MySQL root password: ")

connection = mysql.connector.connect(
    host="127.0.0.1",
    port=3306,
    user="root",
    password=password,
    database="sentinel_ai"
)

cursor = connection.cursor()

# A dedicated table preserves the original prototype tables.
cursor.execute("""
CREATE TABLE IF NOT EXISTS unsw_nb15_detections (
    dataset_id INT NOT NULL PRIMARY KEY,
    actual_label TINYINT NOT NULL,
    predicted_label TINYINT NOT NULL,
    anomaly_score DOUBLE NOT NULL,
    attack_cat VARCHAR(100),
    status VARCHAR(20) NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
connection.commit()

sql = """
INSERT INTO unsw_nb15_detections
(dataset_id, actual_label, predicted_label, anomaly_score,
 attack_cat, status)
VALUES (%s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
actual_label = VALUES(actual_label),
predicted_label = VALUES(predicted_label),
anomaly_score = VALUES(anomaly_score),
attack_cat = VALUES(attack_cat),
status = VALUES(status),
detected_at = CURRENT_TIMESTAMP
"""

records = [
    (
        int(row.id),
        int(row.actual_label),
        int(row.predicted_label),
        float(row.anomaly_score),
        str(row.attack_cat)[:100],
        str(row.status)
    )
    for row in df[required].itertuples(index=False)
]

try:
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        cursor.executemany(sql, batch)
        connection.commit()
        print(
            f"Saved {min(start + len(batch), len(records))}"
            f" / {len(records)} records"
        )

    cursor.execute("""
        SELECT COUNT(*) FROM unsw_nb15_detections
    """)
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT status, COUNT(*)
        FROM unsw_nb15_detections
        GROUP BY status
        ORDER BY status
    """)
    summaries = cursor.fetchall()

    print("\n===== MYSQL STORAGE SUMMARY =====")
    print("Total records in table:", total)
    for status, count in summaries:
        print(f"{status}: {count}")

    print("\nMySQL import completed successfully.")

except Exception:
    connection.rollback()
    raise

finally:
    cursor.close()
    connection.close()
