# SentinelAI Live Alert Pipeline Test

## Objective
Validate the existing network detector, MySQL alert storage,
Streamlit dashboard display, and test-data cleanup.

## Detection rule
- Rule ID: LIVE-PORT-001
- Rule: Possible TCP/UDP port scan
- Threshold: 25 distinct destination ports within 60 seconds

## Test results
1. Initial alert count: 0.
2. Inserted 25 marked synthetic TCP flow records.
3. Existing find_candidates() function detected 25 distinct ports.
4. Existing save_candidate() function saved alert ID 1.
5. Saved alert severity: Medium.
6. Saved alert status: New.
7. Dashboard displayed the alert under Live Network Threat Alerts.
8. Cleanup verification: 0 marked test flows remaining.
9. Cleanup verification: 0 matching test alerts remaining.

## Outcome
PASS — controlled detector-to-database-to-dashboard pipeline
validated, and synthetic test records removed.

## Limitations
This was a synthetic test using documentation-only IP addresses.
It does not prove a real attack occurred or validate live packet
capture end to end. The detection threshold and ML models were
not changed or retrained. A non-fatal datetime.utcnow() deprecation
warning appeared during testing.

## Evidence
Retain the dashboard screenshot showing the alert, terminal output
showing detection and saving, and MySQL output showing cleanup.
