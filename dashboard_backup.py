import os
import pandas as pd
import streamlit as st
import mysql.connector

st.set_page_config(
    page_title="SentinelAI Security Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SentinelAI")
st.subheader("Intelligent Threat Detection Dashboard")
st.caption("UNSW-NB15 | Isolation Forest | MySQL")

# Database connection settings
st.sidebar.header("Database Connection")
db_password = st.sidebar.text_input(
    "MySQL root password",
    type="password"
)

if not db_password:
    st.info("Enter your MySQL password in the sidebar to load security data.")
    st.stop()

try:
    conn = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password=db_password,
        database="sentinel_ai"
    )

    # Summary statistics
    summary_query = """
        SELECT
            COUNT(*) AS total,
            SUM(predicted_label = 1) AS suspicious,
            SUM(predicted_label = 0) AS normal_count,
            SUM(actual_label = 1 AND predicted_label = 1) AS true_positive,
            SUM(actual_label = 0 AND predicted_label = 1) AS false_positive,
            SUM(actual_label = 1 AND predicted_label = 0) AS false_negative,
            SUM(actual_label = 0 AND predicted_label = 0) AS true_negative
        FROM unsw_nb15_detections
    """
    summary = pd.read_sql(summary_query, conn).iloc[0]

    total = int(summary["total"] or 0)
    suspicious = int(summary["suspicious"] or 0)
    normal_count = int(summary["normal_count"] or 0)
    tp = int(summary["true_positive"] or 0)
    fp = int(summary["false_positive"] or 0)
    fn = int(summary["false_negative"] or 0)
    tn = int(summary["true_negative"] or 0)

    if total == 0:
        st.warning("The detection table is empty.")
        conn.close()
        st.stop()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", f"{total:,}")
    col2.metric("Suspicious Events", f"{suspicious:,}")
    col3.metric("Normal Predictions", f"{normal_count:,}")
    col4.metric("Detection Rate (Recall)", f"{tp / (tp + fn):.2%}" if tp + fn else "N/A")

    st.divider()
    st.subheader("Model Evaluation")

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )
    fpr = fp / (fp + tn) if fp + tn else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{precision:.2%}")
    m2.metric("Recall", f"{recall:.2%}")
    m3.metric("F1-score", f"{f1:.2%}")
    m4.metric("False-positive Rate", f"{fpr:.2%}")

    left, right = st.columns(2)

    with left:
        st.subheader("Prediction Summary")
        prediction_data = pd.DataFrame({
            "Prediction": ["Normal", "Suspicious"],
            "Events": [normal_count, suspicious]
        }).set_index("Prediction")
        st.bar_chart(prediction_data)

    with right:
        st.subheader("Confusion Matrix")
        matrix = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual Normal", "Actual Attack"],
            columns=["Predicted Normal", "Predicted Attack"]
        )
        st.dataframe(matrix, use_container_width=True)

    st.divider()
    st.subheader("Suspicious Events by Attack Category")

    category_query = """
        SELECT attack_cat, COUNT(*) AS event_count
        FROM unsw_nb15_detections
        WHERE predicted_label = 1
        GROUP BY attack_cat
        ORDER BY event_count DESC
    """
    categories = pd.read_sql(category_query, conn)
    st.bar_chart(
        categories.set_index("attack_cat")["event_count"]
    )

    st.divider()
    st.subheader("Suspicious Event Explorer")

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        category_options = ["All"] + sorted(
            categories["attack_cat"].dropna().astype(str).unique().tolist()
        )
        selected_category = st.selectbox(
            "Filter by attack category",
            category_options
        )

    with filter_col2:
        min_score = st.number_input(
            "Minimum anomaly score",
            value=-1.0,
            min_value=-1.0,
            max_value=1.0,
            step=0.05
        )

    events_query = """
        SELECT dataset_id, actual_label, predicted_label,
               anomaly_score, attack_cat, status, detected_at
        FROM unsw_nb15_detections
        WHERE predicted_label = 1 AND anomaly_score >= %s
    """
    params = [min_score]

    if selected_category != "All":
        events_query += " AND attack_cat = %s"
        params.append(selected_category)

    events_query += " ORDER BY anomaly_score DESC LIMIT 500"

    events = pd.read_sql(events_query, conn, params=params)
    st.caption("Showing up to 500 matching events, ordered by anomaly score.")
    st.dataframe(events, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Alert Summary")

    if not events.empty:
        st.warning(
            f"{len(events):,} matching suspicious events are displayed. "
            "Review these records for investigation."
        )
    else:
        st.success("No events match the current filters.")

    conn.close()

except mysql.connector.Error as exc:
    st.error(
        "Could not connect to MySQL or read the detection table. "
        "Check that MySQL is running, the password is correct, "
        "and the table exists."
    )
    st.code(str(exc))
except Exception as exc:
    st.error("The dashboard encountered an unexpected error.")
    st.code(str(exc))
