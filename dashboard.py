import json
from datetime import datetime, timezone
from pathlib import Path

import mysql.connector
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


MODEL_LABEL = "Weighted Binary Random Forest"
MODEL_KEY = "binary_fuzzer_shellcode_x4"
CALIBRATION_FPR_TARGET = 0.0075
METRICS_FILE = Path("sentinelai_rf_binary_attack_experiment_metrics.csv")
CATEGORY_FILE = Path("sentinelai_rf_binary_attack_experiment_by_category.csv")

st.set_page_config(
    page_title="SentinelAI Security Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ SentinelAI")
st.subheader("Security Operations Dashboard")
st.caption(f"Primary classifier: {MODEL_LABEL} | FastAPI | MySQL")
st_autorefresh(interval=59_000, key="sentinelai_dashboard_refresh")

st.sidebar.header("Database Connection")
db_password = st.sidebar.text_input("MySQL root password", type="password")
if not db_password:
    st.info("Enter your MySQL password in the sidebar to load the dashboard.")
    st.stop()

try:
    # Display validation results for the model selected as primary.
    if not METRICS_FILE.exists() or not CATEGORY_FILE.exists():
        st.error("The weighted Random Forest validation reports are missing.")
        st.stop()

    metrics = pd.read_csv(METRICS_FILE)
    selected = metrics.loc[
        (metrics["model"] == MODEL_KEY)
        & (metrics["calibration_fpr_target"] == CALIBRATION_FPR_TARGET)
    ]
    if selected.empty:
        st.error("The selected Random Forest validation result was not found.")
        st.stop()
    model_result = selected.iloc[0]

    tn = int(model_result["validation_TN"])
    fp = int(model_result["validation_FP"])
    fn = int(model_result["validation_FN"])
    tp = int(model_result["validation_TP"])
    precision = float(model_result["validation_precision"])
    recall = float(model_result["validation_attack_recall"])
    f1 = float(model_result["validation_F1"])
    fpr = float(model_result["validation_FPR"])
    threshold = float(model_result["threshold"])

    st.header("Primary Model Performance")
    st.caption(
        "Weighted Binary Random Forest · held-out validation split · "
        "attack score threshold selected on a separate calibration slice. "
        "These are validation metrics, not live-event counts."
    )
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Precision", f"{precision:.2%}")
    metric2.metric("Attack Recall", f"{recall:.2%}")
    metric3.metric("F1-score", f"{f1:.2%}")
    metric4.metric("False-positive Rate", f"{fpr:.2%}")
    st.caption(
        f"Primary classifier threshold: {threshold:.6f}  ·  "
        "Scores at or above the threshold are classified as potential attacks."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Validation Confusion Matrix")
        matrix = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual Normal", "Actual Attack"],
            columns=["Predicted Normal", "Predicted Attack"],
        )
        st.dataframe(matrix, use_container_width=True)
    with right:
        st.subheader("Attack-category recall")
        categories = pd.read_csv(CATEGORY_FILE)
        categories = categories.loc[
            (categories["model"] == MODEL_KEY)
            & (categories["calibration_fpr_target"] == CALIBRATION_FPR_TARGET)
            & (categories["category"] != "Normal")
        ].copy()
        if categories.empty:
            st.info("No category-level validation results are available.")
        else:
            categories["recall_percent"] = categories["recall"] * 100
            st.bar_chart(categories.set_index("category")["recall_percent"])

    if not categories.empty:
        st.subheader("Category-level results")
        st.dataframe(
            categories[["category", "records", "detected", "missed", "recall_percent"]]
            .rename(columns={"recall_percent": "Recall (%)"}),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.header("🔴 Live Security Monitoring")
    st.caption(
        "Events submitted to the SentinelAI API. UNSW-NB15 demo sources are "
        "excluded so validation samples are not presented as live telemetry."
    )

    conn = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password=db_password,
        database="sentinel_ai",
    )
    live_query = """
        SELECT event_id, event_timestamp, source, event_type,
               classifier_score, classifier_threshold, classifier_prediction,
               status, alert_severity, detection_reason, detection_methods,
               matched_rules, alert_status, assigned_to, investigation_notes,
               raw_data, normalized_fields
        FROM live_security_events
        WHERE LEFT(LOWER(source), 9) <> 'unsw_nb15'
        ORDER BY event_timestamp DESC, event_id DESC
        LIMIT 1000
    """
    live_events = pd.read_sql(live_query, conn)
    conn.close()

    if not live_events.empty:
        live_events["event_timestamp_utc"] = pd.to_datetime(
            live_events["event_timestamp"], utc=True, errors="coerce"
        )
        live_events["classifier_prediction"] = pd.to_numeric(
            live_events["classifier_prediction"], errors="coerce"
        )
        live_events["classifier_score"] = pd.to_numeric(
            live_events["classifier_score"], errors="coerce"
        )
        live_events["rf_score_display"] = live_events["classifier_score"].map(
            lambda score: f"{score:.4f}" if pd.notna(score) else "Not recorded"
        )
        live_events["rf_result"] = live_events["classifier_prediction"].map(
            {0: "Normal", 1: "Potential attack"}
        ).fillna("Unavailable for this event")

        def matched_rule_names(value):
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    return ""
            if not isinstance(value, list):
                return ""
            return ", ".join(
                item.get("rule_id", "")
                for item in value
                if isinstance(item, dict) and item.get("rule_id")
            )

        live_events["matched_rule_names"] = live_events["matched_rules"].map(
            matched_rule_names
        )
        live_events["detection"] = live_events.apply(
            lambda row: (
                "Binary Random Forest"
                if row["classifier_prediction"] == 1
                else (
                    f"Security rule: {row['matched_rule_names']}"
                    if row["matched_rule_names"]
                    else "No alert"
                )
            ),
            axis=1,
        )
    else:
        live_events["event_timestamp_utc"] = pd.Series(dtype="datetime64[ns, UTC]")
        live_events["rf_score_display"] = pd.Series(dtype="object")
        live_events["rf_result"] = pd.Series(dtype="object")
        live_events["matched_rule_names"] = pd.Series(dtype="object")
        live_events["detection"] = pd.Series(dtype="object")

    rf_alerts = int(live_events["classifier_prediction"].eq(1).sum())
    active_alerts = int(
        live_events["alert_status"].isin(["New", "Investigating"]).sum()
    )
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Recent API Events", f"{len(live_events):,}")
    kpi2.metric("RF Attack Detections", f"{rf_alerts:,}")
    kpi3.metric("Active Alerts", f"{active_alerts:,}")

    st.subheader("Threat hunting")
    st.caption(
        "Search and narrow the latest 1,000 non-dataset API events by time, "
        "source, type, detection, severity, or investigation status."
    )
    filter1, filter2, filter3, filter4, filter5, filter6, search_col = st.columns(
        [1.1, 1, 1, 1, 1.2, 1, 2]
    )
    with filter1:
        time_range = st.selectbox(
            "Time range",
            ["All loaded events", "Last 15 minutes", "Last hour", "Last 24 hours", "Last 7 days"],
            index=0,
        )
    with filter2:
        sources = ["All"] + sorted(
            live_events["source"].dropna().astype(str).unique().tolist()
        )
        source_filter = st.selectbox("Source", sources)
    with filter3:
        event_types = ["All"] + sorted(
            live_events["event_type"].dropna().astype(str).unique().tolist()
        )
        event_type_filter = st.selectbox("Event type", event_types)
    with filter4:
        status_filter = st.selectbox("Detection result", ["All", "Suspicious", "Normal"])
    with filter5:
        workflow_filter = st.selectbox(
            "Investigation status",
            ["All", "New", "Investigating", "Resolved", "False Positive"],
        )
    with filter6:
        severity_options = ["All"] + sorted(
            live_events["alert_severity"].dropna().astype(str).unique().tolist()
        )
        severity_filter = st.selectbox("Severity", severity_options)
    with search_col:
        search = st.text_input(
            "Search live events", placeholder="Event ID, source, rule, or type..."
        )

    filtered = live_events.copy()
    range_durations = {
        "Last 15 minutes": pd.Timedelta(minutes=15),
        "Last hour": pd.Timedelta(hours=1),
        "Last 24 hours": pd.Timedelta(hours=24),
        "Last 7 days": pd.Timedelta(days=7),
    }
    if time_range in range_durations:
        cutoff = pd.Timestamp.now(tz="UTC") - range_durations[time_range]
        filtered = filtered[filtered["event_timestamp_utc"] >= cutoff]
    if source_filter != "All":
        filtered = filtered[filtered["source"].astype(str) == source_filter]
    if event_type_filter != "All":
        filtered = filtered[filtered["event_type"].astype(str) == event_type_filter]
    if status_filter != "All":
        filtered = filtered[filtered["status"] == status_filter]
    if workflow_filter != "All":
        filtered = filtered[filtered["alert_status"] == workflow_filter]
    if severity_filter != "All":
        filtered = filtered[filtered["alert_severity"] == severity_filter]
    if search.strip():
        searchable = (
            filtered["event_id"].astype(str)
            + " " + filtered["source"].astype(str)
            + " " + filtered["event_type"].astype(str)
            + " " + filtered["matched_rule_names"].astype(str)
            + " " + filtered["detection_reason"].fillna("").astype(str)
            + " " + filtered["detection_methods"].fillna("").astype(str)
            + " " + filtered["raw_data"].fillna("").astype(str)
            + " " + filtered["normalized_fields"].fillna("").astype(str)
        )
        filtered = filtered[searchable.str.contains(
            search.strip(), case=False, regex=False, na=False
        )]

    st.caption(f"Hunt results: {len(filtered):,} matching events")
    display_columns = [
        "event_id", "event_timestamp", "source", "event_type",
        "rf_score_display", "rf_result", "detection", "alert_severity",
        "alert_status",
    ]
    if filtered.empty:
        st.info(
            "No live events match this view yet. Events sent to the API will "
            "appear here; this dashboard refreshes every minute."
        )
    st.dataframe(
        filtered[display_columns].rename(
            columns={"rf_score_display": "RF attack score"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    def display_value(value, fallback="Not recorded"):
        return fallback if pd.isna(value) else str(value)

    st.subheader("Alert investigation")
    investigatable = filtered[filtered["status"] == "Suspicious"]
    if investigatable.empty:
        st.info("No suspicious events match the current filters. Adjust the filters to investigate an alert.")
    else:
        event_choices = investigatable["event_id"].astype(int).tolist()
        selected_event_id = st.selectbox(
            "Select an alert", event_choices,
            format_func=lambda event_id: (
                f"E-{event_id} · "
                f"{display_value(investigatable.loc[investigatable['event_id'] == event_id, 'alert_severity'].iloc[0], 'Unrated')} · "
                f"{investigatable.loc[investigatable['event_id'] == event_id, 'source'].iloc[0]}"
            ),
        )
        selected_event = investigatable.loc[
            investigatable["event_id"] == selected_event_id
        ].iloc[0]
        detail_left, detail_right = st.columns(2)
        detail_left.write(f"**Detection:** {selected_event['detection']}")
        detail_left.write(f"**RF result:** {selected_event['rf_result']}")
        detail_left.write(f"**RF attack score:** {selected_event['rf_score_display']}")
        detail_left.write(
            f"**Severity:** {display_value(selected_event['alert_severity'], 'Unrated')}"
        )
        reason = display_value(selected_event["detection_reason"], "No explanation recorded.")
        # Keep the primary dashboard explanation focused on the active RF/rule detection.
        reason = reason.split("Isolation Forest anomaly signal:", 1)[0].strip()
        detail_right.write("**Why it was flagged**")
        detail_right.write(reason or "No primary detection explanation recorded.")

        def parse_json_field(value):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value

        with st.expander("Event context"):
            st.write("**Matched rules**")
            st.json(parse_json_field(selected_event["matched_rules"]) or [])
            st.write("**Normalized fields**")
            st.json(parse_json_field(selected_event["normalized_fields"]) or {})
            st.write("**Raw event data**")
            st.json(parse_json_field(selected_event["raw_data"]) or {})

        previous_notes = (
            str(selected_event["investigation_notes"])
            if pd.notna(selected_event["investigation_notes"])
            else ""
        )
        st.caption("Investigation notes are appended to the event record with a UTC timestamp.")
        with st.form(f"investigation_{selected_event_id}"):
            status_options = ["New", "Investigating", "Resolved", "False Positive"]
            current_status = selected_event["alert_status"]
            status_index = status_options.index(current_status) if current_status in status_options else 0
            new_status = st.selectbox("Update status", status_options, index=status_index)
            assigned_to = st.text_input(
                "Assigned to",
                value=(
                    str(selected_event["assigned_to"])
                    if pd.notna(selected_event["assigned_to"])
                    else ""
                ),
            )
            new_note = st.text_area("Add investigation note")
            submitted = st.form_submit_button("Save investigation")
        if previous_notes:
            with st.expander("Previous investigation notes"):
                st.text(previous_notes)
        if submitted:
            note = new_note.strip()
            updated_notes = previous_notes
            if note:
                timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                entry = f"[{timestamp}] {note}"
                updated_notes = f"{previous_notes}\n{entry}".strip()
            update_conn = mysql.connector.connect(
                host="127.0.0.1", port=3306, user="root",
                password=db_password, database="sentinel_ai",
            )
            try:
                with update_conn.cursor() as update_cursor:
                    update_cursor.execute(
                        """UPDATE live_security_events
                           SET alert_status = %s, assigned_to = %s,
                               investigation_notes = %s
                           WHERE event_id = %s""",
                        (new_status, assigned_to.strip() or None, updated_notes or None,
                         int(selected_event_id)),
                    )
                update_conn.commit()
                st.success("Investigation updated.")
                st.rerun()
            finally:
                update_conn.close()

        st.subheader("Safe response simulation")
        st.caption(
            "Records a proposed response for this alert. It does not block traffic, "
            "change accounts, isolate devices, or contact external systems."
        )
        with st.form(f"response_simulation_{selected_event_id}"):
            action_type = st.selectbox(
                "Simulated action",
                [
                    "Block source (simulation only)",
                    "Isolate host (simulation only)",
                    "Disable account (simulation only)",
                    "Collect additional logs (simulation only)",
                    "Other (simulation only)",
                ],
            )
            action_reason = st.text_area(
                "Why would this response be considered?",
                key=f"response_reason_{selected_event_id}",
            )
            response_owner = st.text_input(
                "Recorded by",
                key=f"response_owner_{selected_event_id}",
            )
            simulate = st.form_submit_button("Record simulated response")
        if simulate:
            outcome = (
                "Simulation recorded only. No external system action was performed."
            )
            response_conn = mysql.connector.connect(
                host="127.0.0.1", port=3306, user="root",
                password=db_password, database="sentinel_ai",
            )
            try:
                with response_conn.cursor() as response_cursor:
                    response_cursor.execute(
                        """INSERT INTO sentinelai_response_simulations
                           (event_id, action_type, reason, performed_by, outcome, created_at)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        (
                            int(selected_event_id), action_type,
                            action_reason.strip() or None,
                            response_owner.strip() or None,
                            outcome,
                            datetime.now(timezone.utc).replace(tzinfo=None),
                        ),
                    )
                response_conn.commit()
                st.success(outcome)
                st.rerun()
            finally:
                response_conn.close()

        response_conn = mysql.connector.connect(
            host="127.0.0.1", port=3306, user="root",
            password=db_password, database="sentinel_ai",
        )
        try:
            with response_conn.cursor() as response_cursor:
                response_cursor.execute(
                    """SELECT action_type, reason, performed_by, outcome, created_at
                       FROM sentinelai_response_simulations
                       WHERE event_id = %s
                       ORDER BY created_at DESC, simulation_id DESC""",
                    (int(selected_event_id),),
                )
                response_history = response_cursor.fetchall()
            if response_history:
                st.write("**Recorded simulations for this alert**")
                st.dataframe(
                    pd.DataFrame(
                        response_history,
                        columns=["Action", "Reason", "Recorded by", "Outcome", "Recorded at (UTC)"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No response simulations have been recorded for this alert.")
        finally:
            response_conn.close()

except mysql.connector.Error as exc:
    st.error("Could not connect to MySQL or load live SentinelAI events.")
    st.code(str(exc))
except Exception as exc:
    st.error("The dashboard could not load the Random Forest report or events.")
    st.code(str(exc))
