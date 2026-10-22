from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
import mysql.connector
from streamlit_autorefresh import st_autorefresh


st.set_page_config(
    page_title="SentinelAI Live Security Operations",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ SentinelAI Security Operations")
st.caption("Live API events, detections, and incident investigations")
st_autorefresh(interval=59_000, key="sentinelai_dashboard_refresh")

st.sidebar.header("Live view")
time_range = st.sidebar.selectbox(
    "Show events from",
    ["Last hour", "Last 24 hours", "Last 7 days"],
    index=1,
)
range_durations = {
    "Last hour": timedelta(hours=1),
    "Last 24 hours": timedelta(hours=24),
    "Last 7 days": timedelta(days=7),
}
start_time = datetime.now(timezone.utc) - range_durations[time_range]

st.sidebar.header("Database connection")
db_password = st.sidebar.text_input("MySQL root password", type="password")
if not db_password:
    st.info("Enter your MySQL password in the sidebar to load live events.")
    st.stop()

conn = None
try:
    conn = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password=db_password,
        database="sentinel_ai",
    )

    live_query = """
        SELECT event_id, event_timestamp, source, event_type, raw_data,
               normalized_fields, anomaly_score, status, prediction,
               classifier_score, classifier_threshold, classifier_prediction,
               isolation_forest_flagged, alert_severity, detection_reason,
               detection_methods, matched_rules, alert_status, assigned_to,
               investigation_notes, detected_at
        FROM live_security_events
        WHERE event_timestamp >= %s
          AND LEFT(LOWER(source), 9) <> 'unsw_nb15'
        ORDER BY event_timestamp DESC, event_id DESC
        LIMIT 1000
    """
    live_events = pd.read_sql(live_query, conn, params=[start_time.replace(tzinfo=None)])

    st.caption(
        f"Showing API events from the {time_range.lower()}. "
        "UNSW-NB15 demo submissions and historical batch evaluations are excluded."
    )

    if live_events.empty:
        st.info(
            "No live telemetry has arrived in this time range. SentinelAI "
            "receives events only when a sender posts them to the API; it "
            "does not capture network traffic by itself. UNSW-NB15 demo "
            "submissions are intentionally hidden from this operational view. "
            "Connect a live sender or submit a non-demo event at "
            "http://127.0.0.1:8000/docs."
        )

    live_suspicious = int(live_events["status"].eq("Suspicious").sum())
    live_normal = int(live_events["status"].eq("Normal").sum())
    active_alerts = int(
        live_events["alert_status"].isin(["New", "Investigating"]).sum()
    )
    rule_alerts = int(
        live_events["detection_methods"].astype(str).str.contains(
            "security_rule", regex=False, na=False
        ).sum()
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(f"Events · {time_range}", f"{len(live_events):,}")
    kpi2.metric("Potential Threats", f"{live_suspicious:,}")
    kpi3.metric("Active Alerts", f"{active_alerts:,}")
    kpi4.metric("Rule-Generated Alerts", f"{rule_alerts:,}")

    st.subheader("Live detection summary")
    summary = pd.DataFrame(
        {"Events": [live_normal, live_suspicious]},
        index=["Normal", "Suspicious"],
    )
    st.bar_chart(summary)

    st.subheader("Live alert explorer")
    filter1, filter2, filter3 = st.columns(3)
    with filter1:
        status_filter = st.selectbox(
            "Detection status", ["All", "Suspicious", "Normal"], key="live_status"
        )
    with filter2:
        alert_filter = st.selectbox(
            "Alert workflow status",
            ["All", "New", "Investigating", "Resolved", "False Positive"],
            key="live_alert_status",
        )
    with filter3:
        severity_filter = st.selectbox(
            "Severity", ["All", "Low", "Medium", "High", "Critical"],
            key="live_severity",
        )

    search = st.text_input(
        "Search events",
        placeholder="Event ID, source, rule, or explanation...",
    )
    filtered = live_events.copy()
    if status_filter != "All":
        filtered = filtered[filtered["status"] == status_filter]
    if alert_filter != "All":
        filtered = filtered[filtered["alert_status"] == alert_filter]
    if severity_filter != "All":
        filtered = filtered[filtered["alert_severity"] == severity_filter]
    if search.strip():
        searchable = (
            filtered["event_id"].astype(str)
            + " " + filtered["source"].astype(str)
            + " " + filtered["event_type"].astype(str)
            + " " + filtered["detection_reason"].astype(str)
            + " " + filtered["detection_methods"].astype(str)
            + " " + filtered["matched_rules"].astype(str)
        )
        filtered = filtered[searchable.str.contains(
            search.strip(), case=False, regex=False, na=False
        )]

    st.caption(f"{len(filtered):,} matching events · maximum 1,000 loaded")
    display_columns = [
        "event_id", "event_timestamp", "source", "event_type",
        "classifier_score", "alert_severity", "status", "alert_status",
        "detection_methods", "detection_reason",
    ]
    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
    )
    if filtered.empty:
        st.caption("New events will appear here after they are submitted to the API.")

    if not filtered.empty:
        selected_event_id = st.selectbox(
            "Event details",
            filtered["event_id"].tolist(),
            format_func=lambda event_id: f"E-{event_id}",
        )
        selected_event = filtered.loc[
            filtered["event_id"] == selected_event_id
        ].iloc[0]
        with st.expander(f"Details for event E-{selected_event_id}"):
            st.write("Detection explanation")
            st.write(selected_event["detection_reason"])
            st.write("Matched rules")
            st.json(selected_event["matched_rules"] or [])
            st.write("Raw event")
            st.json(selected_event["raw_data"] or {})

    st.subheader("Investigate alert")
    alert_rows = live_events[live_events["status"] == "Suspicious"]
    if alert_rows.empty:
        st.info("No suspicious live events are available to investigate.")
    else:
        alert_ids = alert_rows["event_id"].tolist()
        with st.form("manage_live_alert"):
            selected_alert_id = st.selectbox(
                "Select alert",
                alert_ids,
                format_func=lambda event_id: (
                    f"E-{event_id} · "
                    f"{alert_rows.loc[alert_rows['event_id'] == event_id, 'alert_severity'].iloc[0]} · "
                    f"{alert_rows.loc[alert_rows['event_id'] == event_id, 'event_type'].iloc[0]}"
                ),
            )
            current_alert = alert_rows.loc[
                alert_rows["event_id"] == selected_alert_id
            ].iloc[0]
            current_status = current_alert["alert_status"] or "New"
            current_owner = current_alert["assigned_to"] or ""
            existing_notes = current_alert["investigation_notes"] or ""
            status_options = ["New", "Investigating", "Resolved", "False Positive"]
            selected_status = st.selectbox(
                "Status",
                status_options,
                index=status_options.index(current_status)
                if current_status in status_options else 0,
            )
            assigned_to = st.text_input(
                "Assigned to", value=current_owner, max_chars=128,
                placeholder="Analyst name (optional)",
            )
            if existing_notes:
                st.text_area(
                    "Investigation history", value=existing_notes,
                    height=140, disabled=True,
                )
            new_note = st.text_area(
                "Add investigation note", max_chars=2000,
                placeholder="Evidence, checks performed, or next steps...",
            )
            save_investigation = st.form_submit_button("Save investigation")

        if save_investigation:
            note_entry = None
            if new_note.strip():
                note_entry = (
                    f"[{datetime.now(timezone.utc).isoformat()}] {new_note.strip()}"
                )
            updated_notes = existing_notes
            if note_entry:
                updated_notes = (
                    f"{existing_notes}\n\n{note_entry}"
                    if existing_notes else note_entry
                )
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    UPDATE live_security_events
                    SET alert_status = %s, assigned_to = %s,
                        investigation_notes = %s
                    WHERE event_id = %s AND status = 'Suspicious'
                    """,
                    (
                        selected_status,
                        assigned_to.strip() or None,
                        updated_notes or None,
                        int(selected_alert_id),
                    ),
                )
                conn.commit()
                st.success(f"Investigation for E-{selected_alert_id} saved.")
                st.rerun()
            finally:
                cursor.close()

    conn.close()

except mysql.connector.Error as exc:
    if conn is not None and conn.is_connected():
        conn.close()
    st.error("Could not connect to MySQL or load live security events.")
    st.code(str(exc))
except Exception as exc:
    if conn is not None and conn.is_connected():
        conn.close()
    st.error("The live dashboard encountered an unexpected error.")
    st.code(str(exc))
