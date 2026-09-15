"""
NetSentry SOC & Threat Intelligence Dashboard.
Displays live request metrics, attack breakdown table, and integrates directly with Cloudflare R2 and Aiven PostgreSQL.
"""

import os
import io
import time
import json
import pandas as pd
import streamlit as st
import psycopg2
import boto3
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="NetSentry - Threat Intelligence & Gateway Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 15px;
        color: white;
        border-left: 5px solid #3b82f6;
    }
    .badge-blocked {
        background-color: #dc2626;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-allowed {
        background-color: #16a34a;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def get_db_connection():
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    if uri.startswith("postgresql+psycopg2://"):
        uri = uri.replace("postgresql+psycopg2://", "postgresql://", 1)
    return psycopg2.connect(uri)


def get_r2_client():
    endpoint = os.getenv("R2_ENDPOINT_URL")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not (endpoint and access_key and secret_key):
        return None
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


@st.cache_data(ttl=5)
def load_data_from_aiven(limit: int = 500):
    try:
        conn = get_db_connection()
        if not conn:
            return pd.DataFrame()
        query = f"""
            SELECT id, timestamp, client_ip, event_type, status_code,
                   threat_probability, is_attack, user_agent, headers,
                   flow_metrics, synced_to_r2
            FROM security_events
            ORDER BY id DESC
            LIMIT {limit};
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching data from Aiven: {e}")
        return pd.DataFrame()


def sync_to_r2_action():
    from netsentry.serving.sync_engine import sync_aiven_to_r2
    return sync_aiven_to_r2(sync_all=False, attacks_only=True)


# --- UI HEADER ---
col_logo, col_header = st.columns([1, 6])
with col_logo:
    st.markdown("<h1 style='text-align: center;'>🛡️</h1>", unsafe_allow_html=True)
with col_header:
    st.title("NetSentry Security Gateway & Live SOC Dashboard")
    st.caption("Real-Time Threat Detection, Packet Inspection & Cloudflare R2 Sync")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    auto_refresh = st.checkbox("Auto-refresh every 5s", value=True)
    limit = st.slider("Record Limit", min_value=50, max_value=2000, value=200, step=50)

    st.markdown("---")
    st.subheader("Cloudflare R2 Sync Engine")
    bucket = os.getenv("R2_BUCKET_NAME", "netsentry")
    st.write(f"**Target Bucket:** `{bucket}`")
    if st.button("🚀 Trigger R2 Backup Sync Now", use_container_width=True):
        with st.spinner("Exporting attack packets to Cloudflare R2..."):
            res = sync_to_r2_action()
            if res.get("status") == "SUCCESS":
                st.success(f"Synced {res.get('synced_records')} packets to `{res.get('r2_key')}`")
            else:
                st.error("Sync failed.")
    
    st.markdown("---")
    st.markdown("""
    **Architecture Flow:**
    1. Wire traffic passes through `/gateway/...`
    2. ML evaluates 65 statistical flow metrics in RAM
    3. Blocked threats (403) & Benign requests logged to Aiven DB
    4. Sync engine packages attack records into CSV in Cloudflare R2
    """)

# --- LOAD DATA ---
df = load_data_from_aiven(limit=limit)

if df.empty:
    st.warning("No security events recorded yet in Aiven PostgreSQL. Send traffic via `python scripts/simulate_traffic.py` to populate data.")
else:
    total_requests = len(df)
    total_attacks = int(df["is_attack"].sum())
    total_benign = total_requests - total_attacks
    attack_rate = (total_attacks / total_requests * 100) if total_requests > 0 else 0.0
    synced_count = int(df["synced_to_r2"].sum())

    # --- TOP METRIC CARDS ---
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(label="Total Wire Requests", value=total_requests)
    with m2:
        st.metric(label="Legitimate / Benign (200 OK)", value=total_benign)
    with m3:
        st.metric(label="Malicious Attacks Blocked (403)", value=total_attacks, delta=f"{attack_rate:.1f}% Threat Rate", delta_color="inverse")
    with m4:
        avg_threat = df["threat_probability"].mean()
        st.metric(label="Avg Threat Probability", value=f"{avg_threat:.3f}")
    with m5:
        st.metric(label="Synced to Cloudflare R2", value=f"{synced_count} / {total_requests}")

    st.markdown("---")

    # --- ATTACK BREAKDOWN & CHARTS ---
    c_chart1, c_chart2 = st.columns([1, 1])

    with c_chart1:
        st.subheader("📊 Breakdown by Attack & Traffic Type")
        type_counts = df["event_type"].value_counts().reset_index()
        type_counts.columns = ["Attack / Traffic Type", "Count"]
        st.dataframe(type_counts, use_container_width=True, hide_index=True)

    with c_chart2:
        st.subheader("📈 Real-time Traffic Composition")
        st.bar_chart(data=type_counts.set_index("Attack / Traffic Type"), use_container_width=True)

    st.markdown("---")

    # --- DETAILED SECURITY EVENTS TABLE ---
    st.subheader("🔍 Captured Request Log & Threat Feed")

    # Filter by Event Type
    event_filter = st.multiselect(
        "Filter by Traffic / Attack Type:",
        options=list(df["event_type"].unique()),
        default=list(df["event_type"].unique()),
    )

    filtered_df = df[df["event_type"].isin(event_filter)].copy()

    # Format presentation table
    display_cols = ["id", "timestamp", "client_ip", "event_type", "status_code", "threat_probability", "is_attack", "synced_to_r2"]
    st.dataframe(
        filtered_df[display_cols].style.map(
            lambda v: "background-color: #fee2e2; color: #991b1b; font-weight: bold;" if v == 1 else "",
            subset=["is_attack"]
        ).map(
            lambda v: "color: #dc2626; font-weight: bold;" if v == 403 else "color: #16a34a; font-weight: bold;",
            subset=["status_code"]
        ),
        use_container_width=True,
        height=400,
        hide_index=True,
    )

    # --- PACKET INSPECTION DRAWER ---
    with st.expander("📦 Inspect Individual Network Flow Payloads & Headers"):
        selected_id = st.selectbox("Select Event ID to inspect:", filtered_df["id"].tolist())
        selected_row = filtered_df[filtered_df["id"] == selected_id].iloc[0]
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**HTTP Headers Captured:**")
            st.json(selected_row["headers"])
        with c2:
            st.markdown("**Extracted Flow Statistical Metrics:**")
            st.json(selected_row["flow_metrics"])

# Auto refresh mechanism
if auto_refresh:
    time.sleep(5)
    st.rerun()
