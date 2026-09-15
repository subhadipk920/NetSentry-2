"""
NetSentry Security Ingestion & Sync Engine (Aiven -> Cloudflare R2).
Extracts captured security packets / attack flows from Aiven PostgreSQL,
converts them into timestamped partition CSV files, and uploads to Cloudflare R2.
"""

import os
import io
import sys
import logging
from datetime import datetime, timezone
import pandas as pd
import psycopg2
import boto3
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NetSentrySyncEngine")

load_dotenv()


def get_db_connection():
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise ValueError("MLFLOW_TRACKING_URI is not set.")
    if uri.startswith("postgresql+psycopg2://"):
        uri = uri.replace("postgresql+psycopg2://", "postgresql://", 1)
    return psycopg2.connect(uri)


def get_r2_client():
    endpoint = os.getenv("R2_ENDPOINT_URL")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not (endpoint and access_key and secret_key):
        raise ValueError("Cloudflare R2 credentials (R2_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) missing.")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def sync_aiven_to_r2(sync_all: bool = False, attacks_only: bool = True):
    """
    Reads attack records from Aiven PostgreSQL, serializes to CSV, and uploads to R2.
    Marks processed records as synced.
    """
    bucket_name = os.getenv("R2_BUCKET_NAME", "netsentry")
    logger.info(f"Starting Aiven to R2 sync (Bucket: {bucket_name}, attacks_only={attacks_only})")

    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        SELECT id, timestamp, client_ip, event_type, status_code,
               threat_probability, is_attack, user_agent, headers, flow_metrics
        FROM security_events
    """
    conditions = []
    if not sync_all:
        conditions.append("synced_to_r2 = FALSE")
    if attacks_only:
        conditions.append("is_attack = 1")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id ASC;"

    cur.execute(query)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

    if not rows:
        logger.info("No unsynced attack packets found in Aiven PostgreSQL.")
        cur.close()
        conn.close()
        return {"status": "SUCCESS", "synced_records": 0, "r2_key": None}

    df = pd.DataFrame(rows, columns=columns)
    record_ids = df["id"].tolist()
    total_records = len(record_ids)
    logger.info(f"Retrieved {total_records} attack record(s) from Aiven.")

    # Flatten flow_metrics into separate columns for ML analysis if available
    if "flow_metrics" in df.columns:
        metrics_df = pd.json_normalize(df["flow_metrics"].apply(lambda x: x if isinstance(x, dict) else {}))
        metrics_df.columns = [f"flow_{col}" for col in metrics_df.columns]
        df = pd.concat([df.drop(columns=["flow_metrics"]), metrics_df], axis=1)

    # Convert headers to string to store cleanly in CSV
    if "headers" in df.columns:
        df["headers"] = df["headers"].astype(str)

    # Convert to CSV buffer
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    # Generate partition key: captured_traffic/YYYY/MM/DD/attacks_YYYYMMDD_HHMMSS.csv
    now_utc = datetime.now(timezone.utc)
    date_path = now_utc.strftime("%Y/%m/%d")
    filename = f"attack_packets_{now_utc.strftime('%Y%m%d_%H%M%S')}_{total_records}pkts.csv"
    r2_key = f"captured_traffic/{date_path}/{filename}"

    # Also upload a latest pointer for streaming dashboards
    latest_r2_key = "captured_traffic/latest_attacks.csv"

    # Upload to Cloudflare R2
    r2_client = get_r2_client()
    logger.info(f"Uploading CSV to Cloudflare R2 -> s3://{bucket_name}/{r2_key}")
    r2_client.put_object(
        Bucket=bucket_name,
        Key=r2_key,
        Body=csv_bytes,
        ContentType="text/csv",
    )
    r2_client.put_object(
        Bucket=bucket_name,
        Key=latest_r2_key,
        Body=csv_bytes,
        ContentType="text/csv",
    )

    # Mark rows as synced in Aiven Postgres
    update_query = "UPDATE security_events SET synced_to_r2 = TRUE WHERE id = ANY(%s);"
    cur.execute(update_query, (record_ids,))
    conn.commit()

    cur.close()
    conn.close()

    logger.info(f"✅ Successfully exported {total_records} packets to R2 and updated Aiven database.")
    return {"status": "SUCCESS", "synced_records": total_records, "r2_key": r2_key}


if __name__ == "__main__":
    force_all = "--all" in sys.argv
    all_traffic = "--include-benign" in sys.argv
    result = sync_aiven_to_r2(sync_all=force_all, attacks_only=not all_traffic)
    print(result)
