"""
NetSentry Security Event Logger & Sync Utilities.
Logs security requests (benign vs attack) to Aiven PostgreSQL and syncs records to Cloudflare R2.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import psycopg2
import boto3
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("NetSentrySecurityDB")


def get_db_connection():
    """Connects to Aiven PostgreSQL instance."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    if uri.startswith("postgresql+psycopg2://"):
        uri = uri.replace("postgresql+psycopg2://", "postgresql://", 1)
    return psycopg2.connect(uri)


def get_r2_client():
    """Initializes Boto3 S3 client for Cloudflare R2."""
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


def log_security_event(
    client_ip: str,
    event_type: str,
    status_code: int,
    threat_probability: float,
    is_attack: int,
    user_agent: str,
    headers: Dict[str, Any],
    flow_metrics: Optional[Dict[str, Any]] = None,
):
    """
    Persists a network security event directly into Aiven PostgreSQL.
    Fails safely without breaking runtime request flow.
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return
        with conn.cursor() as cur:
            insert_query = """
                INSERT INTO security_events (
                    client_ip, event_type, status_code, threat_probability,
                    is_attack, user_agent, headers, flow_metrics, synced_to_r2
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE)
            """
            cur.execute(
                insert_query,
                (
                    client_ip,
                    event_type,
                    status_code,
                    threat_probability,
                    is_attack,
                    user_agent,
                    json.dumps(headers),
                    json.dumps(flow_metrics or {}),
                ),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to log security event to Aiven: {e}")
    finally:
        if conn:
            conn.close()
