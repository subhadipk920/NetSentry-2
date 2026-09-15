import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("MLFLOW_TRACKING_URI")
if not uri:
    raise ValueError("MLFLOW_TRACKING_URI not found in environment.")

if uri.startswith("postgresql+psycopg2://"):
    uri = uri.replace("postgresql+psycopg2://", "postgresql://", 1)

conn = psycopg2.connect(uri)
cur = conn.cursor()

create_table_query = """
CREATE TABLE IF NOT EXISTS security_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    client_ip VARCHAR(64),
    event_type VARCHAR(64),
    status_code INT,
    threat_probability FLOAT,
    is_attack INT,
    user_agent TEXT,
    headers JSONB,
    flow_metrics JSONB,
    synced_to_r2 BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_security_events_ts ON security_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_security_events_synced ON security_events (synced_to_r2);
"""

cur.execute(create_table_query)
conn.commit()
print("Table security_events created/verified successfully in Aiven PostgreSQL!")
cur.close()
conn.close()
