import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
uri = os.getenv("MLFLOW_TRACKING_URI").replace("postgresql+psycopg2://", "postgresql://")
conn = psycopg2.connect(uri)
cur = conn.cursor()

cur.execute("""
    SELECT pid, datname, usename, client_addr, state
    FROM pg_stat_activity
    WHERE pid != pg_backend_pid() AND datname IS NOT NULL;
""")
rows = cur.fetchall()
print(f"Found {len(rows)} connections to terminate:")
for r in rows:
    print(r)

cur.execute("""
    SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE pid != pg_backend_pid() AND datname IN ('airflow_db', 'mlflow_db');
""")
cur.fetchall()
conn.commit()

cur.execute("SELECT count(*) FROM pg_stat_activity;")
print("Remaining active connections:", cur.fetchone()[0])

cur.close()
conn.close()
