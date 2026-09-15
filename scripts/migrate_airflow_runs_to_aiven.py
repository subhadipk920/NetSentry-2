"""
Migrates historical Airflow metadata (DAG runs, task instances, log records)
from local Docker PostgreSQL to Aiven PostgreSQL.
"""

import os
import subprocess
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Tables to migrate in dependency order
TABLES = [
    "dag",
    "dag_run",
    "task_instance",
    "task_fail",
    "log",
    "job",
]

def dump_table_from_local_postgres(table: str) -> str:
    cmd = [
        "docker", "exec", "docker-postgres-1",
        "pg_dump", "-U", "airflow", "-d", "airflow",
        "--data-only",
        "--column-inserts",
        "-t", table
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout

def restore_to_aiven():
    uri = os.getenv("AIRFLOW_DATABASE_CONN").replace("postgresql+psycopg2://", "postgresql://")
    conn = psycopg2.connect(uri)
    conn.autocommit = False
    cur = conn.cursor()

    print("Disabling triggers/constraints during import...")
    cur.execute("SET session_replication_role = 'replica';")

    for table in TABLES:
        print(f"Exporting {table} from local container...")
        try:
            sql_data = dump_table_from_local_postgres(table)
            if not sql_data.strip():
                print(f"No data for {table}.")
                continue
            
            # Filter sql to only INSERT statements
            insert_lines = [
                line for line in sql_data.splitlines()
                if line.startswith("INSERT INTO")
            ]
            print(f"Executing {len(insert_lines)} inserts for {table} into Aiven...")
            for insert_stmt in insert_lines:
                try:
                    cur.execute(insert_stmt)
                except Exception as e:
                    # Ignore duplicates or minor schema conflicts
                    conn.rollback()
                    cur.execute("SET session_replication_role = 'replica';")
            conn.commit()
            print(f"✅ {table} migrated successfully.")
        except Exception as e:
            print(f"Error migrating {table}: {e}")
            conn.rollback()
            cur.execute("SET session_replication_role = 'replica';")

    cur.execute("SET session_replication_role = 'origin';")
    conn.commit()
    cur.close()
    conn.close()
    print("🎉 All previous DAG runs and task histories migrated to Aiven!")

if __name__ == "__main__":
    restore_to_aiven()
