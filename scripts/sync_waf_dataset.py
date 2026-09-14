import os
import urllib.request
import boto3
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://huggingface.co/datasets/puyang2025/waf_data_v2/resolve/main"
FILES = ["train.parquet", "eval.parquet", "test.parquet"]
DATA_DIR = "data/waf"
os.makedirs(DATA_DIR, exist_ok=True)

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("R2_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
bucket = os.getenv("R2_BUCKET_NAME", "netsentry")

for fname in FILES:
    url = f"{BASE_URL}/{fname}"
    local_path = os.path.join(DATA_DIR, fname)
    r2_key = f"data/waf/{fname}"
    
    print(f"\n1. Downloading {fname} from Hugging Face...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(local_path, "wb") as out_f:
        out_f.write(resp.read())
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"Downloaded {fname} ({size_mb:.2f} MB)")
    
    print(f"2. Uploading {fname} to Cloudflare R2: s3://{bucket}/{r2_key}...")
    s3.upload_file(local_path, bucket, r2_key)
    print(f"Successfully uploaded {r2_key} to R2!")

print("\nAll WAF Data v2 files successfully uploaded to Cloudflare R2!")
