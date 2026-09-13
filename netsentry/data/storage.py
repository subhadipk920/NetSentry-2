import os
import sys
import boto3
from botocore.config import Config
from dotenv import load_dotenv
load_dotenv()


class ProgressPercentage:
    def __init__(self, filename: str, size: float = None):
        self._filename = filename
        if size is not None:
            self._size = float(size)
        else:
            self._size = float(os.path.getsize(filename)) if os.path.exists(filename) else 0.0
        self._seen_so_far = 0

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        if self._size > 0:
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                f"\r{self._filename}: {self._seen_so_far / (1024*1024):.2f}MB / {self._size / (1024*1024):.2f}MB ({percentage:.1f}%)"
            )
        else:
            sys.stdout.write(f"\r{self._filename}: {self._seen_so_far / (1024*1024):.2f}MB downloaded")
        sys.stdout.flush()

def upload_file_to_r2(local_path: str, r2_path: str):
    r2 = boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )
    bucket = os.getenv("R2_BUCKET_NAME", "netsentry")
    print(f"Connecting to Cloudflare R2 bucket '{bucket}'...")
    r2.upload_file(local_path, bucket, r2_path, Callback=ProgressPercentage(local_path))
    print(f"\nSuccessfully uploaded to {bucket}/{r2_path}!")


def download_file_from_r2(r2_path: str, local_path: str):
    """Downloads a file directly from Cloudflare R2 using R2_* environment variables."""
    # Ensure destination directory exists
    dir_path = os.path.dirname(local_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    r2 = boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )
    bucket = os.getenv("R2_BUCKET_NAME", "netsentry")
    print(f"Downloading from Cloudflare R2 bucket '{bucket}'...")
    
    # Get remote file size for accurate progress calculation
    try:
        response = r2.head_object(Bucket=bucket, Key=r2_path)
        total_size = response.get("ContentLength", 0)
    except Exception:
        total_size = 0

    r2.download_file(bucket, r2_path, local_path, Callback=ProgressPercentage(local_path, size=total_size))
    print(f"\nSuccessfully downloaded {bucket}/{r2_path} -> {local_path}")