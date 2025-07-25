import boto3
import os
from botocore.exceptions import ClientError
from botocore.config import Config
import hashlib

class S3Utils:
    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            config=Config(retries={"max_attempts": 5})
        )
        self.bucket = os.getenv("S3_BUCKET", "ragbot-conversations")

    def init_bucket(self):
        try:
            self.s3_client.create_bucket(Bucket=self.bucket)
        except ClientError as e:
            if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                raise

    def list_keys(self, prefix=""):
        response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def check_exists(self, key):
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def delete_file(self, key):
        self.s3_client.delete_object(Bucket=self.bucket, Key=key)

    def get_file_hash(self, file_path):
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def upload_if_changed(self, key, file_path):
        if self.check_exists(key):
            s3_hash = self.s3_client.head_object(Bucket=self.bucket, Key=key).get("ETag", "").strip('"')
            local_hash = self.get_file_hash(file_path)
            if s3_hash == local_hash:
                return False
        self.s3_client.upload_file(file_path, self.bucket, key)
        return True