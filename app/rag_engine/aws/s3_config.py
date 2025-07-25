import os
from botocore.config import Config

class S3Config:
    def __init__(self):
        self.access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket = os.getenv("S3_BUCKET", "ragbot-conversations")
        self.retry_config = Config(retries={"max_attempts": 5})
        self.validate()

    def validate(self):
        if not all([self.access_key_id, self.secret_access_key, self.bucket, self.region]):
            raise ValueError("Missing required AWS configuration in environment variables.")

    def get_client(self):
        import boto3
        return boto3.client(
            "s3",
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region,
            config=self.retry_config
        )
