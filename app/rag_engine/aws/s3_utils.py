import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import hashlib
from app.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

class S3Utils:
    def __init__(self, bucket_name=None):
        self.access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.session_token = os.getenv("AWS_SESSION_TOKEN")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket = bucket_name or os.getenv("S3_BUCKET", "ragbot-conversations")
        self.endpoint_url = os.getenv("AWS_ENDPOINT_URL")
        self.disable_s3 = os.getenv("DISABLE_S3", "false").lower() == "true"
        self.retry_config = Config(
            retries={"max_attempts": int(os.getenv("AWS_MAX_RETRIES", 5)), "mode": "standard"},
            region_name=self.region
        )
        self.s3_client = self._create_client()
        if not self.disable_s3:
            self._validate()
            self.init_bucket()

    def _create_client(self):
        try:
            session = boto3.Session(
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                aws_session_token=self.session_token,
                region_name=self.region
            )
            client = session.client("s3", endpoint_url=self.endpoint_url, config=self.retry_config)
            logger.debug(f"S3 client created for region {self.region}, bucket {self.bucket}")
            return client
        except Exception as e:
            logger.error(f"Failed to create S3 client: {str(e)}")
            raise RuntimeError(f"Failed to initialize S3 client: {str(e)}")

    def _validate(self):
        required_vars = {
            "AWS_ACCESS_KEY_ID": self.access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.secret_access_key,
            "AWS_REGION": self.region,
            "S3_BUCKET": self.bucket
        }
        missing = [key for key, value in required_vars.items() if not value]
        if missing:
            error_msg = f"Missing required environment variables: {', '.join(missing)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            logger.info(f"Validated access to S3 bucket: {self.bucket}")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.error(f"Failed to validate bucket {self.bucket}: {error_code} - {str(e)}")
            raise ValueError(f"Cannot access S3 bucket {self.bucket}: {str(e)}")

    def init_bucket(self):
        if self.disable_s3:
            logger.debug("Bucket initialization skipped due to DISABLE_S3=true")
            return
        try:
            self.s3_client.create_bucket(
                Bucket=self.bucket,
                CreateBucketConfiguration={"LocationConstraint": self.region}
            )
            logger.info(f"Created S3 bucket: {self.bucket}")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code not in ["BucketAlreadyOwnedByYou", "BucketAlreadyExists"]:
                logger.error(f"Failed to create bucket {self.bucket}: {str(e)}")
                raise
            logger.debug(f"Bucket {self.bucket} already exists")

    def list_keys(self, prefix=""):
        if self.disable_s3:
            logger.debug(f"List keys skipped due to DISABLE_S3=true, prefix: {prefix}")
            return []
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            keys = [obj["Key"] for obj in response.get("Contents", [])]
            logger.debug(f"Listed {len(keys)} keys with prefix '{prefix}' in bucket {self.bucket}")
            return keys
        except ClientError as e:
            logger.error(f"Failed to list keys with prefix '{prefix}': {str(e)}")
            raise

    def check_exists(self, key):
        if self.disable_s3:
            logger.debug(f"Check exists skipped for key {key} due to DISABLE_S3=true")
            return False
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            logger.debug(f"Object exists: s3://{self.bucket}/{key}")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.debug(f"Object does not exist: s3://{self.bucket}/{key}")
                return False
            logger.error(f"Error checking existence of key {key}: {str(e)}")
            raise

    def delete_file(self, key):
        if self.disable_s3:
            logger.debug(f"Delete file skipped for key {key} due to DISABLE_S3=true")
            return
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=key)
            logger.info(f"Deleted object: s3://{self.bucket}/{key}")
        except ClientError as e:
            logger.error(f"Failed to delete key {key}: {str(e)}")
            raise

    def get_file_hash(self, file_path):
        try:
            hasher = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            file_hash = hasher.hexdigest()
            logger.debug(f"Calculated MD5 hash for {file_path}: {file_hash}")
            return file_hash
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {str(e)}")
            raise

    def upload_if_changed(self, key, file_path):
        if self.disable_s3:
            logger.debug(f"Upload skipped for key {key} due to DISABLE_S3=true")
            return False
        try:
            if self.check_exists(key):
                s3_hash = self.s3_client.head_object(Bucket=self.bucket, Key=key).get("ETag", "").strip('"')
                local_hash = self.get_file_hash(file_path)
                if s3_hash == local_hash:
                    logger.debug(f"No changes detected for key {key}, skipping upload")
                    return False
            self.s3_client.upload_file(file_path, self.bucket, key)
            logger.info(f"Uploaded file to s3://{self.bucket}/{key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload file {file_path} to key {key}: {str(e)}")
            raise