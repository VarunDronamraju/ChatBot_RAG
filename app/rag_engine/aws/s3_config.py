import os
from botocore.config import Config
from botocore.exceptions import ClientError
import boto3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Config:
    """Handles AWS S3 configuration using environment variables and Boto3."""
    
    def __init__(self, bucket_name=None):
        """Initialize S3 configuration with environment variables."""
        # Load credentials and configuration from environment variables
        self.access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.session_token = os.getenv("AWS_SESSION_TOKEN")  # Support temporary credentials
        self.region = os.getenv("AWS_REGION", "us-east-1")  # Default region
        self.bucket = bucket_name or os.getenv("S3_BUCKET", "ragbot-conversations")  # Allow override
        self.endpoint_url = os.getenv("AWS_ENDPOINT_URL")  # Support custom endpoints (e.g., LocalStack, MinIO)
        
        # Configure retry behavior
        self.retry_config = Config(
            retries={
                "max_attempts": int(os.getenv("AWS_MAX_RETRIES", 5)),
                "mode": "standard"  # Options: legacy, standard, adaptive
            },
            region_name=self.region
        )
        
        # Validate configuration
        self.validate()

    def validate(self):
        """Validate required configuration parameters."""
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
        
        # Verify bucket exists and is accessible
        try:
            client = self.get_client()
            client.head_bucket(Bucket=self.bucket)
            logger.info(f"Successfully validated access to bucket: {self.bucket}")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.error(f"Failed to validate bucket {self.bucket}: {error_code} - {str(e)}")
            raise ValueError(f"Cannot access S3 bucket {self.bucket}: {str(e)}")

    def get_client(self):
        """Create and return an S3 client with the configured settings."""
        try:
            session = boto3.Session(
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                aws_session_token=self.session_token,
                region_name=self.region
            )
            client = session.client(
                "s3",
                endpoint_url=self.endpoint_url,
                config=self.retry_config
            )
            return client
        except Exception as e:
            logger.error(f"Failed to create S3 client: {str(e)}")
            raise RuntimeError(f"Failed to initialize S3 client: {str(e)}")

    def get_resource(self):
        """Create and return an S3 resource with the configured settings."""
        try:
            session = boto3.Session(
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                aws_session_token=self.session_token,
                region_name=self.region
            )
            resource = session.resource(
                "s3",
                endpoint_url=self.endpoint_url,
                config=self.retry_config
            )
            return resource
        except Exception as e:
            logger.error(f"Failed to create S3 resource: {str(e)}")
            raise RuntimeError(f"Failed to initialize S3 resource: {str(e)}")
def get_s3_client():
    return S3Config().get_client()
