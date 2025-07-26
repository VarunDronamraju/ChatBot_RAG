"""
S3 uploader for RAGBot.
Handles uploading and downloading documents, metadata, conversations, and messages to/from S3.
Location: app/rag_engine/aws/s3_uploader.py
"""

import os
import json
from botocore.exceptions import ClientError
from app.rag_engine.aws.s3_utils import S3Utils
from app.utils.logger import get_logger

logger = get_logger(__name__)

class S3Uploader:
    """Handles S3 uploads and downloads for RAGBot documents and conversations."""

    def __init__(self, bucket_name=None):
        self.s3_utils = S3Utils(bucket_name)
        self.disable_s3 = os.getenv("DISABLE_S3", "false").lower() == "true"
        if self.disable_s3:
            logger.info("S3 uploads disabled via DISABLE_S3 environment variable")

    def upload_document(self, user_id, doc_id, file_path):
        """Upload a document to S3."""
        if self.disable_s3:
            logger.debug(f"S3 upload skipped for document {doc_id} due to DISABLE_S3")
            return None
        key = f"users/{user_id}/uploads/{doc_id}.pdf"
        try:
            self.s3_utils.s3_client.upload_file(file_path, self.s3_utils.bucket, key)
            logger.info(f"Uploaded document {doc_id} to s3://{self.s3_utils.bucket}/{key}")
            return key
        except ClientError as e:
            logger.error(f"Failed to upload document {doc_id}: {str(e)}")
            raise

    def download_document(self, user_id, doc_id, local_path):
        """Download a document from S3."""
        if self.disable_s3:
            logger.debug(f"S3 download skipped for document {doc_id} due to DISABLE_S3")
            return None
        key = f"users/{user_id}/uploads/{doc_id}.pdf"
        try:
            self.s3_utils.s3_client.download_file(self.s3_utils.bucket, key, local_path)
            logger.info(f"Downloaded document {doc_id} from s3://{self.s3_utils.bucket}/{key} to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download document {doc_id}: {str(e)}")
            raise

    def upload_metadata(self, doc_id, metadata):
        """Upload document metadata to S3 as JSON."""
        if self.disable_s3:
            logger.debug(f"S3 upload skipped for metadata {doc_id} due to DISABLE_S3")
            return None
        key = f"metadata/{doc_id}.json"
        try:
            self.s3_utils.s3_client.put_object(
                Bucket=self.s3_utils.bucket,
                Key=key,
                Body=json.dumps(metadata, indent=2),
                ContentType="application/json"
            )
            logger.info(f"Uploaded metadata {doc_id} to s3://{self.s3_utils.bucket}/{key}")
            return key
        except ClientError as e:
            logger.error(f"Failed to upload metadata {doc_id}: {str(e)}")
            raise

    def upload_conversation(self, user_id, conv_id, data):
        """Upload conversation data to S3 as JSON."""
        if self.disable_s3:
            logger.debug(f"S3 upload skipped for conversation {conv_id} due to DISABLE_S3")
            return None
        key = f"users/{user_id}/conversations/{conv_id}.json"
        try:
            self.s3_utils.s3_client.put_object(
                Bucket=self.s3_utils.bucket,
                Key=key,
                Body=json.dumps(data, indent=2),
                ContentType="application/json"
            )
            logger.info(f"Uploaded conversation {conv_id} to s3://{self.s3_utils.bucket}/{key}")
            return key
        except ClientError as e:
            logger.error(f"Failed to upload conversation {conv_id}: {str(e)}")
            raise

    def upload_message(self, user_id, conv_id, message_id, data):
        """Upload message data to S3 as JSON."""
        if self.disable_s3:
            logger.debug(f"S3 upload skipped for message {message_id} due to DISABLE_S3")
            return None
        key = f"users/{user_id}/messages/{conv_id}/{message_id}.json"
        try:
            self.s3_utils.s3_client.put_object(
                Bucket=self.s3_utils.bucket,
                Key=key,
                Body=json.dumps(data, indent=4),
                ContentType="application/json"
            )
            logger.info(f"Uploaded message {message_id} to s3://{self.s3_utils.bucket}/{key}")
            return key
        except ClientError as e:
            logger.error(f"Failed to upload message {message_id}: {str(e)}")
            raise