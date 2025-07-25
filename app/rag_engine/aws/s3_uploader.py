from app.rag_engine.aws.s3_utils import S3Utils
import os
import json

class S3Uploader:
    def __init__(self):
        self.s3_utils = S3Utils()

    def upload_document(self, user_id, doc_id, file_path):
        if os.getenv("DISABLE_S3", "false").lower() == "true":
            return
        key = f"users/{user_id}/uploads/{doc_id}.pdf"
        self.s3_utils.s3_client.upload_file(file_path, self.s3_utils.bucket, key)
        return key

    def download_document(self, user_id, doc_id, local_path):
        if os.getenv("DISABLE_S3", "false").lower() == "true":
            return
        key = f"users/{user_id}/uploads/{doc_id}.pdf"
        self.s3_utils.s3_client.download_file(self.s3_utils.bucket, key, local_path)

    def upload_metadata(self, doc_id, metadata):
        if os.getenv("DISABLE_S3", "false").lower() == "true":
            return
        key = f"metadata/{doc_id}.json"
        self.s3_utils.s3_client.put_object(
            Bucket=self.s3_utils.bucket,
            Key=key,
            Body=json.dumps(metadata)
        )

    def upload_conversation(self, user_id, conv_id, data):
        if os.getenv("DISABLE_S3", "false").lower() == "true":
            return
        key = f"users/{user_id}/conversations/{conv_id}.json"
        self.s3_utils.s3_client.put_object(
            Bucket=self.s3_utils.bucket,
            Key=key,
            Body=json.dumps(data)
        )

    def upload_message(self, user_id, conv_id, message_id, data):
        if os.getenv("DISABLE_S3", "false").lower() == "true":
            return
        key = f"users/{user_id}/messages/{conv_id}/{message_id}.json"
        self.s3_utils.s3_client.put_object(
            Bucket=self.s3_utils.bucket,
            Key=key,
            Body=json.dumps(data)
        )
