import boto3
import os
from src.constants import (
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    AWS_ACCESS_KEY_ID_ENV_KEY,
    REGION_NAME,
    WASABI_ENDPOINT_URL  # ✅ Add this to your constants
)


class S3Client:

    s3_client = None
    s3_resource = None

    def __init__(self, region_name=REGION_NAME):
        """
        This Class gets Wasabi credentials from environment variables and creates a connection with S3-compatible bucket.
        Raises exception when required environment variables are not set.
        """

        if S3Client.s3_resource is None or S3Client.s3_client is None:
            __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)

            if __access_key_id is None:
                raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not set.")
            if __secret_access_key is None:
                raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set.")

            S3Client.s3_resource = boto3.resource(
                service_name='s3',
                region_name=region_name,
                endpoint_url=WASABI_ENDPOINT_URL,  # ✅ Required for Wasabi
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key
            )

            S3Client.s3_client = boto3.client(
                service_name='s3',
                region_name=region_name,
                endpoint_url=WASABI_ENDPOINT_URL,  # ✅ Required for Wasabi
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key
            )

        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
