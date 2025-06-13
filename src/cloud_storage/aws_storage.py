import boto3
from src.configuration.aws_connection import S3Client
from io import StringIO
from typing import Union, List
import os
import sys
from src.logger import logging
from mypy_boto3_s3.service_resource import Bucket
from src.exception import MyException
from botocore.exceptions import ClientError
from pandas import DataFrame, read_csv
import pickle


class SimpleStorageService:
    """
    A class for interacting with Wasabi (S3-compatible) storage using boto3.
    Supports uploading/downloading files, models, and data frames.
    """

    def __init__(self):
        try:
            s3_client = S3Client()
            self.s3_resource = s3_client.s3_resource
            self.s3_client = s3_client.s3_client
            logging.info(f"Connected to S3 via: {self.s3_client.meta.endpoint_url}")
        except Exception as e:
            raise MyException(e, sys)

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [obj for obj in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects) > 0
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_object(object_name: object, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str]:
        try:
            # Handle case where object_name is already the content
            if hasattr(object_name, 'read'):
                content = object_name.read()
                if decode and isinstance(content, bytes):
                    content = content.decode()
                return StringIO(content) if make_readable else content
                
            # Handle case where object_name is a dict with a 'Body' key
            if isinstance(object_name, dict) and 'Body' in object_name:
                content = object_name['Body'].read()
                if decode and isinstance(content, bytes):
                    content = content.decode()
                return StringIO(content) if make_readable else content
                
            # Handle case where object_name is an S3 object
            if hasattr(object_name, 'get') and callable(getattr(object_name, 'get')):
                content = object_name.get()["Body"].read()
                if decode and isinstance(content, bytes):
                    content = content.decode()
                return StringIO(content) if make_readable else content
                
            # If we get here, we don't know how to handle the object
            raise ValueError(f"Unsupported object type: {type(object_name)}")
            
        except Exception as e:
            raise MyException(e, sys) from e

    def get_bucket(self, bucket_name: str) -> Bucket:
        try:
            return self.s3_resource.Bucket(bucket_name)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:
        try:
            bucket = self.get_bucket(bucket_name)
            objects = [obj for obj in bucket.objects.filter(Prefix=filename)]
            return objects[0] if len(objects) == 1 else objects
        except Exception as e:
            raise MyException(e, sys) from e

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        try:
            model_key = f"{model_dir}/{model_name}" if model_dir else model_name
            file_object = self.get_file_object(model_key, bucket_name)
            model_obj = self.read_object(file_object, decode=False)
            model = pickle.loads(model_obj)
            logging.info("Production model loaded from S3 bucket.")
            return model
        except Exception as e:
            raise MyException(e, sys) from e

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        try:
            # Attempt to load the folder to check if it exists
            self.s3_resource.Object(bucket_name, folder_name).load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.s3_client.put_object(Bucket=bucket_name, Key=f"{folder_name}/")

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):
        try:
            logging.info(f"Uploading {from_filename} to {to_filename} in {bucket_name}")
            self.s3_client.upload_file(from_filename, bucket_name, to_filename)  # ✅ For Wasabi use client
            if remove:
                os.remove(from_filename)
                logging.info(f"Removed local file {from_filename} after upload")
        except Exception as e:
            raise MyException(e, sys) from e

    def upload_df_as_csv(self, data_frame: DataFrame, local_filename: str, bucket_filename: str, bucket_name: str):
        try:
            data_frame.to_csv(local_filename, index=False)
            self.upload_file(local_filename, bucket_filename, bucket_name)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_df_from_object(self, object_: object) -> DataFrame:
        try:
            content = self.read_object(object_, make_readable=True)
            return read_csv(content, na_values="na")
        except Exception as e:
            raise MyException(e, sys) from e

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            return self.get_df_from_object(csv_obj)
        except Exception as e:
            raise MyException(e, sys) from e
