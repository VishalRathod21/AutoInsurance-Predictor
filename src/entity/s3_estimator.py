from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.entity.estimator import MyModel
from pandas import DataFrame
import sys


class Proj1Estimator:
    """
    This class handles saving, loading, and using a model stored in a Wasabi S3-compatible bucket.
    """

    def __init__(self, bucket_name: str, model_path: str):
        """
        Initializes the estimator with the S3 bucket name and model path.

        Args:
            bucket_name (str): The name of the S3 bucket where the model is stored.
            model_path (str): The full S3 key path to the model file.
        """
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self.loaded_model: MyModel = None

    def is_model_present(self) -> bool:
        """
        Checks whether the model file exists in the S3 bucket.

        Returns:
            bool: True if model exists, else False.
        """
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=self.model_path)
        except MyException as e:
            print(e)
            return False

    def load_model(self) -> MyModel:
        """
        Loads the model from the S3 bucket into memory.

        Returns:
            MyModel: The loaded model object.
        """
        try:
            model = self.s3.load_model(model_name=self.model_path, bucket_name=self.bucket_name)
            return model
        except Exception as e:
            raise MyException(e, sys)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Uploads a model file from the local system to the S3 bucket.

        Args:
            from_file (str): Local path to the model file.
            remove (bool): Whether to delete the local file after upload.
        """
        try:
            self.s3.upload_file(
                from_filename=from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: DataFrame):
        """
        Performs prediction using the loaded model.

        Args:
            dataframe (DataFrame): Input features for prediction.

        Returns:
            Any: Model prediction results.
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise MyException(e, sys)
