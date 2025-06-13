import json
import sys
import os
import yaml

import pandas as pd

from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config =read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns in the dataframe matches the schema
        
        Output      :   Returns bool indicating if column counts match
        On Failure  :   Logs the mismatch and returns False
        """
        try:
            if 'columns' not in self._schema_config:
                logging.warning("No 'columns' key found in schema config")
                return False
                
            schema_columns = [list(col.keys())[0] for col in self._schema_config["columns"]]
            df_columns = dataframe.columns.tolist()
            
            # Check if number of columns match
            if len(df_columns) != len(schema_columns):
                logging.warning(
                    f"Column count mismatch. Expected {len(schema_columns)} columns "
                    f"but found {len(df_columns)} columns"
                )
                logging.info(f"Schema columns: {schema_columns}")
                logging.info(f"DataFrame columns: {df_columns}")
                return False
                
            # Check if all required columns are present (even if in different order)
            missing_columns = [col for col in schema_columns if col not in df_columns]
            if missing_columns:
                logging.warning(f"Missing required columns: {missing_columns}")
                return False
                
            logging.info(f"All {len(schema_columns)} required columns are present")
            return True
            
        except Exception as e:
            logging.error(f"Error in validate_number_of_columns: {str(e)}")
            logging.exception("Detailed error:")
            return False

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence and types of numerical and categorical columns
        
        Output      :   Returns bool indicating if all required columns with correct types exist
        On Failure  :   Logs detailed error messages and returns False
        """
        try:
            dataframe_columns = set(df.columns)
            validation_errors = []
            
            # Check numerical columns
            if 'numerical_columns' not in self._schema_config:
                logging.warning("No 'numerical_columns' key found in schema config")
                return False
                
            schema_numerical_columns = set(self._schema_config["numerical_columns"])
            missing_numerical = schema_numerical_columns - dataframe_columns
            if missing_numerical:
                validation_errors.append(f"Missing numerical columns: {sorted(missing_numerical)}")
            
            # Check categorical columns
            if 'categorical_columns' not in self._schema_config:
                logging.warning("No 'categorical_columns' key found in schema config")
                return False
                
            schema_categorical_columns = set(self._schema_config["categorical_columns"])
            missing_categorical = schema_categorical_columns - dataframe_columns
            if missing_categorical:
                validation_errors.append(f"Missing categorical columns: {sorted(missing_categorical)}")
            
            # Check for extra columns not in schema
            all_schema_columns = schema_numerical_columns.union(schema_categorical_columns)
            extra_columns = dataframe_columns - all_schema_columns
            if extra_columns:
                logging.info(f"Extra columns found (not in schema): {sorted(extra_columns)}")
            
            # Log detailed column information
            logging.info(f"DataFrame columns: {sorted(dataframe_columns)}")
            logging.info(f"Schema numerical columns: {sorted(schema_numerical_columns)}")
            logging.info(f"Schema categorical columns: {sorted(schema_categorical_columns)}")
            
            if validation_errors:
                for error in validation_errors:
                    logging.warning(error)
                return False
                
            logging.info("All required numerical and categorical columns are present")
            return True
            
        except Exception as e:
            logging.error(f"Error in is_column_exist: {str(e)}")
            logging.exception("Detailed error:")
            return False

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns DataValidationArtifact with validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            validation_errors = []
            logging.info("Starting data validation")
            
            # Read train and test data
            train_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            
            logging.info(f"Training data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")

            # Validate number of columns in train and test data
            train_cols_status = self.validate_number_of_columns(dataframe=train_df)
            test_cols_status = self.validate_number_of_columns(dataframe=test_df)
            
            if not train_cols_status:
                validation_errors.append("Number of columns in training data does not match schema")
            if not test_cols_status:
                validation_errors.append("Number of columns in test data does not match schema")

            # Validate column existence and types in train and test data
            train_columns_status = self.is_column_exist(df=train_df)
            test_columns_status = self.is_column_exist(df=test_df)
            
            if not train_columns_status:
                validation_errors.append("Missing or invalid columns in training data")
            if not test_columns_status:
                validation_errors.append("Missing or invalid columns in test data")
            
            # Check if all validations passed
            validation_passed = len(validation_errors) == 0
            validation_message = "Data validation passed successfully" if validation_passed else ". ".join(validation_errors)
            
            # Create validation report
            validation_report = {
                "validation_status": validation_passed,
                "message": validation_message,
                "train_columns": list(train_df.columns),
                "test_columns": list(test_df.columns),
                "schema_columns": [list(col.keys())[0] for col in self._schema_config["columns"]] if 'columns' in self._schema_config else []
            }
            
            # Ensure the directory for validation_report_file_path exists
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)
            
            # Save validation report
            with open(self.data_validation_config.validation_report_file_path, 'w') as f:
                yaml.dump(validation_report, f)
            
            logging.info(f"Validation completed. Status: {'Passed' if validation_passed else 'Failed'}")
            logging.info(f"Validation message: {validation_message}")
            
            # Create and return validation artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_passed,
                message=validation_message,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            logging.info("Data validation artifact created and saved to YAML file.")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e