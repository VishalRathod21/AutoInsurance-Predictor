import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from dataclasses import asdict
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging, log_function_call
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

class DataTransformationError(Exception):
    """Custom exception for data transformation errors."""
    pass


class DataTransformation:
    """
    Class for handling all data transformation operations.
    
    This class is responsible for:
    - Loading and validating input data
    - Applying feature engineering and preprocessing
    - Handling class imbalance
    - Saving transformed data and preprocessing objects
    """
    
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        Initialize DataTransformation with required artifacts and configurations.
        
        Args:
            data_ingestion_artifact: Artifact containing paths to input data files
            data_transformation_config: Configuration for data transformation
            data_validation_artifact: Artifact containing validation results
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            
            # Load schema configuration
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            if not self._schema_config:
                raise DataTransformationError("Failed to load schema configuration")
                
            logging.info("DataTransformation initialized with provided configurations")
            
        except Exception as e:
            error_msg = f"Error initializing DataTransformation: {str(e)}"
            logging.error(error_msg)
            raise DataTransformationError(error_msg) from e

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Read data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded DataFrame
            
        Raises:
            DataTransformationError: If file not found or invalid format
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
                
            logging.info(f"Reading data from {file_path}")
            df = pd.read_csv(file_path)
            
            if df.empty:
                raise ValueError(f"Empty DataFrame loaded from {file_path}")
                
            logging.info(f"Successfully loaded data with shape: {df.shape}")
            return df
            
        except Exception as e:
            error_msg = f"Error reading data from {file_path}: {str(e)}"
            logging.error(error_msg)
            raise DataTransformationError(error_msg) from e

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data.
        
        The transformer includes:
        - Standard scaling for numerical features
        - Min-max scaling for specified columns
        
        Returns:
            Pipeline: Configured scikit-learn pipeline
            
        Raises:
            DataTransformationError: If there's an error creating the transformer
        """
        method_name = self.get_data_transformer_object.__name__
        logging.info(f"Entered {method_name} of DataTransformation class")

        try:
            # Validate schema configuration
            if 'num_features' not in self._schema_config or 'mm_columns' not in self._schema_config:
                raise ValueError("Missing required schema configurations: 'num_features' or 'mm_columns'")
                
            # Get features from schema
            num_features = self._schema_config.get('num_features', [])
            mm_columns = self._schema_config.get('mm_columns', [])
            
            logging.info(f"Configuring transformers for {len(num_features)} numerical and {len(mm_columns)} min-max features")
            
            # Initialize transformers with validation
            transformers = []
            
            if num_features:
                numeric_transformer = StandardScaler()
                transformers.append(("StandardScaler", numeric_transformer, num_features))
                logging.info(f"Added StandardScaler for features: {num_features}")
                
            if mm_columns:
                min_max_scaler = MinMaxScaler()
                transformers.append(("MinMaxScaler", min_max_scaler, mm_columns))
                logging.info(f"Added MinMaxScaler for features: {mm_columns}")
            
            if not transformers:
                logging.warning("No transformers were configured. Check your schema configuration.")
            
            # Create preprocessor with dynamic transformers
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'  # Preserve columns not specified in transformers
            )
            
            # Create and return the pipeline
            pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            
            logging.info(f"Successfully created data transformation pipeline with {len(transformers)} transformers")
            return pipeline
            
        except Exception as e:
            error_msg = f"Error in {method_name}: {str(e)}"
            logging.error(error_msg, exc_info=True)
            raise DataTransformationError(error_msg) from e

    def _map_gender_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map Gender column to binary values (0 for Female, 1 for Male).
        
        Args:
            df: Input DataFrame containing the 'Gender' column
            
        Returns:
            DataFrame with transformed 'Gender' column
            
        Raises:
            DataTransformationError: If the 'Gender' column is missing or transformation fails
        """
        method_name = self._map_gender_column.__name__
        logging.info(f"Starting {method_name}")
        
        try:
            column_name = 'Gender'
            
            # Check if column exists
            if column_name not in df.columns:
                error_msg = f"Required column '{column_name}' not found in dataframe"
                logging.error(error_msg)
                raise DataTransformationError(error_msg)
            
            # Log initial value distribution
            value_counts = df[column_name].value_counts(dropna=False)
            logging.info(f"Initial {column_name} value counts:\n{value_counts}")
            
            # Handle missing values
            if df[column_name].isnull().any():
                missing_count = df[column_name].isnull().sum()
                logging.warning(f"Found {missing_count} missing values in '{column_name}'. Filling with mode.")
                mode_value = df[column_name].mode()[0]
                df[column_name] = df[column_name].fillna(mode_value)
                logging.info(f"Filled {missing_count} missing values with mode: {mode_value}")
            
            # Convert to string and strip whitespace for consistent processing
            df[column_name] = df[column_name].astype(str).str.strip().str.title()
            
            # Define valid mappings (case-insensitive)
            valid_mappings = {
                'Female': 0,
                'Male': 1,
                'F': 0,
                'M': 1,
                '0': 0,
                '1': 1
            }
            
            # Map values
            df[column_name] = df[column_name].map(valid_mappings)
            
            # Handle any remaining invalid values (None values from unmapped keys)
            if df[column_name].isnull().any():
                invalid_count = df[column_name].isnull().sum()
                logging.warning(f"Found {invalid_count} invalid values in '{column_name}'. Replacing with mode.")
                mode_value = df[column_name].mode()[0] if not df[column_name].mode().empty else 0
                df[column_name] = df[column_name].fillna(mode_value)
            
            # Convert to int8 to save memory
            df[column_name] = df[column_name].astype('int8')
            
            # Log final distribution
            final_counts = df[column_name].value_counts()
            logging.info(f"Final {column_name} value counts (0=Female, 1=Male):\n{final_counts}")
            
            return df
            
        except Exception as e:
            error_msg = f"Error in {method_name}: {str(e)}"
            logging.error(error_msg, exc_info=True)
            raise DataTransformationError(error_msg) from e

    def _create_dummy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create dummy variables for categorical features.
        
        Args:
            df: Input DataFrame containing categorical columns
            
        Returns:
            DataFrame with dummy-encoded categorical columns
            
        Raises:
            DataTransformationError: If there's an error during dummy variable creation
        """
        method_name = self._create_dummy_columns.__name__
        logging.info(f"Starting {method_name}")
        
        try:
            # Get categorical columns from schema
            categorical_columns = self._schema_config.get('categorical_columns', [])
            
            if not categorical_columns:
                logging.warning("No categorical columns defined in schema for dummy creation")
                return df
                
            logging.info(f"Categorical columns from schema: {categorical_columns}")
            
            # Filter out columns that don't exist in the dataframe
            missing_columns = [col for col in categorical_columns if col not in df.columns]
            if missing_columns:
                logging.warning(f"Categorical columns in schema but not in dataframe: {missing_columns}")
                
            categorical_columns = [col for col in categorical_columns if col in df.columns]
            
            if not categorical_columns:
                logging.warning("No valid categorical columns found in dataframe for dummy creation")
                return df
                
            logging.info(f"Creating dummy variables for columns: {categorical_columns}")
            
            # Store original dtypes for reference
            original_dtypes = df[categorical_columns].dtypes
            logging.info(f"Original dtypes of categorical columns:\n{original_dtypes}")
            
            # Convert all categorical columns to string type to avoid issues with mixed types
            for col in categorical_columns:
                # Handle potential mixed types by converting to string first
                df[col] = df[col].astype(str).str.strip()
                
                # Log value counts for each categorical column
                value_counts = df[col].value_counts(dropna=False)
                logging.info(f"Value counts for {col} before dummies:\n{value_counts}")
            
            # Create dummy variables
            df = pd.get_dummies(
                df, 
                columns=categorical_columns, 
                drop_first=True,
                dtype='int8'  # Use int8 to save memory
            )
            
            # Log the newly created dummy columns
            new_columns = [col for col in df.columns if any(f"{cat_col}_" in col for cat_col in categorical_columns)]
            logging.info(f"Created {len(new_columns)} dummy columns")
            
            return df
            
        except Exception as e:
            error_msg = f"Error in {method_name}: {str(e)}"
            logging.error(error_msg, exc_info=True)
            raise DataTransformationError(error_msg) from e

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename specific columns and ensure proper data types for dummy columns.
        
        Args:
            df: Input DataFrame with columns to be renamed
            
        Returns:
            DataFrame with renamed and properly typed columns
            
        Raises:
            DataTransformationError: If there's an error during column renaming or type conversion
        """
        method_name = self._rename_columns.__name__
        logging.info(f"Starting {method_name}")
        
        try:
            # Define column mappings for renaming
            column_mappings = {
                "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
                "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years",
                "Vehicle_Damage_Yes": "Vehicle_Damaged"
            }
            
            # Log columns before renaming
            logging.info(f"Columns before renaming: {df.columns.tolist()}")
            
            # Apply renames for columns that exist
            rename_dict = {k: v for k, v in column_mappings.items() if k in df.columns}
            if rename_dict:
                logging.info(f"Renaming columns: {rename_dict}")
                df = df.rename(columns=rename_dict)
            
            # Define columns that should be converted to int
            int_columns = [
                "Vehicle_Age_lt_1_Year", 
                "Vehicle_Age_gt_2_Years", 
                "Vehicle_Damaged"
            ]
            
            # Process each column that needs type conversion
            for col in int_columns:
                if col in df.columns:
                    try:
                        # Log current dtype and sample values
                        logging.info(f"Converting column '{col}' to int. Current dtype: {df[col].dtype}")
                        
                        # Handle potential non-numeric values
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            logging.warning(f"Column '{col}' is not numeric. Attempting conversion...")
                            
                            # Convert to string, then to numeric, coerce errors to NaN
                            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), 
                                                 errors='coerce')
                            
                            # Count and handle any NaN values
                            nan_count = df[col].isna().sum()
                            if nan_count > 0:
                                logging.warning(f"Found {nan_count} NaN values in '{col}' after conversion. Filling with 0.")
                                df[col] = df[col].fillna(0)
                        
                        # Convert to int8 to save memory
                        df[col] = df[col].astype('int8')
                        
                        # Log conversion result
                        logging.info(f"Successfully converted '{col}' to {df[col].dtype}")
                        
                    except Exception as e:
                        error_msg = f"Error converting column '{col}' to int: {str(e)}"
                        logging.error(error_msg, exc_info=True)
                        raise DataTransformationError(error_msg) from e
            
            # Log final column names and dtypes
            logging.info(f"Final columns after renaming: {df.columns.tolist()}")
            logging.info(f"Final dtypes after conversion:\n{df.dtypes}")
            
            return df
            
        except Exception as e:
            error_msg = f"Error in {method_name}: {str(e)}"
            logging.error(error_msg, exc_info=True)
            raise DataTransformationError(error_msg) from e

    def _drop_id_column(self, df):
        """Drop the specified columns if they exist."""
        try:
            drop_cols = self._schema_config.get('drop_columns', [])
            if not isinstance(drop_cols, list):
                drop_cols = [drop_cols]
                
            for col in drop_cols:
                if col in df.columns:
                    logging.info(f"Dropping column: {col}")
                    df = df.drop(col, axis=1)
                else:
                    logging.warning(f"Column '{col}' not found in dataframe to drop")
            return df
        except Exception as e:
            logging.error(f"Error in _drop_id_column: {str(e)}")
            raise

    def _validate_dataframe_columns(self, df: pd.DataFrame, df_name: str = 'DataFrame') -> None:
        """Validate that required columns exist in the dataframe."""
        required_columns = set()
        
        # Get columns from schema
        if 'columns' in self._schema_config:
            required_columns.update(col for col_dict in self._schema_config['columns'] for col in col_dict.keys())
        
        # Add target column if not in schema columns
        if TARGET_COLUMN not in required_columns:
            required_columns.add(TARGET_COLUMN)
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Missing required columns in {df_name}: {missing_columns}"
            logging.error(error_msg)
            logging.error(f"Available columns in {df_name}: {df.columns.tolist()}")
            raise ValueError(error_msg)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            
            # Validate data validation status
            if not self.data_validation_artifact.validation_status:
                error_msg = f"Data validation failed: {self.data_validation_artifact.message}"
                logging.error(error_msg)
                raise ValueError(error_msg)

            # Load train and test data
            logging.info(f"Loading training data from: {self.data_ingestion_artifact.trained_file_path}")
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            logging.info(f"Training data shape: {train_df.shape}")
            
            logging.info(f"Loading test data from: {self.data_ingestion_artifact.test_file_path}")
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info(f"Test data shape: {test_df.shape}")
            
            # Validate columns in both dataframes
            self._validate_dataframe_columns(train_df, 'training data')
            self._validate_dataframe_columns(test_df, 'test data')

            # Separate features and target
            logging.info("Splitting features and target")
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            logging.info(f"Training features shape: {input_feature_train_df.shape}, target shape: {target_feature_train_df.shape}")
            logging.info(f"Test features shape: {input_feature_test_df.shape}, target shape: {target_feature_test_df.shape}")

            # Apply custom transformations in specified sequence
            try:
                logging.info("Applying transformations to training data")
                input_feature_train_df = self._map_gender_column(input_feature_train_df)
                input_feature_train_df = self._drop_id_column(input_feature_train_df)
                input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
                input_feature_train_df = self._rename_columns(input_feature_train_df)
                logging.info("Transformations applied to training data")
                
                logging.info("Applying transformations to test data")
                input_feature_test_df = self._map_gender_column(input_feature_test_df)
                input_feature_test_df = self._drop_id_column(input_feature_test_df)
                input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
                input_feature_test_df = self._rename_columns(input_feature_test_df)
                logging.info("Transformations applied to test data")
                
                # Ensure both train and test have the same columns after transformations
                train_cols = set(input_feature_train_df.columns)
                test_cols = set(input_feature_test_df.columns)
                
                # Add missing columns to test data with 0 values
                for col in (train_cols - test_cols):
                    input_feature_test_df[col] = 0
                
                # Reorder test columns to match train columns
                input_feature_test_df = input_feature_test_df[input_feature_train_df.columns]
                
            except Exception as e:
                logging.error(f"Error during data transformation: {str(e)}")
                raise

            # Get and apply preprocessor
            logging.info("Initializing data preprocessor")
            preprocessor = self.get_data_transformer_object()
            
            logging.info("Fitting preprocessor on training data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Transforming test data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            logging.info(f"Transformed shapes - Train: {input_feature_train_arr.shape}, Test: {input_feature_test_arr.shape}")

            # Handle class imbalance with SMOTEENN
            logging.info("Applying SMOTEENN for handling class imbalance")
            try:
                smt = SMOTEENN(sampling_strategy="minority", random_state=42)
                
                logging.info("Resampling training data")
                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )
                
                logging.info("Resampling test data")
                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )
                
                logging.info(f"After SMOTEENN - Train shape: {input_feature_train_final.shape}, "
                             f"Test shape: {input_feature_test_final.shape}")
                
            except Exception as e:
                logging.error(f"Error during SMOTEENN resampling: {str(e)}")
                # Fallback to original data if SMOTEENN fails
                input_feature_train_final, target_feature_train_final = input_feature_train_arr, target_feature_train_df
                input_feature_test_final, target_feature_test_final = input_feature_test_arr, target_feature_test_df
                logging.warning("Using original data (without SMOTEENN) due to resampling error")

            # Combine features and targets
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_file_path), exist_ok=True)
            
            # Save transformed data and preprocessor
            logging.info("Saving transformation artifacts")
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Data transformation completed successfully")
            
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                is_transformed=True,
                message="Data transformation completed successfully"
            )

        except Exception as e:
            error_msg = f"Error in data transformation: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return DataTransformationArtifact(
                is_transformed=False,
                message=error_msg,
                error=str(e)
            )