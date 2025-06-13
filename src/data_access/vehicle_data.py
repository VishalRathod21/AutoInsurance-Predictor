import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException

class Proj1Data:
    """
    A class to export MongoDB records as a pandas DataFrame.
    """

    def __init__(self) -> None:
        """
        Initializes the MongoDB client connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Exports an entire MongoDB collection as a pandas DataFrame.

        Parameters:
        ----------
        collection_name : str
            The name of the MongoDB collection to export.
        database_name : Optional[str]
            Name of the database (optional). Defaults to DATABASE_NAME.


        Returns:
        -------
        pd.DataFrame
            DataFrame containing the collection data, with '_id' column removed and 'na' values replaced with NaN.
        """
        try:
            print(f"\n=== Starting data export from MongoDB ===")
            print(f"Database: {database_name or 'default'}, Collection: {collection_name}")
            
            # Access specified collection from the default or specified database
            try:
                if database_name is None:
                    collection = self.mongo_client.database[collection_name]
                else:
                    collection = self.mongo_client[database_name][collection_name]
                print("Successfully accessed collection")
            except Exception as e:
                print(f"Error accessing collection: {str(e)}")
                raise

            # First, let's check if the collection exists and has documents
            try:
                doc_count = collection.count_documents({})
                print(f"Number of documents in collection: {doc_count}")
                
                if doc_count == 0:
                    print("Warning: The collection exists but is empty!")
                    return pd.DataFrame()
                    
            except Exception as e:
                print(f"Error counting documents: {str(e)}")
                raise

            # Try to fetch a sample document to check the structure
            try:
                sample_doc = collection.find_one()
                if sample_doc:
                    print("Sample document keys:", list(sample_doc.keys()))
                else:
                    print("Warning: Could not fetch a sample document!")
            except Exception as e:
                print(f"Error fetching sample document: {str(e)}")

            # Fetch all documents
            print("Starting to fetch all documents...")
            try:
                cursor = collection.find()
                print("Cursor created, converting to list...")
                docs = list(cursor)
                print(f"Successfully converted {len(docs)} documents to list")
                
                if not docs:
                    print("No documents found in the cursor!")
                    return pd.DataFrame()
                    
                print("Creating DataFrame...")
                df = pd.DataFrame(docs)
                print(f"Created DataFrame with shape: {df.shape}")
                
                # Clean up the DataFrame
                if "_id" in df.columns:
                    print("Dropping _id column")
                    df = df.drop(columns=["_id"], axis=1)
                if "id" in df.columns:
                    print("Dropping id column")
                    df = df.drop(columns=["id"], axis=1)
                
                print("Replacing 'na' strings with np.nan...")
                df.replace({"na": np.nan}, inplace=True)
                
                print("=== Data export completed successfully ===\n")
                return df
                
            except Exception as e:
                print(f"Error during data fetching/processing: {str(e)}")
                raise

        except Exception as e:
            print(f"Unexpected error in export_collection_as_dataframe: {str(e)}")
            raise MyException(e, sys)