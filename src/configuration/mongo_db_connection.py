import os
import sys
import pymongo
import certifi
from urllib.parse import quote_plus, urlparse, parse_qs, urlunparse

from src.exception import MyException
from src.logger import logging
from src.constants import DATABASE_NAME, MONGODB_URL_KEY

# Load the certificate authority file to avoid timeout errors when connecting to MongoDB
ca = certifi.where()

class MongoDBClient:
    """
    MongoDBClient is responsible for establishing a connection to the MongoDB database.

    Attributes:
    ----------
    client : MongoClient
        A shared MongoClient instance for the class.
    database : Database
        The specific database instance that MongoDBClient connects to.

    Methods:
    -------
    __init__(database_name: str) -> None
        Initializes the MongoDB connection using the given database name.
    """

    client = None  # Shared MongoClient instance across all MongoDBClient instances

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        """
        Initializes a connection to the MongoDB database. If no existing connection is found, it establishes a new one.

        Parameters:
        ----------
        database_name : str, optional
            Name of the MongoDB database to connect to. Default is set by DATABASE_NAME constant.

        Raises:
        ------
        MyException
            If there is an issue connecting to MongoDB or if the environment variable for the MongoDB URL is not set.
        """
        try:
            # Check if a MongoDB client connection has already been established; if not, create a new one
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)  # Retrieve MongoDB URL from environment variables
                if mongo_db_url is None:
                    raise Exception(f"Environment variable '{MONGODB_URL_KEY}' is not set.")
                
                # Remove any surrounding quotes from the URL if present
                mongo_db_url = mongo_db_url.strip('"\'')
                
                # Parse the URL to handle encoding properly
                parsed = urlparse(mongo_db_url)
                
                # Rebuild the URL with proper encoding
                if parsed.scheme == 'mongodb+srv':
                    # For MongoDB Atlas SRV connection string
                    if parsed.username or parsed.password:
                        # If username/password exists, ensure they're properly encoded
                        username = quote_plus(parsed.username or '')
                        password = quote_plus(parsed.password or '')
                        netloc = f"{username}:{password}@{parsed.hostname}"
                        if parsed.port:
                            netloc = f"{netloc}:{parsed.port}"
                        
                        # Rebuild the URL with encoded credentials
                        mongo_db_url = urlunparse((
                            parsed.scheme,
                            netloc,
                            parsed.path,
                            parsed.params,
                            parsed.query,
                            parsed.fragment
                        ))
                
                # Print the URL for debugging (remove in production)
                print(f"Connecting to MongoDB with URL: {mongo_db_url.split('@')[-1]}")
                
                # Establish a new MongoDB client connection
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
                
            # Use the shared MongoClient for this instance
            self.client = MongoDBClient.client
            self.database = self.client[database_name]  # Connect to the specified database
            self.database_name = database_name
            logging.info("MongoDB connection successful.")
            
        except Exception as e:
            # Raise a custom exception with traceback details if connection fails
            raise MyException(e, sys)