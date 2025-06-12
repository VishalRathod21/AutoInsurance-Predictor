import os
import pandas as pd
from dotenv import load_dotenv
import pymongo

# Load environment variables
load_dotenv()

# MongoDB connection string from .env file
MONGODB_URL = os.getenv('MONGODB_URL')
if not MONGODB_URL:
    raise ValueError("MONGODB_URL not found in .env file")

# Database and collection names
DATABASE_NAME = "Vehicle-Insurance"
COLLECTION_NAME = "Vehicle-Insurance-Data"

def import_data():
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(MONGODB_URL)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Check if collection already has data
        if collection.count_documents({}) > 0:
            print(f"Collection '{COLLECTION_NAME}' already contains data. Dropping existing data...")
            collection.drop()
            # Recreate the collection
            collection = db[COLLECTION_NAME]
        
        # Read the CSV file
        print("Reading data from CSV file...")
        df = pd.read_csv('notebook/data.csv')
        
        # Convert DataFrame to dictionary and insert into MongoDB
        print(f"Importing {len(df)} records to MongoDB...")
        data = df.to_dict('records')
        collection.insert_many(data)
        
        # Verify the import
        count = collection.count_documents({})
        print(f"Successfully imported {count} records to {DATABASE_NAME}.{COLLECTION_NAME}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    import_data()
