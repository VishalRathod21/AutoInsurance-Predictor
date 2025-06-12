import os
import pymongo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB URL from environment variables
mongo_url = os.getenv('MONGODB_URL')
if not mongo_url:
    print("Error: MONGODB_URL not found in environment variables")
    exit(1)

try:
    # Connect to MongoDB
    client = pymongo.MongoClient(mongo_url)
    
    # List all databases
    print("\nAvailable databases:")
    for db_name in client.list_database_names():
        print(f"- {db_name}")
    
    # Check the specific database and collection
    db_name = "Vehicle-Insurance"
    collection_name = "Vehicle-Insurance-Data"
    
    db = client[db_name]
    
    print(f"\nCollections in database '{db_name}':")
    for coll_name in db.list_collection_names():
        print(f"- {coll_name}")
    
    # Check if the collection exists and count documents
    if collection_name in db.list_collection_names():
        collection = db[collection_name]
        count = collection.count_documents({})
        print(f"\nNumber of documents in '{collection_name}': {count}")
        
        # Show first document if exists
        if count > 0:
            print("\nFirst document:")
            print(collection.find_one())
        else:
            print("The collection is empty.")
    else:
        print(f"\nCollection '{collection_name}' does not exist in database '{db_name}'")
    
except Exception as e:
    print(f"\nError connecting to MongoDB: {e}")
    print("\nPlease verify your MongoDB connection string in the .env file.")
    print("Make sure the URL is correct and your IP is whitelisted in MongoDB Atlas.")
    print("The URL should look like: mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/")

print("\nNote: If you see authentication errors, please check your MongoDB Atlas credentials and IP whitelist settings.")
