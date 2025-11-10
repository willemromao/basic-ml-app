import os
from pymongo import MongoClient

# Read from environment variables
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

def get_mongo_collection(collection_name: str):
    """
    Connects to MongoDB and returns a collection.
    
    Args:
        collection_name: Name of the collection to access
        
    Returns:
        MongoDB collection object
        
    Raises:
        ValueError: If MONGO_URI or MONGO_DB not set
    """
    if not MONGO_URI:
        raise ValueError("MONGO_URI environment variable not set")
    if not MONGO_DB:
        raise ValueError("MONGO_DB environment variable not set")
    
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[collection_name]
    
    # Test connection
    client.server_info()
    
    return collection