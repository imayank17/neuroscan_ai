import os
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("MONGODB_URI")

if not DATABASE_URL:
    print("Error: MONGODB_URI not found in .env")
    exit(1)

try:
    client = MongoClient(DATABASE_URL, tlsCAFile=certifi.where())
    db = client["seizure_detection"]
    
    collections = ["users", "predictions", "feedbacks"]
    
    print(f"Clearing database: {db.name}")
    for coll in collections:
        count = db[coll].count_documents({})
        db[coll].delete_many({})
        print(f" - Deleted {count} documents from '{coll}' collection.")
    
    print("Database cleared successfully!")
except Exception as e:
    print(f"Failed to clear database: {e}")
