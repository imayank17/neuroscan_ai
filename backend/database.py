import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB setup
DATABASE_URL = os.getenv("MONGODB_URI")

if not DATABASE_URL:
    raise ValueError("MONGODB_URI must be set in the .env file")

# Create a database connection
try:
    from logger import app_logger
    client = MongoClient(DATABASE_URL)
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
    app_logger.info("Successfully connected to MongoDB cluster.")
except Exception as e:
    from logger import app_logger
    app_logger.error(f"Failed to connect to MongoDB: {e}")
    raise e

db = client["seizure_detection"] # Use the seizure_detection database

# Collection references
users_collection = db["users"]
predictions_collection = db["predictions"]
feedbacks_collection = db["feedbacks"]

def get_db():
    try:
        yield db
    finally:
        pass
