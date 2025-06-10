import os
from dotenv import load_dotenv

def setup_environment():
    load_dotenv()

def get_api_key(index):
    return os.getenv(f"G{index}")