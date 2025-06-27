"""
Configuration settings for the Swiss Legal Chatbot application.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application constants
DEV_MODE = False
CHROMA_DB_PATH = "./chroma_db"
LAW_COLLECTION_PREFIX = "law_"

# Model settings
DEFAULT_MODEL = "gpt-4o-mini"
MAX_RESULTS = 5
MAX_CONVERSATION_LENGTH = 2000
MAX_RECENT_MESSAGES = 4

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# UI settings
APP_TITLE = "Schwizerischer Rechts-Chatbot im Migrationsrecht" 