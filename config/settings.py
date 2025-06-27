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

# Reflection pattern settings
ENABLE_REFLECTION = True
MAX_REFLECTION_ITERATIONS = 3
REFLECTION_ADDITIONAL_SOURCES_LIMIT = 3

# Collection querying optimization settings
ENABLE_PARALLEL_QUERYING = True          # Enable parallel collection querying
MAX_PARALLEL_WORKERS = 4                 # Maximum number of parallel workers for collection queries
ENABLE_SMART_COLLECTION_FILTERING = True # Enable intelligent collection filtering based on query intent
COLLECTION_FILTERING_THRESHOLD = 3       # Only apply filtering if more than this many collections
LAZY_LOADING_ENABLED = True              # Enable lazy loading of collection metadata

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# UI settings
APP_TITLE = "Schwizerischer Rechts-Chatbot im Migrationsrecht" 