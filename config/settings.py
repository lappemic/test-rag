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

# Streaming settings
ENABLE_STREAMING = True                  # Enable response streaming
STREAMING_CHUNK_SIZE = 25               # Number of words per streaming chunk
ENABLE_STAGE_NOTIFICATIONS = True       # Show processing stage updates to user

# Model settings
DEFAULT_MODEL = "gpt-4o"
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

# Max-Marginal Relevance (MMR) settings
ENABLE_MMR = True                        # Enable MMR re-ranking for diverse results
MMR_LAMBDA = 0.5                        # Lambda parameter for relevance vs diversity trade-off (0.0 = max diversity, 1.0 = max relevance)
MMR_FETCH_K = 15                        # Number of documents to fetch before MMR re-ranking (should be > MAX_RESULTS)
MMR_USE_FAST_MODE = True                 # Use fast heuristic-based MMR instead of full embedding-based MMR

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# UI settings
APP_TITLE = "Schwizerischer Rechts-Chatbot im Migrationsrecht" 