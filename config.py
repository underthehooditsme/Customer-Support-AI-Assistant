import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Config:
    """
    Configuration class to load and manage all configuration variables.
    """
    # Base directory for the project
    BASE_DIR = Path(__file__).resolve().parent

    # Directories for data and logs
    DATA_DIR = BASE_DIR / "data_files"
    LOG_DIR = BASE_DIR / "logs"
    DB_DIR = BASE_DIR / "database"

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    # Default dataset URL and processed data path
    DATASET_URL = "https://huggingface.co/datasets/MohammadOthman/mo-customer-support-tweets-945k"
    DATASET_NAME = "MohammadOthman/mo-customer-support-tweets-945k"
    PROCESSED_DATA_PATH = DATA_DIR / "processed_customer_support.csv"
    VECTOR_DB_PATH = DATA_DIR / "vectordb"

    # Embedding settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

    # LLM settings
    DEFAULT_LOCAL_MODEL = os.getenv("DEFAULT_LOCAL_MODEL", "Llama3-8b-8192")
    DEFAULT_GROQ_MODEL = os.getenv("DEFAULT_GROQ_MODEL", "Llama3-70b-8192")
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 500))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

    # Retriever
    TOP_K = 3

    # Database settings
    DB_PATH = DB_DIR / "customer_support.db"

    # API settings
    API_TITLE = os.getenv("API_TITLE", "Customer Support RAG API")
    API_DESCRIPTION = os.getenv("API_DESCRIPTION", "API for answering customer support queries using RAG")
    API_VERSION = os.getenv("API_VERSION", "1.0.0")

    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOG_DIR / "app.log"

    # Cache settings
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", 100))

    # App
    HOST = '127.0.0.1'
    PORT = 8000
    DEBUG = True


CONFIG = Config()

