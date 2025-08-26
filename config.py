import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration"""
    
    # Google Gemini API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDNo_VDkus7erQ-pkYd6fFQOVGaspFmzeI")
    
    # PDF Path Configuration
    PDF_PATH = os.getenv("PDF_PATH", "documents/20240716890312078.pdf")
    
    # Model Configuration
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
    
    # RAG Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.75"))
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Create config instance
config = Config()
