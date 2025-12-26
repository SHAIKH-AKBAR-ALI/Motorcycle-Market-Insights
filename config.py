import os
from typing import Optional

class Settings:
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Streamlit Configuration
    STREAMLIT_HOST: str = os.getenv("STREAMLIT_HOST", "0.0.0.0")
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/motorcycle_model.pkl")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")
    
    # Data Configuration
    DATA_PATH: str = os.getenv("DATA_PATH", "data/")
    UPLOAD_PATH: str = os.getenv("UPLOAD_PATH", "uploads/")
    
    # Database Configuration (if using database)
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    
    # Feature Engineering
    CURRENT_YEAR: int = int(os.getenv("CURRENT_YEAR", "2025"))
    
    # Model Parameters
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
    TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
    
    # API Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    class Config:
        case_sensitive = True

settings = Settings()