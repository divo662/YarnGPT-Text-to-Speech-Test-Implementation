# backend/config.py
import os
import secrets
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Centralized configuration management
    """
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Server Configuration
    PORT = int(os.getenv('PORT', 5000))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    # Model Paths
    BASE_MODEL_PATH = os.getenv('BASE_MODEL_PATH', './downloaded_models')
    HF_MODEL_PATH = os.getenv('HF_MODEL_PATH', 'saheedniyi/YarnGPT')
    
    # CORS Configuration
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'tts_app.log')
    
    # Rate Limiting
    RATE_LIMIT_DEFAULT = os.getenv('RATE_LIMIT_DEFAULT', '100 per day')
    RATE_LIMIT_GENERATE = os.getenv('RATE_LIMIT_GENERATE', '20 per hour')

    @classmethod
    def is_production(cls):
        """
        Check if the application is running in production mode
        """
        return not cls.DEBUG

    @classmethod
    def get_model_paths(cls):
        """
        Retrieve model-related paths
        """
        return {
            'base_path': cls.BASE_MODEL_PATH,
            'hf_path': cls.HF_MODEL_PATH,
            'config_path': os.path.join(cls.BASE_MODEL_PATH, 'wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml'),
            'model_path': os.path.join(cls.BASE_MODEL_PATH, 'wavtokenizer_large_speech_320_24k.ckpt')
        }