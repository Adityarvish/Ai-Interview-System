import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / '.env')


class Config:
    POSTGRES_HOST     = os.environ.get('POSTGRES_HOST',     'localhost')
    POSTGRES_PORT     = int(os.environ.get('POSTGRES_PORT', 5432))
    POSTGRES_USER     = os.environ.get('POSTGRES_USER',     'postgres')
    POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD', '')
    POSTGRES_DB       = os.environ.get('POSTGRES_DB',       'ai_interview_db')

   
    DATABASE_URL = os.environ.get(
        'DATABASE_URL',
        f"postgresql+psycopg2://"
        f"{os.environ.get('POSTGRES_USER', 'postgres')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', '')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'ai_interview_db')}"
    )

    # ── Groq Cloud API ────────────────────────────────────────────────────────
    GROQ_API_KEY        = os.environ.get('GROQ_API_KEY', '')
    GROQ_PRIMARY_MODEL  = os.environ.get('GROQ_PRIMARY_MODEL',  'llama-3.3-70b-versatile')
    GROQ_FALLBACK_MODEL = os.environ.get('GROQ_FALLBACK_MODEL', 'llama-3.1-8b-instant')

    # ── Sarvam AI (Hindi / Marathi TTS) ──────────────────────────────────────
    SARVAM_API_KEY = os.environ.get('SARVAM_API_KEY', '')

    HOST = os.environ.get('HOST', os.environ.get('FLASK_HOST', '0.0.0.0'))
    PORT = int(os.environ.get('PORT', os.environ.get('FLASK_PORT', 5000)))

    # Interview settings
    MAX_INTERVIEW_DURATION = int(os.environ.get('MAX_INTERVIEW_DURATION', 2700))  

    # File paths
    UPLOAD_FOLDER  = ROOT_DIR / 'uploads'
    RESUME_FOLDER  = UPLOAD_FOLDER / 'resumes'
    AUDIO_FOLDER   = UPLOAD_FOLDER / 'audio'

    # Allowed extensions
    ALLOWED_RESUME_EXTENSIONS = {'pdf', 'txt'}
    ALLOWED_AUDIO_EXTENSIONS  = {'webm', 'wav', 'ogg', 'mp3'}

    # Max file sizes (in bytes)
    MAX_RESUME_SIZE = 10 * 1024 * 1024   # 10 MB
    MAX_AUDIO_SIZE  = 50 * 1024 * 1024   # 50 MB



Config.UPLOAD_FOLDER.mkdir(exist_ok=True)
Config.RESUME_FOLDER.mkdir(exist_ok=True)
Config.AUDIO_FOLDER.mkdir(exist_ok=True)
