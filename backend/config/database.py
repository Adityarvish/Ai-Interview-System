from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import Config
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

engine = create_engine(
    Config.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=10,
    max_overflow=20,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables if they don't exist."""
    from config import models  
    Base.metadata.create_all(bind=engine)
    _run_lightweight_migrations()
    logger.info("PostgreSQL tables created / verified.")


_COLUMN_MIGRATIONS = [
    ("evaluations",              "decision_score",   "FLOAT DEFAULT 0"),
    ("per_question_evaluations", "star_score",       "FLOAT DEFAULT 0.0"),
    ("per_question_evaluations", "star_components",  "JSONB DEFAULT '{}'"),
    ("per_question_evaluations", "star_missing",     "JSONB DEFAULT '[]'"),
]


def _run_lightweight_migrations():
    with engine.begin() as conn:
        for table, column, ddl_type in _COLUMN_MIGRATIONS:
            try:
                conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {ddl_type}"
                )
            except Exception:
                logger.exception(
                    f"[DB MIGRATION] Failed to ensure column {table}.{column} exists"
                )
                raise


def get_db():
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
