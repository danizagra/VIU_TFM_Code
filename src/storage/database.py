"""
Database connection and session management.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import settings
from src.storage.models import Base


# Create engine
engine = create_engine(
    settings.database_url,
    echo=False,  # Set to True for SQL debugging
    pool_pre_ping=True,  # Verify connections before use
    pool_size=5,
    max_overflow=10
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def create_tables() -> None:
    """
    Create all tables in the database.

    Usage:
        from src.storage.database import create_tables
        create_tables()
    """
    Base.metadata.create_all(bind=engine)


def drop_tables() -> None:
    """
    Drop all tables in the database.
    WARNING: This will delete all data!
    """
    Base.metadata.drop_all(bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Usage:
        from src.storage.database import get_db

        with get_db() as db:
            articles = db.query(Article).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get a new database session.
    Remember to close it when done.

    Usage:
        db = get_db_session()
        try:
            # do stuff
            db.commit()
        finally:
            db.close()
    """
    return SessionLocal()
