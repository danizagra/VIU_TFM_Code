"""
Journalist Agent REST API.

Usage:
    uvicorn src.api.main:app --reload --port 8000
"""

from src.api.main import app

__all__ = ["app"]
