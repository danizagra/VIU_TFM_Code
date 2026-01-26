#!/usr/bin/env python3
"""
Script to set up the database tables.

Usage:
    poetry run python scripts/setup_database.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import create_tables, engine
from src.storage.models import Base


def main():
    """Create all database tables."""
    print("Creating database tables...")
    print(f"Database URL: {engine.url}")

    try:
        create_tables()
        print("\nTables created successfully:")
        for table_name in Base.metadata.tables.keys():
            print(f"  - {table_name}")
        print("\nDatabase setup complete!")
    except Exception as e:
        print(f"\nError creating tables: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
