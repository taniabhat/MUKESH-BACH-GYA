"""
Shared database engine and session factory.

Single source of truth — avoids dual connection pools that existed
when both main.py and models/db.py independently created engines.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine

from config import get_settings


settings = get_settings()


engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    future=True,
    echo=False
)


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for DB sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
