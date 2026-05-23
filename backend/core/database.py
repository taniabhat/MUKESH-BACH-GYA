"""
Shared database engine and session factory.

Single source of truth — avoids dual connection pools that existed
when both main.py and models/db.py independently created engines.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from config import get_settings


settings = get_settings()


def _make_engine(use_null_pool: bool = False):
    kwargs = {
        "future": True,
        "echo": False,
    }
    if use_null_pool:
        kwargs["poolclass"] = NullPool
    else:
        kwargs["pool_pre_ping"] = True
    return create_async_engine(settings.DATABASE_URL, **kwargs)


engine = _make_engine()


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


def make_worker_session():
    """Returns (session_factory, engine) with NullPool asyncpg.
    Must be called inside an active event loop."""
    from sqlalchemy.pool import NullPool
    worker_engine = create_async_engine(
        settings.DATABASE_URL,
        poolclass=NullPool,
        future=True,
        echo=False
    )
    session_factory = async_sessionmaker(
        bind=worker_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    return session_factory, worker_engine


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for DB sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
