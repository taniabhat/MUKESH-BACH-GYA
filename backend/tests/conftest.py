import asyncio
from typing import AsyncGenerator
from typing import Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker

from config import get_settings
from main import app
from models.db import Base
from api.dependencies import get_db

settings = get_settings()

# Use an in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    future=True
)

TestingSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def setup_db() -> AsyncGenerator[None, None]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def db_session(setup_db) -> AsyncGenerator[AsyncSession, None]:
    async with TestingSessionLocal() as session:
        yield session


@pytest.fixture(scope="function")
def client(db_session: AsyncSession) -> TestClient:
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    
    # Patch out ALL production service interactions in the lifespan.
    # The lifespan calls on_startup() which opens real asyncpg connections
    # via the production engine; on subsequent tests a new event loop is
    # created but the global engine's pool is still bound to the old
    # (now-closed) loop, causing "attached to a different loop" errors.
    from unittest.mock import AsyncMock, patch
    with (
        patch("main.on_startup", new_callable=AsyncMock),
        patch("main.on_shutdown", new_callable=AsyncMock),
    ):
        with TestClient(app) as test_client:
            yield test_client
            
    app.dependency_overrides.clear()
