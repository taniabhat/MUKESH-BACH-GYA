from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from neo4j import AsyncGraphDatabase
from qdrant_client import QdrantClient
from sqlalchemy import text

from config import get_settings
from core.logging import get_logger, setup_logging

settings = get_settings()

setup_logging(
    json_output=not settings.DEBUG,
    log_level="DEBUG" if settings.DEBUG else "INFO"
)

from api.routes import router as api_router
from api.websocket import router as websocket_router
from core.database import engine
from models.db import init_db
from core.llm import warmup_llm
from core.rag import ensure_qdrant_collections

logger = get_logger(__name__)

logger = get_logger(__name__)


# -------------------------------------------------------------------
# Redis
# -------------------------------------------------------------------


redis_client = redis.from_url(
    settings.REDIS_URL,
    decode_responses=True
)


# -------------------------------------------------------------------
# Neo4j
# -------------------------------------------------------------------


neo4j_driver = AsyncGraphDatabase.driver(
    settings.NEO4J_URL,
    auth=(
        settings.NEO4J_USER,
        settings.NEO4J_PASSWORD
    )
)


# -------------------------------------------------------------------
# Qdrant
# -------------------------------------------------------------------


if settings.QDRANT_URL:
    qdrant_client = QdrantClient(
        url=settings.QDRANT_URL,
        check_compatibility=False
    )
else:
    qdrant_client = QdrantClient(
        path=settings.QDRANT_PATH,
        check_compatibility=False
    )


# -------------------------------------------------------------------
# Health Checks
# -------------------------------------------------------------------


async def run_database_healthcheck() -> None:

    async with engine.begin() as connection:

        await connection.execute(
            text("SELECT 1")
        )

    logger.info("startup.database", status="healthy")


async def verify_neo4j_connectivity() -> None:

    await neo4j_driver.verify_connectivity()

    logger.info("startup.neo4j", status="connected")


# -------------------------------------------------------------------
# Database Initialization
# -------------------------------------------------------------------


async def initialize_database() -> None:

    logger.info("startup.database_init", status="started")

    await init_db()

    logger.info("startup.database_init", status="complete")


# -------------------------------------------------------------------
# Qdrant Initialization
# -------------------------------------------------------------------


async def initialize_qdrant() -> None:

    logger.info("startup.qdrant_init", status="started")

    ensure_qdrant_collections()

    logger.info("startup.qdrant_init", status="complete")


# -------------------------------------------------------------------
# LLM Warmup
# -------------------------------------------------------------------


async def warmup_services() -> None:

    logger.info("startup.llm_warmup", status="started")

    try:
        await warmup_llm()

        logger.info("startup.llm_warmup", status="complete")

    except Exception as error:

        logger.error(
            "startup.llm_warmup",
            status="failed",
            error=str(error)
        )


# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------


async def on_startup() -> None:

    logger.info("startup", status="starting", app=settings.APP_NAME, version=settings.APP_VERSION)

    await run_database_healthcheck()

    await initialize_database()

    await initialize_qdrant()

    await verify_neo4j_connectivity()

    await warmup_services()

    logger.info("startup", status="complete")


# -------------------------------------------------------------------
# Shutdown
# -------------------------------------------------------------------


async def on_shutdown() -> None:

    logger.info("shutdown", status="starting")

    await engine.dispose()

    await neo4j_driver.close()

    from core.graph import get_driver
    driver = get_driver()
    if driver:
        await driver.close()

    await redis_client.aclose()

    logger.info("shutdown", status="complete")


# -------------------------------------------------------------------
# Lifespan
# -------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):

    await on_startup()

    yield

    await on_shutdown()


# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------


def create_app() -> FastAPI:

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    app.include_router(
        api_router,
        prefix="/api/v1"
    )

    app.include_router(
        websocket_router,
        prefix="/ws"
    )

    @app.get("/health")
    async def healthcheck():

        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION
        }

    return app


# -------------------------------------------------------------------
# App Instance
# -------------------------------------------------------------------


app = create_app()