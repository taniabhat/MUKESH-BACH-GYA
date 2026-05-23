"""
Structured logging configuration using structlog.

Provides JSON output for production and colored console for development.
Every module should use: logger = get_logger(__name__)
"""

import logging
import sys
import json
import redis

import structlog

from config import get_settings

# Global sync redis client for logging
_redis_client = None

def get_redis_client():
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client

def redis_publish_processor(logger, log_method, event_dict):
    project_id = event_dict.get("project_id")
    if project_id:
        try:
            r = get_redis_client()
            # Prepare payload for frontend
            payload = {
                "event": event_dict.get("event"),
                "status": event_dict.get("level", "info"),
                "details": {k: v for k, v in event_dict.items() if k not in ("event", "level", "project_id", "timestamp")}
            }
            r.publish(f"project:{project_id}:events", json.dumps(payload))
        except Exception:
            pass
    return event_dict


def setup_logging(
    json_output: bool = True,
    log_level: str = "INFO"
) -> None:
    """
    Configure structlog for the entire application.

    Args:
        json_output: True for JSON (production), False for colored console (dev).
        log_level: Minimum log level.
    """

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        redis_publish_processor,
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Quiet noisy third-party loggers
    for noisy in [
        "httpx",
        "httpcore",
        "urllib3",
        "asyncio",
        "sqlalchemy.engine",
        "neo4j",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a bound logger for a module."""
    return structlog.get_logger(name)
