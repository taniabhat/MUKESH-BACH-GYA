"""
Global configuration for Research Discovery Platform.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Provider Types
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    TOGETHER = "together"
    GROQ = "groq"
    LMSTUDIO = "lmstudio"


class EmbeddingProvider(str, Enum):
    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

@dataclass
class HTTPConfig:

    timeout: int = 30

    max_retries: int = 3

    max_connections: int = 100

    user_agent: str = (
        "ResearchDiscovery/1.0"
    )


# ---------------------------------------------------------------------------
# LLM Provider Config
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:

    provider: LLMProvider = field(
        default_factory=lambda: LLMProvider(
            os.getenv(
                "LLM_PROVIDER",
                "ollama",
            )
        )
    )

    model_name: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL",
            "qwen3:14b",
        )
    )

    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "LLM_API_KEY",
        )
    )

    base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BASE_URL",
            "http://localhost:11434",
        )
    )

    temperature: float = field(
        default_factory=lambda: float(
            os.getenv(
                "LLM_TEMPERATURE",
                "0.3",
            )
        )
    )

    max_tokens: int = field(
        default_factory=lambda: int(
            os.getenv(
                "LLM_MAX_TOKENS",
                "1024",
            )
        )
    )

    request_timeout: int = field(
        default_factory=lambda: int(
            os.getenv(
                "LLM_TIMEOUT",
                "60",
            )
        )
    )


# ---------------------------------------------------------------------------
# Embedding Config
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:

    provider: EmbeddingProvider = field(
        default_factory=lambda: EmbeddingProvider(
            os.getenv(
                "EMBEDDING_PROVIDER",
                "local",
            )
        )
    )

    model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL",
            "BAAI/bge-m3",
        )
    )

    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_API_KEY",
        )
    )

    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_BASE_URL",
        )
    )

    dimension: int = field(
        default_factory=lambda: int(
            os.getenv(
                "EMBEDDING_DIMENSION",
                "1024",
            )
        )
    )

    batch_size: int = field(
        default_factory=lambda: int(
            os.getenv(
                "EMBEDDING_BATCH_SIZE",
                "32",
            )
        )
    )

    device: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_DEVICE",
            "cuda",
        )
    )

    cache_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "MODEL_CACHE_DIR",
                "/tmp/models",
            )
        )
    )


# ---------------------------------------------------------------------------
# Retrieval APIs
# ---------------------------------------------------------------------------

@dataclass
class OpenAlexConfig:

    enabled: bool = True

    max_retries: int = 3

    base_url: str = (
        "https://api.openalex.org"
    )

    email: str = field(
        default_factory=lambda: os.getenv(
            "OPENALEX_EMAIL",
            "researcher@example.com",
        )
    )


@dataclass
class SemanticScholarConfig:

    enabled: bool = True

    max_retries: int = 5

    base_url: str = (
        "https://api.semanticscholar.org/graph/v1"
    )

    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "SEMANTIC_SCHOLAR_API_KEY",
        )
    )

    rps_limit: int = 5


@dataclass
class ArxivConfig:

    enabled: bool = True

    max_retries: int = 3

    base_url: str = (
        "http://export.arxiv.org/api/query"
    )


@dataclass
class CrossRefConfig:

    enabled: bool = True

    max_retries: int = 3

    base_url: str = (
        "https://api.crossref.org/works"
    )

    mailto: str = field(
        default_factory=lambda: os.getenv(
            "CROSSREF_MAILTO",
            "researcher@example.com",
        )
    )


# ---------------------------------------------------------------------------
# Retrieval Pipeline
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:

    num_expansion_queries: int = 10

    results_per_query: int = 20

    citation_expansion_top_k: int = 20

    fuzzy_dedup_threshold: float = 0.94

    final_corpus_min: int = 50

    final_corpus_max: int = 150

    mmr_lambda: float = 0.7


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

@dataclass
class RankingConfig:

    semantic_similarity: float = 0.60

    citation_boost: float = 0.15

    recency_boost: float = 0.10

    venue_score: float = 0.10

    keyword_overlap: float = 0.05


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

@dataclass
class StorageConfig:

    storage_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "STORAGE_DIR",
                "/tmp/research_discovery",
            )
        )
    )


# ---------------------------------------------------------------------------
# Root Settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:

    debug: bool = field(
        default_factory=lambda: (
            os.getenv(
                "DEBUG",
                "false",
            ).lower()
            == "true"
        )
    )

    log_level: str = field(
        default_factory=lambda: os.getenv(
            "LOG_LEVEL",
            "INFO",
        )
    )

    http: HTTPConfig = field(
        default_factory=HTTPConfig
    )

    llm: LLMConfig = field(
        default_factory=LLMConfig
    )

    embedding: EmbeddingConfig = field(
        default_factory=EmbeddingConfig
    )

    retrieval: RetrievalConfig = field(
        default_factory=RetrievalConfig
    )

    ranking: RankingConfig = field(
        default_factory=RankingConfig
    )

    storage: StorageConfig = field(
        default_factory=StorageConfig
    )

    openalex: OpenAlexConfig = field(
        default_factory=OpenAlexConfig
    )

    semantic_scholar: SemanticScholarConfig = (
        field(
            default_factory=(
                SemanticScholarConfig
            )
        )
    )

    arxiv: ArxivConfig = field(
        default_factory=ArxivConfig
    )

    crossref: CrossRefConfig = field(
        default_factory=CrossRefConfig
    )


settings = Settings()