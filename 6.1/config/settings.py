"""
Global configuration and settings for Research Discovery Module.

Open-source stack:
- LLM:        Qwen3 via Ollama (dev) or vLLM (production)
- Embeddings: BGE-M3 via sentence-transformers (local)
- APIs:       OpenAlex, Semantic Scholar (free), arXiv (free), CrossRef (free)
- No paid API keys required.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIConfig:
    # -----------------------------------------------------------------------
    # LLM — Local inference (Qwen3)
    # -----------------------------------------------------------------------
    # Set USE_VLLM=true in production (vLLM server).
    # Default: Ollama (local dev).
    use_vllm: bool = field(
        default_factory=lambda: os.getenv("USE_VLLM", "false").lower() == "true"
    )

    # Ollama endpoint (local development)
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # vLLM endpoint (production — OpenAI-compatible)
    vllm_base_url: str = field(
        default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8080/v1")
    )

    # Model name for query expansion
    # Ollama: "qwen3:14b"  |  vLLM: model name as loaded in vLLM server
    qwen3_model: str = field(
        default_factory=lambda: os.getenv("QWEN3_MODEL", "qwen3:14b")
    )

    # LLM generation parameters
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # -----------------------------------------------------------------------
    # Semantic Scholar (free tier — no key needed, key gives higher rate limit)
    # -----------------------------------------------------------------------
    semantic_scholar_api_key: str = field(
        default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    )
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    semantic_scholar_rps: int = 5  # semaphore concurrency limit

    # -----------------------------------------------------------------------
    # OpenAlex (completely free — polite pool via email header)
    # -----------------------------------------------------------------------
    openalex_base_url: str = "https://api.openalex.org"
    openalex_email: str = field(
        default_factory=lambda: os.getenv("OPENALEX_EMAIL", "researcher@example.com")
    )

    # -----------------------------------------------------------------------
    # arXiv (completely free — no key)
    # -----------------------------------------------------------------------
    arxiv_base_url: str = "http://export.arxiv.org/api/query"

    # -----------------------------------------------------------------------
    # CrossRef (completely free — polite pool via mailto header)
    # -----------------------------------------------------------------------
    crossref_base_url: str = "https://api.crossref.org/works"
    crossref_mailto: str = field(
        default_factory=lambda: os.getenv("CROSSREF_MAILTO", "researcher@example.com")
    )

    # -----------------------------------------------------------------------
    # HTTP
    # -----------------------------------------------------------------------
    http_timeout: int = 30
    max_retries: int = 3


@dataclass
class RankingWeights:
    semantic_similarity: float = 0.60
    citation_boost: float = 0.15
    recency_boost: float = 0.10
    venue_score: float = 0.10
    keyword_overlap: float = 0.05


@dataclass
class RetrievalConfig:
    num_expansion_queries: int = 10
    results_per_query: int = 20
    citation_expansion_top_k: int = 20
    fuzzy_dedup_threshold: float = 0.94
    final_corpus_min: int = 50
    final_corpus_max: int = 150
    mmr_lambda: float = 0.7


@dataclass
class EmbeddingConfig:
    # BGE-M3: best open-source multilingual embedding model
    # Supports dense + sparse + hybrid retrieval
    model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    )
    dimension: int = 1024
    batch_size: int = 32
    # Use GPU if available
    device: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cuda")
    )
    # Cache model in memory after first load
    cache_dir: str = field(
        default_factory=lambda: os.getenv("MODEL_CACHE_DIR", "/tmp/models")
    )


@dataclass
class Settings:
    api: APIConfig = field(default_factory=APIConfig)
    ranking: RankingWeights = field(default_factory=RankingWeights)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )


# Singleton — import this everywhere
settings = Settings()