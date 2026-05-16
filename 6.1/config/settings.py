"""
Global configuration and settings for Research Discovery Module.
All API keys, timeouts, and tuning parameters live here.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIConfig:
    # Semantic Scholar
    semantic_scholar_api_key: str = field(
        default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    )
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    semantic_scholar_rps: int = 5  # requests per second (semaphore limit)

    # OpenAlex
    openalex_base_url: str = "https://api.openalex.org"
    openalex_email: str = field(
        default_factory=lambda: os.getenv("OPENALEX_EMAIL", "researcher@example.com")
    )  # Polite pool

    # arXiv
    arxiv_base_url: str = "http://export.arxiv.org/api/query"

    # CrossRef
    crossref_base_url: str = "https://api.crossref.org/works"
    crossref_mailto: str = field(
        default_factory=lambda: os.getenv("CROSSREF_MAILTO", "researcher@example.com")
    )

    # LLM (for query expansion)
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    llm_model: str = "claude-sonnet-4-20250514"

    # HTTP
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
    # How many queries to generate from one research idea
    num_expansion_queries: int = 10
    # Max results per query per source
    results_per_query: int = 20
    # Top-K for citation expansion
    citation_expansion_top_k: int = 20
    # Dedup fuzzy threshold
    fuzzy_dedup_threshold: float = 0.94
    # Final corpus target
    final_corpus_min: int = 50
    final_corpus_max: int = 150
    # MMR diversity lambda
    mmr_lambda: float = 0.7


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"
    dimension: int = 1024
    batch_size: int = 32


@dataclass
class Settings:
    api: APIConfig = field(default_factory=APIConfig)
    ranking: RankingWeights = field(default_factory=RankingWeights)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


# Singleton
settings = Settings()