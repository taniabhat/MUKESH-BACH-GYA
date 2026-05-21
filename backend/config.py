from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------

    APP_NAME: str = "ResearchOS"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    SECRET_KEY: str = Field(..., min_length=32)

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    DATABASE_URL: str
    REDIS_URL: str

    QDRANT_URL: str

    NEO4J_URL: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str

    # ------------------------------------------------------------------
    # OpenRouter / LLM
    # ------------------------------------------------------------------

    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    QWEN3_MODEL: str = "nvidia/nemotron-3-super-120b-a12b:free"
    DEEPSEEK_MODEL: str = "openai/gpt-oss-120b:free"
    PHI3_MODEL: str = "z-ai/glm-4.5-air:free"
    HF_API_KEY: str
    # Embeddings

    EMBEDDING_MODEL: str = (
        "BAAI/bge-m3"
    )

    RERANKER_MODEL: str = (
        "BAAI/bge-reranker-v2-m3"
    )

    IMAGE_EMBED_MODEL: str = (
        "google/siglip-base-patch16-224"
    )

    CODE_EMBED_MODEL: str = (
        "microsoft/codebert-base"
    )

    # Chunking

    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200

    # Retrieval

    TOP_K_RETRIEVAL: int = 10
    MAX_CONTEXT_TOKENS: int = 8000

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------

    GROBID_URL: str

    DATA_DIR: str = str(BASE_DIR / "data")

    SANDBOX_IMAGE: str = "research-os-sandbox:latest"

    # ------------------------------------------------------------------
    # APIs
    # ------------------------------------------------------------------

    S2_API_KEY: str = ""
    CROSSREF_EMAIL: str

    # ------------------------------------------------------------------
    # Retrieval / Pipeline
    # ------------------------------------------------------------------

    MAX_CONTEXT_TOKENS: int = 8000
    RETRIEVAL_TOP_K: int = 15
    RERANK_TOP_K: int = 5

    REQUEST_TIMEOUT_SECONDS: int = 300
    MAX_RETRIES: int = 3

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def papers_dir(self) -> Path:
        return Path(self.DATA_DIR) / "papers"

    @property
    def figures_dir(self) -> Path:
        return Path(self.DATA_DIR) / "figures"

    @property
    def generated_dir(self) -> Path:
        return Path(self.DATA_DIR) / "generated"

    @property
    def exports_dir(self) -> Path:
        return Path(self.DATA_DIR) / "exports"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def create_directories(self) -> None:
        directories = [
            self.papers_dir,
            self.figures_dir,
            self.generated_dir,
            self.exports_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.create_directories()
    return settings


settings = get_settings()