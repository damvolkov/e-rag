"""RAG pipeline settings."""

from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Unified settings for e-rag pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    PG_CONN_STR: str = "postgresql://postgres:postgres@localhost:45010/e_rag"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    PARENT_SPLIT_LENGTH: int = 500
    PARENT_SPLIT_OVERLAP: int = 50
    CHILD_SPLIT_LENGTH: int = 100
    CHILD_SPLIT_OVERLAP: int = 20
    SEMANTIC_WEIGHT: float = 0.9
    KEYWORD_WEIGHT: float = 0.1
    TOP_K: int = 10
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"

    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parents[3]
    PARENT_TABLE: ClassVar[str] = "parent_documents"
    CHILD_TABLE: ClassVar[str] = "child_documents"


settings = Settings()
