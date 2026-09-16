from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Gemini
    gemini_api_key: str = ""
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    generation_model: str = "gemini-3.5-flash"

    # Pinecone (optional — falls back to the local store when unset)
    pinecone_api_key: str = ""
    pinecone_index_name: str = "rag-pipeline"

    # Retrieval tuning
    dense_top_k: int = 25
    sparse_top_k: int = 25
    rrf_k: int = 60
    rerank_top_k: int = 8

    # Chunking
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120

    # Storage
    data_dir: Path = Path("./data")
    index_dir: Path = Path("./.indexes")

    @property
    def use_pinecone(self) -> bool:
        return bool(self.pinecone_api_key)

    @property
    def use_gemini(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()
