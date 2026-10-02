import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

CONFIG_FILENAME = "config.yaml"


def _find_config() -> Path:
    """Walk up from this file so it works on the host and in the container."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{CONFIG_FILENAME} not found above {__file__}")


class LLMSettings(BaseModel):

    model: str
    base_url: str
    api_key_env: str = ""
    temperature: float = 0.0
    max_tokens: int | None = 2048
    timeout_seconds: float = 120.0
    max_retries: int = 0
    streaming: bool = True


class FieldNames(BaseModel):
    id: str
    image: str
    price: str
    rating: str
    description: str


class RetrievalSettings(BaseModel):
    collection: str
    top_k: int = 5
    max_top_k: int = 10
    fusion: str = "rrf"
    prefetch_limit: int = 20
    dense_vector: str
    sparse_vector: str
    sparse_model: str
    fields: FieldNames


class EmbeddingSettings(BaseModel):
    api_key_env: str
    model: str
    input_type: str
    dimensions: int
    embedding_types: List[str]


class StorageSettings(BaseModel):
    qdrant_url_env: str
    qdrant_url_default: str
    database_url_env: str


class APISettings(BaseModel):
    cors_origins: List[str]
    cors_allow_credentials: bool = True


class AgentSettings(BaseModel):
    max_iterations: int = 12
    prompts: Dict[str, str]


class Settings(BaseModel):
    llm: LLMSettings
    retrieval: RetrievalSettings
    embedding: EmbeddingSettings
    storage: StorageSettings
    api: APISettings
    agent: AgentSettings

    @property
    def qdrant_url(self) -> str:
        return os.getenv(self.storage.qdrant_url_env) or self.storage.qdrant_url_default

    @property
    def database_url(self) -> str | None:
        return os.getenv(self.storage.database_url_env) or None

    def provider_api_key(self, env_name: str) -> str:
        return (os.getenv(env_name) or "").strip()


def load_env() -> None:
    load_dotenv(encoding="utf-8-sig")
    load_dotenv(dotenv_path=_find_config().parent / ".env", encoding="utf-8-sig")


_ENV_REF = re.compile(r"\$\{(?P<name>[A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:

    def replace(match: re.Match) -> str:
        name, fallback = match.group("name"), match.group(2)
        found = (os.getenv(name) or "").strip()
        return found if found else (fallback or "")

    if isinstance(value, str):
        return _ENV_REF.sub(replace, value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_env()
    with open(_find_config(), "r", encoding="utf-8-sig") as handle:
        return Settings(**_expand_env(yaml.safe_load(handle)))
