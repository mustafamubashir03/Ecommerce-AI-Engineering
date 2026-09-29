"""Single source of truth for every changeable value.

Values come from `config.yaml` at the repository root. Secrets are never stored
there: each entry only names the env var to read the secret from.
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

CONFIG_FILENAME = "config.yaml"


def _find_config() -> Path:
    """Walk up from this file so it works on the host and in the container."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{CONFIG_FILENAME} not found above {__file__}")


class RetrySettings(BaseModel):
    backoff_seconds: float = 0.5


class GoogleSettings(BaseModel):
    """The external provider fallback, a different provider than the pool."""

    enabled: bool = False
    model: str = ""
    api_key_env: str = "GOOGLE_API_KEY"
    timeout_seconds: float = 90.0


class GroqSettings(BaseModel):
    """Groq's OpenAI compatible endpoint, with its own ordered model list."""

    enabled: bool = False
    base_url: str = "https://api.groq.com/openai/v1"
    api_key_env: str = "GROQ_API_KEY"
    models: str = ""
    primary_model: str = ""
    timeout_seconds: float = 60.0
    strict_structured_output: List[str] = Field(default_factory=list)

    def model_pool(self) -> List[str]:
        """The configured ids, primary first, without duplicates."""
        candidates = [item.strip() for item in (self.models or "").split(",") if item.strip()]
        ordered: List[str] = []
        primary = (self.primary_model or "").strip()
        if primary:
            ordered.append(primary)
        for candidate in candidates:
            if candidate not in ordered:
                ordered.append(candidate)
        return ordered


class LLMSettings(BaseModel):
    active: str = "auto"
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    max_retries: int = 0
    max_tokens: int | None = 2048
    retries: RetrySettings = Field(default_factory=RetrySettings)
    pool: str = ""
    primary_model: str = ""
    model_prefix: str = ""
    base_urls: Dict[str, str] = Field(default_factory=dict)
    auto_detect: List[str] = Field(default_factory=list)
    key_envs: Dict[str, str] = Field(default_factory=dict)
    google: GoogleSettings = Field(default_factory=GoogleSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    fallback_order: List[str] = Field(default_factory=list)


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

    def provider_api_key_env(self, provider: str) -> str:
        return self.llm.key_envs.get(provider, "")

    def provider_api_key(self, provider: str) -> str:
        env_name = self.provider_api_key_env(provider)
        return (os.getenv(env_name) or "").strip() if env_name else ""

    def model_pool(self) -> List[str]:
        """The configured model ids, primary first, without duplicates.

        Parsed once and reused: the router needs a stable order for the whole
        process, not a fresh read per request.
        """
        separator = "," if "," in self.llm.pool else None
        candidates = [item.strip() for item in (self.llm.pool or "").split(separator or None)]
        ordered: List[str] = []

        primary = (self.llm.primary_model or "").strip()
        if primary:
            ordered.append(primary)
        for candidate in candidates:
            if candidate and candidate not in ordered:
                ordered.append(candidate)
        return ordered


def load_env() -> None:
    """Load .env, tolerating a UTF-8 BOM.

    Editors that write a BOM silently rename the first variable (it arrives as
    '\\ufeffGOOGLE_API_KEY'), which makes that key invisible to the app. Reading
    with utf-8-sig drops the BOM instead.
    """
    load_dotenv(encoding="utf-8-sig")
    load_dotenv(dotenv_path=_find_config().parent / ".env", encoding="utf-8-sig")


_ENV_REF = re.compile(r"\$\{(?P<name>[A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Replace ${VAR} and ${VAR:-fallback} in config values from the environment.

    This is what keeps a changeable value in one place: config.yaml names the
    variable, .env holds the value, and no code edit is needed to change it.
    """

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
