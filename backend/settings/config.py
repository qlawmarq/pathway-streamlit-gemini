"""
Application Configuration Management with Pydantic Settings
"""

import os
import yaml
from pathlib import Path
from typing import List, Any, Dict
from functools import lru_cache

from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    max_retries: int = Field(ge=0)


class EmbeddingSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str


class TextProcessingSettings(BaseModel):
    max_tokens_per_chunk: int = Field(gt=0)


class FileFilteringSettings(BaseModel):
    allowed_extensions: List[str]
    allowed_filenames: List[str]
    excluded_patterns: List[str]


class DataSourceSettings(BaseModel):
    path: str
    mode: str = Field(pattern="^(streaming|static)$")
    file_filtering: FileFilteringSettings


class RAGSettings(BaseModel):
    search_topk: int = Field(gt=0)
    prompt_template: str


class ServerSettings(BaseModel):
    host: str
    port: int = Field(gt=0, le=65535)
    with_cache: bool


class DevelopmentSettings(BaseModel):
    debug_logging: bool
    show_file_list: bool


class Settings(BaseSettings):
    llm: LLMSettings
    embedding: EmbeddingSettings
    text_processing: TextProcessingSettings
    data_source: DataSourceSettings
    rag: RAGSettings
    server: ServerSettings
    development: DevelopmentSettings

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_config_settings_source,
            file_secret_settings,
        )


def yaml_config_settings_source() -> Dict[str, Any]:
    config_file = os.getenv("RAG_CONFIG_FILE", "app/config/settings.yaml")

    if not os.path.isabs(config_file):
        current_dir = Path(__file__).parent.parent.parent
        config_file = current_dir / config_file
    else:
        config_file = Path(config_file)

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
