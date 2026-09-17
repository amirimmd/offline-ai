"""Configuration loading for workspace, models, and policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from offline_ai.utils.paths import ensure_dir, project_root, resolve_path


class DatabaseConfig(BaseModel):
    filename: str = "db/offline_ai.db"
    echo: bool = False
    foreign_keys: bool = True


class VectorsConfig(BaseModel):
    index_path: str = "vectors/faiss.index"
    id_map_path: str = "vectors/id_map.json"
    dimension: int = 768


class LoggingConfig(BaseModel):
    level: str = "INFO"
    dir: str = "logs"
    json_logs: bool = True
    log_document_text: bool = False


class RetrievalConfig(BaseModel):
    weight_semantic: float = 0.45
    weight_lexical: float = 0.30
    weight_entity: float = 0.10
    weight_metadata: float = 0.05
    weight_graph: float = 0.10
    top_k: int = 20
    rerank_top_k: int = 8
    min_evidence_score: float = 0.15


class ChunkingConfig(BaseModel):
    min_chars_for_chunking: int = 1200
    chunk_size: int = 800
    chunk_overlap: int = 120


class IngestionConfig(BaseModel):
    near_duplicate_threshold: float = 0.92


class HardwareConfig(BaseModel):
    prefer_gpu: bool = True
    min_vram_mb_for_gpu: int = 4096


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class TrainingConfig(BaseModel):
    replay_ratio: float = 0.3
    replay_strategy: str = "stratified"


class AppConfig(BaseModel):
    workspace_name: str = "offline_ai_workspace"
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    vectors: VectorsConfig = Field(default_factory=VectorsConfig)
    models_config_file: str = "models.yaml"
    models_cache_dir: str = "models"
    adapters_root: str = "adapters"
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


class Policies(BaseModel):
    citation_required: bool = True
    allow_unsupported_claims: bool = False
    preserve_original_documents: bool = True
    if_evidence_missing: str = "UNKNOWN"
    insufficient_evidence_message: str = "Insufficient evidence in stored memory."
    grounding: dict[str, Any] = Field(default_factory=dict)
    source_priority: list[str] = Field(default_factory=list)
    source_priority_as_truth: bool = False
    conversation: dict[str, Any] = Field(default_factory=dict)
    logging: dict[str, Any] = Field(default_factory=dict)
    security: dict[str, Any] = Field(default_factory=dict)
    duplicates: dict[str, Any] = Field(default_factory=dict)


class Settings(BaseSettings):
    """Runtime settings bound to a workspace directory."""

    workspace: Path
    config: AppConfig
    policies: Policies
    models_raw: dict[str, Any]
    project_root: Path

    model_config = {"arbitrary_types_allowed": True}

    @property
    def db_path(self) -> Path:
        return resolve_path(self.workspace, self.config.database.filename)

    @property
    def vectors_index_path(self) -> Path:
        return resolve_path(self.workspace, self.config.vectors.index_path)

    @property
    def vectors_id_map_path(self) -> Path:
        return resolve_path(self.workspace, self.config.vectors.id_map_path)

    @property
    def models_dir(self) -> Path:
        return resolve_path(self.workspace, self.config.models_cache_dir)

    @property
    def adapters_dir(self) -> Path:
        return resolve_path(self.workspace, self.config.adapters_root)

    @property
    def log_dir(self) -> Path:
        return resolve_path(self.workspace, self.config.logging.dir)

    def ensure_layout(self) -> None:
        for d in (
            self.workspace,
            self.db_path.parent,
            self.vectors_index_path.parent,
            self.models_dir,
            self.adapters_dir,
            self.log_dir,
            self.workspace / "conversations",
            self.workspace / "training",
            self.workspace / "backups",
        ):
            ensure_dir(d)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _parse_logging(raw: dict[str, Any]) -> LoggingConfig:
    data = dict(raw)
    if "json" in data and "json_logs" not in data:
        data["json_logs"] = data.pop("json")
    return LoggingConfig(**data)


def _parse_app_config(raw: dict[str, Any]) -> AppConfig:
    workspace = raw.get("workspace") or {}
    models = raw.get("models") or {}
    return AppConfig(
        workspace_name=workspace.get("name", "offline_ai_workspace"),
        database=DatabaseConfig(**(raw.get("database") or {})),
        vectors=VectorsConfig(**(raw.get("vectors") or {})),
        models_config_file=models.get("config_file", "models.yaml")
        if isinstance(models, dict)
        else "models.yaml",
        models_cache_dir=models.get("cache_dir", "models")
        if isinstance(models, dict)
        else "models",
        adapters_root=(raw.get("adapters") or {}).get("root", "adapters"),
        logging=_parse_logging(raw.get("logging") or {}),
        retrieval=RetrievalConfig(**(raw.get("retrieval") or {})),
        chunking=ChunkingConfig(**(raw.get("chunking") or {})),
        ingestion=IngestionConfig(**(raw.get("ingestion") or {})),
        hardware=HardwareConfig(**(raw.get("hardware") or {})),
        api=ApiConfig(**(raw.get("api") or {})),
        training=TrainingConfig(**(raw.get("training") or {})),
    )


def load_settings(workspace: str | Path, config_path: Path | None = None) -> Settings:
    """Load settings for a workspace. Copies defaults from project if missing."""
    root = project_root()
    ws = Path(workspace).expanduser().resolve()
    ensure_dir(ws)

    cfg_src = config_path or (root / "config.yaml")
    ws_cfg = ws / "config.yaml"
    if not ws_cfg.exists() and cfg_src.exists():
        ws_cfg.write_text(cfg_src.read_text(encoding="utf-8"), encoding="utf-8")

    policies_src = root / "config" / "policies.yaml"
    ws_policies_dir = ensure_dir(ws / "config")
    ws_policies = ws_policies_dir / "policies.yaml"
    if not ws_policies.exists() and policies_src.exists():
        ws_policies.write_text(policies_src.read_text(encoding="utf-8"), encoding="utf-8")

    models_src = root / "models.yaml"
    ws_models = ws / "models.yaml"
    if not ws_models.exists() and models_src.exists():
        ws_models.write_text(models_src.read_text(encoding="utf-8"), encoding="utf-8")

    raw_cfg = _load_yaml(ws_cfg if ws_cfg.exists() else cfg_src)
    raw_policies = _load_yaml(ws_policies if ws_policies.exists() else policies_src)
    raw_models = _load_yaml(ws_models if ws_models.exists() else models_src)

    settings = Settings(
        workspace=ws,
        config=_parse_app_config(raw_cfg),
        policies=Policies(**raw_policies),
        models_raw=raw_models,
        project_root=root,
    )
    settings.ensure_layout()
    return settings
