"""
Application facade for the offline knowledge system.

``LocalAI`` wires persistence, retrieval, extraction, and the grounded ask
pipeline. Prefer this class from application code; lower-level modules are for
extension and testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from offline_ai.core.config import Settings, load_settings
from offline_ai.core.hardware import HardwareDetector, HardwareInfo, InferenceRecommendation
from offline_ai.database.session import Database
from offline_ai.ingestion.base import IngestItem, IngestionPipeline
from offline_ai.ingestion.tweets import tweets_to_items
from offline_ai.memory.manager import MemoryManager
from offline_ai.memory.raw import RawMemory
from offline_ai.retrieval.hybrid import RetrievalWeights
from offline_ai.utils.console import configure_stdio
from offline_ai.utils.logging import get_logger, setup_logging
from offline_ai.utils.paths import resolve_path

logger = get_logger(__name__)


class LocalAI:
    """
    Primary entry point.

    Documents and embeddings are stored under the workspace directory. Model
    weights are optional and replaceable without losing memory.
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        config_path: Path | None = None,
        auto_setup_logging: bool = True,
        init_db: bool = True,
        auto_embed: bool = True,
    ) -> None:
        self.settings: Settings = load_settings(workspace, config_path=config_path)
        configure_stdio()
        if auto_setup_logging:
            setup_logging(
                level=self.settings.config.logging.level,
                log_dir=self.settings.log_dir,
                json_logs=self.settings.config.logging.json_logs,
            )
        self._hardware = HardwareDetector()
        self._hw_info: HardwareInfo | None = None
        self._recommendation: InferenceRecommendation | None = None
        self.auto_embed = auto_embed

        self.db: Database | None = None
        self.raw_memory: RawMemory | None = None
        self.ingestion: IngestionPipeline | None = None
        self.memory_manager: MemoryManager | None = None
        self.memory = None
        self.llm = None
        self.embedder = None
        self.vector_store = None
        self.retriever = None
        self.evidence = None
        self.graph = None
        self.extractor = None

        if init_db:
            self._init_persistence()

        logger.info(
            "LocalAI initialized",
            extra={"event": "engine_init", "component": "engine"},
        )

    def _embedding_model_path(self) -> str | None:
        models = self.settings.models_raw.get("models") or {}
        defaults = self.settings.models_raw.get("defaults") or {}
        key = defaults.get("embedding", "multilingual-e5-base")
        cfg = models.get(key) or {}
        local = cfg.get("local_path")
        if not local:
            return None
        return str(resolve_path(self.workspace, local))

    def _init_persistence(self) -> None:
        from offline_ai.evidence.engine import EvidenceEngine
        from offline_ai.extraction.pipeline import ExtractionPipeline
        from offline_ai.llm.factory import create_llm_backend
        from offline_ai.memory.graph import KnowledgeGraph

        self.db = Database(
            self.settings.db_path,
            echo=self.settings.config.database.echo,
        )
        chunking = self.settings.config.chunking.model_dump()
        self.raw_memory = RawMemory(self.db, chunking=chunking)

        hw = self.detect_hardware()
        device = hw["recommendation"]["device"]
        rec = hw["recommendation"]
        weights = RetrievalWeights(
            semantic=self.settings.config.retrieval.weight_semantic,
            lexical=self.settings.config.retrieval.weight_lexical,
            entity=self.settings.config.retrieval.weight_entity,
            metadata=self.settings.config.retrieval.weight_metadata,
            graph=self.settings.config.retrieval.weight_graph,
        )
        self.memory_manager = MemoryManager.create(
            self.db,
            self.raw_memory,
            index_path=self.settings.vectors_index_path,
            id_map_path=self.settings.vectors_id_map_path,
            dimension=self.settings.config.vectors.dimension,
            embedding_model_path=self._embedding_model_path(),
            device=device if device == "cuda" else "cpu",
            weights=weights,
        )
        self.memory = self.memory_manager
        self.embedder = self.memory_manager.embedder
        self.vector_store = self.memory_manager.vector_store
        self.retriever = self.memory_manager.retriever

        self.extractor = ExtractionPipeline(self.db)
        self.graph = KnowledgeGraph(self.db)
        self.llm = create_llm_backend(
            self.settings.models_raw,
            workspace=self.workspace,
            device=device,
            n_ctx=int(rec.get("context_length", 4096)),
            n_gpu_layers=int(rec.get("n_gpu_layers", 0)),
            n_threads=int(rec.get("n_threads", 4)),
        )
        self.evidence = EvidenceEngine(
            self.db,
            self.memory_manager,
            self.llm,
            self.settings.policies,
            top_k=self.settings.config.retrieval.top_k,
            rerank_top_k=self.settings.config.retrieval.rerank_top_k,
            min_evidence_score=self.settings.config.retrieval.min_evidence_score,
        )

        def _after(doc: Any, created: bool) -> None:
            if not created:
                return
            if self.auto_embed and self.memory_manager is not None:
                self.memory_manager.index_document(doc.document_id)
            if self.extractor is not None:
                self.extractor.process_document(doc.document_id, doc.original_text)

        self.ingestion = IngestionPipeline(self.db, self.raw_memory, after_document=_after)

    @property
    def workspace(self) -> Path:
        return self.settings.workspace

    def detect_hardware(self) -> dict[str, Any]:
        info, rec = self._hardware.recommend(
            prefer_gpu=self.settings.config.hardware.prefer_gpu,
            min_vram_mb_for_gpu=self.settings.config.hardware.min_vram_mb_for_gpu,
        )
        self._hw_info = info
        self._recommendation = rec
        return {
            "hardware": {
                "os": info.os_name,
                "python": info.python_version,
                "cpu_count": info.cpu_count,
                "ram_mb": info.ram_mb,
                "cuda_available": info.cuda_available,
                "gpu_name": info.gpu_name,
                "vram_mb": info.vram_mb,
                "gpus": info.gpus,
            },
            "recommendation": {
                "device": rec.device,
                "profile": rec.profile,
                "quantization": rec.quantization,
                "context_length": rec.context_length,
                "batch_size": rec.batch_size,
                "n_gpu_layers": rec.n_gpu_layers,
                "n_threads": rec.n_threads,
                "notes": rec.notes,
            },
        }

    def doctor(self, *, offline: bool = False) -> dict[str, Any]:
        report: dict[str, Any] = {
            "workspace": str(self.workspace),
            "workspace_exists": self.workspace.exists(),
            "db_path": str(self.settings.db_path),
            "db_exists": self.settings.db_path.exists(),
            "vectors_index": str(self.settings.vectors_index_path),
            "vectors_exist": self.settings.vectors_index_path.exists(),
            "policies": {
                "citation_required": self.settings.policies.citation_required,
                "allow_unsupported_claims": self.settings.policies.allow_unsupported_claims,
                "preserve_original_documents": self.settings.policies.preserve_original_documents,
            },
            "hardware": self.detect_hardware(),
            "offline_mode_requested": offline,
            "ok": True,
            "issues": [],
        }
        if offline:
            if self.settings.policies.security.get("allow_network", False):
                report["issues"].append("policies.security.allow_network is true")
                report["ok"] = False
            if self.settings.policies.security.get("telemetry", False):
                report["issues"].append("telemetry enabled in policies")
                report["ok"] = False
        return report

    def add(
        self,
        knowledge: str | list[str] | dict[str, Any] | list[dict[str, Any]],
        *,
        source: str = "manual",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Store knowledge in persistent memory.

        Accepts a string, a list of strings, a mapping with a ``text`` field,
        or a list of such mappings.

        Examples:
            ai.add("Company X VPN was compromised.")
            ai.add(["fact one", "fact two"])
            ai.add({"text": "...", "author": "alice", "source_url": "https://..."})
        """
        assert self.ingestion is not None
        items: list[IngestItem] = []

        def _from_dict(obj: dict[str, Any]) -> IngestItem:
            text = obj.get("text") or obj.get("content") or obj.get("knowledge")
            if not text:
                raise ValueError("dict knowledge must include text/content/knowledge")
            return IngestItem(
                text=str(text),
                source=str(obj.get("source") or source),
                source_url=obj.get("source_url") or obj.get("url") or kwargs.get("source_url"),
                author=obj.get("author") or kwargs.get("author"),
                timestamp=obj.get("timestamp") or kwargs.get("timestamp"),
                language=obj.get("language") or kwargs.get("language"),
                metadata={
                    k: v
                    for k, v in obj.items()
                    if k
                    not in {
                        "text",
                        "content",
                        "knowledge",
                        "source",
                        "source_url",
                        "url",
                        "author",
                        "timestamp",
                        "language",
                    }
                },
                source_type=obj.get("source_type") or kwargs.get("source_type") or "manual",
                source_name=obj.get("source_name") or kwargs.get("source_name"),
                collection_method=obj.get("collection_method") or "api",
            )

        if isinstance(knowledge, str):
            items.append(
                IngestItem(
                    text=knowledge,
                    source=source,
                    source_url=kwargs.get("source_url"),
                    author=kwargs.get("author"),
                    timestamp=kwargs.get("timestamp"),
                    language=kwargs.get("language"),
                    metadata=kwargs.get("metadata") or {},
                    source_type=kwargs.get("source_type") or "manual",
                    source_name=kwargs.get("source_name"),
                    collection_method=kwargs.get("collection_method", "api"),
                )
            )
        elif isinstance(knowledge, dict):
            items.append(_from_dict(knowledge))
        elif isinstance(knowledge, list):
            for entry in knowledge:
                if isinstance(entry, str):
                    items.append(
                        IngestItem(
                            text=entry,
                            source=source,
                            author=kwargs.get("author"),
                            source_url=kwargs.get("source_url"),
                            metadata=kwargs.get("metadata") or {},
                            source_type=kwargs.get("source_type") or "manual",
                            collection_method="api",
                        )
                    )
                elif isinstance(entry, dict):
                    items.append(_from_dict(entry))
                else:
                    raise TypeError(f"Unsupported knowledge item type: {type(entry)}")
        else:
            raise TypeError("knowledge must be str, dict, or list")

        stats = self.ingestion.ingest_items(items, source_description=source)
        return stats.to_dict()

    def learn(
        self,
        knowledge: str | list[str] | dict[str, Any] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Alias for :meth:`add`."""
        return self.add(knowledge, **kwargs)

    def ingest_text(
        self,
        text: str,
        source: str = "text",
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.add(
            text,
            source=source,
            metadata=metadata or {},
            **kwargs,
        )

    def ingest_tweets(self, tweets: list[dict] | list[str]) -> dict[str, Any]:
        assert self.ingestion is not None
        items = tweets_to_items(tweets)
        stats = self.ingestion.ingest_items(items, source_description="tweets")
        return stats.to_dict()

    def ingest_file(self, path: str | Path) -> dict[str, Any]:
        assert self.ingestion is not None
        stats = self.ingestion.ingest_file(Path(path))
        return stats.to_dict()

    def rebuild_knowledge(self) -> dict[str, int]:
        """
        Re-extract entities, claims, and graph relationships from all documents.

        Use after upgrading the relation engine so older workspaces get strong links.
        """
        assert self.extractor is not None
        return self.extractor.reprocess_all()

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        assert self.raw_memory is not None
        doc = self.raw_memory.get_document(document_id)
        if doc is None:
            return None
        return RawMemory.document_to_dict(doc)

    def ask(
        self,
        query: str,
        *,
        text_only: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        """
        Answer a question using retrieved evidence only.

        Returns a structured dict by default. Pass ``text_only=True`` to receive
        only the answer string.
        """
        assert self.evidence is not None
        result = self.evidence.ask(query, **kwargs)
        if text_only:
            return str(result.get("answer") or "")
        return result

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        assert self.memory_manager is not None
        top_k = kwargs.get("top_k", self.settings.config.retrieval.top_k)
        rerank_top_k = kwargs.get("rerank_top_k", self.settings.config.retrieval.rerank_top_k)
        return self.memory_manager.search(
            query,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            source=kwargs.get("source"),
            author=kwargs.get("author"),
            after=kwargs.get("after"),
            before=kwargs.get("before"),
        )

    def get_claim(self, claim_id: str) -> dict[str, Any] | None:
        from offline_ai.evidence.citation import CitationManager

        assert self.db is not None
        return CitationManager(self.db).get_source("claim", claim_id)

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        from offline_ai.evidence.citation import CitationManager

        assert self.db is not None
        return CitationManager(self.db).get_source("entity", entity_id)

    def feedback(
        self,
        answer_id: str,
        rating: int | None = None,
        correction: str | None = None,
    ) -> dict[str, Any]:
        from sqlalchemy import select

        from offline_ai.database.ids import next_id
        from offline_ai.database.models import AnswerRecord, Feedback

        assert self.db is not None
        with self.db.session() as session:
            ans = session.execute(
                select(AnswerRecord).where(AnswerRecord.answer_id == answer_id)
            ).scalar_one_or_none()
            if ans is None:
                raise KeyError(f"Unknown answer_id: {answer_id}")
            fid = next_id(session, "FB")
            retrieved = []
            if isinstance(ans.structured_json, dict):
                retrieved = [
                    d.get("id") for d in (ans.structured_json.get("documents") or []) if d.get("id")
                ]
            fb = Feedback(
                feedback_id=fid,
                answer_id=answer_id,
                rating=rating,
                correction=correction,
                original_answer=ans.answer_text,
                retrieved_document_ids=retrieved,
            )
            session.add(fb)
            session.commit()
            return {
                "feedback_id": fid,
                "answer_id": answer_id,
                "rating": rating,
                "correction": correction,
            }

    def stats(self) -> dict[str, Any]:
        doc_count = 0
        if self.db is not None:
            from offline_ai.database.repositories import DocumentRepository

            with self.db.session() as session:
                doc_count = DocumentRepository(session).count()
        return {
            "workspace": str(self.workspace),
            "version": __import__("offline_ai").__version__,
            "documents": doc_count,
            "vectors": self.vector_store.size if self.vector_store is not None else 0,
            "embedder": getattr(self.embedder, "model_name", None),
            "llm": getattr(self.llm, "model_name", None),
            "llm_deep": "extractive" not in str(getattr(self.llm, "model_name", "")).lower(),
            "components_ready": {
                "db": self.db is not None,
                "memory": self.memory_manager is not None,
                "llm": self.llm is not None,
                "embedder": self.embedder is not None,
                "vector_store": self.vector_store is not None,
                "retriever": self.retriever is not None,
            },
        }
