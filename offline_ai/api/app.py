"""Optional local FastAPI application (bind localhost by default)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from offline_ai.core.engine import LocalAI


class IngestTextRequest(BaseModel):
    text: str
    source: str = "api"
    metadata: dict[str, Any] = Field(default_factory=dict)
    author: str | None = None
    source_url: str | None = None


class AskRequest(BaseModel):
    query: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20


class FeedbackRequest(BaseModel):
    answer_id: str
    rating: int | None = None
    correction: str | None = None


def create_app(workspace: str | Path = "./workspace") -> FastAPI:
    ai = LocalAI(workspace)
    app = FastAPI(title="Offline AI", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "workspace": str(ai.workspace)}

    @app.post("/ingest")
    def ingest(req: IngestTextRequest) -> dict[str, Any]:
        return ai.ingest_text(
            req.text,
            source=req.source,
            metadata=req.metadata,
            author=req.author,
            source_url=req.source_url,
        )

    @app.post("/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        return ai.ask(req.query)

    @app.post("/search")
    def search(req: SearchRequest) -> dict[str, Any]:
        return ai.search(req.query, top_k=req.top_k)

    @app.get("/documents/{document_id}")
    def get_document(document_id: str) -> dict[str, Any]:
        doc = ai.get_document(document_id)
        if not doc:
            raise HTTPException(404, "document not found")
        return doc

    @app.get("/claims/{claim_id}")
    def get_claim(claim_id: str) -> dict[str, Any]:
        claim = ai.get_claim(claim_id)
        if not claim:
            raise HTTPException(404, "claim not found")
        return claim

    @app.get("/entities/{entity_id}")
    def get_entity(entity_id: str) -> dict[str, Any]:
        ent = ai.get_entity(entity_id)
        if not ent:
            raise HTTPException(404, "entity not found")
        return ent

    @app.get("/sources/{document_id}")
    def get_source(document_id: str) -> dict[str, Any]:
        return get_document(document_id)

    @app.get("/memory/stats")
    def memory_stats() -> dict[str, Any]:
        return ai.stats()

    @app.get("/models")
    def models() -> dict[str, Any]:
        return ai.settings.models_raw

    @app.get("/adapters")
    def adapters() -> dict[str, Any]:
        root = ai.settings.adapters_dir
        items = [p.name for p in root.glob("adapter_v*") if p.is_dir()] if root.exists() else []
        return {"adapters": items}

    @app.post("/feedback")
    def feedback(req: FeedbackRequest) -> dict[str, Any]:
        try:
            return ai.feedback(req.answer_id, rating=req.rating, correction=req.correction)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    return app
