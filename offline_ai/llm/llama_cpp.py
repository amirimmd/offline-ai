"""llama.cpp GGUF backend."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from offline_ai.llm.base import LLMBackend, LLMMessage, LLMResult
from offline_ai.utils.logging import get_logger

logger = get_logger(__name__)


class LlamaCppBackend(LLMBackend):
    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        chat_format: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.chat_format = chat_format
        self._llm = None

    @property
    def model_name(self) -> str:
        return self.model_path.name

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        if self._llm is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {self.model_path}")
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_threads=self.n_threads,
            verbose=False,
            chat_format=self.chat_format,
        )
        logger.info(
            "llama.cpp model loaded",
            extra={"event": "llm_load", "component": "llama_cpp"},
        )

    def unload(self) -> None:
        self._llm = None

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> LLMResult:
        self.load()
        assert self._llm is not None
        out = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        text = out["choices"][0]["text"]
        return LLMResult(text=text, model=self.model_name, raw=out)

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> LLMResult:
        self.load()
        assert self._llm is not None
        # Cap completion so prompt + output stays inside n_ctx.
        prompt_chars = sum(len(m.content or "") for m in messages)
        prompt_tokens_est = max(1, prompt_chars // 2)
        room = max(32, self.n_ctx - prompt_tokens_est - 64)
        max_tokens = max(32, min(int(max_tokens), room))
        payload = [{"role": m.role, "content": m.content} for m in messages]
        try:
            out = self._llm.create_chat_completion(
                messages=payload,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
        except Exception as exc:
            msg = str(exc)
            if "context" in msg.lower() or "token" in msg.lower():
                raise RuntimeError(
                    f"Prompt too large for model context ({self.n_ctx}): {msg}"
                ) from exc
            raise
        text = out["choices"][0]["message"]["content"]
        return LLMResult(text=text or "", model=self.model_name, raw=out)

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        self.load()
        assert self._llm is not None
        stream = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            text = chunk["choices"][0].get("text") or ""
            if text:
                yield text
