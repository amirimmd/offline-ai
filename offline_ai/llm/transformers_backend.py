"""Transformers / PyTorch LLM backend."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from offline_ai.llm.base import LLMBackend, LLMMessage, LLMResult
from offline_ai.utils.logging import get_logger

logger = get_logger(__name__)


class TransformersBackend(LLMBackend):
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        load_in_4bit: bool = False,
        max_context: int = 4096,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_context = max_context
        self._model = None
        self._tokenizer = None

    @property
    def model_name(self) -> str:
        return self.model_path.name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Transformers model not found: {self.model_path}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        kwargs = {"local_files_only": True}
        if self.load_in_4bit and self.device == "cuda":
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **kwargs)
        if "device_map" not in kwargs:
            self._model.to(self.device)
        self._model.eval()
        logger.info(
            "Transformers model loaded",
            extra={"event": "llm_load", "component": "transformers"},
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> LLMResult:
        self.load()
        assert self._model is not None and self._tokenizer is not None
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        text = self._tokenizer.decode(gen, skip_special_tokens=True)
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        return LLMResult(text=text, model=self.model_name)

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> LLMResult:
        self.load()
        assert self._tokenizer is not None
        payload = [{"role": m.role, "content": m.content} for m in messages]
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                payload, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        # Simple non-token streaming fallback: yield full generation
        result = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        yield result.text
