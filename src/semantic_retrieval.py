from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from src.backends import configure_hf_environment

DEFAULT_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_EMBEDDING_MAX_LENGTH = 512
RETRIEVAL_MODES = {"tag", "semantic"}


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    device: str = DEFAULT_EMBEDDING_DEVICE
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    max_length: int = DEFAULT_EMBEDDING_MAX_LENGTH


def embedding_config_from_env(
    model: str | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
) -> EmbeddingConfig:
    resolved_model = (
        model
        if model and model != DEFAULT_EMBEDDING_MODEL
        else os.getenv("SEAL_EMBEDDING_MODEL", model or DEFAULT_EMBEDDING_MODEL)
    )
    resolved_device = (
        device
        if device and device != DEFAULT_EMBEDDING_DEVICE
        else os.getenv("SEAL_EMBEDDING_DEVICE", device or DEFAULT_EMBEDDING_DEVICE)
    )
    resolved_batch_size = (
        batch_size
        if batch_size and batch_size != DEFAULT_EMBEDDING_BATCH_SIZE
        else int(os.getenv("SEAL_EMBEDDING_BATCH_SIZE", str(batch_size or DEFAULT_EMBEDDING_BATCH_SIZE)))
    )
    resolved_max_length = (
        max_length
        if max_length and max_length != DEFAULT_EMBEDDING_MAX_LENGTH
        else int(os.getenv("SEAL_EMBEDDING_MAX_LENGTH", str(max_length or DEFAULT_EMBEDDING_MAX_LENGTH)))
    )
    return EmbeddingConfig(
        model=resolved_model,
        device=resolved_device,
        batch_size=resolved_batch_size,
        max_length=resolved_max_length,
    )


def case_retrieval_text(record: dict[str, Any]) -> str:
    return _first_text(
        record.get("retrieval_text"),
        record.get("patient_case"),
        record.get("question"),
        record.get("condition_name"),
    )


def experience_retrieval_text(record: dict[str, Any]) -> str:
    return _first_text(
        record.get("retrieval_text"),
        record.get("patient_case"),
        record.get("question"),
        "\n".join(
            str(value)
            for value in (
                record.get("condition_name"),
                record.get("failure_mode"),
                record.get("reflection"),
            )
            if value
        ),
    )


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class LocalTextEmbedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._tokenizer: object | None = None
        self._model: object | None = None
        self._torch: object | None = None
        self._device: str | None = None

    def encode(self, texts: list[str]):
        cleaned = [text.strip() for text in texts if text and text.strip()]
        if not cleaned:
            self._lazy_load()
            assert self._torch is not None
            return self._torch.empty((0, 0), dtype=self._torch.float32)

        self._lazy_load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None

        torch = self._torch
        batches = []
        for start in range(0, len(cleaned), self.config.batch_size):
            batch = cleaned[start : start + self.config.batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}
            with torch.inference_mode():
                output = self._model(**inputs)
            token_embeddings = output.last_hidden_state
            attention_mask = inputs["attention_mask"]
            pooled = self._mean_pool(token_embeddings, attention_mask)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.detach().cpu().to(dtype=torch.float32))
        return torch.cat(batches, dim=0)

    def _lazy_load(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return

        configure_hf_environment()
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic retrieval requires torch and transformers. Install "
                "requirements-hf.txt or rerun with --retrieval_mode tag."
            ) from exc

        requested_device = self.config.device.lower()
        if requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested embedding device {self.config.device!r}, but CUDA is unavailable."
            )
        else:
            device = requested_device

        tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        model = AutoModel.from_pretrained(self.config.model)
        model.to(device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._device = device

    def _mean_pool(self, token_embeddings: Any, attention_mask: Any):
        assert self._torch is not None
        torch = self._torch
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


class SemanticRecordIndex:
    def __init__(
        self,
        records: list[dict[str, Any]],
        embedder: LocalTextEmbedder,
        text_builder: Callable[[dict[str, Any]], str],
    ) -> None:
        self.records = records
        self.embedder = embedder
        self.text_builder = text_builder
        self._embedded_count = 0
        self._embeddings: Any = None
        self._indexed_records: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        if self._embeddings is None:
            return
        self._append_embeddings([record])

    def search(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        if limit <= 0 or not self.records:
            return []
        if not query_text.strip():
            return self.records[-limit:][::-1]

        self._ensure_current()
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            return []

        query_embedding = self.embedder.encode([query_text.strip()])
        return self.search_embedding(query_embedding, limit)

    def search_embedding(self, query_embedding: Any, limit: int) -> list[dict[str, Any]]:
        if limit <= 0 or not self.records:
            return []
        self._ensure_current()
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            return []
        if query_embedding is None or query_embedding.shape[0] == 0:
            return []

        if len(query_embedding.shape) == 2:
            query_vector = query_embedding[0]
        else:
            query_vector = query_embedding
        scores = self._embeddings @ query_vector
        k = min(limit, scores.shape[0])
        top_indices = scores.topk(k=k).indices.tolist()
        return [self._indexed_records[index] for index in top_indices]

    def _ensure_current(self) -> None:
        if self._embedded_count >= len(self.records):
            return
        self._append_embeddings(self.records[self._embedded_count :])

    def _append_embeddings(self, records: list[dict[str, Any]]) -> None:
        pairs = []
        for record in records:
            text = self.text_builder(record).strip()
            if text:
                pairs.append((record, text))
        if not pairs:
            self._embedded_count = len(self.records)
            return
        indexed_records = [record for record, _ in pairs]
        texts = [text for _, text in pairs]
        embeddings = self.embedder.encode(texts)
        if embeddings.shape[0] == 0:
            self._embedded_count = len(self.records)
            return
        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = self.embedder._torch.cat([self._embeddings, embeddings], dim=0)
        self._indexed_records.extend(indexed_records)
        self._embedded_count = len(self.records)
