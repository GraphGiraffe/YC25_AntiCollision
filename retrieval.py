# -*- coding: utf-8 -*-
"""
retrieval.py
============
Stage 7 of the OSS-RAG pipeline: *vector-based first-pass search*.

• Формирует эмбеддинг запроса (тот же SentenceTransformer, что и на этапе 5)
• Ищет top-k (по умолчанию 30) чанков по cosine similarity в выбранных
  FAISS-индексах
• Возвращает список словарей:
    {
      "text": …,
      "metadata": …,
      "vec_score": float   # cosine ∈ [-1, 1]
    }

NB: для работы нужен доступ к *.chunks.jsonl* (тексты) и *.faiss / *.meta.json*
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from functools import lru_cache

import numpy as np
from tqdm import tqdm

# optional dependencies (проверяем только 1 раз)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore

DEFAULT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"


def _load_chunks(slug: str, chunks_dir: Path) -> List[Dict[str, Any]]:
    fp = chunks_dir / f"{slug}.chunks.jsonl"
    return [json.loads(line) for line in fp.open()]


@lru_cache(maxsize=1)
def _get_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    """Singleton-обёртка для SentenceTransformer."""
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers не установлен")
    return SentenceTransformer(model_name, device="cpu")


def retrieve_candidates(
    query: str,
    index_paths: List[Path],
    chunks_dir: Path,
    top_k: int = 30,
    model_name: str = DEFAULT_EMBED_MODEL,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    model = _get_embedder(model_name)          # ← берём «живой» объект из кэша
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")

    all_hits: List[Dict[str, Any]] = []
    for idx_path in tqdm(index_paths, desc="FAISS search", disable=verbose == False):
        slug = idx_path.stem  # mercury_(planet)
        index = faiss.read_index(str(idx_path))
        D, I = index.search(q_emb, top_k)  # (1, k)

        # тексты и метаданные лежат в chunks.jsonl (строка i соответствует id i)
        chunk_rows = _load_chunks(slug, chunks_dir)

        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue  # FAISS возвратил «пустые» места
            row = chunk_rows[idx]
            all_hits.append(
                {
                    "text": row["text"],
                    "metadata": row["metadata"],
                    "vec_score": float(score),  # ∈ [-1, 1] (embeddings нормированы)
                }
            )
    return all_hits
