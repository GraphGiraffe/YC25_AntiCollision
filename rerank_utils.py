# -*- coding: utf-8 -*-
"""
rerank_utils.py
===============
Stage 8 of the OSS-RAG pipeline: *LLM-based re-ranking*.

• Делит кандидатов на пачки по 3 («mini-batch»)  
• Для каждой пачки → prompt в open-source LLM (HF transformers, чат-режим)  
• Модель возвращает JSON-лист из 3 оценок relevance ∈ [0, 1]  
• Итоговый скор = 0.3·vector_norm + 0.7·llm_score

Модель по умолчанию: TinyLlama-1.1B-Chat — лёгкая, но даёт ощутимый прирост
precision без коммерческих API.
"""
# from __future__ import annotations

# import json
# import math
# from pathlib import Path
# from typing import List, Dict, Any

# from functools import lru_cache

# import numpy as np
# from tqdm import tqdm

# try:
#     from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# except ImportError:  # pragma: no cover
#     pipeline = None  # type: ignore

# import transformers
# transformers.logging.set_verbosity_error()

# DEFAULT_RERANK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# # DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
# # DEFAULT_RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"

# _ALPHA = 0.3  # вес similarity-score (LLM = 0.7)


# def _build_prompt(query: str, passages: List[str]) -> str:
#     tpl = [
#         "You are a helpful assistant that scores the relevance of passages to a user's question.",
#         f"Question: {query}",
#         "",
#         "Passages:",
#     ]
#     for i, p in enumerate(passages, 1):
#         tpl.append(f"[{i}] \"\"\"{p}\"\"\"")
#     tpl.append(
#         "\nReturn ONLY a JSON array with "
#         f"{len(passages)} numbers between 0 and 1 — relevance for each passage in order."
#     )
#     return "\n".join(tpl)


# def _norm_vec(score: float) -> float:
#     """Cosine ∈ [-1, 1] → [0, 1]."""
#     return (score + 1.0) / 2.0


# @lru_cache(maxsize=1)
# def _get_rerank_pipe(model_name: str = DEFAULT_RERANK_MODEL, max_new_tokens: int = 20):
#     if pipeline is None:
#         raise RuntimeError("transformers не установлен")
#     tok = AutoTokenizer.from_pretrained(model_name)
#     mdl = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
#     return pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=max_new_tokens)


# def rerank_candidates(
#     query: str,
#     candidates: List[Dict[str, Any]],
#     model_name: str = DEFAULT_RERANK_MODEL,
#     batch_size: int = 3,
# ) -> List[Dict[str, Any]]:
#     """
#     Add field 'final_score' к каждому кандидату и вернуть отсортированный список.
#     """
#     llm = _get_rerank_pipe(model_name)

#     # --- шаг 1. пакетами по batch_size ---
#     for i in tqdm(range(0, len(candidates), batch_size), desc="LLM rerank"):
#         batch = candidates[i : i + batch_size]
#         prompt = _build_prompt(query, [b["text"] for b in batch])
#         raw = llm(prompt, do_sample=False)[0]["generated_text"]
#         try:
#             scores = json.loads(raw.split(prompt, 1)[-1].strip())
#         except json.JSONDecodeError:
#             # если модель «сбилась» – fallback = 0.5
#             scores = [0.5] * len(batch)

#         # --- шаг 2. комбинируем ---
#         for cand, s in zip(batch, scores):
#             cand["llm_score"] = float(s)
#             cand["final_score"] = _ALPHA * _norm_vec(cand["vec_score"]) + (1 - _ALPHA) * cand["llm_score"]

#     # --- шаг 3. сортировка по итоговому скору ---
#     return sorted(candidates, key=lambda x: x["final_score"], reverse=True)

# rerank_ce.py  (новая версия Stage-8)

from __future__ import annotations
from functools import lru_cache
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
import torch

from sentence_transformers import CrossEncoder

# DEFAULT_RERANK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"     # 1.1 B, max_len=512
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"                 # 312 M, max_len=512
# DEFAULT_RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"     # 312 M, max_len=512
_ALPHA = 0.3                                                     # вес от векторного поиска


@lru_cache(maxsize=1)
def _get_ce(model_name: str = DEFAULT_RERANK_MODEL) -> CrossEncoder:
    """Singleton – грузим cross-encoder ровно один раз за процесс."""
    return CrossEncoder(model_name, device="cuda" if torch.cuda.is_available() else "cpu")


def _norm_vec(score: float) -> float:
    return (score + 1.0) / 2.0          # cosine −1…1 → 0…1


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    model_name: str = DEFAULT_RERANK_MODEL,
    batch_size: int = 16,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    ce = _get_ce(model_name=model_name)

    pairs = [[query, cand["text"]] for cand in candidates]
    ce_scores: np.ndarray = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    for cand, s in zip(candidates, ce_scores):
        cand["llm_score"] = float(s)                       # 0…1
        cand["final_score"] = (
            _ALPHA * _norm_vec(cand["vec_score"]) + (1 - _ALPHA) * cand["llm_score"]
        )

    return sorted(candidates, key=lambda x: x["final_score"], reverse=True)
