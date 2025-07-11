# -*- coding: utf-8 -*-
"""
answer_generation.py
====================
Stages 9 & 10 – context building + answer generation.

• build_context() — формирует контекст с <<PAGE n>>.
• generate_answer() — отдаёт структурированный JSON-ответ.

Поддерживаемые источники модели:
  1. 🤗 transformers repo_id / локальный каталог — обычная FP16-модель;
  2. 🤗 repo_id, оканчивающийся на '-GGUF'            – авто-скачка *.gguf;
  3. Локальный путь к конкретному *.gguf              – сразу в llama.cpp.

Зависимости (CPU-режим):
    pip install transformers>=4.42 llama-cpp-python>=0.2.64 huggingface-hub
"""
from __future__ import annotations
from functools import lru_cache

import json
import re
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple

from tqdm import tqdm

from prompts import build_final_prompt

# ──────────────────────────────────────────────────────────────────────────────
#  Опциональные зависимости (загружаем «лениво», чтобы не падать без GPU)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:  # pragma: no cover
    pipeline = AutoTokenizer = AutoModelForCausalLM = None  # type: ignore

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover
    Llama = None  # type: ignore

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:  # pragma: no cover
    hf_hub_download = list_repo_files = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
#  Модели по умолчанию
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_GGUF_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
_GGUF_PREFERRED = ("Q3_K_M.gguf", "Q4_K_M.gguf", "Q4_K.gguf", "Q5_0.gguf")  # что скачивать прежде всего

# ──────────────────────────────────────────────────────────────────────────────
#  9. CONTEXT BUILDING
# ──────────────────────────────────────────────────────────────────────────────

# def build_context(
#     ranked_chunks: List[Dict[str, Any]],
#     top_n: int = 10,
# ) -> Tuple[str, List[int]]:
#     ctx_lines, pages = [], []
#     for i, ch in enumerate(ranked_chunks[:top_n], 1):
#         ctx_lines.append(f"<<PAGE {i}>>\n{ch['text'].strip()}")
#         pages.append(i)
#     return "\n\n".join(ctx_lines), pages

# optional tokenizer ─────────────────────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _enc = None  # fallback → будем считать «token = word»

KEEP_CTX_TOKENS = 2048        # максимум токенов, которые передадим модели
# ---------------------------------------------------------------------------


def _encode(text: str) -> list[int]:
    if _enc:
        return _enc.encode(text)
    return text.split()        # грубая оценка, но лучше, чем ничего


def _decode(tokens: list[int] | list[str]) -> str:
    if _enc:
        return _enc.decode(tokens)          # type: ignore[arg-type]
    return " ".join(tokens)                 # fallback


def build_context(
    ranked_chunks: List[Dict[str, Any]],
    top_n: int = 10,
) -> Tuple[str, List[int]]:
    ctx_lines, pages = [], []
    for i, ch in enumerate(ranked_chunks[:top_n], 1):
        ctx_lines.append(f"<<PAGE {i}>>\n{ch['text'].strip()}")
        pages.append(i)

    ctx = "\n\n".join(ctx_lines)

    # -------- trim to KEEP_CTX_TOKENS ----------
    toks = _encode(ctx)
    if len(toks) > KEEP_CTX_TOKENS:
        toks = toks[-KEEP_CTX_TOKENS:]      # оставляем ТОЛЬКО хвост
        ctx = _decode(toks)

    return ctx, pages

# ──────────────────────────────────────────────────────────────────────────────
#  helpers: загрузка gguf, выбор backend
# ──────────────────────────────────────────────────────────────────────────────


def _download_best_gguf(repo_id: str, gguf_quantization: List[str] = _GGUF_PREFERRED) -> Path:
    if hf_hub_download is None or list_repo_files is None:
        raise RuntimeError("huggingface-hub не установлен")

    files = list_repo_files(repo_id)
    # ищем «предпочтительный» файл
    for pat in gguf_quantization:
        match = next((f for f in files if f.endswith(pat)), None)
        if match:
            break
    else:
        # берём любой .gguf
        ggufs = [f for f in files if f.endswith(".gguf")]
        if not ggufs:
            raise FileNotFoundError("в репо нет *.gguf")
        match = ggufs[0]

    local_path = hf_hub_download(repo_id, match, local_dir="~/.cache/gguf_models")
    return Path(local_path)


def _load_llm(model_ref: str | Path, max_ctx=4096, max_tok=512, gguf_quantization: List[str] = _GGUF_PREFERRED, verbose: bool = False):
    """
    Выбирает backend:
      • .gguf → llama.cpp
      • repo_id с «-GGUF» → download+llama.cpp
      • иначе → transformers.pipeline
    """
    model_ref = Path(model_ref) if not isinstance(model_ref, Path) else model_ref

    # ---------------- llama.cpp branch ----------------
    if str(model_ref).endswith(".gguf") or str(model_ref).endswith("-GGUF"):
        if Llama is None:
            raise RuntimeError("llama-cpp-python не установлен")

        if str(model_ref).endswith("-GGUF"):
            model_ref = _download_best_gguf(str(model_ref), gguf_quantization)  # repo_id → Path

        return Llama(
            model_path=str(model_ref),
            n_ctx=max_ctx,
            n_threads=4,
            n_gpu_layers=35,  # измените, если собрали CUDA/Metal
            verbose=verbose
        )

    # ---------------- transformers branch --------------
    if pipeline is None:
        raise RuntimeError("transformers не установлен")

    tok = AutoTokenizer.from_pretrained(str(model_ref))
    mdl = AutoModelForCausalLM.from_pretrained(str(model_ref), device_map="auto")
    return pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=max_tok,
        do_sample=False,
        verbose=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 10. ANSWER GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def _safe_json_extract(raw: str) -> Dict[str, Any]:
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"final_answer": "N/A"}


# def generate_answer(
#     query: str,
#     ranked_chunks: List[Dict[str, Any]],
#     model_ref: str | Path = DEFAULT_GGUF_REPO,
#     top_n_ctx: int = 10,
#     max_new_tokens: int = 512,
# ) -> Dict[str, Any]:
#     """Return structured dict as specified in prompts.JSON_SCHEMA."""
#     ctx, _ = build_context(ranked_chunks, top_n=top_n_ctx)
#     prompt = build_final_prompt(ctx, query)

#     llm = _load_llm(model_ref, max_ctx=4096, max_tok=max_new_tokens)

#     # llama.cpp → объект Llama (callable), transformers → pipeline
#     if isinstance(llm, Llama):
#         out = llm(prompt, max_tokens=max_new_tokens, temperature=0)["choices"][0]["text"]
#     else:
#         out = llm(prompt)[0]["generated_text"]

#     result = _safe_json_extract(out)
#     for k in ("step_by_step_analysis", "reasoning_summary", "citations", "final_answer"):
#         result.setdefault(k, "" if k != "citations" else [])
#     return result


@lru_cache(maxsize=1)
def _get_answer_llm(model_ref: str | Path, max_ctx=4096, max_tok=512, gguf_quantization: List[str] = _GGUF_PREFERRED, verbose=False):
    """Единоразово загружает Llama или transformers-pipeline."""
    return _load_llm(model_ref, max_ctx=max_ctx, max_tok=max_tok, gguf_quantization=gguf_quantization, verbose=verbose)

# ─────────────────────────────────────────────────────────────────────────────


def generate_answer(
    query: str,
    ranked_chunks: List[Dict[str, Any]],
    model_ref: str | Path = DEFAULT_GGUF_REPO,
    top_n_ctx: int = 10,
    max_new_tokens: int = 512,
    gguf_quantization: List[str] = _GGUF_PREFERRED,
    verbose: int = False,
) -> Dict[str, Any]:
    ctx, _ = build_context(ranked_chunks, top_n=top_n_ctx)
    prompt = build_final_prompt(ctx, query)

    llm = _get_answer_llm(model_ref, max_ctx=4096, max_tok=max_new_tokens, gguf_quantization=gguf_quantization, verbose=verbose)

    if isinstance(llm, Llama):
        # llama.cpp путь
        out = llm(prompt, max_tokens=max_new_tokens, temperature=0)["choices"][0]["text"]
    else:                                                # transformers.pipeline
        out = llm(prompt)[0]["generated_text"]

    result = _safe_json_extract(out)
    if verbose > 1:
        result['prompt'] = prompt  # для отладки
    for k in ("step_by_step_analysis", "reasoning_summary", "citations", "final_answer"):
        result.setdefault(k, "" if k != "citations" else [])
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  CLI demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Generate structured answer (stages 9-10)")
    parser.add_argument("query", help="User question")
    parser.add_argument("ranked_pickle", type=Path, help="Pickle with ranked chunks (list[dict])")
    parser.add_argument(
        "--model",
        default=DEFAULT_GGUF_REPO,
        help="HF repo_id, gguf path, или transformers-модель",
    )
    args = parser.parse_args()

    ranked = pickle.loads(args.ranked_pickle.read_bytes())
    ans = generate_answer(args.query, ranked, model_ref=args.model, gguf_quantization=_GGUF_PREFERRED)
    print(json.dumps(ans, indent=2, ensure_ascii=False))
