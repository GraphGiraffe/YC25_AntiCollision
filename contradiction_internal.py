# -*- coding: utf-8 -*-
"""
contradiction_internal.py
=========================
Stage-12 – detect contradictions **inside** a single long text
(without external KB).
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

import re, faiss, numpy as np
from tqdm import tqdm

from chunking import chunk_section, split_into_sections           # 12-a
from retrieval import _get_embedder                                # 12-b
from answer_generation import _get_answer_llm                      # 12-e
from prompts import build_pair_prompt                       # 12-e

# ---------- hyper-params --------------------------------------------------- #
EMBED_MODEL   = "BAAI/bge-base-en-v1.5"
ANSWER_MODEL  = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
GGUF_PREF     = ("Q4_K_M.gguf", "Q4_K.gguf")
CHUNK_TOKS    = 70               # ≈ 2-3 предложения
TOP_K_NEIGHB  = 8
SIM_THRESHOLD = 0.25             # cosine ≥ 0.25 → проверяем LLM

# ---------- helpers -------------------------------------------------------- #
def _sent_chunk(text: str) -> List[str]:
    """Rough sentence split keeping punctuation."""
    # primitive splitter – можно заменить на nltk / spacy
    return re.split(r"(?<=[.!?])\s+", text.strip())

def _chunk_doc(text: str) -> List[str]:
    """Return list of small claim-chunks (~70 tokens)."""
    claims: List[str] = []
    for sec_title, sec_txt, *_ in split_into_sections(text):
        toks, buf = 0, []
        for sent in _sent_chunk(sec_txt):
            s_toks = len(sent.split())
            if toks + s_toks > CHUNK_TOKS and buf:
                claims.append(" ".join(buf))
                buf, toks = [], 0
            buf.append(sent)
            toks += s_toks
        if buf:
            claims.append(" ".join(buf))
    return claims

def _nli_llm(llm, a: str, b: str) -> Dict[str, Any]:
    prompt = build_pair_prompt(a, b)
    raw = llm(prompt, max_tokens=512, temperature=0)["choices"][0]["text"]
    import json, re
    try:
        return json.loads(re.search(r"\{.*\}", raw, re.S).group())
    except Exception:
        return {"status":"neutral","reason":"parse_error"}

# ---------- main ----------------------------------------------------------- #
def find_internal_contradictions(text: str, verbose: bool = False) -> List[Dict[str, Any]]:
    # 1) chunk + embed
    claims = _chunk_doc(text)
    embedder = _get_embedder(EMBED_MODEL)
    embs = embedder.encode(claims, normalize_embeddings=True).astype("float32")

    # 2) build temp FAISS index
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    llm = _get_answer_llm(ANSWER_MODEL, gguf_quantization=GGUF_PREF, verbose=verbose)

    contradictions: List[Dict[str, Any]] = []
    for i, claim in tqdm(list(enumerate(claims)), desc="Scanning pairs"):
        D, I = index.search(embs[i:i+1], TOP_K_NEIGHB+1)  # +1 self
        for score, j in zip(D[0][1:], I[0][1:]):          # skip self idx 0
            if score < SIM_THRESHOLD: break
            verdict = _nli_llm(llm, claim, claims[j])
            if verdict.get("status") == "contradict":
                contradictions.append({
                    "claim_id": i,
                    "counter_id": int(j),
                    "claim": claim,
                    "counter_claim": claims[j],
                    "reason": verdict.get("reason","")
                })
    return contradictions

# ---------------- CLI ------------------------------------------------------ #
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Find contradictions inside a text")
    parser.add_argument("file", help="path to .txt")
    parser.add_argument(
        '-v',
        '--verbose',
        dest="verbose",
        action = 'append_const',
        const = 1,
        help="Enable verbose output (tqdm progress bars and debug info)",
    )
    args = parser.parse_args()

    verbose = sum(args.verbose) if args.verbose else 0

    txt = Path(args.file).read_text()
    result = find_internal_contradictions(txt, verbose=verbose)
    print(json.dumps(result, indent=2, ensure_ascii=False))
