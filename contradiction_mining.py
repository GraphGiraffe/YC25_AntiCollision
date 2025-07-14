# -*- coding: utf-8 -*-
"""
contradiction_mining.py
=======================
Stage-11 – find contradictions between *input_text* and the KB built at stages 1-5.

Pipeline (per chunk):
    1) embed chunk
    2) retrieve & rerank evidence passages
    3) ask LLM: support / contradict / neutral
Outputs list[dict]:
    {
      "chunk_id": str,
      "chunk": str,
      "status": "contradict" | "support" | "neutral",
      "evidence": list[str],          # KB passages
      "reason": str                   # short LLM explanation
    }
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from chunking import split_into_sections, chunk_section           # 11-a
from query_router import route_query                              # 6
from retrieval import retrieve_candidates                         # 7
from rerank_utils import rerank_candidates                        # 8
from answer_generation import _get_answer_llm                     # 11-d
from prompts import build_contradiction_prompt                    # new prompt builder

# ---- runtime constants ---------------------------------------------------- #
INDEX_DIR   = Path("wiki/faiss")
CHUNKS_DIR  = Path("wiki/chunks")

CHUNK_TOKS    = 70               # ≈ 2-3 предложения
TOP_K_SEARCH  = 30               # top-k for retrieval
TOP_EVIDENCE  = 5                # top-k for rerank

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
ANSWER_MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
GGUF_PREF   = ("Q4_K_M.gguf", "Q4_K.gguf")

# ------------------------ helpers ----------------------------------------- #
def _chunk_input(text: str) -> List[Dict[str, Any]]:
    """Return list of Chunk-like dicts from raw user text (planet='input')."""
    chunks: List[Dict[str, Any]] = []
    for sec_title, sec_txt, sec_start, _ in split_into_sections(text):
        for ch in chunk_section("input", sec_title, sec_txt, sec_start, chunk_size=CHUNK_TOKS, overlap=50):
            chunks.append({"id": ch.id, "text": ch.text, "meta": ch.metadata})
    return chunks


def _ask_llm(chunk: str, ctx: str, llm) -> Dict[str, Any]:
    """LLM classification support / contradict / neutral."""
    prompt = build_contradiction_prompt(chunk, ctx)
    raw = llm(prompt, max_tokens=512, temperature=0)["choices"][0]["text"]
    # minimal post-processing
    import json, re
    try:
        obj = json.loads(re.search(r"\{.*\}", raw, re.S).group())
    except Exception:
        obj = {"status": "neutral", "reason": "LLM parse error"}
    return obj


# ---------------------- MAIN ENTRY ---------------------------------------- #
def mine_contradictions(
    input_text: str,
    top_k_search: int = 30,
    top_evidence: int = 5,
    verbose: int = 0
) -> List[Dict[str, Any]]:
    llm = _get_answer_llm(ANSWER_MODEL, gguf_quantization=GGUF_PREF, verbose=verbose)
    chunks = _chunk_input(input_text)

    results: List[Dict[str, Any]] = []
    for ch in tqdm(chunks, desc="Contradiction mining"):
        # --- retrieval & rerank ---
        idx_paths = route_query(ch["text"], INDEX_DIR, verbose=verbose)
        cands = retrieve_candidates(ch["text"], idx_paths, CHUNKS_DIR,
                                    top_k=top_k_search, model_name=EMBED_MODEL, verbose=verbose)
        ranked = rerank_candidates(ch["text"], cands, model_name=RERANK_MODEL, verbose=verbose)
        evidence = [r["text"] for r in ranked[:top_evidence]]
        ctx = "\n\n".join(evidence)

        # --- ask LLM ---
        verdict = _ask_llm(ch["text"], ctx, llm)
        results.append({
            "chunk_id": ch["id"],
            "chunk": ch["text"],
            "evidence": evidence,
            **verdict            # status + reason
        })
    return results


# ---------------- CLI demo ------------------------------------------------- #
if __name__ == "__main__":
    import argparse, textwrap, json

    parser = argparse.ArgumentParser(description="Find contradictions inside a text")
    parser.add_argument("file", help="Path to .txt with narrative to check")
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
    out = mine_contradictions(txt, top_k_search=TOP_K_SEARCH, top_evidence=TOP_EVIDENCE, verbose=verbose)
    print(json.dumps(out, indent=2, ensure_ascii=False))
