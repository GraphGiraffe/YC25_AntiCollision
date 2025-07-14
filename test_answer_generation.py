# test_answer_generation.py
# ============================
# Stage 9 & 10 of the OSS-RAG pipeline: *Context Building* and *Answer Generation*.
# # 1. build_context() – takes top-N chunks, prefixes with markers `<<PAGE n>>`.
# # 2. generate_answer() – feeds prompt to a local model and returns parsed dictionary.

from pathlib import Path
import json

from query_router import route_query
from retrieval import retrieve_candidates
from rerank_utils import rerank_candidates
from answer_generation import generate_answer

from text_dicts import questions


INDEX_DIR = Path("./wiki/faiss")
CHUNKS_DIR = Path("./wiki/chunks")

# EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

TOP_K = 50

# RERANK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_MODEL = "BAAI/bge-reranker-base"
# RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"  # latest, best quality

# RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"

ANSWER_MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
GGUF_PREFERRED = ["Q4_K_M.gguf", "Q4_K.gguf", "Q5_0.gguf"]  # preferred quantization formats

def test(question: str, verbose: int = 0) -> None:
    """Test the full OSS-RAG pipeline for a given question."""
    # 6. routing
    idx_files = route_query(question, INDEX_DIR)

    # 7. first-pass vector search
    candidates = retrieve_candidates(question, idx_files, CHUNKS_DIR, top_k=TOP_K, model_name=EMBED_MODEL, verbose=verbose)

    # 8. LLM rerank
    ranked = rerank_candidates(question, candidates, model_name=RERANK_MODEL)

    # 9-10. context → answer
    answer = generate_answer(question, ranked, max_new_tokens=512, model_ref=ANSWER_MODEL, gguf_quantization=tuple(GGUF_PREFERRED), verbose=verbose)

    print(json.dumps(answer, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the OSS-RAG pipeline (stages 6-10)")
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
    for q in questions[:20]:  # test only first 5 queries
        print(f"\nЗапрос: {q}")
        test(q, verbose=verbose)
        print("-----")

    print("✔️  Test finished.")
