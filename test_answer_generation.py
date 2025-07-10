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

from query_utils import questions


INDEX_DIR = Path("./wiki/faiss")
CHUNKS_DIR = Path("./wiki/chunks")

EMBED_MODEL = "BAAI/bge-large-en-v1.5"

RERANK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# RERANK_MODEL = "BAAI/bge-reranker-base"
# RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"

ANSWER_MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

def test(question: str) -> None:
    """Test the full OSS-RAG pipeline for a given question."""
    # 6. routing
    idx_files = route_query(question, INDEX_DIR)

    # 7. first-pass vector search
    candidates = retrieve_candidates(question, idx_files, CHUNKS_DIR, top_k=30, model_name=EMBED_MODEL)

    # 8. LLM rerank
    ranked = rerank_candidates(question, candidates, model_name=RERANK_MODEL)

    # 9-10. context → answer
    answer = generate_answer(question, ranked, model_ref=ANSWER_MODEL)

    print(json.dumps(answer, indent=2, ensure_ascii=False))


if __name__ == "__main__":

    for q in questions[:5]:  # test only first 5 queries
        print(f"\nЗапрос: {q}")
        test(q)
        print("-----")
