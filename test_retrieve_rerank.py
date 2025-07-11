# test_retrieve_rerank.py
# ================
# Stage 7 and 8 of the OSS-RAG pipeline: *Retrieval* and *Re-ranking*.
# # • Формирует эмбеддинг запроса (тот же SentenceTransformer, что и на этапе 5)
# # • Ищет top-k (по умолчанию 30) чанков по cosine similarity в выбранных
# #   FAISS-индексах
# # • Возвращает список словарей:
# #     {
# #       "text": …,
# #       "metadata": …,
# #       "vec_score": float   # cosine ∈ [-1, 1]
# #     }

from pathlib import Path
from query_router import route_query
from retrieval import retrieve_candidates
from rerank_utils import rerank_candidates

from text_dicts import questions

INDEX_DIR = Path("./wiki/faiss")
CHUNKS_DIR = Path("./wiki/chunks")

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 30

# RERANK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# RERANK_MODEL = "BAAI/bge-reranker-base"
# RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"  # latest, best quality

def test_retrieve_rerank(query: str, verbose: bool = True) -> None:
    """Test the retrieval and re-ranking stages of the OSS-RAG pipeline.
    Parameters
    ----------
    query : str
        User's question to be processed.
    """
    # stage 6
    idx_paths = route_query(query, INDEX_DIR, verbose=verbose)

    # stage 7
    cands = retrieve_candidates(query, idx_paths, CHUNKS_DIR, top_k=TOP_K, model_name=EMBED_MODEL, verbose=verbose)

    # stage 8
    ranked = rerank_candidates(query, cands, batch_size=3, model_name=RERANK_MODEL, verbose=verbose)

    print("TOP-5 after rerank:")
    for i, hit in enumerate(ranked[:5], 1):
        print(f"{i:>2}. {hit['final_score']:.3f} — {hit['metadata']['section']!r}")
        print("   ", hit['text'][:120].replace("\n", " "), "…\n")


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
    
    for q in questions[:5]:  # test only first 5 queries
        print(f"\nЗапрос: {q}")
        test_retrieve_rerank(q, verbose=args.verbose)
        print("-----")
