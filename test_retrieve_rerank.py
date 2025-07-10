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

from query_utils import questions

INDEX_DIR = Path("./wiki/faiss")
CHUNKS_DIR = Path("./wiki/chunks")


def test_retrieve_rerank(query: str) -> None:
    """Test the retrieval and re-ranking stages of the OSS-RAG pipeline.
    Parameters
    ----------
    query : str
        User's question to be processed.
    """
    # stage 6
    idx_paths = route_query(query, INDEX_DIR)

    # stage 7
    cands = retrieve_candidates(query, idx_paths, CHUNKS_DIR, top_k=30)

    # stage 8
    ranked = rerank_candidates(query, cands, batch_size=3)

    print("TOP-5 after rerank:")
    for i, hit in enumerate(ranked[:5], 1):
        print(f"{i:>2}. {hit['final_score']:.3f} — {hit['metadata']['section']!r}")
        print("   ", hit['text'][:120].replace("\n", " "), "…\n")


if __name__ == "__main__":
    for q in questions[:5]:  # test only first 5 queries
        print(f"\nЗапрос: {q}")
        test_retrieve_rerank(q)
        print("-----")
