# test_routing.py
# ================
# Stage 6 of the OSS-RAG pipeline: *Routing*.
# # • Определяет, о какой планете спрашивает пользователь
# • Если определена → ищет только в соответствующих индексах
# • Если не определена → ищет во всех индексах (бэкап)

from pathlib import Path
from query_router import route_query

index_dir = Path("./wiki/faiss")  # где лежат индексы (stage 5)

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
    
    from text_dicts import questions

    for q in questions:
        
        selected = route_query(q, index_dir, verbose=args.verbose)
        print(f"\nЗапрос: {q}")
        print("Будем искать в:", *selected, sep="\n  • ")
        print("-----")
