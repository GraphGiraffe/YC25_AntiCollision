# test_routing.py
# ================
# Stage 6 of the OSS-RAG pipeline: *Routing*.
# # • Определяет, о какой планете спрашивает пользователь
# • Если определена → ищет только в соответствующих индексах
# • Если не определена → ищет во всех индексах (бэкап)

from pathlib import Path
from query_router import route_query

index_dir = Path("./wiki/faiss")  # где лежат индексы (stage 5)

from query_utils import questions

for q in questions:
    selected = route_query(q, index_dir)
    print(f"\nЗапрос: {q}")
    print("Будем искать в:", *selected, sep="\n  • ")
    print("-----")
