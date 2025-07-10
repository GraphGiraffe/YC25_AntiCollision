# -*- coding: utf-8 -*-
"""
query_router.py
===============
Stage 6 – *Routing* for the OSS-RAG pipeline.

Основная идея:
• Пытаемся определить, о какой(-их) планете(-ах) спрашивает пользователь.
• Если нашли → ограничиваем поиск только нужными FAISS-индексами.
• Если не нашли → «бэкап» – возвращаем все индексы (8 шт.).

Файл будет расширяться шагами 7-10 (retrieval, rerank, answer generation),
поэтому API сделан модульно.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from routing_utils import detect_planets  # наша «подсказка» из алиасов

# ──────────────────────────────────────────────────────────────────────────────
#  Основная точка входа стадии 6
# ──────────────────────────────────────────────────────────────────────────────
def route_query(
    query: str,
    index_dir: Path,
    fallback_all: bool = True,
) -> List[Path]:
    """
    Decide which *.faiss index files to use for *query*.

    Parameters
    ----------
    query : str
        Original user question.
    index_dir : Path
        Directory produced at stage 5 (one *.faiss per planet).
    fallback_all : bool, default=True
        If no planet detected – return **all** index paths;
        otherwise return empty list.

    Returns
    -------
    List[Path]
        Ordered list of FAISS index files to search.
    """
    matched_slugs = detect_planets(query)
    if matched_slugs:
        idx_paths = [
            index_dir / f"{slug}.faiss" for slug in matched_slugs if (index_dir / f"{slug}.faiss").exists()
        ]
        return idx_paths
    # fallback
    if fallback_all:
        return sorted(index_dir.glob("*.faiss"))
    return []


# ──────────────────────────────────────────────────────────────────────────────
#  CLI-демонстрация (можно убрать в проде)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Route user query to planet-specific FAISS indices")
    parser.add_argument("query", help="User question")
    parser.add_argument("index_dir", type=Path, help="Directory with *.faiss")
    parser.add_argument("--no-fallback", action="store_true", help="Disable return of all indices")

    args = parser.parse_args()
    paths = route_query(args.query, args.index_dir, fallback_all=not args.no_fallback)
    print("\n".join(str(p) for p in paths) if paths else "(no indices selected)")
