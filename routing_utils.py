# -*- coding: utf-8 -*-
"""
routing_utils.py
================
Helper utilities for query routing (stage 6 of the OSS-RAG pipeline).

• PLANET_ALIASES – список возможных упоминаний (синонимы/эпитеты) → slug
• compile_patterns() – готовит regex-паттерны под каждую планету
• detect_planets(query) – вернёт список slug-ов, встречающихся в запросе

Файл не зависит от FAISS или sentence-transformers – только «чистый» regex.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List

# ──────────────────────────────────────────────────────────────────────────────
#  1. Алиасы планет – можно дополнять / локализовать
# ──────────────────────────────────────────────────────────────────────────────
# Ключ – slug как генерируется в build_wiki_dataset (re.sub r"[() ]","_")
PLANET_ALIASES: Dict[str, List[str]] = {
    "mercury": ["mercury", "mercury planet", "mercury (planet)", "mercury_(planet)"],
    "venus": ["venus"],
    "earth": ["earth", "terra", "blue planet"],
    "mars": ["mars", "red planet"],
    "jupiter": ["jupiter", "gas giant", "largest planet"],
    "saturn": ["saturn", "ringed planet"],
    "uranus": ["uranus", "ice giant uranus"],
    "neptune": ["neptune", "ice giant neptune"],
}

# ──────────────────────────────────────────────────────────────────────────────
#  2. Компиляция regex-ов
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache
def _compiled_patterns() -> Dict[str, re.Pattern]:
    compiled: Dict[str, re.Pattern] = {}
    for slug, aliases in PLANET_ALIASES.items():
        # «\b» – словоцела, (?i) – case-insensitive, группируем алиасы «|»
        pattern = r"(?i)\b(" + "|".join(re.escape(a) for a in aliases) + r")\b"
        compiled[slug] = re.compile(pattern)
    return compiled


# ──────────────────────────────────────────────────────────────────────────────
#  3. Публичная функция распознавания
# ──────────────────────────────────────────────────────────────────────────────
def detect_planets(query: str) -> List[str]:
    """
    Return list of slug(s) whose alias is found in *query*
    (order preserved as in PLANET_ALIASES).
    """
    query = query.lower()
    hits: List[str] = []
    for slug, pat in _compiled_patterns().items():
        if pat.search(query):
            hits.append(slug)
    return hits
