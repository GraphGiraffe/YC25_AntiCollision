# -*- coding: utf-8 -*-
"""
routing_utils.py – auto-aliases & routing regex for stage-6
"""
from __future__ import annotations
import re, unicodedata
from pathlib import Path
from functools import lru_cache
from typing import Dict, List

# --------------------------------------------------------------------------- #
#  1. Источник истин – единый список страниц                                  #
# --------------------------------------------------------------------------- #
from text_dicts import SOLAR_SYSTEM_PAGES   # ← тот, что вы расширили выше

# кастомные «народные» синонимы
EXTRA_ALIASES: Dict[str, List[str]] = {
    "mars":     ["red planet"],
    "jupiter":  ["gas giant", "largest planet"],
    "saturn":   ["ringed planet"],
    "earth":    ["terra", "blue planet"],
}

# --------------------------------------------------------------------------- #
#  2. Автогенерация алиас-словаря                                             #
# --------------------------------------------------------------------------- #
def _slugify(title: str) -> str:
    """Same rule as dataset builder: spaces & () → '_' + lower-case."""
    slug = re.sub(r"[() ]", "_", title.lower())
    return unicodedata.normalize("NFKD", slug)

def _auto_aliases() -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}
    for title, url in SOLAR_SYSTEM_PAGES:
        slug = _slugify(title)
        base = title.lower()
        url_low = url.lower()
        simple = re.sub(r"\s*\(.*?\)", "", url_low)          # без скобок
        # набор базовых вариантов
        vals = {slug, base, url_low, simple}
        # добиваем «пробелы→_» и обратно
        vals |= {v.replace(" ", "_") for v in vals}
        vals |= {v.replace("_", " ") for v in vals}
        # + кастомные
        vals |= set(EXTRA_ALIASES.get(slug, []))
        aliases[slug] = sorted(vals, key=len)    # короткие вперёд
    return aliases

PLANET_ALIASES: Dict[str, List[str]] = _auto_aliases()

# --------------------------------------------------------------------------- #
#  3. Компиляция regex-ов                                                     #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _compiled_patterns() -> Dict[str, re.Pattern]:
    pats = {}
    for slug, alist in PLANET_ALIASES.items():
        pattern = r"(?i)\b(" + "|".join(re.escape(a) for a in alist) + r")\b"
        pats[slug] = re.compile(pattern)
    return pats

# --------------------------------------------------------------------------- #
#  4. Публичная функция                                                       #
# --------------------------------------------------------------------------- #
def detect_objects(query: str) -> List[str]:
    """
    Return list of slug(s) detected in *query*.
    Order = order of SOLAR_SYSTEM_PAGES list.
    """
    hits: List[str] = []
    for slug, pat in _compiled_patterns().items():
        if pat.search(query):
            hits.append(slug)
    return hits
