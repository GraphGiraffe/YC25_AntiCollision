from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any

import wptools
import requests


API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "oss-rag-bot/0.1 (+https://github.com/your/repo)"}

# def fetch_wikitext(title: str, url: str) -> Dict[str, Any]:
#     """Fetch raw wiki markup and minimal metadata for *title* via **wptools**."""
#     page = wptools.page(url, silent=False)
#     parsed_page = page.get_parse(show=False)

#     wikitext_field = parsed_page.data.get("wikitext", "")
#     wiki_text = wikitext_field.get("*", "") if isinstance(wikitext_field, dict) else wikitext_field

#     fullurl = (
#         parsed_page.data.get("fullurl")
#         or f"https://en.wikipedia.org/wiki/{url.replace(' ', '_')}"
#     )
#     timestamp_raw = parsed_page.data.get("modified")
#     timestamp = (
#         datetime.strptime(timestamp_raw, "%Y-%m-%dT%H:%M:%SZ").isoformat()
#         if timestamp_raw else datetime.utcnow().isoformat()
#     )

#     return {
#         "title": title,
#         "url": fullurl,
#         "timestamp": timestamp,
#         "wikitext": wiki_text
#     }

def fetch_wikitext(title: str, url: str) -> Dict[str, Any]:
    """
    Return dict with raw wikitext + minimal metadata
    — **без** подзапроса картинок.
    """
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",      # ← ровно один prop!
        "format": "json",
        "formatversion": 2,
    }

    res = requests.get(API, params=params, headers=HEADERS, timeout=30)
    res.raise_for_status()
    data = res.json()["parse"]

    return {
        "title":        data.get("title", title),
        "url":          f"https://en.wikipedia.org/wiki/{url.replace(' ', '_')}",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "wikitext":     data["wikitext"],
    }