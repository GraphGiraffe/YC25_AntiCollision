from __future__ import annotations

from datetime import datetime
from typing import Dict, Any

import wptools


def fetch_wikitext(title: str, url: str) -> Dict[str, Any]:
    """Fetch raw wiki markup and minimal metadata for *title* via **wptools**."""
    page = wptools.page(url)
    parsed_page = page.get_parse(show=False)

    wikitext_field = parsed_page.data.get("wikitext", "")
    wiki_text = wikitext_field.get("*", "") if isinstance(wikitext_field, dict) else wikitext_field

    fullurl = (
        parsed_page.data.get("fullurl")
        or f"https://en.wikipedia.org/wiki/{url.replace(' ', '_')}"
    )
    timestamp_raw = parsed_page.data.get("modified")
    timestamp = (
        datetime.strptime(timestamp_raw, "%Y-%m-%dT%H:%M:%SZ").isoformat()
        if timestamp_raw else datetime.utcnow().isoformat()
    )

    return {"title": title, "url": fullurl, "timestamp": timestamp, "wikitext": wiki_text}
