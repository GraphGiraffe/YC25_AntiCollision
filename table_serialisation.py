from __future__ import annotations

import json
from typing import List
import pandas as pd


def extract_tables(title: str) -> List[pd.DataFrame]:
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    try:
        return pd.read_html(url, flavor="bs4")
    except ValueError:
        return []


def serialise_table_row(df: pd.DataFrame) -> List[str]:
    if df.shape[1] < 2:
        return []
    result = []
    for _, row in df.iloc[:, :2].iterrows():
        subj, val = (str(x).strip() for x in row[:2])
        if subj and val and subj.lower() != val.lower():
            result.append(json.dumps({"subject": subj, "value": val}, ensure_ascii=False))
    return result
