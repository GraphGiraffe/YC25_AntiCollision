from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# Optional dependencies ------------------------------------------------------
try:
    from pint import UnitRegistry  # unit conversion

    _ureg = UnitRegistry()
except ImportError:
    _ureg = None

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _enc = None  # fallback to naive whitespace tokenisation


@dataclass
class Chunk:
    id: str          # unique id within planet
    text: str
    metadata: Dict[str, Any]


def _token_count(text: str) -> int:
    if _enc:
        return len(_enc.encode(text))
    return len(text.split())


def split_into_sections(text: str) -> List[Tuple[str, str, int, int]]:
    """Return list of (section_title, section_text, start_char, end_char)."""
    lines = text.splitlines(keepends=True)
    sections: List[Tuple[str, List[str], int]] = []  # (title, lines, start_char)
    current_title = "intro"
    current_start = 0
    buf: List[str] = []

    for ln in lines:
        heading = re.match(r"^(==+)(.+?)(==+)\s*$", ln)
        if heading:
            # flush previous
            if buf:
                sections.append((current_title, buf.copy(), current_start))
                current_start += sum(len(x) for x in buf)
                buf.clear()
            current_title = heading.group(2).strip()
            continue
        buf.append(ln)
    # last section
    if buf:
        sections.append((current_title, buf.copy(), current_start))

    result: List[Tuple[str, str, int, int]] = []
    for title, lines_, start in sections:
        txt = "".join(lines_)
        end = start + len(txt)
        result.append((title, txt.strip(), start, end))
    return result


def chunk_section(planet: str, sec_title: str, sec_text: str, sec_start: int,
                  chunk_size: int = 300, overlap: int = 50) -> List[Chunk]:
    if not sec_text:
        return []
    enc = _enc
    if enc is None:
        # naive fallback – whitespace tokens
        tokens = sec_text.split()
        token_to_char = []
        pos = 0
        for tok in tokens:
            pos = sec_text.find(tok, pos)
            token_to_char.append(pos)
            pos += len(tok)
    else:
        tokens = enc.encode(sec_text)
        token_to_char = []
        pos = 0
        for tok in tokens:
            piece = enc.decode([tok])
            idx = sec_text.find(piece, pos)
            token_to_char.append(idx if idx >= 0 else pos)
            pos = (idx if idx >= 0 else pos) + len(piece)

    chunks: List[Chunk] = []
    step = chunk_size - overlap
    for i in range(0, len(tokens), step):
        window_toks = tokens[i: i + chunk_size]
        if not window_toks:
            break

        # --- границы чанка в символах --------------------------------------
        start_char_rel = token_to_char[i]
        end_token_idx = i + len(window_toks) - 1
        if enc:
            last_piece = enc.decode([tokens[end_token_idx]])
            token_len = len(last_piece)
            chunk_text = enc.decode(window_toks)
        else:
            token_len = len(tokens[end_token_idx])
            chunk_text = " ".join(window_toks)

        end_char_rel = token_to_char[end_token_idx] + token_len
        global_start = sec_start + start_char_rel
        global_end = sec_start + end_char_rel
        # -------------------------------------------------------------------

        chunk_id = f"{planet}:{len(chunks)}"
        meta = {
            "planet": planet,
            "section": sec_title,
            "char_start": global_start,
            "char_end": global_end,
        }
        chunks.append(Chunk(chunk_id, chunk_text.strip(), meta))

        if i + chunk_size >= len(tokens):
            break
    return chunks
