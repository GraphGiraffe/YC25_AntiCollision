# -*- coding: utf-8 -*-
"""
Pipeline: build_wiki_dataset.py
================================
Full pipeline to build a small-scale Wikipedia corpus about the eight
planets of the Solar System for an OSS RAG prototype.

Stages implemented
------------------
1. Raw collection        – wiki markup + metadata (wptools)
2. Parsing & cleaning    – plain text (mwparserfromhell)
3. Table serialisation   – row-wise JSON blocks (pandas.read_html)
4. Chunking              – ~300 tokens, 50-token overlap (tiktoken)
5. Vectorisation         – open-source embeddings + FAISS per planet

Open-source stack only – no OpenAI calls.

Dependencies (Python 3.11):
    pip install wptools mwparserfromhell pandas requests tqdm python-dateutil pint \
                beautifulsoup4 lxml tiktoken sentence-transformers faiss-cpu

Inspired by:
    • Habr article «RAG Challenge #2»
    • https://github.com/IlyaRice/RAG-Challenge-2
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Iterable

from tqdm import tqdm

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

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None  # type: ignore

from raw_collection import fetch_wikitext
from parsing_cleaning import parse_wikitext
from chunking import split_into_sections, chunk_section, Chunk
from table_serialisation import extract_tables, serialise_table_row
# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

PLANETS = [
    ["Mercury", "Mercury (planet)"],
    ["Venus", "Venus"],
    ["Earth", "Earth"],
    ["Mars", "Mars"],
    ["Jupiter", "Jupiter"],
    ["Saturn", "Saturn"],
    ["Uranus", "Uranus"],
    ["Neptune", "Neptune"],
]

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # 1024-d open model
TOKENS_PER_CHUNK = 300
TOKEN_OVERLAP = 50


###############################################################################
# 1. RAW COLLECTION (WIKI MARKUP + METADATA)
###############################################################################


def stage1_collect_raw(out_dir: Path, title_url_list: Iterable[List[str, str]] = PLANETS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for title, url in tqdm(list(title_url_list), desc="Fetching wiki pages"):
        try:
            data = fetch_wikitext(title, url)
        except Exception as exc:
            tqdm.write(f"⚠️  Failed to fetch {title}: {exc}")
            continue
        slug = re.sub(r"[() ]", "_", title.lower())
        (out_dir / f"{slug}.raw.wiki").write_text(data["wikitext"], encoding="utf-8")
        meta = {k: v for k, v in data.items() if k != "wikitext"}
        (out_dir / f"{slug}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


###############################################################################
# 2. PARSING & CLEANING
###############################################################################


def stage2_clean(in_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in tqdm(list(in_dir.glob("*.raw.wiki")), desc="Cleaning text"):
        raw_text = raw_path.read_text()
        clean_text = parse_wikitext(raw_text)
        slug = raw_path.stem.replace(".raw", "")
        (out_dir / f"{slug}.clean.txt").write_text(clean_text)


###############################################################################
# 3. TABLE SERIALISATION (OPTIONAL AUGMENTATION)
###############################################################################


def stage3_tables(in_dir: Path, out_dir: Path, title_url_list: Iterable[List[str, str]] = PLANETS) -> None:
    for title, url in tqdm(list(title_url_list), desc="Serialising tables"):
        slug = re.sub(r"[() ]", "_", title.lower())
        cleaned_path = out_dir / f"{slug}.clean.txt"
        if not cleaned_path.exists():
            continue
        text = cleaned_path.read_text()
        tables = extract_tables(title)
        lines: List[str] = []
        for tbl in tables:
            lines.extend(serialise_table_row(tbl))
        if lines:
            cleaned_path.write_text(text + "\n\n" + "\n".join(lines))


###############################################################################
# 4. CHUNKING (300 tokens, 50 overlap)
###############################################################################


def stage4_chunk(in_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cleaned_path in tqdm(list(in_dir.glob("*.clean.txt")), desc="Chunking"):
        planet_slug = cleaned_path.stem.replace(".clean", "")
        full_text = cleaned_path.read_text()
        sections = split_into_sections(full_text)
        all_chunks: List[Chunk] = []
        for sec_title, sec_txt, sec_start, _ in sections:
            all_chunks.extend(chunk_section(planet_slug, sec_title, sec_txt, sec_start, chunk_size=TOKENS_PER_CHUNK, overlap=TOKEN_OVERLAP))
        # persist as JSONL
        chunks_path = out_dir / f"{planet_slug}.chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as fp:
            for ch in all_chunks:
                fp.write(json.dumps({"id": ch.id, "text": ch.text, "metadata": ch.metadata}, ensure_ascii=False))
                fp.write("\n")


###############################################################################
# 5. VECTORIZATION (EMBEDDINGS + FAISS)
###############################################################################


def stage5_vectorise(chunks_dir: Path, index_dir: Path, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    if faiss is None:
        raise RuntimeError("faiss-cpu not installed")

    index_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, device="cpu")

    for chunks_path in tqdm(list(chunks_dir.glob("*.chunks.jsonl")), desc="Embedding + FAISS"):
        planet_slug = chunks_path.stem.replace(".chunks", "")
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        with chunks_path.open() as fp:
            for line in fp:
                obj = json.loads(line)
                texts.append(obj["text"])
                metadatas.append(obj["metadata"])
        if not texts:
            continue
        embs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        # FAISS expects float32
        embs = embs.astype("float32")
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        faiss.write_index(index, str(index_dir / f"{planet_slug}.faiss"))
        # save metadata alongside
        (index_dir / f"{planet_slug}.meta.json").write_text(json.dumps(metadatas, indent=2, ensure_ascii=False))


###############################################################################
#  CLI ENTRY
###############################################################################


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Wikipedia planet corpus (stages 1-5)")
    parser.add_argument("out", type=Path, help="Output directory for artefacts")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["stage1", "stage2", "stage3", "stage4", "stage5"],
        help="Which stages to skip",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="HF sentence-transformers model for embeddings",
    )

    args = parser.parse_args()

    raw_dir = args.out / "raw"
    clean_dir = args.out / "clean"
    chunks_dir = args.out / "chunks"
    index_dir = args.out / "faiss"

    if "stage1" not in args.skip:
        stage1_collect_raw(raw_dir, title_url_list=PLANETS)
    if "stage2" not in args.skip:
        stage2_clean(raw_dir, clean_dir)
    if "stage3" not in args.skip:
        stage3_tables(raw_dir, clean_dir)
    if "stage4" not in args.skip:
        stage4_chunk(clean_dir, chunks_dir)
    if "stage5" not in args.skip:
        stage5_vectorise(chunks_dir, index_dir, model_name=args.model)

    print("✔️  Pipeline finished. Outputs in", args.out)
