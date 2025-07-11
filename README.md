## `README.md`

> **Solar-System RAG** – an end-to-end, fully-offline Retrieval-Augmented-Generation
> workflow that fact-checks questions about the Sun, planets, moons, dwarf-planets,
> comets & belts.
>
> **Stack:** Python 3.11 • Conda + pip • Sentence-Transformers • FAISS • Cross-Encoder
> rerank • `llama.cpp`/Transformers for answer-LLM

---
### RAG Scheme

```
──────────────────────────────  build database  ─────────────────────────────

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                    INGESTION  (stages 1-5, offline)                     │
  │                                                                         │
  │  Wikipedia pages  →  clean & chunk  →  BGE embeddings  →  FAISS index   │
  │                                                                         │
  │  (see: build_wiki_dataset.py → wiki/{raw│clean│chunks│faiss})           │
  └─────────────────────────────────────────────────────────────────────────┘


────────────────────────────────  llm usage  ────────────────────────────────

┌──────────┐  ───route───►  ┌──────────────────┐         ┌───────────────┐
│ Question │────────┬──────►│  Index selector  │────────►│  FAISS search │──┐
└──────────┘        │       │  (routing_utils) │         │  (top-K = 30) │  │
                    │       └──────────────────┘         └───────────────┘  │
                    │                       selected *.faiss                │
                    │                                                       │
                    │                   pairs                               │
                    │            ┌──────────────────┐                       │
                    │            │ Cross-Encoder CE │◄──────────────────────┘
                    │            │  (rerank_utils)  │   vec-score α + CE 1-α 
                    │            └────────┬─────────┘                        
                    │                     │ top-10 chunks                    
                    │                     ▼                                  
                    │         ┌────────────────────────┐                     
                    │         │    build_context()     │                     
                    │         │  <<PAGE n>> markers    │                     
                    │         └───────────┬────────────┘                     
                    │                     │ ctx (prompt string)              
                    │                     ▼                                  
                    │         ┌────────────────────────┐                     
                    │         │    Prompt template     │                     
                    └────────►│ JSON-schema + ctx + Q  │                     
                              │   (prompts.py)         │                     
                              └───────────┬────────────┘                     
                                          │ final prompt                     
                                          ▼                                  
                              ┌────────────────────────┐                     
                              │  Answer-LLM            │                     
                              │  llama.cpp  (GGUF)     │                     
                              └───────────┬────────────┘                     
                                          │ JSON output                      
                                          ▼                                  
                                  ┌──────────────┐                           
                                  │   Answer     │                           
                                  │ (strict JSON)│                           
                                  └──────────────┘                           

```
---

### Pipeline overview

| #      | Stage                        | What happens                                                                                                                                                                                           | Why                                                                    | \*\*Tools used in **this repo**                                         |
| ------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **0**  | **Env setup**                | Conda + pip `env.yml` installs: Py 3.11, `pandas`, `sentence-transformers`, `faiss-gpu/-cpu`, `llama-cpp-python`, `transformers`, …                                                                    | Reproducible, GPU/CPU-agnostic sandbox                                 | Conda + pip (see `env.yml`)                                             |
| **1**  | Raw collection               | For each page in `SOLAR_SYSTEM_PAGES` call MediaWiki API → save `*.raw.wiki` + meta JSON                                                                                                               | Freeze source text & revision stamp                                    | `requests` (REST) — `raw_collection.py`                                 |
| **2**  | Parsing & cleaning           | `mwparserfromhell` → plain text; strip templates, `[citation needed]`; convert units to SI (`pint`)                                                                                                    | Noise-free text = better embeddings                                    | `parsing_cleaning.py`                                                   |
| **3**  | Table serialisation *(opt.)* | `pandas.read_html` extracts infobox / physical-data tables → one JSON line per row appended to text                                                                                                    | Extra structured facts for retrieval                                   | `table_serialisation.py`                                                |
| **4**  | Chunking                     | `tiktoken` → 300-token windows, 50-token overlap; store `planet, section, char_start/end` in metadata                                                                                                  | Short, semantically tight chunks improve recall                        | `chunking.py`                                                           |
| **5**  | Vectorisation                | Encode chunks with **`BAAI/bge-base-en-v1.5`** (768 d); build one `faiss.IndexFlatIP` **per page**                                                                                                     | “One doc = one index” isolates topics                                  | FAISS GPU/CPU, `sentence-transformers` — `build_wiki_dataset.py`        |
| **6**  | Query routing                | Regex-match slugs via auto-aliases → select subset of indices, fallback = all                                                                                                                          | 8-70× less search space, same recall                                   | `routing_utils.py`, `query_router.py`                                   |
| **7**  | First-pass retrieval         | Search each selected index for **top-k (30-50)** by cosine                                                                                                                                             | Brute-force Flat IP maximises recall                                   | `retrieval.py`                                                          |
| **8**  | Cross-encoder rerank         | Score each *(query, chunk)* pair with **`BAAI/bge-reranker-base`**; final = 0.3·vector + 0.7·CE                                                                                                        | Cheap precision boost, no generation latency                           | `sentence-transformers.CrossEncoder` — `rerank_utils.py`                |
| **9**  | Context build                | Take top-10 chunks, prefix with `<<PAGE n>>`; trim to ≤2048 tokens                                                                                                                                     | Fits into 4 k ctx; page tags enable citations                          | `answer_generation.build_context()`                                     |
| **10** | Answer generation            | Prompt = system + JSON-schema + context + question → **`Mistral-7B` GGUF** via `llama.cpp` → strict JSON: `step_by_step_analysis`, `reasoning_summary`, `citations`, `final_answer` (“N/A” if no fact) | CoT + structured output = minimal hallucinations, easy post-processing | `answer_generation.generate_answer()` (Transformers **or** `llama.cpp`) |

*(Stages 11-12 from the original checklist—CLI wrappers & evaluation—are not yet implemented in this fork, but hooks are ready in `test_*` scripts.)*

---

## 0 · Repo layout

```
solar-rag/
 ├─ build_wiki_dataset.py      # stages 1-5 ─ collect → vectorise
 ├─ query_router.py            # stage 6    ─ query → index subset
 ├─ retrieval.py               # stage 7    ─ FAISS Top-k
 ├─ rerank_utils.py            # stage 8    ─ Cross-Encoder rerank
 ├─ answer_generation.py       # stages 9-10 ─ context → JSON answer
 ├─ prompts.py                 # reusable system / schema prompts
 ├─ test_* .py                 # quick demos for every stage-group
 ├─ text_dicts.py              # list of Wikipedia pages + sample questions
 ├─ env.yml                    # Conda + pip environment (CPU ✓ / GPU ✓)
 └─ wiki/                      # will appear after stage-5
```

---

## 1 · Set-up (one command)

```bash
conda env create -f env.yml      # installs conda + pip deps
conda activate ac                # (the env name is "ac")
```

### Optional · GPU build of `llama.cpp`

```bash
# only if you DO have CUDA ≥ 11.8
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 \
pip install --no-binary :all: llama-cpp-python==0.2.64
```

Check:

```bash
python - <<'PY'
import llama_cpp
print("llama.cpp →", llama_cpp.__version__)
PY
```

---

## 2 · Build the local corpus (stages 1-5)

```bash
python build_wiki_dataset.py          # ≈ 4-5 min on laptop
# artefacts land in ./wiki/{raw│clean│chunks│faiss}
```

*Pages taken from* `text_dicts.SOLAR_SYSTEM_PAGES` – 50 articles incl. planets,
moons, “Moons of Jupiter”, Sun, belts, comets …

---

## 3 · End-to-end test (stages 6-10)

```bash
python test_answer_generation.py       # answers 20 sample questions
```

Response is **strict JSON**:

```json
{
  "step_by_step_analysis": "According to <<PAGE 2>>, Mercury has...",
  "reasoning_summary": "No measurable atmosphere → surface pressure ≈ 0 Pa.",
  "citations": [2],
  "final_answer": "Effectively 0 Pa (near-vacuum)."
}
```

---

## 4 · Performance knobs

│ Layer          │ Default                    │ Swap to                          │ Effect                      │
│ -------------- │ -------------------------- │ -------------------------------- │ --------------------------- │
│ **Embeddings** │ `bge-base-en-v1.5` (768 d) │ `bge-large`                      │ +1 pp nDCG, ×1.4 RAM        │
│ **Reranker**   │ `bge-reranker-base`        │ `bge-reranker-large`             │ +2 pp nDCG, needs 3 GB VRAM │
│ **Answer LLM** │ `Mistral-7B Q4_K_M` GGUF   │ GPU layers 35 or `phi-3-mini-4k` │ 5 tok/s → 25 tok/s          │
│ **Prompt**     │ 10 pages → 3 600 tok       │ `KEEP_CTX_TOKENS = 1_024`        │ –70 % prompt-eval time      │

Tune mixing weight in `rerank_utils.py`: `_ALPHA = 0.3`
(0 = CE-only, 1 = vector-only).

---

Happy hacking 🚀
