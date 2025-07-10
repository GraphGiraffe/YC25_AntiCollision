## `README.md`

> **OSS RAG prototype** – fact-checking about the Solar-System planets
> Open-source only • Python 3.11 • Conda + pip hybrid environment

---

## 1 · Quick install

```bash
# clone the repo
git clone https://github.com/you/ycrag.git && cd ycrag

# create the conda environment from the lock file
conda env create -f env.yml
conda activate ycrag
```

If you do **not** plan to use a GPU, you can already run the whole pipeline (it will use CPU-only *llama-cpp-python*).

### (optional) Enable CUDA acceleration for **llama.cpp**

```bash
# CUDA ≥ 11.8 is installed on the host
conda install cudatoolkit -c nvidia          # only if it is not on the system

# rebuild llama-cpp-python with cuBLAS support
CMAKE_ARGS="-DLLAMA_CUBLAS=on -DLLAMA_METAL=off -DLLAMA_CLBLAST=off" \
FORCE_CMAKE=1 \
pip install --no-binary :all: llama-cpp-python==0.2.64
```

Verify:

```bash
python - <<'PY'
from llama_cpp import Llama
print("llama.cpp version:", Llama.__version__)
PY
```

---

## 2 · Build the local Wikipedia corpus (stages 1-5)

```bash
python build_wiki_dataset.py ./wiki         # ≈ 1-2 min on a laptop
```

*Output tree*

```
wiki/
 ├─ raw/       *.raw.wiki + *.meta.json
 ├─ clean/     *.clean.txt             (after parsing)
 ├─ chunks/    *.chunks.jsonl          (300-token windows)
 └─ faiss/     *.faiss + *.meta.json   (per-planet indexes)
```

---

## 3 · Run an interactive test (stages 6-10)

```bash
python main_demo.py
# → answers five sample questions from query_utils.py
```

Or ask your own:

```python
from pathlib import Path
from main_demo import ask        # tiny helper around the full pipeline

ask("How long does it take for Saturn to orbit the Sun?")
```

The assistant returns a **strict JSON** object:

```json
{
  "step_by_step_analysis": "...",
  "reasoning_summary": "...",
  "citations": [1, 3],
  "final_answer": "29.46 Earth years"
}
```

---

## 4 · Tips for speed & memory

| Component      | Default                  | Possible swap                                          | Effect                          |
| -------------- | ------------------------ | ------------------------------------------------------ | ------------------------------- |
| **Embeddings** | `BAAI/bge-large-en-v1.5` | `bge-base-en`                                          | –25 % RAM / faster index build  |
| **Reranker**   | Chat LLM (*TinyLlama*)   | Cross-encoder `BAAI/bge-reranker-base`                 | ×10 faster, no generation flags |
| **Answer LLM** | `Mistral-7B Q4_K_M`      | GPU layers (`n_gpu_layers=35`) or `phi-3-mini` (3.8 B) | 5 → 25 tok/s                    |
| **Context**    | 10 pages, 3 650 tok      | keep last 1 024 tok (`KEEP_CTX_TOKENS`)                | −70 % prompt-eval time          |

---

## 5 · Troubleshooting

* **`ImportError: …`** – check that you are inside `conda activate ycrag`.
* **`undefined reference to cuda…` while installing llama-cpp** – rebuild with `-DLLAMA_CUBLAS=off`, or ensure CUDA toolkit headers are reachable.
* **`generation flags are not valid`** – harmless; remove `temperature` from the `GenerationConfig` or switch to a cross-encoder reranker.

---

## 6 · Directory overview

```
ycrag/
 ├─ build_wiki_dataset.py       # stages 1-5
 ├─ retrieval.py                # stage 7
 ├─ rerank_utils.py             # stage 8
 ├─ answer_generation.py        # stages 9-10
 ├─ prompts.py                  # reusable prompt blocks
 ├─ query_router.py             # stage 6
 ├─ …
 └─ env.yml                     # this environment file
```

Happy hacking 🚀
