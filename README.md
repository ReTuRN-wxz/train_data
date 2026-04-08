# train_data – Automated Paper Dataset Builder

This repository stores high-quality STEM research papers in a structured format
suitable for training or fine-tuning language models.

The pipeline is split into two steps:

1. **`scripts/build_paper_dataset.py`** — query OpenAlex and save metadata to `papers.jsonl`
2. **`scripts/generate_paper_files.py`** — read `papers.jsonl` and generate per-paper folders with AI-enriched content

---

## Directory layout

```
train_data/
├── <arxiv_id_or_paper_id>/        ← one folder per paper (created by step 2)
│   ├── <id>.md                    ← AI-generated Markdown introduction
│   ├── <id>.parsed.json           ← {file_name, title, full_text}
│   ├── <id>.pdf                   ← PDF (downloaded when pdf_url is available)
│   ├── <id>.raw.txt               ← plain-text summary with layer tags
│   └── <id>.summary.json          ← {concept_layer, detail_layer, application_layer}
├── papers.jsonl                   ← paper metadata produced by step 1
├── scripts/
│   ├── build_paper_dataset.py     ← step 1: fetch & save paper metadata
│   ├── generate_paper_files.py    ← step 2: generate per-paper folders
│   ├── config.yaml                ← reference configuration (values / comments)
│   └── modules/
│       ├── fetcher.py             ← OpenAlex API client
│       ├── file_generator.py      ← file writer for each paper
│       ├── llm_client.py          ← Kimi / Moonshot LLM abstraction
│       └── utils.py               ← helpers (filename sanitization, etc.)
└── requirements.txt
```

---

## Installation

```bash
# Clone the repo (if you haven't already)
git clone https://github.com/ReTuRN-wxz/train_data.git
cd train_data

# Install Python dependencies (Python 3.9+ recommended)
pip install -r requirements.txt
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `KIMI_API_KEY` | **yes** (for LLM) | – | Moonshot/Kimi API key |
| `KIMI_BASE_URL` | no | `https://api.moonshot.cn/v1` | API base URL |
| `KIMI_MODEL` | no | `kimi2.5thinking` | Model name |

> **Without `KIMI_API_KEY`** step 2 runs in **metadata-only mode**: per-paper
> folders are still created with real citation/abstract data, but summaries and
> full-text contain placeholder stubs instead of AI-generated content.

```bash
export KIMI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
export KIMI_BASE_URL="https://api.moonshot.cn/v1"   # optional
export KIMI_MODEL="kimi2.5thinking"                  # optional
```

---

## Usage

### Step 1 — Fetch paper metadata

`build_paper_dataset.py` queries OpenAlex and writes results to a JSONL file.

```bash
# Basic usage – writes to papers.jsonl in the current directory
python scripts/build_paper_dataset.py

# Custom keywords, date range and output path
python scripts/build_paper_dataset.py \
    --keywords "transformer" "large language model" "BERT" \
    --year-start 2020 \
    --total 100 \
    --min-citations 200 \
    --output-dir . \
    --output-file papers.jsonl
```

#### `build_paper_dataset.py` CLI flags

| Flag | Default | Description |
|---|---|---|
| `--keywords` | STEM list | One or more search terms (space-separated) |
| `--total` | `100` | Number of papers to fetch |
| `--page-size` | `25` | Results per API page |
| `--year-start` | `2018` | Earliest publication year |
| `--year-end` | _(none)_ | Latest publication year |
| `--min-citations` | `100` | Minimum citation count |
| `--source` | `openalex` | Data source (currently only `openalex`) |
| `--mailto` | _(none)_ | Email for OpenAlex "polite pool" (recommended) |
| `--output-dir` | `.` | Directory where the JSONL file is saved |
| `--output-file` | `papers.jsonl` | JSONL output filename |

---

### Step 2 — Generate per-paper folders

`generate_paper_files.py` reads the JSONL file produced in step 1 and creates
a folder for each paper containing Markdown, JSON, and plain-text files.

```bash
# Basic usage (reads papers.jsonl from current directory)
python scripts/generate_paper_files.py

# Full run with LLM summaries and parallel workers
export KIMI_API_KEY="sk-..."
python scripts/generate_paper_files.py \
    --input papers.jsonl \
    --output-dir . \
    --max-workers 4

# Skip PDF downloads (faster)
python scripts/generate_paper_files.py --no-pdf

# Regenerate folders that already exist
python scripts/generate_paper_files.py --overwrite

# Preview which papers would be processed without writing any files
python scripts/generate_paper_files.py --dry-run
```

#### `generate_paper_files.py` CLI flags

| Flag | Default | Description |
|---|---|---|
| `--input` | `papers.jsonl` | Path to the input JSONL file |
| `--output-dir` | `.` | Root directory for per-paper folders |
| `--max-workers` | `1` | Parallel worker threads |
| `--overwrite` | `false` | Re-generate files even if folder exists |
| `--no-pdf` | `false` | Skip PDF download |
| `--dry-run` | `false` | List papers without writing files |

---

## End-to-end quickstart

```bash
# 1. Fetch 20 highly-cited ML papers (no API key needed)
python scripts/build_paper_dataset.py \
    --keywords "deep learning" "transformer" \
    --total 20 \
    --min-citations 500

# 2. Generate per-paper files (metadata-only, no LLM key required)
python scripts/generate_paper_files.py --input papers.jsonl
```

```bash
# Full run with AI-generated summaries
export KIMI_API_KEY="sk-..."

python scripts/build_paper_dataset.py --total 100 --min-citations 200
python scripts/generate_paper_files.py --max-workers 4
```

---

## Output structure example

After running both steps you will see:

```
./
├── papers.jsonl               ← metadata index (step 1 output)
├── 2310.06825v2/
│   ├── 2310.06825v2.md
│   ├── 2310.06825v2.parsed.json
│   ├── 2310.06825v2.pdf
│   ├── 2310.06825v2.raw.txt
│   └── 2310.06825v2.summary.json
└── 2305.10601v1/
    └── ...
```

### File formats

**`<id>.summary.json`**
```json
{
  "concept_layer": "One-two sentence high-level contribution summary.",
  "detail_layer": "Technical details of the method (~100-150 words).",
  "application_layer": "Real-world applications and future directions."
}
```

**`<id>.parsed.json`**
```json
{
  "file_name": "<id>.pdf",
  "title": "<id>",
  "full_text": "## Paper Title\n\n**Authors:** ...\n\n## Abstract\n\n..."
}
```

**`<id>.raw.txt`**
```
[concept_layer]
One-two sentence high-level contribution summary.

[detail_layer]
Technical details of the method (~100-150 words).

[application_layer]
Real-world applications and future directions.
```

---

## Extending / customising

* **Add a new LLM backend** – subclass `LLMClient` in `scripts/modules/llm_client.py`
  and register it in `build_llm_client()`.
* **Add a new data source** – implement a class with a `.search()` method
  returning `List[PaperRecord]` and wire it in `build_paper_dataset.py`.
* **Change prompt templates** – edit the prompt constants at the top of
  `scripts/modules/llm_client.py`.
