# train_data – Automated Paper Dataset Builder

This repository stores high-quality STEM research papers in a structured format
suitable for training or fine-tuning language models.

---

## Directory layout

```
train_data/
├── <arxiv_id_or_paper_id>/        ← one folder per paper
│   ├── <id>.md                    ← full-text / AI-generated Markdown
│   ├── <id>.parsed.json           ← {file_name, title, full_text}
│   ├── <id>.pdf                   ← PDF (downloaded when available)
│   ├── <id>.raw.txt               ← plain-text summary with layer tags
│   └── <id>.summary.json          ← {concept_layer, detail_layer, application_layer}
├── scripts/
│   ├── build_paper_dataset.py     ← main automation script
│   ├── config.yaml                ← default configuration
│   └── modules/
│       ├── fetcher.py             ← Semantic Scholar / OpenAlex API
│       ├── file_generator.py      ← template-driven file writer
│       ├── llm_client.py          ← Kimi / Moonshot LLM abstraction
│       └── utils.py               ← helpers (sanitization, retries, schema)
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

## Configuration

Configuration is driven by **`scripts/config.yaml`**.  You can:

1. Edit `scripts/config.yaml` directly.
2. Supply a custom YAML file via `--config my_config.yaml`.
3. Override individual values with CLI flags (highest priority).

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `source` | `semantic_scholar` | API backend: `semantic_scholar` or `openalex` |
| `start_year` | `2018` | Earliest publication year |
| `end_year` | current year | Latest publication year |
| `keywords` | STEM list | List of search terms |
| `fields_of_study` | major sciences | Semantic Scholar field filter |
| `min_citations` | `100` | Minimum citation count |
| `total_papers` | `200` | Number of papers to collect |
| `output_dir` | `.` (repo root) | Where to write paper folders |
| `overwrite` | `false` | Regenerate existing folders |
| `download_pdf` | `true` | Download open-access PDFs |
| `max_workers` | `4` | Parallel threads for generation |
| `llm_provider` | `kimi` | LLM backend for AI summaries |

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `KIMI_API_KEY` | **yes** (for LLM) | – | Moonshot/Kimi API key |
| `KIMI_BASE_URL` | no | `https://api.moonshot.cn/v1` | API base URL |
| `KIMI_MODEL` | no | `kimi2.5thinking` | Model name |
| `SEMANTIC_SCHOLAR_API_KEY` | no | – | Higher rate-limit S2 key |
| `OPENALEX_EMAIL` | no | – | Enables OpenAlex "polite pool" |

> **Without `KIMI_API_KEY`** the script runs in **metadata-only mode**: paper
> folders are created with real citation/abstract data, but the summary fields
> contain placeholder text instead of AI-generated content.

Set variables before running:

```bash
export KIMI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
export KIMI_BASE_URL="https://api.moonshot.cn/v1"   # optional
export KIMI_MODEL="kimi2.5thinking"                  # optional
```

---

## Running examples

### Quickstart (metadata-only, no LLM key needed)

```bash
python scripts/build_paper_dataset.py \
    --total 20 \
    --min-citations 500 \
    --output-dir ./data
```

### Full run with LLM summaries

```bash
export KIMI_API_KEY="sk-..."

python scripts/build_paper_dataset.py \
    --keywords "transformer,attention,large language model,BERT,GPT" \
    --start-year 2020 \
    --total 100 \
    --min-citations 200 \
    --output-dir ./data \
    --max-workers 4
```

### Use OpenAlex backend

```bash
export OPENALEX_EMAIL="yourname@example.com"

python scripts/build_paper_dataset.py \
    --source openalex \
    --keywords "protein folding,molecular dynamics" \
    --total 50
```

### Overwrite existing folders

```bash
python scripts/build_paper_dataset.py --overwrite
```

### Skip PDF downloads (faster)

```bash
python scripts/build_paper_dataset.py --no-pdf
```

### Custom config file

```bash
python scripts/build_paper_dataset.py --config my_run.yaml
```

---

## Output structure example

After a successful run you will see:

```
./
├── 2310.06825v2/
│   ├── 2310.06825v2.md
│   ├── 2310.06825v2.parsed.json
│   ├── 2310.06825v2.pdf
│   ├── 2310.06825v2.raw.txt
│   └── 2310.06825v2.summary.json
├── 2305.10601v1/
│   ├── ...
└── failed_papers.jsonl     ← any papers that could not be generated
```

### File formats

**`<id>.summary.json`**
```json
{
  "concept_layer": "One-two sentence high-level contribution summary.",
  "detail_layer": "Technical details of the method (~100-150 chars).",
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
Technical details of the method (~100-150 chars).

[application_layer]
Real-world applications and future directions.
```

---

## Extending / customising

* **Add a new LLM backend** – subclass `LLMClient` in `scripts/modules/llm_client.py`
  and register it in `build_llm_client()`.
* **Add a new data source** – implement a class with a `.search()` method
  returning `List[PaperRecord]` and wire it in `build_paper_dataset.py:fetch_papers()`.
* **Change templates** – edit the prompt constants in `llm_client.py` or
  the `_default_full_text()` / `_write_*` methods in `file_generator.py`.

---

## Failed-paper log

Papers that could not be generated are written to `failed_papers.jsonl`
(one JSON object per line) in the output directory.  Re-run with `--overwrite`
to retry them.
