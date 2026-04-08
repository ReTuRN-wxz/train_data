#!/usr/bin/env python3
"""
build_paper_dataset.py – Automated high-citation STEM paper dataset builder.

Usage examples
--------------
# Basic run with defaults (uses config.yaml):
    python scripts/build_paper_dataset.py

# Custom keywords and output:
    python scripts/build_paper_dataset.py \\
        --keywords "transformer,attention,BERT" \\
        --total 50 --min-citations 200 \\
        --output-dir ./data

# Use OpenAlex instead of Semantic Scholar:
    python scripts/build_paper_dataset.py --source openalex

# Enable LLM-generated summaries (set env var before running):
    export KIMI_API_KEY=sk-...
    python scripts/build_paper_dataset.py --total 10

# Overwrite existing folders:
    python scripts/build_paper_dataset.py --overwrite

Environment variables
---------------------
    KIMI_API_KEY    API key for Kimi/Moonshot  (required for LLM features)
    KIMI_BASE_URL   API base URL               (default: https://api.moonshot.cn/v1)
    KIMI_MODEL      Model name                 (default: kimi2.5thinking)
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Make sure the scripts directory is on sys.path so that `modules` is found
# when the script is run from any working directory.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from modules.fetcher import (  # noqa: E402
    OpenAlexFetcher,
    PaperRecord,
    SemanticScholarFetcher,
    deduplicate,
)
from modules.file_generator import PaperFileGenerator  # noqa: E402
from modules.llm_client import build_llm_client  # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_paper_dataset")

# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = _SCRIPTS_DIR / "config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning(
            "PyYAML is not installed – skipping config file loading. "
            "Install it with: pip install pyyaml"
        )
        return {}

    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the YAML config file, returning an empty dict on failure."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        logger.debug("Config file not found at %s; using defaults.", path)
        return {}
    try:
        return _load_yaml(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load config from %s: %s", path, exc)
        return {}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a high-citation STEM paper dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        metavar="PATH",
        help="Path to a YAML config file (default: scripts/config.yaml).",
    )
    p.add_argument(
        "--source",
        choices=["semantic_scholar", "openalex"],
        help="Paper metadata source (default: semantic_scholar).",
    )
    p.add_argument(
        "--keywords",
        metavar="TERMS",
        help="Comma-separated keywords (overrides config).",
    )
    p.add_argument(
        "--start-year",
        type=int,
        metavar="YEAR",
        help="Earliest publication year (default: 2018).",
    )
    p.add_argument(
        "--end-year",
        type=int,
        metavar="YEAR",
        help="Latest publication year (default: current year).",
    )
    p.add_argument(
        "--min-citations",
        type=int,
        metavar="N",
        help="Minimum citation count (default: 100).",
    )
    p.add_argument(
        "--total",
        type=int,
        metavar="N",
        help="Total number of papers to collect (default: 200).",
    )
    p.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Root output directory (default: current directory).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Regenerate files even when the folder already exists.",
    )
    p.add_argument(
        "--no-pdf",
        action="store_true",
        default=False,
        help="Skip PDF downloads.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        metavar="N",
        help="Number of parallel worker threads (default: 4).",
    )
    p.add_argument(
        "--llm-provider",
        metavar="PROVIDER",
        help="LLM provider to use for summaries (default: kimi).",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    p.add_argument(
        "--failed-log",
        metavar="PATH",
        help="File to write failed paper IDs/titles to (default: failed_papers.jsonl).",
    )
    return p


def merge_config(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI args into config (CLI values take precedence)."""

    def _set(key: str, value: Any) -> None:
        if value is not None:
            cfg[key] = value

    _set("source", args.source)
    _set("start_year", args.start_year)
    _set("end_year", args.end_year)
    _set("min_citations", args.min_citations)
    _set("total_papers", args.total)
    _set("output_dir", args.output_dir)
    _set("max_workers", args.max_workers)
    _set("llm_provider", args.llm_provider)

    if args.keywords:
        cfg["keywords"] = [k.strip() for k in args.keywords.split(",") if k.strip()]
    if args.overwrite:
        cfg["overwrite"] = True
    if args.no_pdf:
        cfg["download_pdf"] = False

    # Apply defaults
    cfg.setdefault("source", "semantic_scholar")
    cfg.setdefault("start_year", 2018)
    cfg.setdefault("end_year", datetime.date.today().year)
    cfg.setdefault("min_citations", 100)
    cfg.setdefault("total_papers", 200)
    cfg.setdefault("output_dir", ".")
    cfg.setdefault("overwrite", False)
    cfg.setdefault("download_pdf", True)
    cfg.setdefault("max_workers", 4)
    cfg.setdefault("llm_provider", "kimi")
    cfg.setdefault("api_requests_per_second", 1.0)
    cfg.setdefault("retry_attempts", 3)
    cfg.setdefault("api_timeout", 30)
    cfg.setdefault("llm_timeout", 120)
    cfg.setdefault("fields_of_study", [])
    cfg.setdefault("keywords", [
        "deep learning",
        "machine learning",
        "neural network",
        "computer vision",
        "natural language processing",
        "reinforcement learning",
        "quantum computing",
        "bioinformatics",
        "genomics",
        "climate science",
        "materials science",
        "particle physics",
    ])
    cfg.setdefault("failed_log", str(Path(cfg["output_dir"]) / "failed_papers.jsonl"))

    if args.failed_log:
        cfg["failed_log"] = args.failed_log

    return cfg


# ---------------------------------------------------------------------------
# Fetching helpers
# ---------------------------------------------------------------------------

def fetch_papers(cfg: Dict[str, Any]) -> List[PaperRecord]:
    """Fetch and deduplicate papers according to *cfg*."""
    source = cfg["source"]
    keywords: List[str] = cfg["keywords"]
    start_year: int = cfg["start_year"]
    end_year: int = cfg["end_year"]
    min_citations: int = cfg["min_citations"]
    total: int = cfg["total_papers"]
    rps: float = cfg["api_requests_per_second"]
    timeout: int = cfg["api_timeout"]
    fields_of_study: List[str] = cfg.get("fields_of_study", [])

    logger.info(
        "Fetching papers: source=%s, keywords=%d, years=%d-%d, "
        "min_citations=%d, total=%d",
        source, len(keywords), start_year, end_year, min_citations, total,
    )

    all_records: List[PaperRecord] = []

    if source == "semantic_scholar":
        ss_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        fetcher = SemanticScholarFetcher(
            api_key=ss_api_key,
            requests_per_second=rps,
            timeout=timeout,
        )
        # Batch keywords into groups to get broader coverage
        batch_size = 5
        num_batches = max(1, (len(keywords) + batch_size - 1) // batch_size)
        per_batch = max(50, total // num_batches)

        for i in range(0, len(keywords), batch_size):
            batch = keywords[i: i + batch_size]
            logger.info("Querying S2 with keywords: %s …", batch)
            try:
                records = fetcher.search(
                    keywords=batch,
                    start_year=start_year,
                    end_year=end_year,
                    min_citations=min_citations,
                    limit=per_batch,
                    fields_of_study=fields_of_study or None,
                )
                all_records.extend(records)
                logger.info("  → %d records so far (batch returned %d).", len(all_records), len(records))
            except Exception as exc:  # noqa: BLE001
                logger.warning("S2 batch failed (%s): %s", batch, exc)

    elif source == "openalex":
        oa_email = os.environ.get("OPENALEX_EMAIL", "")
        fetcher_oa = OpenAlexFetcher(
            email=oa_email,
            requests_per_second=rps,
            timeout=timeout,
        )
        for keyword in keywords:
            logger.info("Querying OpenAlex for: %s …", keyword)
            try:
                records = fetcher_oa.search(
                    keywords=[keyword],
                    start_year=start_year,
                    end_year=end_year,
                    min_citations=min_citations,
                    limit=max(50, total // len(keywords)),
                )
                all_records.extend(records)
                logger.info("  → %d records so far (keyword returned %d).", len(all_records), len(records))
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenAlex keyword '%s' failed: %s", keyword, exc)

    else:
        raise ValueError(f"Unknown source: {source!r}")

    # Deduplicate and sort
    unique = deduplicate(all_records)
    unique.sort(key=lambda r: r.citation_count, reverse=True)

    logger.info(
        "After deduplication: %d unique records (from %d total fetched).",
        len(unique), len(all_records),
    )

    return unique[:total]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cfg: Dict[str, Any]) -> int:
    """Run the full pipeline. Returns exit code (0 = success)."""

    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # 1. Fetch papers
    papers = fetch_papers(cfg)
    if not papers:
        logger.error("No papers found. Check your keywords and API connectivity.")
        return 1

    logger.info("Starting file generation for %d papers …", len(papers))

    # 2. Build LLM client
    llm = build_llm_client(provider=cfg["llm_provider"])

    # 3. Build generator
    generator = PaperFileGenerator(
        output_dir=output_dir,
        llm_client=llm,
        download_pdf=cfg["download_pdf"],
        overwrite=cfg["overwrite"],
    )

    # 4. Generate files (parallel)
    max_workers: int = cfg["max_workers"]
    failed: List[Dict[str, Any]] = []
    succeeded = 0

    failed_log_path = Path(cfg["failed_log"])
    failed_log_path.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {
            executor.submit(generator.generate, record): record
            for record in papers
        }
        total = len(future_to_record)
        done = 0
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            done += 1
            try:
                ok = future.result()
            except Exception as exc:  # noqa: BLE001
                ok = False
                logger.error(
                    "Unhandled exception for '%s': %s", record.folder_name(), exc
                )

            if ok:
                succeeded += 1
            else:
                failed.append(
                    {
                        "folder_name": record.folder_name(),
                        "title": record.title,
                        "arxiv_id": record.arxiv_id,
                        "doi": record.doi,
                        "citation_count": record.citation_count,
                    }
                )
            logger.info("Progress: %d/%d (✓ %d, ✗ %d)", done, total, succeeded, len(failed))

    # 5. Write failed log
    if failed:
        with failed_log_path.open("w", encoding="utf-8") as fh:
            for entry in failed:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.warning(
            "%d papers failed. See %s for details.", len(failed), failed_log_path
        )

    logger.info(
        "Done. Succeeded: %d / %d. Output: %s",
        succeeded,
        len(papers),
        output_dir,
    )
    return 0 if not failed else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    cfg = load_config(args.config)
    cfg = merge_config(cfg, args)

    logger.debug("Effective configuration:\n%s", json.dumps(cfg, indent=2, default=str))

    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
