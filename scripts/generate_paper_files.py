#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_paper_files.py
=======================
Read ``papers.jsonl`` (produced by ``build_paper_dataset.py``) and, for every
paper, generate the per-paper folder with the following files:

    {output_dir}/
      {folder_name}/
        {folder_name}.md
        {folder_name}.parsed.json
        {folder_name}.raw.txt
        {folder_name}.summary.json
        {folder_name}.pdf          (only when pdf_url is present)

LLM generation uses the Kimi / Moonshot API.  Configure via env vars:

    KIMI_API_KEY   – required for AI-generated content
    KIMI_BASE_URL  – optional (default: https://api.moonshot.cn/v1)
    KIMI_MODEL     – optional (default: kimi2.5thinking)

Without KIMI_API_KEY the script runs in *metadata-only* mode: files are still
created but summaries and full-text are placeholder stubs.

Usage
-----
    python scripts/generate_paper_files.py                          # defaults
    python scripts/generate_paper_files.py --input papers.jsonl \\
        --output-dir . --max-workers 4 --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from modules.fetcher import PaperRecord
from modules.file_generator import PaperFileGenerator
from modules.llm_client import build_llm_client

logger = logging.getLogger("generate_paper_files")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-paper files from papers.jsonl using Kimi LLM."
    )
    p.add_argument(
        "--input",
        default="papers.jsonl",
        help="Path to the input JSONL file (default: papers.jsonl)",
    )
    p.add_argument(
        "--output-dir",
        default=".",
        help="Root directory where per-paper folders will be created (default: .)",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel worker threads (default: 4)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate files even if they already exist",
    )
    p.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF download even when pdf_url is present",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list papers that would be processed; do not create any files",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_papers(path: Path) -> List[PaperRecord]:
    records: List[PaperRecord] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(PaperRecord.from_dict(d))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line %d (JSON parse error): %s", lineno, exc)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    papers = load_papers(input_path)
    logger.info("Loaded %d papers from %s", len(papers), input_path)

    if args.dry_run:
        for p in papers:
            logger.info("[dry-run] Would process: %s  →  %s", p.title, p.folder_name())
        return

    # Build LLM client (auto-detects KIMI_API_KEY; falls back to noop)
    llm = build_llm_client(provider="kimi")
    if llm.is_available():
        logger.info("Kimi LLM client ready.")
    else:
        logger.warning(
            "Running in metadata-only mode (KIMI_API_KEY not set). "
            "Set KIMI_API_KEY to enable AI-generated summaries."
        )

    generator = PaperFileGenerator(
        output_dir=output_dir,
        llm_client=llm,
        download_pdf=not args.no_pdf,
        overwrite=args.overwrite,
    )

    success = 0
    failure = 0

    if args.max_workers <= 1:
        for paper in papers:
            ok = generator.generate(paper)
            if ok:
                success += 1
            else:
                failure += 1
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(generator.generate, p): p for p in papers}
            for future in as_completed(futures):
                paper = futures[future]
                try:
                    ok = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Unexpected error for '%s': %s", paper.folder_name(), exc
                    )
                    ok = False
                if ok:
                    success += 1
                else:
                    failure += 1

    logger.info(
        "Done. %d succeeded, %d failed (total %d).", success, failure, len(papers)
    )
    if failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
