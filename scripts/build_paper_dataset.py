#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from modules.fetcher import OpenAlexFetcher

logger = logging.getLogger("build_paper_dataset")


@dataclass
class Config:
    output_dir: str = "."
    output_file: str = "papers.jsonl"

    source: str = "openalex"

    keywords: List[str] = field(
        default_factory=lambda: [
            "deep learning",
            "machine learning",
            "neural network",
            "computer vision",
            "natural language processing",
        ]
    )
    total: int = 100
    page_size: int = 25

    year_start: Optional[int] = 2018
    year_end: Optional[int] = None
    min_citations: int = 100

    # 兼容旧配置（OpenAlex 当前不使用）
    fields_of_study: List[str] = field(default_factory=list)

    # OpenAlex polite pool 建议带邮箱
    mailto: Optional[str] = None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build paper dataset from OpenAlex")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--output-file", default="papers.jsonl")
    p.add_argument("--keywords", nargs="*", default=None)

    p.add_argument("--total", type=int, default=100)
    p.add_argument("--page-size", type=int, default=25)

    p.add_argument("--year-start", type=int, default=2018)
    p.add_argument("--year-end", type=int, default=None)
    p.add_argument("--min-citations", type=int, default=100)

    p.add_argument("--source", default="openalex")
    p.add_argument("--mailto", default=None, help="email for OpenAlex polite pool")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.output_dir = args.output_dir
    cfg.output_file = args.output_file
    cfg.total = args.total
    cfg.page_size = args.page_size
    cfg.year_start = args.year_start
    cfg.year_end = args.year_end
    cfg.min_citations = args.min_citations
    cfg.source = args.source
    cfg.mailto = args.mailto

    if args.keywords:
        cfg.keywords = args.keywords
    return cfg


def fetch_papers(cfg: Config):
    # 修复过的日志格式（兼容 None）
    year_start = cfg.year_start if cfg.year_start is not None else "N/A"
    year_end = cfg.year_end if cfg.year_end is not None else "N/A"

    logger.info(
        "Fetching papers: source=%s, keywords=%d, years=%s-%s, min_citations=%d, total=%d",
        cfg.source,
        len(cfg.keywords),
        year_start,
        year_end,
        cfg.min_citations,
        cfg.total,
    )

    if cfg.source.lower() != "openalex":
        raise ValueError("This version only supports source=openalex")

    logger.info("Querying OpenAlex with keywords: %s ...", cfg.keywords[:5])

    fetcher = OpenAlexFetcher(
        timeout=30.0,
        max_attempts=5,
        base_wait=2.0,
        max_wait=60.0,
        min_interval=1.0,
        mailto=cfg.mailto,
    )

    records = fetcher.search(
        keywords=cfg.keywords,
        total=cfg.total,
        page_size=cfg.page_size,
        fields_of_study=cfg.fields_of_study,
        year_start=cfg.year_start,
        year_end=cfg.year_end,
        min_citations=cfg.min_citations,
    )

    return records


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def run(cfg: Config) -> int:
    out_dir = Path(cfg.output_dir).resolve()
    out_path = out_dir / cfg.output_file

    logger.info("Output directory: %s", out_dir)

    papers = fetch_papers(cfg)
    n = write_jsonl(out_path, papers)

    logger.info("Done. Wrote %d papers to %s", n, out_path)
    return 0


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = build_config(args)
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()