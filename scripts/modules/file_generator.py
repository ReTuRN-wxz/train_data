"""
Template-driven file generator.

For each ``PaperRecord`` this module creates the folder structure that matches
the repository convention::

    {output_dir}/
      {folder_name}/
        {folder_name}.md
        {folder_name}.parsed.json
        {folder_name}.pdf          (only when download succeeds)
        {folder_name}.raw.txt
        {folder_name}.summary.json

File contents are derived from:
  - Paper metadata (fetched via API)
  - LLM-generated summaries (via LLMClient)
  - Optional PDF download from open-access URL
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .fetcher import PaperRecord
from .llm_client import LLMClient, NoopLLMClient
from .utils import (
    PARSED_JSON_SCHEMA,
    SUMMARY_JSON_SCHEMA,
    sanitize_filename,
    validate_schema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PDF_TIMEOUT = 60  # seconds
_PDF_MAX_SIZE_MB = 50
_PDF_MAX_SIZE = _PDF_MAX_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------


class PaperFileGenerator:
    """Generate the per-paper folder and all its files.

    Args:
        output_dir: Root directory where paper folders will be created.
        llm_client: LLM client used for summary and full-text generation.
        download_pdf: Whether to attempt downloading the PDF.
        overwrite: If True, regenerate files even when they already exist.
    """

    def __init__(
        self,
        output_dir: str | Path,
        llm_client: Optional[LLMClient] = None,
        download_pdf: bool = True,
        overwrite: bool = False,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._llm = llm_client or NoopLLMClient()
        self._download_pdf = download_pdf
        self._overwrite = overwrite
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "paper-dataset-builder/1.0 (github.com/ReTuRN-wxz/train_data)"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, record: PaperRecord) -> bool:
        """Generate all files for *record*.

        Returns:
            True on success, False on failure.
        """
        folder_name = record.folder_name()
        paper_dir = self._output_dir / folder_name

        if paper_dir.exists() and not self._overwrite:
            # Check whether all expected non-PDF files already exist
            expected = [
                f"{folder_name}.md",
                f"{folder_name}.parsed.json",
                f"{folder_name}.raw.txt",
                f"{folder_name}.summary.json",
            ]
            if all((paper_dir / f).exists() for f in expected):
                logger.info("Skipping '%s' (already exists).", folder_name)
                return True

        try:
            paper_dir.mkdir(parents=True, exist_ok=True)
            paper_data = self._build_paper_data(record)

            # 1. Generate summary via LLM (or placeholder)
            summary = self._llm.generate_summary(paper_data)
            self._validate_summary(summary, folder_name)

            # 2. Generate full markdown text
            full_text = self._llm.generate_full_text(paper_data)
            if not full_text:
                full_text = self._default_full_text(paper_data)

            # 3. Write files
            self._write_summary_json(paper_dir, folder_name, summary)
            self._write_raw_txt(paper_dir, folder_name, summary)
            self._write_parsed_json(paper_dir, folder_name, full_text)
            self._write_md(paper_dir, folder_name, full_text)

            # 4. Optionally download PDF
            if self._download_pdf and record.pdf_url:
                self._download_pdf_file(paper_dir, folder_name, record.pdf_url)

            logger.info("Generated '%s'.", folder_name)
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to generate '%s': %s", folder_name, exc)
            return False

    # ------------------------------------------------------------------
    # File writers
    # ------------------------------------------------------------------

    def _write_summary_json(
        self,
        paper_dir: Path,
        folder_name: str,
        summary: Dict[str, str],
    ) -> None:
        path = paper_dir / f"{folder_name}.summary.json"
        data = {
            "concept_layer": summary.get("concept_layer", ""),
            "detail_layer": summary.get("detail_layer", ""),
            "application_layer": summary.get("application_layer", ""),
        }
        errors = validate_schema(data, SUMMARY_JSON_SCHEMA)
        if errors:
            logger.warning("summary.json schema issues for '%s': %s", folder_name, errors)
        self._write_json(path, data)

    def _write_raw_txt(
        self,
        paper_dir: Path,
        folder_name: str,
        summary: Dict[str, str],
    ) -> None:
        """Write raw.txt in the same format as the repo examples."""
        path = paper_dir / f"{folder_name}.raw.txt"
        lines = []
        for layer in ("concept_layer", "detail_layer", "application_layer"):
            lines.append(f"[{layer}]")
            lines.append(summary.get(layer, ""))
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_parsed_json(
        self,
        paper_dir: Path,
        folder_name: str,
        full_text: str,
    ) -> None:
        path = paper_dir / f"{folder_name}.parsed.json"
        data = {
            "file_name": f"{folder_name}.pdf",
            "title": folder_name,
            "full_text": full_text,
        }
        errors = validate_schema(data, PARSED_JSON_SCHEMA)
        if errors:
            logger.warning("parsed.json schema issues for '%s': %s", folder_name, errors)
        self._write_json(path, data)

    def _write_md(
        self,
        paper_dir: Path,
        folder_name: str,
        full_text: str,
    ) -> None:
        path = paper_dir / f"{folder_name}.md"
        path.write_text(full_text, encoding="utf-8")

    # ------------------------------------------------------------------
    # PDF downloader
    # ------------------------------------------------------------------

    def _download_pdf_file(
        self, paper_dir: Path, folder_name: str, pdf_url: str
    ) -> None:
        pdf_path = paper_dir / f"{folder_name}.pdf"
        if pdf_path.exists() and not self._overwrite:
            return

        try:
            logger.debug("Downloading PDF from %s …", pdf_url)
            resp = self._session.get(pdf_url, timeout=_PDF_TIMEOUT, stream=True)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "pdf" not in content_type and "octet-stream" not in content_type:
                logger.warning(
                    "Unexpected Content-Type '%s' for PDF URL: %s",
                    content_type,
                    pdf_url,
                )

            size = 0
            with pdf_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    size += len(chunk)
                    if size > _PDF_MAX_SIZE:
                        logger.warning("PDF too large (>%d MB), skipping.", _PDF_MAX_SIZE_MB)
                        f.close()
                        pdf_path.unlink(missing_ok=True)
                        return
                    f.write(chunk)

            logger.debug("Saved PDF to %s (%.1f KB).", pdf_path, size / 1024)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not download PDF for '%s': %s", folder_name, exc)
            pdf_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_paper_data(record: PaperRecord) -> Dict[str, Any]:
        return {
            "arxiv_id": record.arxiv_id,
            "doi": record.doi,
            "title": record.title,
            "abstract": record.abstract,
            "authors": record.authors,
            "year": record.year,
            "venue": record.venue,
            "citation_count": record.citation_count,
            "pdf_url": record.pdf_url,
        }

    @staticmethod
    def _default_full_text(paper_data: Dict[str, Any]) -> str:
        title = paper_data.get("title", "Unknown Title")
        authors = paper_data.get("authors", [])
        abstract = paper_data.get("abstract", "")
        year = paper_data.get("year", "")
        citation_count = paper_data.get("citation_count", 0)
        venue = paper_data.get("venue", "")
        doi = paper_data.get("doi", "")

        author_str = ", ".join(authors[:8]) if authors else "Unknown"
        if len(authors) > 8:
            author_str += " et al."

        lines = [f"## {title}", ""]
        if author_str:
            lines += [f"**Authors:** {author_str}", ""]
        meta_parts = []
        if year:
            meta_parts.append(f"Year: {year}")
        if venue:
            meta_parts.append(f"Venue: {venue}")
        if citation_count:
            meta_parts.append(f"Citations: {citation_count}")
        if doi:
            meta_parts.append(f"DOI: {doi}")
        if meta_parts:
            lines += ["**" + " | ".join(meta_parts) + "**", ""]
        if abstract:
            lines += ["## Abstract", "", abstract, ""]

        return "\n".join(lines)

    @staticmethod
    def _validate_summary(summary: Dict[str, str], folder_name: str) -> None:
        errors = validate_schema(summary, SUMMARY_JSON_SCHEMA)
        if errors:
            logger.warning(
                "Summary validation failed for '%s': %s", folder_name, errors
            )

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
