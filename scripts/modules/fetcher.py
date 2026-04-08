"""
Paper metadata fetcher.

Supports two backends:
  - Semantic Scholar Public API  (primary)
  - OpenAlex REST API            (fallback / alternative)

Both return a unified ``PaperRecord`` typed-dict so the rest of the pipeline
does not need to know which backend was used.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import requests

from .utils import retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified data model
# ---------------------------------------------------------------------------


@dataclass
class PaperRecord:
    """Minimal paper metadata shared across all backends."""

    # Identifiers
    arxiv_id: str = ""          # e.g. "2103.00020v1"  (may be empty)
    doi: str = ""
    semantic_scholar_id: str = ""
    openalex_id: str = ""

    # Bibliographic info
    title: str = ""
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    year: int = 0
    venue: str = ""
    citation_count: int = 0

    # Full-text / URLs
    pdf_url: str = ""
    full_text: str = ""         # markdown / plain text when available

    # Raw metadata blob (for debugging / extra fields)
    raw: Dict[str, Any] = field(default_factory=dict)

    def folder_name(self) -> str:
        """Return the folder name to use on disk.

        Priority: arxiv_id > doi-slug > semantic_scholar_id > title-slug.
        """
        from .utils import sanitize_filename

        if self.arxiv_id:
            return sanitize_filename(self.arxiv_id)
        if self.doi:
            return sanitize_filename(self.doi.replace("/", "_"))
        if self.semantic_scholar_id:
            return sanitize_filename(self.semantic_scholar_id)
        return sanitize_filename(self.title or "unknown_paper")

    def unique_key(self) -> str:
        """Return a deduplication key (lower-cased title normalised)."""
        import re

        t = re.sub(r"\s+", " ", self.title.lower().strip())
        return t


# ---------------------------------------------------------------------------
# Semantic Scholar backend
# ---------------------------------------------------------------------------

_SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_SS_FIELDS = (
    "paperId,externalIds,title,abstract,authors,year,venue,"
    "citationCount,openAccessPdf,publicationDate"
)

_DEFAULT_HEADERS = {
    "User-Agent": "paper-dataset-builder/1.0 (github.com/ReTuRN-wxz/train_data)"
}


class SemanticScholarFetcher:
    """Fetch papers from the Semantic Scholar public API."""

    def __init__(
        self,
        api_key: str = "",
        requests_per_second: float = 1.0,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key
        self._min_interval = 1.0 / max(requests_per_second, 0.1)
        self._timeout = timeout
        self._last_request_time: float = 0.0
        self._session = requests.Session()
        self._session.headers.update(_DEFAULT_HEADERS)
        if api_key:
            self._session.headers["x-api-key"] = api_key

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        keywords: List[str],
        *,
        start_year: int = 2018,
        end_year: Optional[int] = None,
        min_citations: int = 50,
        limit: int = 100,
        fields_of_study: Optional[List[str]] = None,
    ) -> List[PaperRecord]:
        """Search for papers matching *keywords* and filter criteria.

        Args:
            keywords: List of search terms (joined with OR).
            start_year: Earliest publication year (inclusive).
            end_year: Latest publication year (inclusive, defaults to current year).
            min_citations: Minimum citation count.
            limit: Maximum number of results to return.
            fields_of_study: Optional list of S2 field-of-study filters
                             (e.g. ['Computer Science', 'Physics']).

        Returns:
            List of ``PaperRecord`` objects sorted by citation_count descending.
        """
        end_year = end_year or date.today().year
        query = " | ".join(keywords)

        results: List[PaperRecord] = []
        offset = 0
        batch_size = min(100, limit * 3)  # over-fetch to account for filtering

        while len(results) < limit:
            batch = self._search_page(
                query=query,
                offset=offset,
                limit=batch_size,
                fields_of_study=fields_of_study,
            )
            if not batch:
                break

            for record in batch:
                if record.year < start_year or record.year > end_year:
                    continue
                if record.citation_count < min_citations:
                    continue
                results.append(record)

            if len(batch) < batch_size:
                break  # no more pages

            offset += batch_size

            if len(results) >= limit:
                break

        # Sort by citation count (highest first) and truncate
        results.sort(key=lambda r: r.citation_count, reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    @retry(max_attempts=3, delay=5.0, backoff=2.0, exceptions=(requests.RequestException,))
    def _search_page(
        self,
        query: str,
        offset: int,
        limit: int,
        fields_of_study: Optional[List[str]],
    ) -> List[PaperRecord]:
        self._throttle()

        params: Dict[str, Any] = {
            "query": query,
            "offset": offset,
            "limit": limit,
            "fields": _SS_FIELDS,
        }
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        resp = self._session.get(_SS_SEARCH_URL, params=params, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        records: List[PaperRecord] = []
        for item in data.get("data", []):
            record = self._parse_ss_item(item)
            if record:
                records.append(record)
        return records

    @staticmethod
    def _parse_ss_item(item: Dict[str, Any]) -> Optional[PaperRecord]:
        title = (item.get("title") or "").strip()
        if not title:
            return None

        external_ids = item.get("externalIds") or {}
        arxiv_raw = external_ids.get("ArXiv", "")
        arxiv_id = f"{arxiv_raw}v1" if arxiv_raw and not arxiv_raw[0].isalpha() else arxiv_raw

        pdf_info = item.get("openAccessPdf") or {}
        pdf_url = pdf_info.get("url", "")
        # Prefer arXiv PDF when available
        if arxiv_raw and not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_raw}"

        pub_date = item.get("publicationDate") or ""
        year = item.get("year") or 0
        if pub_date and not year:
            try:
                year = int(pub_date[:4])
            except (ValueError, TypeError):
                pass

        authors = [
            a.get("name", "") for a in (item.get("authors") or []) if a.get("name")
        ]

        return PaperRecord(
            arxiv_id=arxiv_id,
            doi=external_ids.get("DOI", ""),
            semantic_scholar_id=item.get("paperId", ""),
            title=title,
            abstract=(item.get("abstract") or "").strip(),
            authors=authors,
            year=year,
            venue=(item.get("venue") or "").strip(),
            citation_count=item.get("citationCount") or 0,
            pdf_url=pdf_url,
            raw=item,
        )


# ---------------------------------------------------------------------------
# OpenAlex backend
# ---------------------------------------------------------------------------

_OA_SEARCH_URL = "https://api.openalex.org/works"


class OpenAlexFetcher:
    """Fetch papers from the OpenAlex REST API (no auth required)."""

    def __init__(
        self,
        email: str = "",
        requests_per_second: float = 5.0,
        timeout: int = 30,
    ) -> None:
        self._email = email
        self._min_interval = 1.0 / max(requests_per_second, 0.1)
        self._timeout = timeout
        self._last_request_time: float = 0.0
        self._session = requests.Session()
        ua = "paper-dataset-builder/1.0"
        if email:
            ua += f" (mailto:{email})"
        self._session.headers["User-Agent"] = ua

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        keywords: List[str],
        *,
        start_year: int = 2018,
        end_year: Optional[int] = None,
        min_citations: int = 50,
        limit: int = 100,
    ) -> List[PaperRecord]:
        end_year = end_year or date.today().year
        query = " ".join(keywords)

        results: List[PaperRecord] = []
        cursor = "*"
        per_page = min(200, limit * 4)

        while len(results) < limit:
            batch, next_cursor = self._search_page(
                query=query,
                start_year=start_year,
                end_year=end_year,
                cursor=cursor,
                per_page=per_page,
            )
            if not batch:
                break

            for record in batch:
                if record.citation_count >= min_citations:
                    results.append(record)

            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

            if len(results) >= limit:
                break

        results.sort(key=lambda r: r.citation_count, reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    @retry(max_attempts=3, delay=5.0, backoff=2.0, exceptions=(requests.RequestException,))
    def _search_page(
        self,
        query: str,
        start_year: int,
        end_year: int,
        cursor: str,
        per_page: int,
    ) -> tuple[List[PaperRecord], str]:
        self._throttle()

        params: Dict[str, Any] = {
            "search": query,
            "filter": (
                f"publication_year:{start_year}-{end_year},"
                "type:article,"
                "is_oa:true"
            ),
            "sort": "cited_by_count:desc",
            "per_page": per_page,
            "cursor": cursor,
            "select": (
                "id,doi,title,abstract_inverted_index,authorships,"
                "publication_year,primary_location,cited_by_count,"
                "best_oa_location,ids"
            ),
        }

        resp = self._session.get(_OA_SEARCH_URL, params=params, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        records: List[PaperRecord] = []
        for item in data.get("results", []):
            record = self._parse_oa_item(item)
            if record:
                records.append(record)

        meta = data.get("meta", {})
        next_cursor = meta.get("next_cursor", "")
        return records, next_cursor

    @staticmethod
    def _parse_oa_item(item: Dict[str, Any]) -> Optional[PaperRecord]:
        title = (item.get("title") or "").strip()
        if not title:
            return None

        # Reconstruct abstract from inverted index
        abstract = ""
        inv = item.get("abstract_inverted_index")
        if inv:
            max_pos = max((p for positions in inv.values() for p in positions), default=-1)
            if max_pos >= 0:
                words = [""] * (max_pos + 1)
                for word, positions in inv.items():
                    for pos in positions:
                        if 0 <= pos <= max_pos:
                            words[pos] = word
                abstract = " ".join(words).strip()

        ids = item.get("ids", {})
        arxiv_raw = ids.get("arxiv", "")
        # OpenAlex returns full URLs like "https://arxiv.org/abs/2103.00020"
        arxiv_id = ""
        if arxiv_raw:
            arxiv_id = arxiv_raw.split("/")[-1]
            # add v1 suffix if no version present
            if not re.search(r"v\d+$", arxiv_id):
                arxiv_id = arxiv_id + "v1"

        doi_raw = item.get("doi", "") or ids.get("doi", "") or ""
        doi = doi_raw.replace("https://doi.org/", "").strip()

        pdf_url = ""
        best_oa = item.get("best_oa_location") or {}
        pdf_url = best_oa.get("pdf_url") or ""
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id.rstrip('v1')}"

        authors = []
        for a in item.get("authorships", []):
            author = a.get("author", {})
            name = author.get("display_name", "")
            if name:
                authors.append(name)

        venue = ""
        primary = item.get("primary_location") or {}
        source = primary.get("source") or {}
        venue = source.get("display_name", "")

        return PaperRecord(
            arxiv_id=arxiv_id,
            doi=doi,
            openalex_id=item.get("id", ""),
            title=title,
            abstract=abstract,
            authors=authors,
            year=item.get("publication_year") or 0,
            venue=venue,
            citation_count=item.get("cited_by_count") or 0,
            pdf_url=pdf_url,
            raw=item,
        )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate(records: List[PaperRecord]) -> List[PaperRecord]:
    """Remove duplicates, keeping the record with the highest citation count."""
    seen_titles: Dict[str, int] = {}  # title -> index in `out`
    seen_dois: Dict[str, int] = {}
    seen_arxiv: Dict[str, int] = {}
    out: List[Optional[PaperRecord]] = []

    for record in records:
        key = record.unique_key()
        doi_key = record.doi.strip().lower()
        arxiv_key = record.arxiv_id.strip().lower()

        existing_idx: Optional[int] = None
        if arxiv_key and arxiv_key in seen_arxiv:
            existing_idx = seen_arxiv[arxiv_key]
        elif doi_key and doi_key in seen_dois:
            existing_idx = seen_dois[doi_key]
        elif key in seen_titles:
            existing_idx = seen_titles[key]

        if existing_idx is not None:
            existing = out[existing_idx]
            if existing and record.citation_count > existing.citation_count:
                out[existing_idx] = record
        else:
            idx = len(out)
            out.append(record)
            seen_titles[key] = idx
            if doi_key:
                seen_dois[doi_key] = idx
            if arxiv_key:
                seen_arxiv[arxiv_key] = idx

    return [r for r in out if r is not None]
