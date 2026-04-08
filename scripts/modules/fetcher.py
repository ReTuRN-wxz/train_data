import re
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PaperRecord — canonical in-memory representation of a fetched paper
# ---------------------------------------------------------------------------

_ARXIV_FROM_DOI = re.compile(r"arxiv[./](\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
_ARXIV_FROM_ID = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)


@dataclass
class PaperRecord:
    """Unified record produced by any fetcher backend."""

    paper_id: str = ""           # source-specific ID (e.g. OpenAlex W-ID or arXiv)
    doi: Optional[str] = None
    title: str = ""
    abstract: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    publication_date: Optional[str] = None
    venue: Optional[str] = None
    citation_count: int = 0
    pdf_url: Optional[str] = None
    external_ids: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def arxiv_id(self) -> Optional[str]:
        """Return a normalised arXiv ID (e.g. '2103.00020v1') when available."""
        # 1. explicit external_ids
        for key in ("ArXiv", "arxiv", "ar5iv"):
            val = self.external_ids.get(key)
            if val:
                return str(val)
        # 2. extract from DOI string
        if self.doi:
            m = _ARXIV_FROM_DOI.search(self.doi)
            if m:
                return m.group(1)
        # 3. extract from paper_id URL
        if self.paper_id:
            m = _ARXIV_FROM_ID.search(self.paper_id)
            if m:
                return m.group(1)
        return None

    def folder_name(self) -> str:
        """Return the folder/file-stem name for this paper.

        Priority:
          1. arXiv ID  (e.g. '2103.00020v1')
          2. DOI slug  (last path segment, e.g. 's41587-019-0209-9')
          3. Sanitised title  (truncated to 80 chars)
          4. paper_id slug
        """
        arxiv = self.arxiv_id
        if arxiv:
            return arxiv.replace("/", "_")

        if self.doi:
            slug = self.doi.rstrip("/").split("/")[-1]
            slug = re.sub(r"[^\w.\-]", "_", slug)
            if slug:
                return slug

        if self.title:
            slug = re.sub(r"[^\w\s\-]", "", self.title)
            slug = re.sub(r"\s+", "_", slug.strip())[:80]
            if slug:
                return slug

        return re.sub(r"[^\w\-]", "_", self.paper_id)[:80] or "unknown_paper"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PaperRecord":
        """Build a PaperRecord from a plain dict (e.g. a row from papers.jsonl)."""
        return cls(
            paper_id=d.get("paper_id", ""),
            doi=d.get("doi"),
            title=d.get("title", ""),
            abstract=d.get("abstract"),
            authors=d.get("authors") or [],
            year=d.get("year"),
            publication_date=d.get("publication_date"),
            venue=d.get("venue"),
            citation_count=d.get("citation_count") or 0,
            pdf_url=d.get("pdf_url"),
            external_ids=d.get("external_ids") or {},
            source=d.get("source", ""),
        )


class OpenAlexFetcher:
    """
    OpenAlex fetcher
    Docs: https://docs.openalex.org/
    """

    BASE_URL = "https://api.openalex.org/works"

    def __init__(
        self,
        timeout: float = 30.0,
        max_attempts: int = 5,
        base_wait: float = 2.0,
        max_wait: float = 60.0,
        min_interval: float = 1.0,
        mailto: Optional[str] = None,
    ):
        self.session = requests.Session()
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.base_wait = base_wait
        self.max_wait = max_wait
        self.min_interval = min_interval
        self._last_req_ts = 0.0

        # polite pool（可选，但强烈建议）
        # 你可以传一个邮箱，OpenAlex 官方建议这样做
        self.mailto = mailto

    def _throttle(self) -> None:
        now = time.time()
        elapsed = now - self._last_req_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_req_ts = time.time()

    def _request_with_retry(self, params: Dict[str, Any]) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                self._throttle()
                resp = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
                time.sleep(0.1)
                return resp.json()

            except requests.exceptions.RequestException as e:
                last_exc = e
                response = getattr(e, "response", None)
                status = response.status_code if response is not None else None

                # 非 429 的 4xx 通常不应重试
                if status is not None and 400 <= status < 500 and status != 429:
                    raise

                if attempt >= self.max_attempts:
                    raise

                wait = min(self.max_wait, self.base_wait * (2 ** (attempt - 1)))
                wait += random.uniform(0, 1.2)

                logger.warning(
                    "Attempt %d/%d for OpenAlex request failed: %s. Retrying in %.1fs ...",
                    attempt,
                    self.max_attempts,
                    f"HTTP {status}" if status else repr(e),
                    wait,
                )
                time.sleep(wait)

        if last_exc:
            raise last_exc
        raise RuntimeError("Unexpected retry state")

    @staticmethod
    def _build_filter(
        year_start: Optional[int],
        year_end: Optional[int],
        min_citations: Optional[int],
    ) -> str:
        filters = []
        if year_start is not None:
            filters.append(f"from_publication_date:{year_start}-01-01")
        if year_end is not None:
            filters.append(f"to_publication_date:{year_end}-12-31")
        if min_citations is not None:
            filters.append(f"cited_by_count:>{min_citations}")
        return ",".join(filters)

    @staticmethod
    def _abstract_from_inverted_index(inv_idx: Optional[Dict[str, List[int]]]) -> Optional[str]:
        if not inv_idx:
            return None
        # OpenAlex abstract_inverted_index -> text
        max_pos = -1
        for positions in inv_idx.values():
            if positions:
                max_pos = max(max_pos, max(positions))
        if max_pos < 0:
            return None
        words = [""] * (max_pos + 1)
        for token, positions in inv_idx.items():
            for pos in positions:
                if 0 <= pos < len(words):
                    words[pos] = token
        return " ".join(words).strip() or None

    def search(
        self,
        keywords: List[str],
        total: int = 100,
        page_size: int = 25,
        fields_of_study: Optional[List[str]] = None,  # 保留签名兼容，不使用
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        min_citations: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not keywords:
            return []

        # 简单关键词拼接
        query = " OR ".join([k.strip() for k in keywords if k and k.strip()])
        if not query:
            return []

        page_size = max(1, min(page_size, 200))
        out: List[Dict[str, Any]] = []
        cursor = "*"

        while len(out) < total:
            remain = total - len(out)
            per_page = min(page_size, remain)

            params: Dict[str, Any] = {
                "search": query,
                "per-page": per_page,
                "cursor": cursor,
                "sort": "cited_by_count:desc",
                "select": ",".join(
                    [
                        "id",
                        "doi",
                        "title",
                        "publication_year",
                        "publication_date",
                        "cited_by_count",
                        "authorships",
                        "primary_location",
                        "abstract_inverted_index",
                    ]
                ),
            }

            f = self._build_filter(year_start, year_end, min_citations)
            if f:
                params["filter"] = f

            if self.mailto:
                params["mailto"] = self.mailto

            payload = self._request_with_retry(params)
            results = payload.get("results", [])
            meta = payload.get("meta", {})

            if not results:
                break

            for r in results:
                out.append(self._normalize_one(r))
                if len(out) >= total:
                    break

            cursor = meta.get("next_cursor")
            if not cursor:
                break

        return out[:total]

    def _normalize_one(self, r: Dict[str, Any]) -> Dict[str, Any]:
        authorships = r.get("authorships") or []
        authors = []
        for a in authorships:
            author_obj = (a or {}).get("author") or {}
            name = author_obj.get("display_name")
            if name:
                authors.append(name)

        primary_location = r.get("primary_location") or {}
        source = (primary_location.get("source") or {})
        venue = source.get("display_name")

        abstract = self._abstract_from_inverted_index(r.get("abstract_inverted_index"))

        return {
            "paper_id": r.get("id"),
            "doi": r.get("doi"),
            "title": r.get("title"),
            "abstract": abstract,
            "authors": authors,
            "year": r.get("publication_year"),
            "publication_date": r.get("publication_date"),
            "venue": venue,
            "citation_count": r.get("cited_by_count"),
            "pdf_url": None,  # OpenAlex 不总是直接给 PDF
            "external_ids": {},
            "source": "openalex",
        }