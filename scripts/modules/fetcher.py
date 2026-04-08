import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import requests


logger = logging.getLogger(__name__)


class SemanticScholarFetcher:
    """
    Semantic Scholar Graph API fetcher with:
    - API key support
    - client-side throttling
    - retry with exponential backoff + jitter
    - 429 Retry-After handling
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    SEARCH_ENDPOINT = f"{BASE_URL}/paper/search"

    def __init__(
        self,
        timeout: float = 30.0,
        max_attempts: int = 6,
        base_wait: float = 3.0,
        max_wait: float = 90.0,
        min_interval: Optional[float] = None,
    ):
        self.session = requests.Session()
        self.timeout = timeout

        # retry settings
        self.max_attempts = max_attempts
        self.base_wait = base_wait
        self.max_wait = max_wait

        # throttling
        self.min_interval = (
            float(os.getenv("S2_MIN_INTERVAL", "1.2"))
            if min_interval is None
            else float(min_interval)
        )
        self._last_req_ts = 0.0

        # auth
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            self.session.headers.update({"x-api-key": api_key})

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _throttle(self) -> None:
        now = time.time()
        elapsed = now - self._last_req_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_req_ts = time.time()

    def _compute_wait(self, attempt: int, response: Optional[requests.Response]) -> float:
        # 429: use Retry-After if available
        if response is not None and response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass

        # fallback: exponential backoff + jitter
        wait = min(self.max_wait, self.base_wait * (2 ** (attempt - 1)))
        wait += random.uniform(0, 1.5)
        return wait

    def _request_with_retry(self, params: Dict[str, Any]) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                self._throttle()
                resp = self.session.get(
                    self.SEARCH_ENDPOINT,
                    params=params,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                # tiny gap after success to avoid bursts
                time.sleep(0.15)
                return resp.json()

            except requests.exceptions.RequestException as e:
                last_exc = e
                response = getattr(e, "response", None)
                status = response.status_code if response is not None else None

                # Non-retriable 4xx (except 429)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise

                if attempt >= self.max_attempts:
                    raise

                wait = self._compute_wait(attempt, response)
                logger.warning(
                    "Attempt %d/%d for '_search_page' failed: %s. Retrying in %.1fs ...",
                    attempt,
                    self.max_attempts,
                    f"HTTP {status}" if status else repr(e),
                    wait,
                )
                time.sleep(wait)

        # Should not reach here, but keeps type checkers happy
        if last_exc:
            raise last_exc
        raise RuntimeError("Unexpected retry state")

    def _search_page(
        self,
        query: str,
        offset: int,
        limit: int,
        fields: Optional[List[str]] = None,
        fields_of_study: Optional[List[str]] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        min_citations: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not fields:
            fields = [
                "paperId",
                "externalIds",
                "title",
                "abstract",
                "authors",
                "year",
                "venue",
                "citationCount",
                "openAccessPdf",
                "publicationDate",
            ]

        params: Dict[str, Any] = {
            "query": query,
            "offset": offset,
            "limit": limit,
            "fields": ",".join(fields),
        }

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        # year filter
        if year_start is not None and year_end is not None:
            params["year"] = f"{year_start}-{year_end}"
        elif year_start is not None:
            params["year"] = f"{year_start}-"
        elif year_end is not None:
            params["year"] = f"-{year_end}"

        # citation filter
        if min_citations is not None:
            params["minCitationCount"] = min_citations

        payload = self._request_with_retry(params)
        return payload.get("data", [])

    # ----------------------------
    # Public API
    # ----------------------------
    def search(
        self,
        keywords: List[str],
        total: int = 100,
        page_size: int = 50,
        fields_of_study: Optional[List[str]] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        min_citations: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search papers and return normalized records.
        """
        if not keywords:
            return []

        # 用 OR 拼接关键词，与你日志里行为一致
        query = " | ".join(k.strip() for k in keywords if k and k.strip())
        if not query:
            return []

        page_size = max(1, min(page_size, 100))  # API 一般上限 100
        records: List[Dict[str, Any]] = []
        offset = 0

        while len(records) < total:
            remaining = total - len(records)
            limit = min(page_size, remaining)

            batch = self._search_page(
                query=query,
                offset=offset,
                limit=limit,
                fields_of_study=fields_of_study,
                year_start=year_start,
                year_end=year_end,
                min_citations=min_citations,
            )
            if not batch:
                break

            records.extend(self._normalize_many(batch))
            offset += limit

        return records[:total]

    def _normalize_many(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._normalize_one(r) for r in rows]

    def _normalize_one(self, r: Dict[str, Any]) -> Dict[str, Any]:
        authors = r.get("authors") or []
        author_names = [a.get("name") for a in authors if isinstance(a, dict) and a.get("name")]

        external_ids = r.get("externalIds") or {}
        open_access_pdf = r.get("openAccessPdf") or {}
        pdf_url = open_access_pdf.get("url") if isinstance(open_access_pdf, dict) else None

        return {
            "paper_id": r.get("paperId"),
            "title": r.get("title"),
            "abstract": r.get("abstract"),
            "authors": author_names,
            "year": r.get("year"),
            "venue": r.get("venue"),
            "citation_count": r.get("citationCount"),
            "publication_date": r.get("publicationDate"),
            "pdf_url": pdf_url,
            "doi": external_ids.get("DOI"),
            "arxiv_id": external_ids.get("ArXiv"),
            "corpus_id": external_ids.get("CorpusId"),
            "external_ids": external_ids,
            "source": "semantic_scholar",
        }