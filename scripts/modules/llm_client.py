"""
LLM client abstraction.

Usage::

    from modules.llm_client import build_llm_client

    client = build_llm_client()          # auto-detects from env vars
    summary = client.generate_summary(paper_data)

Environment variables (all optional – if unset the client gracefully degrades):
    KIMI_API_KEY     – API key for Moonshot/Kimi
    KIMI_BASE_URL    – Base URL  (default: https://api.moonshot.cn/v1)
    KIMI_MODEL       – Model name (default: kimi2.5thinking)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = (
    "你是一个科研论文分析专家。请根据提供的论文信息，生成三层摘要：\n"
    "1. concept_layer：用1-2句话概括核心贡献和方法\n"
    "2. detail_layer：详细描述技术细节和实现方法（约100-150字）\n"
    "3. application_layer：描述实际应用场景和未来拓展方向（约80-120字）\n\n"
    "请以JSON格式返回，包含 concept_layer、detail_layer、application_layer 三个字段。"
)

_SUMMARY_USER_TEMPLATE = """\
论文标题：{title}

作者：{authors}

摘要：{abstract}

发表年份：{year}
引用次数：{citation_count}

请根据以上信息生成三层摘要（JSON格式）：
"""

_FULL_TEXT_SYSTEM_PROMPT = (
    "你是一个科研论文助手，请根据提供的论文摘要和元数据，"
    "生成一篇结构化的论文介绍文档（Markdown格式），包含：\n"
    "- 标题和作者\n"
    "- 摘要\n"
    "- 主要贡献\n"
    "- 方法概述\n"
    "- 实验结果（如有）\n"
    "- 结论\n"
    "使用中英文混合或纯英文均可。"
)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract interface for LLM-based text generation."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when the client has a valid API key and can make calls."""

    @abstractmethod
    def generate_summary(self, paper_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate concept/detail/application layer summaries.

        Args:
            paper_data: Dict with at minimum: title, abstract, authors,
                        year, citation_count.

        Returns:
            Dict with keys: concept_layer, detail_layer, application_layer.
        """

    @abstractmethod
    def generate_full_text(self, paper_data: Dict[str, Any]) -> str:
        """Generate a full Markdown-formatted paper introduction.

        Args:
            paper_data: Same as generate_summary.

        Returns:
            Markdown string.
        """


# ---------------------------------------------------------------------------
# No-op / placeholder client (used when no API key is configured)
# ---------------------------------------------------------------------------


class NoopLLMClient(LLMClient):
    """Falls back to placeholder content when no LLM is configured."""

    def is_available(self) -> bool:
        return False

    def generate_summary(self, paper_data: Dict[str, Any]) -> Dict[str, str]:
        title = paper_data.get("title", "")
        abstract = paper_data.get("abstract", "")
        return {
            "concept_layer": f"[LLM未配置] {title}",
            "detail_layer": abstract[:300] if abstract else "[摘要不可用]",
            "application_layer": "[请配置 KIMI_API_KEY 以获取AI生成的应用层摘要]",
        }

    def generate_full_text(self, paper_data: Dict[str, Any]) -> str:
        title = paper_data.get("title", "Unknown Title")
        authors = paper_data.get("authors", [])
        abstract = paper_data.get("abstract", "")
        year = paper_data.get("year", "")
        citation_count = paper_data.get("citation_count", 0)

        author_str = ", ".join(authors[:5]) if authors else "Unknown"
        if len(authors) > 5:
            author_str += " et al."

        return (
            f"## {title}\n\n"
            f"**Authors:** {author_str}\n\n"
            f"**Year:** {year} | **Citations:** {citation_count}\n\n"
            f"## Abstract\n\n{abstract}\n\n"
            f"---\n\n"
            f"> *Full AI-generated analysis requires KIMI_API_KEY to be set.*\n"
        )


# ---------------------------------------------------------------------------
# Kimi / Moonshot client
# ---------------------------------------------------------------------------

_KIMI_DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
_KIMI_DEFAULT_MODEL = "kimi2.5thinking"


class KimiClient(LLMClient):
    """LLM client backed by the Moonshot/Kimi API (OpenAI-compatible).

    Configuration via environment variables:
        KIMI_API_KEY   – required
        KIMI_BASE_URL  – optional, defaults to https://api.moonshot.cn/v1
        KIMI_MODEL     – optional, defaults to kimi2.5thinking

    A class-level semaphore limits concurrent API calls to 2, staying safely
    under Kimi's default org concurrency limit of 3.
    """

    # Kimi's org-level concurrency limit is 3; keep ≤2 to leave headroom.
    _concurrency_semaphore: threading.Semaphore = threading.Semaphore(2)

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key or os.environ.get("KIMI_API_KEY", "")
        self._base_url = (
            base_url
            or os.environ.get("KIMI_BASE_URL", _KIMI_DEFAULT_BASE_URL)
        ).rstrip("/")
        self._model = model or os.environ.get("KIMI_MODEL", _KIMI_DEFAULT_MODEL)
        self._timeout = timeout
        self._max_retries = max_retries

    def is_available(self) -> bool:
        return bool(self._api_key)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate_summary(self, paper_data: Dict[str, Any]) -> Dict[str, str]:
        user_msg = _SUMMARY_USER_TEMPLATE.format(
            title=paper_data.get("title", ""),
            authors=", ".join(paper_data.get("authors", [])[:5]),
            abstract=paper_data.get("abstract", ""),
            year=paper_data.get("year", ""),
            citation_count=paper_data.get("citation_count", 0),
        )

        raw = self._chat(
            system=_SUMMARY_SYSTEM_PROMPT,
            user=user_msg,
        )

        # Parse JSON from the response (handle markdown code blocks)
        return self._parse_json_response(
            raw,
            fallback={
                "concept_layer": paper_data.get("title", ""),
                "detail_layer": paper_data.get("abstract", "")[:300],
                "application_layer": "[生成失败，请重试]",
            },
        )

    def generate_full_text(self, paper_data: Dict[str, Any]) -> str:
        user_msg = _SUMMARY_USER_TEMPLATE.format(
            title=paper_data.get("title", ""),
            authors=", ".join(paper_data.get("authors", [])[:5]),
            abstract=paper_data.get("abstract", ""),
            year=paper_data.get("year", ""),
            citation_count=paper_data.get("citation_count", 0),
        )
        return self._chat(
            system=_FULL_TEXT_SYSTEM_PROMPT,
            user=user_msg,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat(self, system: str, user: str) -> str:
        """Call the chat completions endpoint and return the assistant message.

        Acquires the class-level semaphore before each attempt to ensure at most
        ``_concurrency_semaphore`` requests are in-flight at the same time.
        """
        import time

        try:
            import openai  # type: ignore
        except ImportError:
            # Fall back to raw requests if openai SDK is not installed
            return self._chat_requests(system, user)

        client = openai.OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

        last_exc: Optional[Exception] = None
        delay = 5.0
        for attempt in range(1, self._max_retries + 1):
            try:
                with self._concurrency_semaphore:
                    response = client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        timeout=self._timeout,
                    )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "KimiClient attempt %d/%d failed: %s. Retrying in %.1fs…",
                    attempt,
                    self._max_retries,
                    exc,
                    delay,
                )
                if attempt < self._max_retries:
                    time.sleep(delay)
                    delay *= 2.0

        raise RuntimeError(
            f"KimiClient: all {self._max_retries} attempts failed"
        ) from last_exc

    def _chat_requests(self, system: str, user: str) -> str:
        """Fallback: call the API using raw requests (no openai SDK)."""
        import time

        import requests

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        last_exc: Optional[Exception] = None
        delay = 5.0
        for attempt in range(1, self._max_retries + 1):
            try:
                with self._concurrency_semaphore:
                    resp = requests.post(
                        url, headers=headers, json=payload, timeout=self._timeout
                    )
                    resp.raise_for_status()
                    data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "KimiClient(requests) attempt %d/%d failed: %s",
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries:
                    time.sleep(delay)
                    delay *= 2.0

        raise RuntimeError(
            f"KimiClient: all {self._max_retries} attempts failed"
        ) from last_exc

    @staticmethod
    def _parse_json_response(
        raw: str, fallback: Dict[str, str]
    ) -> Dict[str, str]:
        """Extract JSON object from raw LLM output."""
        import re

        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find a JSON object substring
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse JSON from LLM response; using fallback.")
        return fallback


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_llm_client(
    provider: str = "kimi",
    **kwargs: Any,
) -> LLMClient:
    """Return the appropriate LLM client.

    Args:
        provider: Currently only 'kimi' is supported.
        **kwargs: Forwarded to the client constructor.

    Returns:
        A configured ``LLMClient`` instance.  If no API key is present,
        returns a ``NoopLLMClient`` that generates placeholder content.
    """
    if provider.lower() == "kimi":
        client: LLMClient = KimiClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")

    if not client.is_available():
        logger.warning(
            "KIMI_API_KEY is not set. "
            "The script will run in metadata-only mode "
            "(LLM-generated summaries will be replaced by placeholders). "
            "Set KIMI_API_KEY to enable full AI-generated content."
        )
        return NoopLLMClient()

    return client
