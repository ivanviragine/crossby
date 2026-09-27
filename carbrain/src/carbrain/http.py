"""HTTP client with retries, polite pacing and resumable downloads."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

USER_AGENT = "carbrain/0.1 (car market research; respects robots.txt and source terms)"
RETRY_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BudgetExhaustedError(FetchError):
    """The run used up the number of requests it was allowed."""


class Http:
    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
        retries: int = 4,
        min_interval: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = httpx.Client(
            transport=transport,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        self.retries = retries
        self.min_interval = min_interval
        self.sleep = sleep
        self.requests = 0
        #: Maximum total requests for this client (None = unlimited).
        self.budget: int | None = None
        self._last_request = 0.0

    def close(self) -> None:
        self.client.close()

    def get(self, url: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        """GET with retries on network errors and 429/5xx; raises FetchError otherwise."""
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._pace()
            try:
                response = self.client.get(url, params=params)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if response.status_code not in RETRY_STATUS:
                    if response.is_error:
                        raise FetchError(
                            f"GET {url} failed with HTTP {response.status_code}: "
                            f"{response.text[:300]}",
                            response.status_code,
                        )
                    return response
                last_error = FetchError(f"GET {url} returned HTTP {response.status_code}")
            if attempt < self.retries:
                self.sleep(min(2**attempt * 2, 30))
        raise FetchError(f"GET {url} failed after {self.retries + 1} attempts: {last_error}")

    def download(self, url: str, dest: Path) -> Path:
        """Stream `url` to `dest`, resuming with a Range request when the transfer drops."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.unlink(missing_ok=True)
        expected: int | None = None
        for attempt in range(self.retries + 1):
            have = dest.stat().st_size if dest.exists() else 0
            if expected is not None and have >= expected:
                return dest
            headers = {"Range": f"bytes={have}-"} if have else {}
            self._pace()
            try:
                with self.client.stream("GET", url, headers=headers) as response:
                    if response.status_code == 416 and expected is not None:
                        return dest
                    if response.status_code not in (200, 206):
                        raise FetchError(
                            f"GET {url} returned HTTP {response.status_code}",
                            response.status_code,
                        )
                    if response.status_code == 200 and have:
                        # Server ignored Range: start over.
                        dest.unlink()
                        have = 0
                    expected = _total_size(response, have) or expected
                    with dest.open("ab") as fh:
                        for chunk in response.iter_bytes(1 << 20):
                            fh.write(chunk)
                size = dest.stat().st_size
                if expected is None or size >= expected:
                    return dest
            except BudgetExhaustedError:
                raise
            except (httpx.TransportError, FetchError):
                if attempt >= self.retries:
                    raise
            self.sleep(min(2**attempt * 2, 30))
        raise FetchError(f"Download of {url} incomplete after {self.retries + 1} attempts")

    def _pace(self) -> None:
        if self.budget is not None and self.requests >= self.budget:
            raise BudgetExhaustedError(f"Request budget of {self.budget} exhausted")
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0 and self._last_request:
            self.sleep(wait)
        self._last_request = time.monotonic()
        self.requests += 1


def _total_size(response: httpx.Response, already_have: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        return int(total) if total.isdigit() else None
    length = response.headers.get("Content-Length")
    return already_have + int(length) if length and length.isdigit() else None
