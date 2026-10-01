"""Vision-LLM OCR Extract adapter (OpenAI-vision-compatible /v1/chat/completions).

One adapter for every server that speaks the OpenAI vision chat API: vLLM serving
olmOCR, or any hosted/local vision model. It renders a scan/image (a PDF page by
page via pymupdf) and asks the vision model to transcribe the text - the
highest-fidelity path for degraded/old scans where classic OCR fails. Opt-in via
``semdex[vision]`` (pymupdf + httpx); needs ``endpoint`` (an OpenAI base URL
ending in ``/v1``) and, for a hosted model, an ``api_key`` (env-only, never
inlined). ``olmocr`` and a generic vision model use this SAME adapter, differing
only in the default model id / prompt.
"""

from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from ...domain.errors import ExtractionError
from ...domain.models import ExtractedDocument
from .._http_retry import DEFAULT_RETRIES, retry_http
from ..discovery.location import from_uri
from ._http import DEFAULT_MAX_BYTES, guess_mime, read_source_bytes

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import httpx

    from ...domain.models import SourceRef

_DEFAULT_MODEL = "olmocr"
_DEFAULT_DPI = 200
_DEFAULT_PROMPT = (
    "You are an OCR engine. Transcribe ALL text visible in this image exactly as "
    "it appears, preserving reading order and layout as Markdown. Output only the "
    "transcription, with no commentary or code fences."
)


class _ChoiceMessage(BaseModel):
    """The assistant message of one chat choice (extra fields ignored)."""

    content: str


class _Choice(BaseModel):
    """One choice of an OpenAI chat-completions response."""

    message: _ChoiceMessage


class _ChatResponse(BaseModel):
    """OpenAI ``/v1/chat/completions`` response envelope (extra fields ignored)."""

    choices: list[_Choice]


class VisionOcrExtractor:
    """OCR a scan/image via an OpenAI-vision-compatible chat server (olmOCR / any VLM)."""

    def __init__(  # noqa: PLR0913 - DI seam: endpoint + vision tunables (model/key/prompt/dpi/limits) + injected client
        self,
        endpoint: str | None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
        dpi: int | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout: float = 120.0,
        retries: int = DEFAULT_RETRIES,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not endpoint:
            raise ExtractionError("vision OCR extractor requires an endpoint (an OpenAI base URL ending in /v1)")
        self._url = endpoint.rstrip("/") + "/chat/completions"
        self._model = model or _DEFAULT_MODEL
        # Bearer only when a key is supplied; local vLLM/olmOCR need none.
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._prompt = prompt or _DEFAULT_PROMPT
        self._dpi = _DEFAULT_DPI if dpi is None else dpi
        self._max_bytes = max_bytes
        self._timeout = timeout
        self._retries = retries
        self._client = client
        # Injected clock so tests exercise the backoff without a real wait.
        self._sleep = sleep

    @property
    def render_dpi(self) -> int:
        """The resolution pages are rasterised at, after the default is applied.

        Public because it decides what a page COSTS a vision model - tokens scale with pixels -
        so a caller sizing a run, or a benchmark recording what it measured, has to be able to
        read it without reaching into the adapter.
        """
        return self._dpi

    def __call__(self, source: SourceRef) -> ExtractedDocument:
        content = read_source_bytes(source, max_bytes=self._max_bytes)
        path = from_uri(source.uri)
        if path.suffix.lower() == ".pdf":
            # One page image in memory at a time (bounded), OCR'd, then joined.
            pages = [self._ocr(image, "image/png") for image in self._render_pdf(content, source)]
            text = "\n\n".join(pages)
        else:
            text = self._ocr(content, guess_mime(source))
        return ExtractedDocument(source=source, text=text)

    def _render_pdf(self, pdf_bytes: bytes, source: SourceRef) -> Iterator[bytes]:
        """Yield one PNG per PDF page (rendered at ``dpi``); bounded memory."""
        try:
            import pymupdf  # type: ignore  # optional dep; see module docstring
        except ImportError as exc:  # pragma: no cover - only without semdex[vision]
            raise ExtractionError("pymupdf is not installed; install semdex[vision]") from exc
        fitz: Any = pymupdf
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            raise ExtractionError(f"vision OCR could not open PDF {source.uri}: {exc}") from exc
        try:
            for page in doc:
                pixmap = page.get_pixmap(dpi=self._dpi)
                png: bytes = pixmap.tobytes("png")
                yield png
        finally:
            doc.close()

    def _ocr(self, image_bytes: bytes, mime: str) -> str:
        data_uri = f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii")
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_uri}}]},
            ],
            "temperature": 0,
        }
        body = self._post(payload)
        try:
            parsed = _ChatResponse.model_validate(body)
        except ValidationError as exc:
            raise ExtractionError(f"vision OCR at {self._url} returned an unexpected response: {exc}") from exc
        if not parsed.choices:
            raise ExtractionError(f"vision OCR at {self._url} returned no choices")
        return parsed.choices[0].message.content

    def _post(self, payload: dict[str, Any]) -> Any:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - only without httpx
            raise ExtractionError("httpx is not installed; install semdex[vision]") from exc
        owns_client = self._client is None
        active = self._client if self._client is not None else httpx.Client(timeout=self._timeout)

        def send() -> httpx.Response:
            response = active.post(self._url, json=payload, headers=self._headers)
            response.raise_for_status()
            return response

        try:
            response = retry_http(send, tries=self._retries, sleep=self._sleep)
            return response.json()
        except httpx.HTTPError as exc:
            raise ExtractionError(f"vision OCR request to {self._url} failed: {exc}") from exc
        except ValueError as exc:
            raise ExtractionError(f"vision OCR at {self._url} returned invalid JSON: {exc}") from exc
        finally:
            if owns_client:
                active.close()


# Static conformance assertion -- pyright verifies VisionOcrExtractor satisfies Extract.
if TYPE_CHECKING:
    from ...application.ports import Extract

    _assert_extract: Extract = VisionOcrExtractor("http://x/v1")


__all__ = [
    "VisionOcrExtractor",
]
