"""OpenAI-compatible speech-to-text client."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


class AsrError(RuntimeError):
    """Base error raised by the STT client."""


class AsrConfigurationError(AsrError):
    """Raised when the STT service is not configured."""


class AsrUpstreamError(AsrError):
    """Raised when the upstream STT service rejects a request."""


@dataclass(frozen=True)
class AsrSettings:
    api_url: str
    api_key: str | None = None
    model: str = "whisper-1"
    language: str | None = "zh"
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "AsrSettings":
        api_url = os.getenv("STT_API_URL", "").strip()
        if not api_url:
            raise AsrConfigurationError("STT_API_URL 未配置")

        api_key = os.getenv("STT_API_KEY", "").strip() or None
        language = os.getenv("STT_LANGUAGE", "zh").strip() or None
        try:
            timeout_seconds = float(os.getenv("STT_TIMEOUT_SECONDS", "60"))
        except ValueError as exc:
            raise AsrConfigurationError("STT_TIMEOUT_SECONDS 必须是数字") from exc

        return cls(
            api_url=api_url,
            api_key=api_key,
            model=os.getenv("STT_MODEL", "whisper-1").strip() or "whisper-1",
            language=language,
            timeout_seconds=timeout_seconds,
        )


class AsrRecognizer:
    """Send recorded audio to an OpenAI-compatible transcription endpoint."""

    def __init__(self, settings: AsrSettings | None = None) -> None:
        self.settings = settings or AsrSettings.from_env()

    def recognize(
        self,
        audio_data: bytes,
        *,
        filename: str = "recording.webm",
        content_type: str = "audio/webm",
    ) -> str:
        headers: dict[str, str] = {}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        form_data = {"model": self.settings.model}
        if self.settings.language:
            form_data["language"] = self.settings.language

        try:
            with httpx.Client(timeout=self.settings.timeout_seconds) as client:
                response = client.post(
                    self.settings.api_url,
                    headers=headers,
                    data=form_data,
                    files={"file": (filename, audio_data, content_type)},
                )
        except httpx.HTTPError as exc:
            raise AsrUpstreamError(f"STT 服务连接失败: {exc}") from exc

        if response.is_error:
            detail = response.text[:500].strip()
            raise AsrUpstreamError(
                f"STT 服务返回 {response.status_code}: {detail or '无详细信息'}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise AsrUpstreamError("STT 服务未返回有效 JSON") from exc

        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise AsrUpstreamError("STT 服务返回中缺少转写文本")
        return text.strip()
