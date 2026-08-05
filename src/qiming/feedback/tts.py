"""OpenAI-compatible text-to-speech client."""

from __future__ import annotations

import os
import base64
import binascii
import json
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().parents[1] / "mllm" / ".env"
load_dotenv(ENV_PATH)


class TtsError(RuntimeError):
    """Base error raised by the TTS client."""


class TtsConfigurationError(TtsError):
    """Raised when the TTS service is not configured."""


class TtsUpstreamError(TtsError):
    """Raised when the upstream TTS service rejects a request."""


@dataclass(frozen=True)
class TtsSettings:
    api_url: str
    api_key: str | None = None
    provider: str = "openai"
    model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    response_format: str = "mp3"
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "TtsSettings":
        api_url = os.getenv("TTS_API_URL", "").strip()
        provider = os.getenv("TTS_PROVIDER", "").strip().lower()
        dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

        if not api_url and dashscope_base_url and dashscope_api_key:
            parsed = urlsplit(dashscope_base_url)
            api_url = (
                f"{parsed.scheme}://{parsed.netloc}"
                "/api/v1/services/audio/tts/SpeechSynthesizer"
            )
            provider = "dashscope"
        if not api_url:
            raise TtsConfigurationError("TTS_API_URL 或 DashScope 配置未提供")

        provider = provider or "openai"
        default_model = (
            "qwen-audio-3.0-tts-flash"
            if provider == "dashscope"
            else "gpt-4o-mini-tts"
        )
        default_voice = "longanhuan_v3.6" if provider == "dashscope" else "alloy"

        try:
            timeout_seconds = float(os.getenv("TTS_TIMEOUT_SECONDS", "60"))
        except ValueError as exc:
            raise TtsConfigurationError("TTS_TIMEOUT_SECONDS 必须是数字") from exc

        return cls(
            api_url=api_url,
            api_key=os.getenv("TTS_API_KEY", "").strip()
            or dashscope_api_key
            or None,
            provider=provider,
            model=os.getenv("TTS_MODEL", default_model).strip() or default_model,
            voice=os.getenv("TTS_VOICE", default_voice).strip() or default_voice,
            response_format=os.getenv("TTS_RESPONSE_FORMAT", "mp3").strip() or "mp3",
            timeout_seconds=timeout_seconds,
        )


class TtsAudioStream:
    """Own an upstream streaming response and its HTTP client."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        response: httpx.Response,
        *,
        provider: str,
        response_format: str,
    ) -> None:
        self._client = client
        self._response = response
        self._provider = provider
        self.content_type = (
            "audio/mpeg"
            if response_format == "mp3"
            else f"audio/{response_format}"
        )

    async def _iter_dashscope_bytes(self) -> AsyncIterator[bytes]:
        async for line in self._response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw_event = line[5:].strip()
            if not raw_event or raw_event == "[DONE]":
                continue
            try:
                event = json.loads(raw_event)
                if event.get("code"):
                    raise TtsUpstreamError(
                        f"TTS 服务返回错误: {event.get('message') or event['code']}"
                    )
                audio_data = event.get("output", {}).get("audio", {}).get("data")
                if audio_data:
                    yield base64.b64decode(audio_data)
            except (ValueError, TypeError, binascii.Error) as exc:
                raise TtsUpstreamError("DashScope TTS 返回了无效的音频流") from exc

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        try:
            chunks = (
                self._iter_dashscope_bytes()
                if self._provider == "dashscope"
                else self._response.aiter_bytes()
            )
            async for chunk in chunks:
                if chunk:
                    yield chunk
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        await self._response.aclose()
        await self._client.aclose()


class TtsSynthesizer:
    """Open a streaming synthesis request to an OpenAI-compatible endpoint."""

    def __init__(self, settings: TtsSettings | None = None) -> None:
        self.settings = settings or TtsSettings.from_env()

    async def open_stream(self, text: str) -> TtsAudioStream:
        headers: dict[str, str] = {"Accept": "audio/mpeg"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        client = httpx.AsyncClient(timeout=self.settings.timeout_seconds)
        payload: dict[str, object]
        if self.settings.provider == "dashscope":
            headers.update(
                {
                    "Accept": "text/event-stream",
                    "X-DashScope-SSE": "enable",
                }
            )
            payload = {
                "model": self.settings.model,
                "input": {
                    "text": text,
                    "voice": self.settings.voice,
                    "format": self.settings.response_format,
                },
            }
        else:
            payload = {
                "model": self.settings.model,
                "voice": self.settings.voice,
                "input": text,
                "response_format": self.settings.response_format,
            }

        request = client.build_request(
            "POST",
            self.settings.api_url,
            headers=headers,
            json=payload,
        )
        try:
            response = await client.send(request, stream=True)
        except httpx.HTTPError as exc:
            await client.aclose()
            raise TtsUpstreamError(f"TTS 服务连接失败: {exc}") from exc

        if response.is_error:
            body = (await response.aread()).decode(errors="replace")[:500].strip()
            status_code = response.status_code
            await response.aclose()
            await client.aclose()
            raise TtsUpstreamError(
                f"TTS 服务返回 {status_code}: {body or '无详细信息'}"
            )

        return TtsAudioStream(
            client,
            response,
            provider=self.settings.provider,
            response_format=self.settings.response_format,
        )


class TtsFeedback:
    """Compatibility facade used by the command-line prototype."""

    def say(self, text: str) -> None:
        print(f"[语音播报] {text}")
