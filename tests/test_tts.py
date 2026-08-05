from __future__ import annotations

import os
import asyncio
import base64
import json
import unittest
from unittest.mock import patch

from src.qiming.feedback.tts import (
    TtsAudioStream,
    TtsConfigurationError,
    TtsSettings,
)


class TtsSettingsTests(unittest.TestCase):
    def test_reads_openai_compatible_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TTS_API_URL": "https://tts.example/v1/audio/speech",
                "TTS_API_KEY": "secret",
                "TTS_MODEL": "speech-model",
                "TTS_VOICE": "nova",
                "TTS_RESPONSE_FORMAT": "mp3",
                "TTS_TIMEOUT_SECONDS": "30",
            },
            clear=True,
        ):
            settings = TtsSettings.from_env()

        self.assertEqual(settings.api_url, "https://tts.example/v1/audio/speech")
        self.assertEqual(settings.api_key, "secret")
        self.assertEqual(settings.model, "speech-model")
        self.assertEqual(settings.voice, "nova")
        self.assertEqual(settings.timeout_seconds, 30)

    def test_requires_api_url(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(TtsConfigurationError):
                TtsSettings.from_env()

    def test_reuses_dashscope_configuration(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "dashscope-secret",
                "DASHSCOPE_BASE_URL": (
                    "https://dashscope.aliyuncs.com/compatible-mode/v1"
                ),
                "TTS_PROVIDER": "dashscope",
            },
            clear=True,
        ):
            settings = TtsSettings.from_env()

        self.assertEqual(settings.api_key, "dashscope-secret")
        self.assertEqual(
            settings.api_url,
            "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/"
            "SpeechSynthesizer",
        )
        self.assertEqual(settings.model, "qwen-audio-3.0-tts-flash")
        self.assertEqual(settings.voice, "longanhuan_v3.6")

    def test_decodes_dashscope_sse_audio_chunks(self) -> None:
        chunks = [b"first", b"second"]

        class FakeResponse:
            async def aiter_lines(self):
                for chunk in chunks:
                    event = {
                        "output": {
                            "audio": {
                                "data": base64.b64encode(chunk).decode("ascii")
                            }
                        }
                    }
                    yield f"data: {json.dumps(event)}"

            async def aclose(self):
                return None

        class FakeClient:
            async def aclose(self):
                return None

        async def collect() -> bytes:
            stream = TtsAudioStream(
                FakeClient(),  # type: ignore[arg-type]
                FakeResponse(),  # type: ignore[arg-type]
                provider="dashscope",
                response_format="mp3",
            )
            return b"".join([chunk async for chunk in stream.iter_bytes()])

        self.assertEqual(asyncio.run(collect()), b"firstsecond")


if __name__ == "__main__":
    unittest.main()
