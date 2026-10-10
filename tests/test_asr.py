from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.qiming.speech.asr import AsrRecognizer, AsrSettings, AsrUpstreamError


class AsrRecognizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recognizer = AsrRecognizer(
            AsrSettings(
                api_url="https://stt.example/v1/audio/transcriptions",
                api_key="secret",
                model="whisper-1",
                language="zh",
            )
        )

    @patch("src.qiming.speech.asr.httpx.Client.post")
    def test_recognize_returns_trimmed_text(self, post: MagicMock) -> None:
        post.return_value.is_error = False
        post.return_value.json.return_value = {"text": "  帮我拿水杯  "}

        result = self.recognizer.recognize(b"audio", filename="voice.webm")

        self.assertEqual(result, "帮我拿水杯")
        self.assertEqual(post.call_args.kwargs["data"]["language"], "zh")
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"], "Bearer secret"
        )

    @patch("src.qiming.speech.asr.httpx.Client.post")
    def test_recognize_rejects_missing_text(self, post: MagicMock) -> None:
        post.return_value.is_error = False
        post.return_value.json.return_value = {"text": ""}

        with self.assertRaises(AsrUpstreamError):
            self.recognizer.recognize(b"audio")

    @patch("src.qiming.speech.asr.httpx.Client.post")
    def test_dashscope_sends_base64_audio(self, post: MagicMock) -> None:
        recognizer = AsrRecognizer(
            AsrSettings(
                api_url=(
                    "https://dashscope.aliyuncs.com/compatible-mode/v1/"
                    "chat/completions"
                ),
                api_key="secret",
                provider="dashscope",
                model="qwen3-asr-flash",
                language="zh",
            )
        )
        post.return_value.is_error = False
        post.return_value.json.return_value = {
            "choices": [{"message": {"content": "  帮我拿水杯  "}}]
        }

        result = recognizer.recognize(
            b"audio",
            filename="voice.webm",
            content_type="audio/webm",
        )

        self.assertEqual(result, "帮我拿水杯")
        payload = post.call_args.kwargs["json"]
        audio = payload["messages"][0]["content"][0]["input_audio"]["data"]
        self.assertTrue(audio.startswith("data:audio/webm;base64,"))
        self.assertEqual(payload["asr_options"]["language"], "zh")


if __name__ == "__main__":
    unittest.main()
