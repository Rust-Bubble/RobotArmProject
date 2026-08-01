from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import app


class SttApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_reports_configuration(self) -> None:
        with patch.dict(os.environ, {"STT_API_URL": "https://stt.example"}):
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["stt_configured"])

    @patch("src.api.app.AsrRecognizer.recognize", return_value="帮我拿水杯")
    def test_transcribe_accepts_browser_audio(self, recognize) -> None:
        with patch.dict(os.environ, {"STT_API_URL": "https://stt.example"}):
            response = self.client.post(
                "/api/stt/transcribe",
                files={"file": ("recording.webm", b"audio", "audio/webm")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "帮我拿水杯"})
        recognize.assert_called_once()

    def test_transcribe_rejects_non_audio(self) -> None:
        response = self.client.post(
            "/api/stt/transcribe",
            files={"file": ("note.txt", b"hello", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)


if __name__ == "__main__":
    unittest.main()
