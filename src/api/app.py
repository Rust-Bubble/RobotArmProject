"""FastAPI entrypoint for browser speech input and output."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.qiming.feedback.tts import (
    TtsConfigurationError,
    TtsSynthesizer,
    TtsUpstreamError,
)

from src.qiming.speech.asr import (
    AsrConfigurationError,
    AsrRecognizer,
    AsrUpstreamError,
)

MAX_AUDIO_BYTES = int(os.getenv("STT_MAX_AUDIO_BYTES", str(15 * 1024 * 1024)))
ALLOWED_AUDIO_TYPES = {
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
}


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4096)


app = FastAPI(title="Qiming Smart Hand API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "stt_configured": bool(
            os.getenv("STT_API_URL", "").strip()
            or (
                os.getenv("DASHSCOPE_BASE_URL", "").strip()
                and os.getenv("DASHSCOPE_API_KEY", "").strip()
            )
        ),
        "stt_model": os.getenv("STT_MODEL", "whisper-1"),
        "tts_configured": bool(
            os.getenv("TTS_API_URL", "").strip()
            or (
                os.getenv("DASHSCOPE_BASE_URL", "").strip()
                and os.getenv("DASHSCOPE_API_KEY", "").strip()
            )
        ),
        "tts_model": os.getenv("TTS_MODEL", "gpt-4o-mini-tts"),
        "tts_voice": os.getenv("TTS_VOICE", "alloy"),
    }


@app.post("/api/stt/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, str]:
    content_type = (file.content_type or "").split(";", 1)[0].lower()
    if content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=415, detail="不支持的音频格式")

    audio = await file.read(MAX_AUDIO_BYTES + 1)
    await file.close()
    if not audio:
        raise HTTPException(status_code=400, detail="音频文件为空")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="音频文件过大")

    filename = Path(file.filename or "recording.webm").name
    try:
        recognizer = AsrRecognizer()
        text = await run_in_threadpool(
            recognizer.recognize,
            audio,
            filename=filename,
            content_type=content_type,
        )
    except AsrConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AsrUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"text": text}


@app.post("/api/tts/speech", response_class=StreamingResponse)
async def synthesize_speech(payload: SpeechRequest) -> StreamingResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="播报文本不能为空")

    try:
        audio = await TtsSynthesizer().open_stream(text)
    except TtsConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TtsUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return StreamingResponse(
        audio.iter_bytes(),
        media_type=audio.content_type,
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )
