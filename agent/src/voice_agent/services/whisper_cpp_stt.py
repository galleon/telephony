"""whisper.cpp STT service.

Whisper.cpp's server exposes POST /inference (multipart form), not the OpenAI-compat
/v1/audio/transcriptions path. This thin subclass of BaseWhisperSTTService keeps all
of pipecat's VAD-triggered batching and transcript handling while posting to the right
endpoint via httpx.
"""

from urllib.parse import urlparse

import httpx
from openai.types.audio import Transcription

from pipecat.services.whisper.base_stt import BaseWhisperSTTService


class WhisperCppSTTService(BaseWhisperSTTService):
    """STT backed by a local whisper.cpp server (POST /inference)."""

    def __init__(self, *, server_url: str, **kwargs):
        # Derive server root from whatever URL form was passed (with or without /v1).
        parsed = urlparse(server_url)
        self._inference_url = f"{parsed.scheme}://{parsed.netloc}/inference"
        # Parent initialises settings / VAD batching; no OpenAI client needed.
        super().__init__(api_key="unused", base_url=None, **kwargs)

    def _create_client(self, api_key, base_url):
        # We bypass the OpenAI SDK and call /inference with httpx directly.
        return None

    async def _transcribe(self, audio: bytes) -> Transcription:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._inference_url,
                files={"file": ("audio.wav", audio, "audio/wav")},
                data={"response_format": "json", "temperature": "0"},
                timeout=30.0,
            )
            resp.raise_for_status()
            return Transcription(text=resp.json().get("text", "").strip())
