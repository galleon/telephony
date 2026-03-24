import asyncio
import os
import re

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

# Whisper artifacts — square-bracket tags that contain no real speech.
# If the transcription is *only* these tags (nothing else after stripping them),
# drop the frame so the LLM never sees the noise.
_ARTIFACT_RE = re.compile(r"\[[^\]]*\]")

from .services.factory import create_ai_services
from .transports.ari import ARITransport

# Must match resampled caller audio from ARITransport (8 kHz μ-law → 16 kHz PCM).
_PIPELINE_INPUT_HZ = 16000


class _FunctionCallFiller(FrameProcessor):
    """Play a filler phrase shortly after the LLM starts generating, if it hasn't spoken yet.

    When the LLM calls a tool it streams JSON tokens for ~2 s before emitting
    FunctionCallsStartedFrame. Waiting for that frame leaves the caller in
    silence long enough to hang up.

    Strategy: on LLMFullResponseStartFrame, arm a short timer (TOOL_CALL_FILLER_DELAY_S,
    default 0.5 s). If a LLMTextFrame arrives first the LLM is producing a direct reply
    and the timer is cancelled. If the timer fires first (no text yet = tool call in
    progress) the filler is pushed to TTS immediately.

    append_to_context=False keeps the filler out of the LLM message history.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._filler = os.getenv("TOOL_CALL_FILLER", "One moment please.")
        self._delay = float(os.getenv("TOOL_CALL_FILLER_DELAY_S", "0.5"))
        self._timer: asyncio.Task | None = None

    def _cancel_timer(self):
        if self._timer and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    async def _fire_after_delay(self):
        try:
            await asyncio.sleep(self._delay)
            await self.push_frame(
                TTSSpeakFrame(text=self._filler, append_to_context=False),
                FrameDirection.DOWNSTREAM,
            )
        except asyncio.CancelledError:
            pass
        finally:
            self._timer = None

    async def process_frame(self, frame, direction):
        if isinstance(frame, StartFrame):
            await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            return
        if isinstance(frame, LLMFullResponseStartFrame):
            self._cancel_timer()
            if self._filler:
                self._timer = asyncio.create_task(self._fire_after_delay())
        elif isinstance(frame, LLMTextFrame):
            # LLM is producing a direct text reply — no filler needed
            self._cancel_timer()
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._cancel_timer()


class _SttTranscriptionLogger(FrameProcessor):
    """Logs and filters STT transcriptions.

    Drops TranscriptionFrames that contain only Whisper noise artifacts
    (e.g. [BLANK_AUDIO], [MUSIC PLAYING]) so the LLM never sees them.
    InterimTranscriptionFrames are passed through unchanged (they don't
    reach the LLM aggregator anyway).
    """

    async def process_frame(self, frame, direction):
        if isinstance(frame, StartFrame):
            await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame):
            real_text = _ARTIFACT_RE.sub("", frame.text).strip()
            if not real_text:
                logger.debug(f"STT artifact dropped: {frame.text!r}")
                return []
            logger.info(f"STT TranscriptionFrame: {frame.text!r}")
        elif direction == FrameDirection.DOWNSTREAM and isinstance(frame, InterimTranscriptionFrame):
            logger.info(f"STT InterimTranscriptionFrame: {frame.text!r}")
        await self.push_frame(frame, direction)
        return []


def configure_bot(asterisk_ip: str, ari_user: str, ari_pass: str):
    # 1. Transport: ARI client → Asterisk; Asterisk → agent media WebSocket (see websocket_client.conf on Asterisk host)
    transport = ARITransport(
        uri=f"http://{asterisk_ip}:8088",
        username=ari_user,
        password=ari_pass,
        app_name="ai-assistant",
    )

    # 2. Services: Fetch local Blackwell-optimized models and tool schemas
    stt, llm, tts, tools = create_ai_services()

    # WhisperSTTService subclasses SegmentedSTTService: it only runs Whisper after
    # VADUserStoppedSpeakingFrame. Without VAD, audio is buffered but never transcribed.
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            sample_rate=_PIPELINE_INPUT_HZ,
            params=VADParams(
                stop_secs=float(os.getenv("VAD_STOP_SECS", "0.35")),
                min_volume=float(os.getenv("VAD_MIN_VOLUME", "0.6")),
                confidence=float(os.getenv("VAD_CONFIDENCE", "0.7")),
            ),
        )
    )

    # 3. Context & Aggregation (system prompt is in LLM Settings)
    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    # 4. The Pipeline Definition
    # Flow: Audio In -> VAD -> STT -> User Agg -> LLM -> Filler -> TTS -> Output -> Agg
    # Echo muting (drop inbound audio while bot speaks) is handled by AlwaysUserMuteStrategy
    # on the LLMUserAggregator above — the STTMuteFilter predecessor was deprecated in 0.0.106.
    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            _SttTranscriptionLogger(),
            user_aggregator,
            llm,
            _FunctionCallFiller(),
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    return pipeline, transport, context
