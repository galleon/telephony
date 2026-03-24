import os

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    FunctionCallsStartedFrame,
    InterimTranscriptionFrame,
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
from pipecat.processors.filters.stt_mute_filter import STTMuteConfig, STTMuteFilter, STTMuteStrategy
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

from .services.factory import create_ai_services
from .transports.ari import ARITransport

# Must match resampled caller audio from ARITransport (8 kHz μ-law → 16 kHz PCM).
_PIPELINE_INPUT_HZ = 16000


class _FunctionCallFiller(FrameProcessor):
    """Speak a short filler phrase the moment the LLM initiates a tool call.

    Without this, the caller hears silence for the full round-trip:
      Whisper STT  (~2 s CPU / ~0.3 s GPU)
    + LLM first pass (tool-call decision, ~2 s)
    + tool execution
    + LLM second pass (response generation, ~2 s)
    + Piper TTS synthesis (~4 s CPU / ~0.2 s GPU)
    = 4–10 s of dead air.  Callers hang up.

    On FunctionCallsStartedFrame we push a TTSSpeakFrame downstream so Piper
    starts speaking immediately while the LLM waits for the tool result.
    Configurable via TOOL_CALL_FILLER env var; set to empty string to disable.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._filler = os.getenv("TOOL_CALL_FILLER", "One moment please.")

    async def process_frame(self, frame, direction):
        await self.push_frame(frame, direction)
        if (
            self._filler
            and direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, FunctionCallsStartedFrame)
        ):
            await self.push_frame(TTSSpeakFrame(text=self._filler), direction)


class _SttTranscriptionLogger(FrameProcessor):
    """Logs final/interim transcriptions so we can confirm caller audio reaches Whisper."""

    async def process_frame(self, frame, direction):
        if isinstance(frame, StartFrame):
            await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, (TranscriptionFrame, InterimTranscriptionFrame)
        ):
            logger.info(f"STT {type(frame).__name__}: {frame.text!r}")
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

    # Drop inbound audio while the bot is speaking (phone echo otherwise hits VAD/STT).
    # BotStarted/BotStoppedSpeakingFrame are broadcast from _ARIOutputProcessor.
    stt_mute = STTMuteFilter(
        config=STTMuteConfig(strategies={STTMuteStrategy.ALWAYS}),
    )

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
    # Flow: Audio In -> STT mute -> VAD -> STT -> User Agg -> LLM -> Filler -> TTS -> Output -> Agg
    pipeline = Pipeline(
        [
            transport.input(),
            stt_mute,
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
