from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import InterimTranscriptionFrame, StartFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .services.factory import create_ai_services
from .transports.ari import ARITransport

# Must match resampled caller audio from ARITransport (8 kHz μ-law → 16 kHz PCM).
_PIPELINE_INPUT_HZ = 16000


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

    # WhisperSTTService subclasses SegmentedSTTService: it only runs Whisper after
    # VADUserStoppedSpeakingFrame. Without VAD, audio is buffered but never transcribed.
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            sample_rate=_PIPELINE_INPUT_HZ,
            params=VADParams(stop_secs=0.35),
        )
    )

    # 3. Context & Aggregation (system prompt is in LLM Settings)
    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    # 4. The Pipeline Definition
    # Flow: Audio In -> VAD -> STT -> User Agg -> LLM -> TTS -> Output -> Assistant Agg
    pipeline = Pipeline([
        transport.input(),
        vad,
        stt,
        _SttTranscriptionLogger(),
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    return pipeline, transport, context
