from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

from .services.factory import create_ai_services
from .transports.ari import ARITransport


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

    # 3. Context & Aggregation (system prompt is in LLM Settings)
    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    # 4. The Pipeline Definition
    # Flow: Audio In -> STT -> User Agg -> LLM -> TTS -> Output -> Assistant Agg
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    return pipeline, transport
