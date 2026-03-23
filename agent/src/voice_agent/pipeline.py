from typing import Optional

from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

from .services.factory import create_ai_services
from .transports.ari import ARITransport


def configure_bot(mac_ip: str, ari_user: str, ari_pass: str, base_url: Optional[str] = None):
    # 1. Transport: ARI client (connects to Mac) + Media server (Asterisk connects to DGX)
    # base_url = Spark/DGX IP so Asterisk knows where to send audio (e.g. http://192.168.1.50)
    transport = ARITransport(
        uri=f"http://{mac_ip}:8088",
        username=ari_user,
        password=ari_pass,
        app_name="ai-assistant",
        base_url=base_url,
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
