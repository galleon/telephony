from pipecat.pipeline.pipeline import Pipeline
from pipecat.transports.network.ari import ARITransport
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from .services.factory import create_ai_services

def configure_bot(mac_ip: str, ari_user: str, ari_pass: str):
    # 1. Transport: Connects DGX Spark to your Mac PBX
    transport = ARITransport(
        uri=f"http://{mac_ip}:8088",
        username=ari_user,
        password=ari_pass,
        app_name="ai-assistant"
    )

    # 2. Services: Fetch local Blackwell-optimized models and tool schemas
    stt, llm, tts, tools = create_ai_services()

    # 3. Context & Aggregation: This manages the conversation memory and tools
    # We use OpenAI-compatible context because vLLM supports it natively
    context = OpenAILLMContext(
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful phone assistant for our company. "
                           "Speak naturally and keep responses concise for audio. "
                           "Use the provided tools to check order statuses or transfer calls."
            }
        ],
        tools=tools
    )
    
    # User context aggregator: Collects transcribed text until a pause is detected
    user_context = context.create_user_context_aggregator()
    
    # Assistant context aggregator: Collects LLM tokens to form complete sentences for TTS
    assistant_context = context.create_assistant_context_aggregator()

    # 4. The Pipeline Definition
    # Flow: Audio In -> STT -> Buffer Text -> LLM (with Tools) -> Buffer Tokens -> TTS -> Audio Out
    pipeline = Pipeline([
        transport.input(),          # Receive audio from Mac
        stt,                        # Local Whisper (Audio -> Text)
        user_context,               # Buffer user speech until finished
        llm,                        # Local vLLM (Brain + Function Calling)
        tts,                        # Local Piper (Text -> Audio)
        assistant_context,          # Buffer AI response for smooth playback
        transport.output()          # Send audio back to Mac
    ])

    return pipeline, transport
