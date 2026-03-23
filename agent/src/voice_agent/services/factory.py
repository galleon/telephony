import os

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.adapters.schemas.function_schema import FunctionSchema
from ..tools.handlers import fetch_order_status, transfer_to_human


def create_ai_services():
    # 1. STT: Local Whisper — defaults tuned for DGX Spark (CUDA). Use cpu on Mac only for dev.
    stt = WhisperSTTService(
        device=os.getenv("WHISPER_DEVICE", "cuda"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
        settings=WhisperSTTService.Settings(
            model=os.getenv("WHISPER_MODEL", "base"),
        ),
    )

    # 2. Define the tool schemas
    order_tool = FunctionSchema(
        name="get_order_status",
        description="Get the shipping status and delivery date for a specific order ID.",
        properties={
            "order_id": {"type": "string", "description": "The 6-digit order number (e.g. 123456)"}
        },
        required=["order_id"],
    )

    transfer_tool = FunctionSchema(
        name="transfer_to_support",
        description="Transfer the caller to a human support agent if you cannot help them.",
        properties={
            "reason": {"type": "string", "description": "Short summary of why the user needs a human"}
        },
    )

    # 3. LLM: vLLM with OpenAI-compatible API
    llm = OpenAILLMService(
        api_key=os.getenv("VLLM_API_KEY", "local-spark"),
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.getenv("LLM_MODEL", "Qwen2-VL-7B-Instruct"),
    )
    llm.register_function("get_order_status", fetch_order_status)
    llm.register_function("transfer_to_support", transfer_to_human)

    # 4. TTS: Local Piper (CUDA on DGX Spark)
    tts = PiperTTSService(
        use_cuda=os.getenv("PIPER_USE_CUDA", "true").lower() == "true",
        settings=PiperTTSService.Settings(
            voice=os.getenv("PIPER_VOICE", "en_US-ryan-high"),
        ),
    )

    tools = [order_tool, transfer_tool]
    return stt, llm, tts, tools
