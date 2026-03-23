import os

from loguru import logger
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from ..tools.handlers import fetch_order_status, transfer_to_human


def create_ai_services():
    # 1. STT: Local Whisper — try CUDA first, fall back to CPU (aarch64 CTranslate2 wheels are CPU-only)
    device = os.getenv("WHISPER_DEVICE", "cuda")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    try:
        stt = WhisperSTTService(
            device=device,
            compute_type=compute_type,
            settings=WhisperSTTService.Settings(
                model=os.getenv("WHISPER_MODEL", "base"),
            ),
        )
    except ValueError as e:
        if "CUDA" in str(e):
            logger.warning(
                f"Whisper CUDA unavailable ({e}). Falling back to CPU. "
                "On aarch64/DGX Spark, CTranslate2 PyPI wheels are CPU-only; build from source for GPU."
            )
            stt = WhisperSTTService(
                device="cpu",
                compute_type="int8",  # int8 is faster on CPU
                settings=WhisperSTTService.Settings(
                    model=os.getenv("WHISPER_MODEL", "base"),
                ),
            )
        else:
            raise

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
        required=[],  # reason is optional
    )

    # 3. LLM: vLLM with OpenAI-compatible API
    llm = OpenAILLMService(
        api_key=os.getenv("VLLM_API_KEY", "local-spark"),
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        settings=OpenAILLMService.Settings(
            model=os.getenv("LLM_MODEL", "Qwen2.5-7B-Instruct"),
            system_instruction=(
                "You are a helpful phone assistant for our company. "
                "When the call first connects and there are no prior messages, immediately greet the caller: 'Hello! How can I help you today?' "
                "Speak naturally and keep responses concise for audio. "
                "Use the provided tools to check order statuses or transfer calls."
            ),
        ),
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

    tools = ToolsSchema(standard_tools=[order_tool, transfer_tool])
    return stt, llm, tts, tools
