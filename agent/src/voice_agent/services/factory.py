import os
from pathlib import Path

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService

from ..tools.handlers import fetch_order_status, transfer_to_human


def create_ai_services():
    # 1. STT: Local Whisper — try CUDA first, fall back to CPU (aarch64 CTranslate2 wheels are CPU-only)
    device = os.getenv("WHISPER_DEVICE", "cuda")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    whisper_model = os.getenv("WHISPER_MODEL", "base")
    try:
        stt = WhisperSTTService(
            device=device,
            compute_type=compute_type,
            settings=WhisperSTTService.Settings(model=whisper_model),
        )
        logger.info(f"STT  | Whisper model={whisper_model!r}  device={device}  compute={compute_type}")
    except ValueError as e:
        if "CUDA" in str(e):
            logger.warning(
                f"Whisper CUDA unavailable ({e}). Falling back to CPU. "
                "On aarch64/DGX Spark, CTranslate2 PyPI wheels are CPU-only; build from source for GPU."
            )
            device, compute_type = "cpu", "int8"
            stt = WhisperSTTService(
                device=device,
                compute_type=compute_type,
                settings=WhisperSTTService.Settings(model=whisper_model),
            )
            logger.info(f"STT  | Whisper model={whisper_model!r}  device={device}  compute={compute_type}  (CPU fallback)")
        else:
            raise

    # 2. Define the tool schemas
    order_tool = FunctionSchema(
        name="get_order_status",
        description="Get the shipping status and delivery date for a specific order ID.",
        properties={"order_id": {"type": "string", "description": "The 6-digit order number (e.g. 123456)"}},
        required=["order_id"],
    )

    transfer_tool = FunctionSchema(
        name="transfer_to_support",
        description="Transfer the caller to a human support agent if you cannot help them.",
        properties={"reason": {"type": "string", "description": "Short summary of why the user needs a human"}},
        required=[],  # reason is optional
    )

    # 3. LLM: vLLM with OpenAI-compatible API
    llm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    llm_model = os.getenv("LLM_MODEL", "Qwen2.5-7B-Instruct")
    logger.info(f"LLM  | model={llm_model!r}  base_url={llm_base_url}")
    llm = OpenAILLMService(
        api_key=os.getenv("VLLM_API_KEY", "local-spark"),
        base_url=llm_base_url,
        settings=OpenAILLMService.Settings(
            model=llm_model,
            system_instruction=(
                "You are a helpful phone assistant for our company. "
                "The caller has already heard a spoken greeting; "
                "do not repeat hello or ask how you can help unless they seem confused. "
                "Do not ask for an order number or use tools until the caller has spoken. "
                "Keep replies concise for phone audio. "
                "Use tools only when the caller gives an order number or asks for status/transfer."
            ),
        ),
    )
    llm.register_function("get_order_status", fetch_order_status)
    llm.register_function("transfer_to_support", transfer_to_human)

    # 4. TTS: Local Piper (CUDA on DGX Spark)
    # Persist Piper voice in cache dir (mounted volume in Docker)
    piper_cache = Path(os.getenv("PIPER_CACHE_DIR", "/app/.cache/piper"))
    piper_cache.mkdir(parents=True, exist_ok=True)
    piper_voice = os.getenv("PIPER_VOICE", "en_US-ryan-high")
    piper_cuda = os.getenv("PIPER_USE_CUDA", "true").lower() == "true"
    tts = PiperTTSService(
        download_dir=piper_cache,
        use_cuda=piper_cuda,
        settings=PiperTTSService.Settings(voice=piper_voice),
    )
    try:
        import onnxruntime as _ort
        _providers = _ort.get_available_providers()
        _using_cuda = "CUDAExecutionProvider" in _providers
    except ImportError:
        _providers, _using_cuda = [], False
    if piper_cuda and not _using_cuda:
        logger.info(
            f"TTS  | Piper voice={piper_voice!r}  cuda=False (onnxruntime-gpu has no aarch64 wheels; "
            "Piper runs on CPU)"
        )
    else:
        logger.info(f"TTS  | Piper voice={piper_voice!r}  cuda={_using_cuda}  cache={piper_cache}")

    tools = ToolsSchema(standard_tools=[order_tool, transfer_tool])
    return stt, llm, tts, tools
