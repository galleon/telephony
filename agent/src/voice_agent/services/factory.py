import os
from pathlib import Path

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService

from ..tools.handlers import escalate_to_engineer, fetch_ticket_status
from .whisper_cpp_stt import WhisperCppSTTService


def create_ai_services():
    # 1. STT
    # If WHISPER_API_URL is set, use the remote whisper.cpp server (CUDA on DGX Spark).
    # whisper.cpp exposes POST /inference; WhisperCppSTTService wraps it with the same
    # VAD-triggered batching as the local CTranslate2 path.
    # Otherwise fall back to local CTranslate2 Whisper on CPU (aarch64 PyPI wheels are CPU-only).
    whisper_model = os.getenv("WHISPER_MODEL", "base")
    whisper_api_url = os.getenv("WHISPER_API_URL", "")
    if whisper_api_url:
        stt = WhisperCppSTTService(server_url=whisper_api_url)
        logger.info(f"STT  | whisper.cpp /inference  server={whisper_api_url}")
    else:
        device = os.getenv("WHISPER_DEVICE", "cuda")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
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
    ticket_tool = FunctionSchema(
        name="get_ticket_status",
        description=(
            "Look up the current status of an IT support ticket by its ticket ID. "
            "Use this when the caller provides a ticket number (e.g. INC001, INC123)."
        ),
        properties={
            "ticket_id": {
                "type": "string",
                "description": "The incident or ticket ID (e.g. INC001). Normalise to uppercase.",
            }
        },
        required=["ticket_id"],
    )

    escalate_tool = FunctionSchema(
        name="escalate_to_engineer",
        description=(
            "Escalate the call to a human engineer when the issue is too complex, "
            "requires hands-on access, or the caller explicitly asks to speak to someone."
        ),
        properties={
            "reason": {
                "type": "string",
                "description": "Brief description of the issue requiring escalation.",
            },
            "team": {
                "type": "string",
                "description": "Target team: 'network', 'desktop', 'security', 'database', or 'general'.",
                "enum": ["network", "desktop", "security", "database", "general"],
            },
        },
        required=["reason"],
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
                "You are an IT support phone assistant. "
                "The caller has already heard a greeting; do not repeat it. "
                "Help with ticket status, troubleshooting steps, and escalations. "
                "Keep replies concise — this is a phone call, not a chat window. "
                "Do not look up a ticket until the caller gives you a ticket ID. "
                "If the issue needs hands-on work or the caller asks for a human, escalate. "
                "When troubleshooting, ask one clarifying question at a time."
            ),
        ),
    )
    llm.register_function("get_ticket_status", fetch_ticket_status)
    llm.register_function("escalate_to_engineer", escalate_to_engineer)

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

    tools = ToolsSchema(standard_tools=[ticket_tool, escalate_tool])
    return stt, llm, tts, tools
