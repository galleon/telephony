import asyncio
import os
import signal
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Use project-local HF cache to avoid permission issues with ~/.cache
_agent_root = Path(__file__).resolve().parent.parent.parent
_cache = _agent_root / ".cache" / "huggingface"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_cache))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_cache))

from .pipeline import configure_bot

async def start_agent():
    # Load settings from .env (agent runs on DGX Spark; MAC_IP = Mac running Asterisk)
    MAC_IP = os.getenv("MAC_IP", "192.168.1.23")
    ARI_USER = os.getenv("ARI_USER", "ai_user")
    ARI_PASS = os.getenv("ARI_PASS", "your_password")

    base_url = os.getenv("SPARK_IP") or os.getenv("DGX_IP") or os.getenv("AGENT_HOST")
    pipeline, transport = configure_bot(MAC_IP, ARI_USER, ARI_PASS, base_url=base_url)

    @transport.event_handler("on_client_connected")
    async def on_connect(trans, client):
        logger.info("📞 CALL CONNECTED: DGX Spark Blackwell cores engaged.")

    logger.info(f"🚀 AI Agent starting. Listening for calls from {MAC_IP}...")
    # Run ARI client + Media server alongside the pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask
    pipecat_task = PipelineTask(pipeline)
    runner = asyncio.create_task(asyncio.gather(
        transport.run(),
        PipelineRunner().run(pipecat_task),
    ))

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, lambda: runner.cancel())
        loop.add_signal_handler(signal.SIGTERM, lambda: runner.cancel())
    except NotImplementedError:
        pass  # Windows has no add_signal_handler

    try:
        await runner
    except asyncio.CancelledError:
        logger.info("Agent shut down.")

if __name__ == "__main__":
    asyncio.run(start_agent())
