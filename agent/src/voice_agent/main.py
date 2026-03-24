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

from pipecat.frames.frames import LLMRunFrame

from .pipeline import configure_bot

async def start_agent():
    # Load settings from .env (agent runs on DGX; ASTERISK_IP = host running Asterisk)
    ASTERISK_IP = os.getenv("ASTERISK_IP") or os.getenv("MAC_IP", "192.168.1.23")
    ARI_USER = os.getenv("ARI_USER", "ai_user")
    ARI_PASS = os.getenv("ARI_PASS", "your_password")

    pipeline, transport, llm_context = configure_bot(ASTERISK_IP, ARI_USER, ARI_PASS)

    logger.info(f"🚀 AI Agent starting. Listening for calls from {ASTERISK_IP}...")
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask

    # Disable idle timeout: we wait for calls, so pipeline has no frames until call connects
    # Disable RTVI: we use ARI/phone, not Pipecat's WebSocket client
    pipecat_task = PipelineTask(
        pipeline,
        idle_timeout_secs=None,
        cancel_on_idle_timeout=False,
        enable_rtvi=False,
    )

    @transport.event_handler("on_client_connected")
    async def on_connect(trans, client):
        logger.info("📞 CALL CONNECTED: DGX Spark Blackwell cores engaged.")
        # Fresh call: clear history so the model greets instead of continuing a prior turn
        llm_context.set_messages([])
        await trans.queue_frame(LLMRunFrame())

    async def run_transport():
        try:
            await transport.run()
        except Exception as e:
            logger.exception(f"Transport failed (ARI/Media): {e}")
            raise

    transport_task = asyncio.create_task(run_transport())
    pipeline_task = asyncio.create_task(
        PipelineRunner(handle_sigint=False, handle_sigterm=False).run(pipecat_task)
    )
    tasks = [transport_task, pipeline_task]

    loop = asyncio.get_running_loop()
    try:
        def on_signal():
            for t in tasks:
                t.cancel()
        loop.add_signal_handler(signal.SIGINT, on_signal)
        loop.add_signal_handler(signal.SIGTERM, on_signal)
    except NotImplementedError:
        pass

    # Wait for first task to finish; if one fails, cancel the other so we can log
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in done:
        try:
            exc = t.exception()
            if exc and not isinstance(exc, asyncio.CancelledError):
                logger.exception(f"Task failed: {exc}")
        except asyncio.CancelledError:
            pass
    for t in pending:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    logger.info("Agent shut down.")

if __name__ == "__main__":
    asyncio.run(start_agent())
