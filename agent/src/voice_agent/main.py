import asyncio
import os
from loguru import logger
from dotenv import load_dotenv
from .pipeline import configure_bot

load_dotenv()

async def start_agent():
    # Load settings from .env (agent runs on DGX Spark; MAC_IP = Mac running Asterisk)
    MAC_IP = os.getenv("MAC_IP", "192.168.1.23")
    ARI_USER = os.getenv("ARI_USER", "ai_user")
    ARI_PASS = os.getenv("ARI_PASS", "your_password")

    pipeline, transport = configure_bot(MAC_IP, ARI_USER, ARI_PASS)

    @transport.event_handler("on_client_connected")
    async def on_connect(transport, client):
        logger.info("📞 CALL CONNECTED: DGX Spark Blackwell cores engaged.")

    logger.info(f"🚀 AI Agent starting. Listening for calls from {MAC_IP}...")
    
    from pipecat.pipeline.runner import PipelineRunner
    await PipelineRunner().run(pipeline)

if __name__ == "__main__":
    try:
        asyncio.run(start_agent())
    except KeyboardInterrupt:
        logger.info("Agent shut down.")
