"""Launcher for the voice agent. Run from agent/: uv run main.py"""

import sys
from pathlib import Path

# Ensure src is on path so voice_agent package is findable
_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

if __name__ == "__main__":
    import asyncio

    from voice_agent.main import start_agent

    try:
        asyncio.run(start_agent())
    except KeyboardInterrupt:
        from loguru import logger

        logger.info("Agent shut down.")
