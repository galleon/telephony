"""
ARI Transport for Asterisk Stasis + Media WebSocket.

Connects to Asterisk ARI (inbound) and runs a Media WebSocket server for audio.
Asterisk must have websocket_client.conf with a connection pointing to this host.
See README and config/sample-websocket_client.conf.

Pipecat does not ship an ARITransport; this is a custom implementation
based on asterisk-websocket-examples.
"""

import asyncio
import audioop
import time
import json
import logging
import os
import uuid
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    from websockets.asyncio.client import connect as ws_connect
    from websockets.asyncio.server import serve as ws_serve
    from websockets.asyncio.server import basic_auth
    from websockets.exceptions import ConnectionClosed
except ImportError:
    raise ImportError(
        "ARI transport requires websockets. Install with: pip install 'pipecat-ai[websocket]'"
    )

# Asterisk sends 8kHz µlaw; Pipecat/Whisper expects 16kHz PCM
ULAW_SAMPLE_RATE = 8000
PIPELINE_SAMPLE_RATE = 16000


class ARIMediaBridge(FrameProcessor):
    """Bridges Asterisk Media WebSocket (µlaw) to Pipecat pipeline (PCM)."""

    def __init__(self, ws, params: TransportParams, role: str, **kwargs):
        super().__init__(**kwargs)
        self._ws = ws
        self._params = params
        self._role = role  # "input" or "output"
        self._lock = asyncio.Lock()
        self._closed = False

    async def process_frame(self, frame, direction):
        if self._closed or not self._ws or self._ws.state != 1:  # 1 = OPEN
            return []

        if self._role == "output" and isinstance(frame, OutputAudioRawFrame):
            # PCM → µlaw, send to Asterisk
            try:
                ulaw = audioop.lin2ulaw(frame.audio, 2)
                await self._ws.send(ulaw)
            except Exception as e:
                logger.error(f"ARI output error: {e}")
                self._closed = True
            return []
        elif self._role == "input" and isinstance(frame, StartFrame):
            await self.push_frame(frame)
            return []
        elif self._role == "input" and isinstance(frame, EndFrame):
            await self.push_frame(frame)
            return []

        return [frame]


class ARITransport(BaseTransport):
    """
    Transport that connects to Asterisk ARI and runs a Media WebSocket server.

    Asterisk (on Mac) connects to the Media server on the DGX when a call enters
    Stasis. Asterisk uses websocket_client.conf: set [media_connection1] uri to
    ws://DGX_IP:8787/media. external_host must be the section name, not an IP.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        app_name: str = "ai-assistant",
        base_url: Optional[str] = None,
        media_host: str = "0.0.0.0",
        media_port: Optional[int] = None,
        params: Optional[TransportParams] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.params = params or TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
            audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
        )
        self._uri = uri.replace("http://", "ws://").replace("https://", "wss://")
        self._username = username
        self._password = password
        self._app_name = app_name
        self._base_url = base_url
        self._media_host = media_host
        self._media_port = media_port or int(os.getenv("ARI_MEDIA_PORT", "8787"))
        self._ari_ws = None
        self._media_ws = None
        self._sessions = {}
        self._input_proc = None
        self._output_proc = None
        self._handlers = {}

    def event_handler(self, event: str):
        def decorator(fn):
            self._handlers[event] = fn
            return fn
        return decorator

    async def _handle_ari_event(self, msg):
        msg_type = msg.get("type", "")
        if msg_type == "RESTResponse":
            return
        if msg_type == "StasisStart":
            await self._on_stasis_start(msg)
        elif msg_type == "StasisEnd":
            await self._on_stasis_end(msg)
        elif msg_type == "ChannelDestroyed":
            await self._on_channel_destroyed(msg)

    async def _on_stasis_start(self, msg):
        ch = msg.get("channel", {})
        dialplan = ch.get("dialplan", {})
        app_data = dialplan.get("app_data", "") or ""
        # External media channel has empty app_data; match by ch["id"] in sessions (our ws_channel)
        is_external_media = ch["id"] in self._sessions and self._sessions.get(ch["id"], {}).get("ws_channel") == ch["id"]
        if "incoming" not in app_data and "websocket" not in app_data and not is_external_media:
            return
        incoming_id = ch["id"]
        incoming_name = ch.get("name", "")
        if "incoming" in app_data:
            sess = {"incoming": incoming_id, "incoming_name": incoming_name}
            self._sessions[incoming_id] = sess
            ws_channel = str(uuid.uuid4())
            sess["ws_channel"] = ws_channel
            self._sessions[ws_channel] = sess
            # For WebSocket transport, external_host must be a websocket_client section name
            # (e.g. media_connection1), not an IP. Asterisk looks up [section] in websocket_client.conf
            # and uses its uri. Do NOT pass IP:port here—that section won't exist.
            external_host = os.getenv("ARI_MEDIA_CONNECTION", "media_connection1")
            if ":" in external_host and any(c.isdigit() for c in external_host):
                logger.warning(
                    f"ARI_MEDIA_CONNECTION looks like IP:port ({external_host}). "
                    "Use the websocket_client section name (e.g. media_connection1) instead."
                )
                external_host = "media_connection1"
            logger.info(f"Creating WebSocket channel for {incoming_name} (external_host={external_host})")
            await self._ari_send_request("POST", "channels/externalMedia", query_strings=[
                {"name": "channelId", "value": ws_channel},
                {"name": "app", "value": msg.get("application", self._app_name)},
                {"name": "data", "value": "websocket"},
                {"name": "external_host", "value": external_host},
                {"name": "transport", "value": "websocket"},
                {"name": "encapsulation", "value": "none"},
                {"name": "format", "value": "ulaw"},
            ])
        else:
            # External media channel: StasisStart for channel we created (id == ws_channel).
            # Asterisk may not set app_data for externally created channels.
            sess = self._sessions.get(ch["id"])
            if sess and sess.get("ws_channel") == ch["id"]:
                sess["ws_channel_name"] = ch.get("name", "")
                logger.info(f"WebSocket channel {ch.get('name')} in Stasis")
                # Bridge created on Dial ANSWER per asterisk-websocket-examples

    async def _on_stasis_end(self, msg):
        ch = msg.get("channel", {})
        dialplan = ch.get("dialplan", {})
        app_data = dialplan.get("app_data", "") or ""
        sess = None
        if "incoming" in app_data:
            sess = self._sessions.get(ch["id"])
            if sess:
                if sess.get("ws_channel"):
                    await self._ari_send_request("DELETE", f"channels/{sess['ws_channel']}")
                self._sessions.pop(sess["incoming"], None)
        elif "websocket" in app_data:
            sess = self._sessions.get(ch["id"])
            if sess:
                if sess.get("incoming"):
                    await self._ari_send_request("DELETE", f"channels/{sess['incoming']}")
                self._sessions.pop(sess["ws_channel"], None)
        if sess and sess.get("bridge_id"):
            await self._ari_send_request("DELETE", f"bridges/{sess['bridge_id']}")
            sess["bridge_id"] = None

    async def _on_channel_destroyed(self, msg):
        pass

    async def _create_bridge_and_answer(self, sess):
        """Create mixing bridge and add channels. Order per asterisk-websocket-examples: bridge, add both, then answer."""
        if sess.get("bridge_id"):
            return  # Already bridged
        bridge_id = str(uuid.uuid4())
        sess["bridge_id"] = bridge_id
        logger.info(f"Creating bridge {bridge_id} for {sess.get('incoming_name', '?')} <-> {sess.get('ws_channel_name', '?')}")
        await self._ari_send_request("POST", f"bridges/{bridge_id}?type=mixing")
        await self._ari_send_request("POST", f"bridges/{bridge_id}/addChannel?channel={sess['incoming']}")
        await self._ari_send_request("POST", f"bridges/{bridge_id}/addChannel?channel={sess['ws_channel']}")
        await self._ari_send_request("POST", f"channels/{sess['incoming']}/answer")

    async def _on_dial(self, msg):
        chan_name = msg.get("peer", {}).get("name", "")
        if "WebSocket/" not in chan_name:
            return
        if msg.get("dialstatus") == "ANSWER":
            sess = self._sessions.get(msg["peer"]["id"])
            if sess:
                logger.info(f"Dial ANSWER for WebSocket channel, creating bridge")
                await self._create_bridge_and_answer(sess)

    async def _ari_send_request(self, method: str, uri: str, query_strings=None):
        if not self._ari_ws or self._ari_ws.state != 1:
            return
        req_id = str(uuid.uuid4())
        req = {"type": "RESTRequest", "request_id": req_id, "method": method, "uri": uri}
        if query_strings:
            qs = "&".join(f"{q['name']}={q['value']}" for q in query_strings)
            req["uri"] = f"{uri}?{qs}"
        await self._ari_ws.send(json.dumps(req))

    async def _run_ari_client(self):
        ari_url = f"{self._uri.rstrip('/')}/ari/events?subscribeAll=false&app={self._app_name}&api_key={self._username}:{self._password}"
        backoff = 5.0
        while True:
            try:
                logger.info(f"Connecting to ARI at {self._uri}")
                async with ws_connect(ari_url) as ws:
                    self._ari_ws = ws
                    backoff = 5.0
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            if msg.get("type") == "StasisStart":
                                await self._on_stasis_start(msg)
                            elif msg.get("type") == "StasisEnd":
                                await self._on_stasis_end(msg)
                            elif msg.get("type") == "Dial":
                                await self._on_dial(msg)
                        except Exception as e:
                            logger.error(f"ARI event error: {e}")
            except Exception as e:
                logger.warning(f"ARI connection lost: {e}. Reconnecting in {backoff:.0f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 60.0)

    async def _process_media(self, ws):
        logger.info(f"Media WebSocket connected from {ws.remote_address}")
        if self._output_proc:
            self._output_proc._ws = ws
        # Bridge is created on Dial ANSWER (_on_dial) per asterisk-websocket-examples
        # Pipeline expects StartFrame before any other frames (LLMRunFrame, audio, etc.)
        if self._input_proc:
            start = StartFrame(
                allow_interruptions=True,
                audio_in_sample_rate=self.params.audio_in_sample_rate,
                audio_out_sample_rate=self.params.audio_out_sample_rate,
            )
            await self._input_proc.queue_frame(start)
        if "on_client_connected" in self._handlers:
            await self._handlers["on_client_connected"](self, ws)
        try:
            optimal_frame_size = 160
            async for message in ws:
                if isinstance(message, str):
                    if "MEDIA_START" in message:
                        for p in message.split()[1:]:
                            kv = p.split(":")
                            if kv[0] == "optimal_frame_size":
                                optimal_frame_size = int(kv[1])
                    elif "MEDIA_XOFF" in message or "MEDIA_XON" in message:
                        continue
                    continue
                if self._input_proc:
                    pcm = audioop.ulaw2lin(message, 2)
                    if PIPELINE_SAMPLE_RATE != ULAW_SAMPLE_RATE:
                        pcm, _ = audioop.ratecv(pcm, 2, 1, ULAW_SAMPLE_RATE, PIPELINE_SAMPLE_RATE, None)
                    frame = InputAudioRawFrame(audio=pcm, sample_rate=PIPELINE_SAMPLE_RATE, num_channels=1)
                    await self._input_proc.queue_frame(frame)
        except Exception as e:
            logger.error(f"Media error: {e}")
        finally:
            if self._input_proc:
                await self._input_proc.queue_frame(EndFrame())
            op = self._output_proc
            sent = getattr(op, "_bytes_sent", 0) if op else 0
            frames = getattr(op, "_frames_received", 0) if op else 0
            logger.info(f"Media WebSocket disconnected (received {frames} audio frames, sent {sent} bytes to Asterisk)")
            if "on_client_disconnected" in self._handlers:
                await self._handlers["on_client_disconnected"](self, ws)

    async def _media_server(self):
        async def handler(ws):
            await self._process_media(ws)
        logger.info(f"Starting Media server on {self._media_host}:{self._media_port}")
        async with ws_serve(handler, self._media_host, self._media_port, subprotocols=["media"]) as server:
            await server.wait_closed()

    async def queue_frame(self, frame):
        """Queue a frame to the pipeline input (e.g. LLMRunFrame after StartFrame)."""
        if self._input_proc:
            await self._input_proc.queue_frame(frame)

    def input(self):
        from pipecat.transports.base_input import BaseInputTransport
        self._input_proc = _ARIInputProcessor(self, self.params)
        return self._input_proc

    def output(self):
        self._output_proc = _ARIOutputProcessor(self, self.params)
        return self._output_proc

    async def run(self):
        """Run ARI client and Media server concurrently."""
        await asyncio.gather(
            self._run_ari_client(),
            self._media_server(),
        )


class _ARIInputProcessor(FrameProcessor):
    """Simple pass-through: receives frames pushed from Media handler and forwards downstream."""

    def __init__(self, transport: ARITransport, params: TransportParams, **kwargs):
        super().__init__(**kwargs)
        self._transport = transport
        self._params = params

    async def process_frame(self, frame, direction):
        # Base FrameProcessor consumes StartFrame (calls __start) without pushing it.
        # We must explicitly push all frames so they reach downstream and the pipeline runs.
        if isinstance(frame, StartFrame):
            await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        return []


# Asterisk expects 160-byte ulaw frames (20ms at 8kHz) for telephony
ULAW_FRAME_SIZE = 160
ULAW_FRAME_MS = 20  # 20ms per frame


class _ARIOutputProcessor(FrameProcessor):
    def __init__(self, transport: ARITransport, params: TransportParams, **kwargs):
        super().__init__(**kwargs)
        self._transport = transport
        self._params = params
        self._ws = None
        self._ulaw_buffer = bytearray()
        self._last_send_time = 0.0
        self._bytes_sent = 0
        self._frames_received = 0

    def set_client_connection(self, ws):
        self._ws = ws

    async def _send_ulaw(self, ulaw: bytes):
        """Send ulaw in 160-byte chunks, paced at 20ms (Asterisk telephony)."""
        if not self._ws or self._ws.state != 1:
            return
        self._ulaw_buffer.extend(ulaw)
        now = time.monotonic()
        while len(self._ulaw_buffer) >= ULAW_FRAME_SIZE:
            # Pace sends at 20ms to match Asterisk's expectation
            elapsed = now - self._last_send_time
            if self._last_send_time > 0 and elapsed < ULAW_FRAME_MS / 1000:
                await asyncio.sleep((ULAW_FRAME_MS / 1000) - elapsed)
            chunk = bytes(self._ulaw_buffer[:ULAW_FRAME_SIZE])
            del self._ulaw_buffer[:ULAW_FRAME_SIZE]
            try:
                await self._ws.send(chunk)
                self._bytes_sent += len(chunk)
                if self._bytes_sent == ULAW_FRAME_SIZE:
                    logger.info("ARI output: first audio frame sent to Asterisk")
                self._last_send_time = time.monotonic()
                now = self._last_send_time
            except ConnectionClosed:
                logger.debug("ARI output: WebSocket closed (call ended)")
                break
            except Exception as e:
                logger.error(f"ARI output send error: {e}")
                break

    async def process_frame(self, frame, direction):
        # Must call super() first so base handles StartFrame and sets __started=True.
        # Otherwise push_frame rejects all frames via _check_started().
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return []
        if not isinstance(frame, OutputAudioRawFrame):
            await self.push_frame(frame, direction)
            return []
        # Diagnostic: log when we receive audio but can't send
        if not self._ws:
            logger.debug("ARI output: received OutputAudioRawFrame but _ws is None")
            await self.push_frame(frame, direction)
            return []
        if self._ws.state != 1:
            logger.debug(f"ARI output: received OutputAudioRawFrame but ws.state={self._ws.state} (need 1=OPEN)")
            await self.push_frame(frame, direction)
            return []
        self._frames_received += 1
        if self._frames_received == 1:
            logger.info(f"ARI output: first OutputAudioRawFrame received (size={len(frame.audio)} bytes, sr={getattr(frame, 'sample_rate', '?')})")
        try:
            pcm = frame.audio
            sr = getattr(frame, "sample_rate", None) or PIPELINE_SAMPLE_RATE
            if len(pcm) == 0:
                logger.warning("ARI output: received empty audio frame")
            else:
                if sr != ULAW_SAMPLE_RATE:
                    pcm, _ = audioop.ratecv(pcm, 2, 1, sr, ULAW_SAMPLE_RATE, None)
                ulaw = audioop.lin2ulaw(pcm, 2)
                await self._send_ulaw(ulaw)
        except Exception as e:
            logger.exception(f"ARI output: error processing audio frame: {e}")
        # Forward frame to assistant aggregator
        await self.push_frame(frame, direction)
        return []
