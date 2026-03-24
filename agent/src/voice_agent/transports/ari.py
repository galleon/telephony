"""
ARI Transport for Asterisk Stasis + Media WebSocket.

Connects to Asterisk ARI (inbound) and runs a Media WebSocket server for audio.
Asterisk must have websocket_client.conf with a connection pointing to this host.
See README and config/sample-websocket_client.conf.

Pipecat does not ship an ARITransport; this is a custom implementation
based on asterisk-websocket-examples.
"""

import asyncio
import json
import os
import time
import uuid

import numpy as np

from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    from websockets.asyncio.client import connect as ws_connect
    from websockets.asyncio.server import serve as ws_serve
    from websockets.exceptions import ConnectionClosed
except ImportError:
    raise ImportError("ARI transport requires websockets. Install with: pip install 'pipecat-ai[websocket]'")

# Asterisk sends 8kHz µlaw; Pipecat/Whisper expects 16kHz PCM
ULAW_SAMPLE_RATE = 8000
PIPELINE_SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# G.711 µ-law codec (replaces deprecated audioop, vectorised with numpy for
# fast execution on aarch64 NEON — audioop is removed in Python 3.13).
# ---------------------------------------------------------------------------

# 256-entry lookup table for ulaw → linear decode (built once at import).
def _build_ulaw2lin_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.int16)
    for i in range(256):
        u = (~i) & 0xFF
        t = ((u & 0x0F) << 3) + 0x84
        t <<= (u & 0x70) >> 4
        table[i] = (0x84 - t) if (u & 0x80) else (t - 0x84)
    return table


_ULAW2LIN_TABLE: np.ndarray = _build_ulaw2lin_table()

# Exponent lookup for linear → ulaw encode (Sun/ITU-T G.711 table).
_G711_EXP_LUT = np.array(
    [
        0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    ],
    dtype=np.int32,
)


def _ulaw2lin(data: bytes) -> bytes:
    """Decode µ-law bytes → 16-bit signed PCM bytes (vectorised)."""
    idx = np.frombuffer(data, dtype=np.uint8)
    return _ULAW2LIN_TABLE[idx].tobytes()


def _lin2ulaw(data: bytes) -> bytes:
    """Encode 16-bit signed PCM bytes → µ-law bytes (vectorised)."""
    CLIP = 32635
    BIAS = 0x84
    s = np.frombuffer(data, dtype=np.int16).astype(np.int32)
    sign = np.where(s < 0, 0x80, 0).astype(np.int32)
    s = np.minimum(np.abs(s), CLIP) + BIAS
    exp = _G711_EXP_LUT[(s >> 7) & 0xFF]
    mantissa = (s >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mantissa)).astype(np.uint8).tobytes()


def _tomono(data: bytes) -> bytes:
    """Stereo interleaved 16-bit PCM → mono by averaging both channels."""
    stereo = np.frombuffer(data, dtype=np.int16).reshape(-1, 2).astype(np.int32)
    mono = np.clip((stereo[:, 0] + stereo[:, 1] + 1) >> 1, -32768, 32767).astype(np.int16)
    return mono.tobytes()


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
                ulaw = _lin2ulaw(frame.audio)
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

    Asterisk connects from your PBX host to the Media server on the DGX when a call enters
    Stasis. Asterisk uses websocket_client.conf: set [media_connection1] uri to
    ws://DGX_IP:8787/media. external_host must be the section name, not an IP.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        app_name: str = "ai-assistant",
        media_host: str = "0.0.0.0",
        media_port: int | None = None,
        params: TransportParams | None = None,
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
        sid = ch["id"]
        is_external_media = sid in self._sessions and self._sessions.get(sid, {}).get("ws_channel") == sid
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
            await self._ari_send_request(
                "POST",
                "channels/externalMedia",
                query_strings=[
                    {"name": "channelId", "value": ws_channel},
                    {"name": "app", "value": msg.get("application", self._app_name)},
                    {"name": "data", "value": "websocket"},
                    {"name": "external_host", "value": external_host},
                    {"name": "transport", "value": "websocket"},
                    {"name": "encapsulation", "value": "none"},
                    {"name": "format", "value": "ulaw"},
                ],
            )
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
        # Channel/bridge DELETE can 404 if Asterisk already destroyed them (caller hung up)

    async def _on_channel_destroyed(self, msg):
        pass

    async def _create_bridge_and_answer(self, sess):
        """Create mixing bridge and add channels.

        Order per asterisk-websocket-examples: bridge, add both, then answer.
        """
        if sess.get("bridge_id"):
            return  # Already bridged
        bridge_id = str(uuid.uuid4())
        sess["bridge_id"] = bridge_id
        in_name = sess.get("incoming_name", "?")
        ws_name = sess.get("ws_channel_name", "?")
        logger.info(f"Creating bridge {bridge_id} for {in_name} <-> {ws_name}")
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
                logger.info("Dial ANSWER for WebSocket channel, creating bridge")
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
        base = self._uri.rstrip("/")
        app = self._app_name
        key = f"{self._username}:{self._password}"
        ari_url = f"{base}/ari/events?subscribeAll=false&app={app}&api_key={key}"
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
                            if msg.get("type") == "RESTResponse":
                                resp = msg.get("response") or msg
                                status = resp.get("status_code")
                                uri = resp.get("uri", "")
                                if status is not None and status >= 400:
                                    # 404 on channel/bridge DELETE is expected when caller hung up
                                    if status == 404 and ("channels/" in uri or "bridges/" in uri):
                                        logger.debug(f"ARI 404 on cleanup (channel/bridge already gone): {uri}")
                                    else:
                                        logger.error(f"ARI REST failed: status={status} uri={uri} msg={msg}")
                                continue
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
        in_chunks = 0
        if self._output_proc:
            self._output_proc._ws = ws
            self._output_proc._buffering_started = False
            self._output_proc._frames_received = 0
            self._output_proc._ulaw_buffer.clear()
            self._output_proc._last_send_time = 0.0
            self._output_proc._bytes_sent = 0
            self._output_proc._pcm_resampler = create_stream_resampler()
            self._output_proc._prepended_outbound_warmup = False
            self._output_proc._cancel_pending_stop_buffering()
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
            in_resampler = create_stream_resampler()
            async for message in ws:
                if isinstance(message, str):
                    # MEDIA_START / flow-control text frames; audio is binary only below
                    if "MEDIA_XOFF" in message or "MEDIA_XON" in message:
                        continue
                    continue
                if self._input_proc:
                    in_chunks += 1
                    pcm = _ulaw2lin(message)
                    if PIPELINE_SAMPLE_RATE != ULAW_SAMPLE_RATE:
                        pcm = await in_resampler.resample(pcm, ULAW_SAMPLE_RATE, PIPELINE_SAMPLE_RATE)
                    frame = InputAudioRawFrame(audio=pcm, sample_rate=PIPELINE_SAMPLE_RATE, num_channels=1)
                    await self._input_proc.queue_frame(frame)
        except Exception as e:
            logger.error(f"Media error: {e}")
        finally:
            if self._output_proc:
                self._output_proc._cancel_pending_stop_buffering()
                try:
                    await self._output_proc._flush_ulaw_tail_frame()
                except Exception:
                    pass
                leftover = len(self._output_proc._ulaw_buffer)
                if leftover:
                    logger.warning(
                        f"ARI output: media session ended with {leftover} unsent μ-law bytes "
                        "(hangup or disconnect before paced playback finished)"
                    )
                await self._output_proc.reset_bot_speaking_state()
            if self._input_proc:
                await self._input_proc.queue_frame(EndFrame())
            op = self._output_proc
            sent = getattr(op, "_bytes_sent", 0) if op else 0
            frames = getattr(op, "_frames_received", 0) if op else 0
            logger.info(
                f"Media WebSocket disconnected (inbound μlaw chunks from Asterisk: {in_chunks}, "
                f"downstream TTS frames: {frames}, sent {sent} bytes to Asterisk)"
            )
            if in_chunks < 25:
                logger.warning(
                    "Very few inbound audio chunks from Asterisk — the phone→PBX path may not be sending "
                    "audio to this WebSocket (RTP, mic, or bridge). STT will not hear the caller."
                )
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


def _ulaw_high_water_bytes() -> int:
    """Backlog threshold: above this, drain faster than real-time to shrink the queue.

    Piper often pushes TTS faster than 8 kHz playback; strict 20 ms pacing builds a
    multi-second buffer. If the caller hangs up, unsent bytes are lost — catch-up
    keeps typical backlog closer to this many ms of audio.
    """
    try:
        ms = float(os.getenv("ARI_ULAW_HIGH_WATER_MS", "280"))
    except ValueError:
        ms = 280.0
    ms = max(float(2 * ULAW_FRAME_MS), ms)
    return int(8000 * (ms / 1000.0))


# ~240ms of 8kHz silence (RTP/bridge + Asterisk buffer; reduces clipped first syllables).
# µ-law encodes 0 PCM as 0xFF — build directly without PCM round-trip.
_ULAW_WARMUP_SILENCE = bytes([0xFF] * (ULAW_FRAME_SIZE * 12))
# Debounce STOP_MEDIA_BUFFERING so back-to-back Piper phrases ("Hello!" + "How can I help…")
# stay in one buffer session; immediate STOP/START between phrases clips the next phrase start.
_STOP_BUFFERING_DEBOUNCE_S = 0.22
# After TTSStoppedFrame, wait this long before BotStoppedSpeakingFrame so multi-sentence
# Piper output does not briefly unmute between phrases (echo would reach VAD/STT).
_BOT_STOP_AFTER_TTS_DEBOUNCE_S = float(os.getenv("BOT_STOP_AFTER_TTS_DEBOUNCE_S", "0.45"))


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
        self._buffering_started = False
        # SOXR resampler: reset on each TTSStopped phrase so the filter delay is flushed (not
        # discarded). Pipecat auto-clears the stream after 0.2 s of inactivity via
        # _soxr_stream.clear(), which discards buffered tail samples — causing the last ~5-30 ms of
        # each phrase to be silently dropped. We flush explicitly instead (see _flush_pcm_resampler).
        self._pcm_resampler = create_stream_resampler()
        self._prepended_outbound_warmup = False
        self._stop_buffering_task: asyncio.Task | None = None
        # Pipecat BaseOutputTransport emits these; we need them for STTMuteFilter / user mute
        # so phone echo during TTS is not treated as the caller speaking.
        self._bot_out_active = False
        self._bot_stop_task: asyncio.Task | None = None

    def _cancel_bot_stop_task(self):
        t = self._bot_stop_task
        if t and not t.done():
            t.cancel()
        self._bot_stop_task = None

    async def reset_bot_speaking_state(self):
        """Call when the media WebSocket ends so Pipecat does not stay 'bot speaking' / muted."""
        self._cancel_bot_stop_task()
        if self._bot_out_active:
            self._bot_out_active = False
            await self.broadcast_frame(BotStoppedSpeakingFrame)

    async def _emit_bot_started(self):
        if self._bot_out_active:
            return
        self._bot_out_active = True
        await self.broadcast_frame(BotStartedSpeakingFrame)

    async def _emit_bot_stopped(self):
        if not self._bot_out_active:
            return
        self._bot_out_active = False
        await self.broadcast_frame(BotStoppedSpeakingFrame)

    async def _debounced_bot_stopped_after_tts(self):
        try:
            await asyncio.sleep(_BOT_STOP_AFTER_TTS_DEBOUNCE_S)
            await self._emit_bot_stopped()
        except asyncio.CancelledError:
            pass
        finally:
            self._bot_stop_task = None

    async def _flush_pcm_resampler(self) -> bytes:
        """Flush the SOXR filter delay so phrase-ending samples are not lost.

        The SOXRStreamAudioResampler holds ~half-filter-length samples inside its
        ResampleStream. If we don't drain them, pipecat's 0.2 s auto-clear discards
        them silently, clipping the last few milliseconds of every spoken phrase.

        Push a short silence block through the resampler to force the held samples
        out, then reset the resampler so the next phrase starts with a clean filter.
        The silence also pads any sub-frame tail to a clean sample boundary.
        """
        # 30 ms at 16 kHz (480 samples) is well above any SOXR HQ filter delay.
        silence = b"\x00\x00" * 480
        try:
            tail_pcm = await self._pcm_resampler.resample(silence, PIPELINE_SAMPLE_RATE, ULAW_SAMPLE_RATE)
        except Exception as exc:
            logger.debug(f"ARI output: resampler flush failed: {exc}")
            tail_pcm = b""
        finally:
            # Fresh filter for the next phrase; avoids stale state between sentences.
            self._pcm_resampler = create_stream_resampler()
        return tail_pcm

    def _cancel_pending_stop_buffering(self):
        t = self._stop_buffering_task
        if t and not t.done():
            t.cancel()
        self._stop_buffering_task = None

    def set_client_connection(self, ws):
        self._ws = ws

    async def _debounced_stop_media_buffering(self):
        try:
            await asyncio.sleep(_STOP_BUFFERING_DEBOUNCE_S)
            if self._buffering_started and self._ws and self._ws.state == 1:
                try:
                    await self._ws.send("STOP_MEDIA_BUFFERING")
                except (ConnectionClosed, Exception):
                    pass
                self._buffering_started = False
        except asyncio.CancelledError:
            pass
        finally:
            self._stop_buffering_task = None

    async def _drain_ulaw_buffer(self):
        """Send full 160-byte μ-law frames; pace at 20 ms unless backlog exceeds high-water."""
        if not self._ws or self._ws.state != 1:
            return
        high = _ulaw_high_water_bytes()
        now = time.monotonic()
        while len(self._ulaw_buffer) >= ULAW_FRAME_SIZE:
            backlog = len(self._ulaw_buffer)
            if backlog <= high:
                elapsed = now - self._last_send_time
                if self._last_send_time > 0 and elapsed < ULAW_FRAME_MS / 1000:
                    await asyncio.sleep((ULAW_FRAME_MS / 1000) - elapsed)
            else:
                await asyncio.sleep(0)
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

    async def _flush_ulaw_tail_frame(self):
        """Pad a fractional tail to one 20 ms frame so phrase endings are not dropped.

        `_drain_ulaw_buffer` only sends multiples of 160 bytes; without this, the last
        few samples of each Piper segment could sit in `_ulaw_buffer` until the next
        utterance or until the WebSocket closes (truncated audio at hangup).
        """
        if not self._ws or self._ws.state != 1:
            return
        rem = len(self._ulaw_buffer) % ULAW_FRAME_SIZE
        if rem == 0:
            return
        pad_samples = ULAW_FRAME_SIZE - rem
        self._ulaw_buffer.extend(bytes([0xFF] * pad_samples))  # µ-law silence = 0xFF
        await self._drain_ulaw_buffer()

    async def _send_ulaw(self, ulaw: bytes):
        """Append μ-law samples and send as many full 20 ms frames as possible."""
        if not self._ws or self._ws.state != 1:
            return
        if not self._prepended_outbound_warmup:
            self._ulaw_buffer.extend(_ULAW_WARMUP_SILENCE)
            self._prepended_outbound_warmup = True
        self._ulaw_buffer.extend(ulaw)
        await self._drain_ulaw_buffer()

    async def process_frame(self, frame, direction):
        # Must call super() first so base handles StartFrame and sets __started=True.
        # Otherwise push_frame rejects all frames via _check_started().
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return []
        if isinstance(frame, TTSStoppedFrame):
            # Drain SOXR filter delay first, then pad the µ-law tail frame.
            tail_pcm = await self._flush_pcm_resampler()
            if tail_pcm:
                await self._send_ulaw(_lin2ulaw(tail_pcm))
            await self._flush_ulaw_tail_frame()
            self._cancel_pending_stop_buffering()
            self._stop_buffering_task = asyncio.create_task(self._debounced_stop_media_buffering())
            # Debounce: Piper often sends TTSStopped between sentences in one reply.
            self._cancel_bot_stop_task()
            self._bot_stop_task = asyncio.create_task(self._debounced_bot_stopped_after_tts())
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
        self._cancel_bot_stop_task()
        await self._emit_bot_started()
        self._cancel_pending_stop_buffering()
        self._frames_received += 1
        if self._frames_received == 1:
            nbytes = len(frame.audio)
            sr = getattr(frame, "sample_rate", "?")
            logger.info(f"ARI output: first OutputAudioRawFrame received (size={nbytes} bytes, sr={sr})")
        if not self._buffering_started and self._ws and self._ws.state == 1:
            try:
                await self._ws.send("START_MEDIA_BUFFERING")
                self._buffering_started = True
            except (ConnectionClosed, Exception):
                pass
        try:
            pcm = frame.audio
            sr = getattr(frame, "sample_rate", None) or PIPELINE_SAMPLE_RATE
            nch = getattr(frame, "num_channels", None) or 1
            if len(pcm) == 0:
                logger.warning("ARI output: received empty audio frame")
            else:
                if nch == 2:
                    pcm = _tomono(pcm)
                    nch = 1
                if sr != ULAW_SAMPLE_RATE:
                    pcm = await self._pcm_resampler.resample(pcm, sr, ULAW_SAMPLE_RATE)
                ulaw = _lin2ulaw(pcm)
                await self._send_ulaw(ulaw)
        except Exception as e:
            logger.exception(f"ARI output: error processing audio frame: {e}")
        # Forward frame to assistant aggregator
        await self.push_frame(frame, direction)
        return []
