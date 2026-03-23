# 🎙️ Sentinel-Voice: On-Premise Multimodal AI Telephony

A high-performance, private AI voice agent bridging **Asterisk IP Telephony** with **NVIDIA DGX Spark** inference. This system enables real-time, low-latency conversational AI with full tool-calling capabilities, running entirely on your local hardware.

## 🏗️ System Architecture

The project implements a "Split-Plane" architecture across two machines:

* **Control Plane (Mac):** Handles SIP signaling, RTP media streams, and the ARI (Asterisk REST Interface) gateway via Docker. **Do not run the agent on the Mac.**
* **Data Plane (DGX Spark):** Runs the AI voice agent—Whisper, vLLM, Piper—orchestrated by Pipecat. **The agent must run on the DGX Spark only**, leveraging the Grace Blackwell architecture for ultra-fast TTFT (Time to First Token).

```mermaid
graph LR
    A[SIP Phone] -->|SIP/RTP| B(Mac: Asterisk 22)
    B -->|ARI Events| C(DGX Spark: AI Agent)
    C -->|vLLM Tools| D[Company DB/API]
    C -->|AudioStream| B
```

---

## 📁 Project Structure

```text
telephony/
├── docker-compose.yaml      # Asterisk PBX Stack (Mac)
├── config/                  # Asterisk Configurations (Mounted to Mac)
│   ├── ari.conf             # ARI Auth & App definitions
│   ├── extensions.conf      # Dialplan for Stasis(ai-assistant)
│   ├── http.conf            # ARI Port 8088 bindings
│   └── modules.conf         # Loadable Asterisk plugins
└── agent/                   # AI Orchestrator — run on DGX Spark only
    ├── pyproject.toml       # Managed by uv
    ├── uv.lock              # Deterministic dependencies
    ├── .env.example         # Template (copy to .env on DGX Spark)
    ├── .env                 # Secrets — set MAC_IP to Mac's address
    └── src/
        └── voice_agent/     # Namespaced source package
            ├── main.py      # Entrypoint & runner
            ├── pipeline.py  # Pipecat flow orchestration
            ├── services/    # STT/LLM/TTS Factory (Blackwell optimized)
            └── tools/       # Function handlers (e.g. Order Lookup)
```

---

## 🚀 Quick Start

### 1. Gateway Setup (Mac)
Ensure your `config/` folder contains the required `.conf` files, then launch the gateway:
```bash
docker compose up -d
```
*Verify with:* `curl -v -u ai_user:your_password http://localhost:8088/ari/asterisk/info`

### 2. Inference Engine Setup (DGX Spark only)
**Run these commands on the DGX Spark machine—not on the Mac.** The agent must run on DGX Spark for GPU-accelerated STT/LLM/TTS.

Copy `agent/.env.example` to `agent/.env` and set `MAC_IP` to your Mac's IP. Ensure **vLLM** is running on port `8000` on the DGX Spark with your chosen model (e.g., `Qwen2-VL-7B`).

```bash
cd agent
uv sync
uv run main.py
```

Environment variables in `agent/.env` (on the DGX Spark):
| Variable | Default | Description |
|----------|---------|-------------|
| `MAC_IP` | `192.168.1.23` | IP of the Mac running Asterisk (agent connects to this) |
| `ARI_USER` / `ARI_PASS` | — | ARI credentials for the Mac gateway |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM API (localhost = DGX Spark) |
| `LLM_MODEL` | `Qwen2-VL-7B-Instruct` | Model name |
| `WHISPER_DEVICE` | `cuda` | `cuda` for DGX, `cpu` only for non-GPU testing |
| `PIPER_USE_CUDA` | `true` | GPU acceleration for TTS on DGX |
| `PIPER_VOICE` | `en_US-ryan-high` | Piper TTS voice |
| `SPARK_IP` / `DGX_IP` | — | Spark IP so Asterisk sends audio here (e.g. `192.168.1.50`) |
| `ARI_MEDIA_PORT` | `8787` | Media WebSocket port on DGX |

### 3. Place a Call
Using **Linphone** (or Zoiper):

| Setting   | Value                         |
|-----------|-------------------------------|
| Server    | `MAC_IP` (e.g. 192.168.1.23)  |
| Port      | 5060                          |
| Username  | 6001                          |
| Password  | password123                   |

Register, then dial extension **600** to reach the AI assistant.

---

## 🛠️ Advanced Features

### ⚡ Blackwell Optimization
The agent is configured to use **NVFP4** quantization via vLLM, reducing memory footprint on the DGX Spark while maintaining intelligence, resulting in a ~35% performance boost over FP8.

### 🛑 Intelligent Barge-in
Uses Pipecat's **VAD (Voice Activity Detection)** to detect when a user interrupts the AI. The system immediately halts the current audio playback and triggers a new inference cycle.

### 🔧 Extensible Tools
Add new business logic to `src/voice_agent/tools/handlers.py`. The AI can:
* Check order statuses via local DB.
* Transfer calls to human queues via ARI bridges.
* Record and summarize calls locally.

---

## 🔒 Security & Privacy
* **No Cloud Egress:** All audio processing (Whisper), reasoning (vLLM), and synthesis (Piper) happen on the DGX Spark.
* **SIP Security:** Configure TLS/SRTP in the `config/` directory for encrypted telephony.

---

## 📜 License
Proprietary. Internal Use Only.
