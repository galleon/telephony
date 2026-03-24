# Echo Test: Isolate No-Audio Issues

If you have no sound on calls, this test checks whether the Asterisk ↔ WebSocket ↔ audio path works. If the echo test plays audio, the issue is in our agent. If not, it's Asterisk config, RTP, or the phone.

## Prerequisites

- Asterisk running on the **Linux host** (Docker, e.g. `pbx-gateway`)
- DGX Spark reachable from that host (e.g. `192.168.1.72`)

## Step 1: Stop the agent

```bash
cd agent && docker compose stop agent
```

This frees port 8787 on the DGX for the echo server.

## Step 2: Clone and run the echo server on the DGX

On the **DGX Spark**:

```bash
cd ~
git clone https://github.com/asterisk/asterisk-websocket-examples.git
cd asterisk-websocket-examples
pip install websockets
```

Edit `mow_echo_test_server.py` line 130: change `"localhost"` to `"0.0.0.0"` so Asterisk (on the **Linux Asterisk host**) can connect:

```python
async with serve(process_media, "0.0.0.0", 8787, subprotocols=["media"],
```

Run it:

```bash
python mow_echo_test_server.py
```

Leave it running. It will play `test.ulaw` to any caller who connects.

## Step 3: Add echo test config on the Asterisk host

Edit `config/websocket_client.conf` – add a new section (or temporarily replace `media_connection1`):

```ini
[echo_test]
type = websocket_client
uri = ws://YOUR_DGX_IP:8787
protocols = media
username = medianame
password = mediapassword
connection_type = per_call_config
connection_timeout = 5000
reconnect_interval = 1000
reconnect_attempts = 3
tls_enabled = no
```

Replace `YOUR_DGX_IP` with the DGX’s IP (e.g. 192.168.1.72).

## Step 4: Add echo extension

Edit `config/extensions.conf` – add under `[internal]`:

```ini
exten => 601,1,NoOp(Echo test - WebSocket)
 same => n,Answer()
 same => n,Dial(WebSocket/echo_test/c(ulaw))
 same => n,Hangup()
```

## Step 5: Reload Asterisk

On the **Linux host** where Asterisk runs:

```bash
docker exec pbx-gateway asterisk -rx "core reload"
```

If Asterisk runs in a different container, use its name instead of `pbx-gateway`.

## Step 6: Call extension 601

From **Telephone** on your Mac (or Linphone, Zoiper), dial **601**. You should hear the test.ulaw audio (a short test clip).

- **If you hear it:** Asterisk ↔ WebSocket ↔ audio works. The problem is in the Sentinel agent (bridge logic, frame handling, etc.).
- **If you don’t:** Check firewall (port 8787 on DGX), RTP ports (10000–10100), NAT, and phone registration.

**Troubleshooting:** If no audio, verify the echo server logs "Media connected" when you dial 601. If not, Asterisk isn't connecting (firewall, wrong IP, or wrong `uri`). Ensure the echo server is bound to `0.0.0.0` and port 8787 is free. See [asterisk-websocket-examples](https://github.com/asterisk/asterisk-websocket-examples).

## Step 7: Restore agent and config

1. Stop the echo server (Ctrl+C).
2. Restore `websocket_client.conf` so `media_connection1` points back to the agent.
3. Remove or comment out the 601 extension if you added it.
4. Run `docker compose start agent`.
