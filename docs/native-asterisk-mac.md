# Running Asterisk Natively on Mac (No Docker)

Use this when the SIP client (e.g. **Telephone** app) runs on the same Mac as Asterisk. Docker’s networking can cause a wrong Contact (192.168.65.1) and no RTP audio. Running Asterisk natively fixes this.

## Prerequisites

- SIP client (Telephone, Linphone, etc.) on the Mac
- Asterisk 20.16+ or 22.x (for WebSocket / `chan_websocket`)

## Install Asterisk on Mac

Asterisk is not in the main Homebrew repo. **Build from source** (the `leedm777/asterisk` tap is broken with recent Homebrew due to deprecated `devel` syntax):

```bash
# 1. Dependencies
brew install automake berkeley-db curl git jansson libedit libtool libuuid pkg-config sqlite speexdsp

# 2. Download Asterisk 22 (or 20.16+ for chan_websocket)
curl -LO https://downloads.asterisk.org/pub/telephony/asterisk/asterisk-22-current.tar.gz
tar xzf asterisk-22-current.tar.gz
cd asterisk-22*

# 3. Configure & build
./bootstrap.sh
./configure
make menuselect   # Optional: enable chan_websocket, res_websocket, res_websocket_client (usually on by default in 22.x)
make
sudo make install
```

## Run Asterisk with project config

1. **Stop Docker Asterisk** (if it was running):

   ```bash
   docker compose down
   ```

2. **Run Asterisk natively**:

   ```bash
   cd /path/to/telephony
   chmod +x scripts/run-asterisk-mac.sh
   ./scripts/run-asterisk-mac.sh
   ```

   The script uses `config/` as the config directory, including `extensions.conf`, `pjsip.conf`, `ari.conf`, `websocket_client.conf`, etc.

3. **Configure Telephone**:
   - Domain: `127.0.0.1` or `192.168.1.23` (your Mac’s IP)
   - Username: `6001`
   - Password: `password123` (from `pjsip.conf`)

## Test

- Dial **602** – Playback (RTP test, no WebSocket)
- Dial **601** – Echo test (WebSocket; requires `mow_echo_test_server.py` on DGX)
- Dial **600** – AI agent (requires agent running on DGX)

## Switching back to Docker

```bash
# Stop native Asterisk (Ctrl+C)
# Start Docker:
docker compose up -d
```

## Notes

- **ARI**: The agent on the DGX connects to `ASTERISK_IP:8088`. Ensure your Asterisk host firewall allows port 8088 from the DGX.
- **WebSocket media**: `websocket_client.conf` points to the DGX for the agent. For the echo test, the script uses `[echo_test]` which points to the DGX as well.
- **External addresses**: `pjsip.conf` uses `external_signaling_address` and `external_media_address`. Update `192.168.1.23` if your Mac’s IP differs.
