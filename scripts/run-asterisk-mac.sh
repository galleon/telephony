#!/bin/bash
# Run Asterisk natively on Mac using project config.
# Fixes RTP when Telephone (or other SIP client) runs on the same Mac - no Docker bridge.
#
# Prereqs: Asterisk installed (tap: brew tap leedm777/asterisk && brew install asterisk,
#          or build from source). Then: ./scripts/run-asterisk-mac.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/config"

# Generate asterisk.conf with resolved config path (asterisk doesn't expand vars)
sed "s|/REPLACED_BY_SCRIPT|$CONFIG_DIR|" "$CONFIG_DIR/asterisk.conf" > "$CONFIG_DIR/asterisk.conf.generated"

echo "Starting Asterisk (config: $CONFIG_DIR)"
exec asterisk -C "$CONFIG_DIR/asterisk.conf.generated" -f
