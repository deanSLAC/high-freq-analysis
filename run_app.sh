#!/bin/bash
# ME-DAQ Raw Pickle Viewer Launcher

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
    source ../venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Pick a bind address: BIND_ADDR env var > beamline IP if locally assigned > localhost.
PORT="${PORT:-5009}"
BEAMLINE_IP="192.168.150.13"

if [ -n "$BIND_ADDR" ]; then
    echo "Serving on $BIND_ADDR (from BIND_ADDR env)"
elif ip -4 addr show 2>/dev/null | grep -qw "$BEAMLINE_IP" || \
     ifconfig 2>/dev/null | grep -qw "$BEAMLINE_IP"; then
    BIND_ADDR="$BEAMLINE_IP"
    echo "Serving on $BIND_ADDR (beamline IP detected)"
else
    BIND_ADDR="localhost"
    echo "Beamline IP not assigned here, serving on localhost"
fi

WS_ORIGIN_PRIMARY="${BIND_ADDR}:${PORT}"
if [ "$BIND_ADDR" = "$BEAMLINE_IP" ]; then
    WS_ORIGIN_PRIMARY="192.168.150.*"
fi

panel serve app.py --show --port "$PORT" --address "$BIND_ADDR" \
    --allow-websocket-origin="$WS_ORIGIN_PRIMARY" \
    --allow-websocket-origin="localhost:$PORT" \
    --static-dirs assets=./static
