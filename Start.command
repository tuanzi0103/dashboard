#!/bin/bash

# Fix permissions for this script
chmod +x "$0" 2>/dev/null

# Move to script directory
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PORT=8620

# -------------------------------
# 1) Find python
# -------------------------------
if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    osascript -e 'display alert "Python Missing" message "Please install Python3 from python.org"'
    exit 1
fi

# -------------------------------
# 2) Create venv if missing
# -------------------------------
if [ ! -d ".venv" ]; then
    echo "🐍 Creating virtual environment..."
    "$PY" -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# -------------------------------
# 3) Install dependencies
# -------------------------------
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# -------------------------------
# 4) Kill ANY process using the port (guaranteed fix)
# -------------------------------
PORT_PID=$(lsof -ti tcp:$PORT)
if [ -n "$PORT_PID" ]; then
    echo "🔪 Killing process using port $PORT (PID: $PORT_PID)"
    kill -9 $PORT_PID 2>/dev/null
    sleep 1
fi

# -------------------------------
# 5) Start streamlit in background
# -------------------------------
echo "🚀 Starting Manly Dashboard..."
nohup streamlit run app.py --server.port=$PORT --server.headless=true > streamlit.log 2>&1 &

# -------------------------------
# 6) WAIT for streamlit to start
# -------------------------------
echo "⏳ Waiting for server to start..."
for i in {1..30}
do
    if lsof -i tcp:$PORT >/dev/null 2>&1; then
        echo "🌐 Server is up!"
        break
    fi
    sleep 1
done

# If still not running:
if ! lsof -i tcp:$PORT >/dev/null 2>&1; then
    echo "❌ Streamlit failed to start. Check streamlit.log."
    exit 1
fi

# -------------------------------
# 7) Open browser
# -------------------------------
open "http://localhost:$PORT"

# -------------------------------
# 8) KEEP TERMINAL OPEN
# -------------------------------
echo
echo "=============================="
echo "Manly Dashboard is running!"
echo "Do NOT close this window."
echo "=============================="
while true; do sleep 3600; done
