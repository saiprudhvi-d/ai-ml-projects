#!/bin/bash
# ─── Start All 8 AI Project APIs ─────────────────────────────────────────────
# Usage: ./start_all.sh [--demo]   (--demo uses tiny fallback models, no GPU needed)
# Logs: ./logs/<project>.log

set -e

DEMO_MODE=false
[[ "$1" == "--demo" ]] && DEMO_MODE=true

PROJECTS=(
  "01_qa_bot:8000"
  "02_contract_analysis:8001"
  "03_code_suggestion:8002"
  "04_empathetic_response:8003"
  "05_meeting_summarizer:8004"
  "06_review_intelligence:8005"
  "07_rag_knowledge_base:8006"
  "08_personal_chat_assistant:8007"
)

mkdir -p logs
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "╔══════════════════════════════════════════╗"
echo "║       🚀  AI Projects Launcher           ║"
echo "║  8 FastAPI services + Streamlit dashboard ║"
echo "╚══════════════════════════════════════════╝"
echo ""
[[ "$DEMO_MODE" == "true" ]] && echo "⚡ DEMO MODE: using small fallback models"
echo ""

# ─── Optional: create a virtual environment ──────────────────────────────────
if [[ ! -d "$BASE_DIR/.venv" ]]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv "$BASE_DIR/.venv"
fi
source "$BASE_DIR/.venv/bin/activate"

# ─── Install base deps ────────────────────────────────────────────────────────
echo "📦 Installing shared dependencies..."
pip install fastapi uvicorn streamlit requests transformers torch \
    sentence-transformers langchain langchain-community \
    faiss-cpu chromadb peft datasets accelerate \
    openai-whisper pypdf python-multipart plotly pandas scikit-learn \
    --quiet --break-system-packages 2>/dev/null || true

# ─── Launch each service ──────────────────────────────────────────────────────
PIDS=()
for entry in "${PROJECTS[@]}"; do
  proj="${entry%%:*}"
  port="${entry##*:}"
  dir="$BASE_DIR/$proj"

  if [[ ! -d "$dir" ]]; then
    echo "⚠️  Skipping $proj (directory not found)"
    continue
  fi

  echo "▶️  Starting $proj on :$port"
  (
    cd "$dir"
    [[ "$DEMO_MODE" == "true" ]] && export DEMO_MODE=true
    uvicorn api:app --host 0.0.0.0 --port "$port" \
      >> "$BASE_DIR/logs/$proj.log" 2>&1
  ) &
  PIDS+=($!)
  sleep 1   # Stagger starts to avoid HF cache collisions
done

echo ""
echo "✅ All services launching. Waiting for health checks..."
sleep 5

# ─── Health check ─────────────────────────────────────────────────────────────
echo ""
echo "📊 Service Status:"
for entry in "${PROJECTS[@]}"; do
  proj="${entry%%:*}"
  port="${entry##*:}"
  if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
    echo "  🟢  $proj  →  http://localhost:$port"
  else
    echo "  🔴  $proj  →  still loading (check logs/$proj.log)"
  fi
done

# ─── Launch dashboard ─────────────────────────────────────────────────────────
echo ""
echo "🖥️  Starting unified dashboard..."
(
  cd "$BASE_DIR"
  streamlit run dashboard.py --server.port 8501 \
    >> "$BASE_DIR/logs/dashboard.log" 2>&1
) &
PIDS+=($!)

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Dashboard → http://localhost:8501       ║"
echo "║  Press Ctrl+C to stop all services       ║"
echo "╚══════════════════════════════════════════╝"

# ─── Graceful shutdown on Ctrl+C ─────────────────────────────────────────────
cleanup() {
  echo ""
  echo "🛑 Shutting down all services..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  pkill -f "uvicorn api:app" 2>/dev/null || true
  pkill -f "streamlit run dashboard.py" 2>/dev/null || true
  echo "✅ All services stopped."
}
trap cleanup EXIT INT TERM

wait
