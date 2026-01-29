#!/bin/bash
# Simple load test to trigger EPLB logging
# Usage: ./load_test.sh [num_requests] [port] [model]

NUM_REQUESTS="${1:-200}"
PORT="${2:-8000}"
MODEL="${3:-deepseek-ai/DeepSeek-V2-Lite}"
BASE_URL="http://localhost:$PORT"

echo "Running EPLB load test..."
echo "  Requests: $NUM_REQUESTS"
echo "  Endpoint: $BASE_URL"
echo "  Model: $MODEL"
echo ""

# Warmup
echo "Warmup request..."
curl -s "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"Hello\", \"max_tokens\": 5}" > /dev/null

echo "Starting load test (log_balancedness_interval=100, so EPLB logs every 100 steps)..."
echo ""

for i in $(seq 1 $NUM_REQUESTS); do
  curl -s "$BASE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Explain AI in one sentence.\", \"max_tokens\": 50}" > /dev/null &
  
  # Print progress every 20 requests
  if [ $((i % 20)) -eq 0 ]; then
    echo "  Sent $i/$NUM_REQUESTS requests..."
  fi
  
  # Limit concurrency to avoid overwhelming
  if [ $((i % 10)) -eq 0 ]; then
    wait
  fi
done

wait
echo ""
echo "Load test complete. Check server logs for EPLB balancedness output:"
echo "  EPLB step: X: avg_tokens=..., max_tokens=..., balancedness=..."
