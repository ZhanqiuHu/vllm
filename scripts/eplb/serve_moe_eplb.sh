#!/bin/bash
# Serve a MoE model with EPLB (Expert-Parallel Load Balancing) enabled
# Usage: ./serve_moe_eplb.sh [model] [tp_size] [port]
# Uses gr (GPU reservation) to allocate GPUs

MODEL="${1:-deepseek-ai/DeepSeek-V2-Lite}"
TP_SIZE="${2:-2}"
PORT="${3:-8000}"

echo "Starting MoE server with EPLB..."
echo "  Model: $MODEL"
echo "  TP Size: $TP_SIZE (using gr $TP_SIZE)"
echo "  Port: $PORT"

gr $TP_SIZE vllm serve "$MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config '{"log_balancedness": true, "log_balancedness_interval": 100, "step_interval": 1000}' \
  --port "$PORT"
