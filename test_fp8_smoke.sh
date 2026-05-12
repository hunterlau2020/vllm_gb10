#!/bin/bash
# Quick FP8 smoke test - just load and run a few tokens
set -e

MODEL=$1
if [ -z "$MODEL" ]; then
    MODEL="/home/hunter/models/Qwen-Qwen3.6-27B-FP8"
fi
export PATH="$PWD/.venv/bin:$PATH"
export NINJA_PATH="$PWD/.venv/bin/ninja"

echo "Testing FP8 model: $MODEL"
echo "This may take a minute to load..."

.venv/bin/vllm bench latency \
    --model "$MODEL" \
    --dtype float16 \
    --enforce-eager \
    --trust-remote-code \
    --num-iters 3 \
    --input-len 128 \
    --output-len 64 \
    --batch-size 1 \
    2>&1

echo ""
echo "Exit code: $?"
