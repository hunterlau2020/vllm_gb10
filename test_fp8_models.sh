#!/bin/bash
# FP8 model test script for vLLM on GB10
# Tests: basic inference via vllm bench latency
set -e

MODELS_DIR=${1:-/home/hunter/models}
OUTPUT_DIR=${2:-fp8_test_results}
export PATH="$PWD/.venv/bin:$PATH"
export NINJA_PATH="$PWD/.venv/bin/ninja"
MODEL_NAMES=(
    "Qwen-Qwen3.6-27B-FP8"
    "Qwen-Qwen3-30B-A3B-FP8"
    "nvidia-NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
)
mkdir -p "$OUTPUT_DIR"

echo "FP8 Model Tests - $(date)" | tee -a "$OUTPUT_DIR/summary.log"
echo "Models dir: $MODELS_DIR" | tee -a "$OUTPUT_DIR/summary.log"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
    if [ ! -d "$MODEL_PATH" ]; then
        echo "[SKIP] $MODEL_NAME not found" | tee -a "$OUTPUT_DIR/summary.log"
        continue
    fi

    echo "" | tee -a "$OUTPUT_DIR/summary.log"
    echo "============================================" | tee -a "$OUTPUT_DIR/summary.log"
    echo "  Testing: $MODEL_NAME" | tee -a "$OUTPUT_DIR/summary.log"
    echo "============================================" | tee -a "$OUTPUT_DIR/summary.log"

    OUTPUT_FILE="$OUTPUT_DIR/${MODEL_NAME//\//_}.log"

    # Use vllm bench latency for basic throughput/latency test
    # Note: this runs a single batch repeatedly (num-iters), not multiple prompts
    set +e
    .venv/bin/vllm bench latency \
        --model "$MODEL_PATH" \
        --dtype float16 \
        --enforce-eager \
        --trust-remote-code \
        --num-iters 3 \
        --num-iters-warmup 3 \
        --input-len 256 \
        --output-len 128 \
        --batch-size 1 \
        > "$OUTPUT_FILE" 2>&1
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[PASS] $MODEL_NAME" | tee -a "$OUTPUT_DIR/summary.log"
    else
        echo "[FAIL] $MODEL_NAME (exit code: $EXIT_CODE)" | tee -a "$OUTPUT_DIR/summary.log"
    fi
done

echo "" | tee -a "$OUTPUT_DIR/summary.log"
echo "============================================" | tee -a "$OUTPUT_DIR/summary.log"
echo "  All tests complete"
echo "============================================" | tee -a "$OUTPUT_DIR/summary.log"
cat "$OUTPUT_DIR/summary.log"
