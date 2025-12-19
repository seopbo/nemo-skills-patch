#!/bin/bash
# Eval script for conversation behavior using output.jsonl format
# Usage: ./eval_conv_v2.sh <results_dir> [options]
#
# Recommended container: gitlab-master.nvidia.com/pzelasko/nemo_containers:25.04-pytorch2.7-28may25
# This container has torchaudio pre-installed which is required for VAD.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default parameters
BARGE_IN_THRESHOLD_SEC=1.5
TT_LATENCY_THRESHOLD_SEC=1.5
TT_PRECISION_BUFFER_SEC=1.0
TT_RECALL_BUFFER_SEC=20.0
VAD_MIN_SILENCE_MS=2000
END_TIME="None"

# Docker container with torchaudio support
DOCKER_IMAGE="gitlab-master.nvidia.com/pzelasko/nemo_containers:25.04-pytorch2.7-28may25"

# Parse arguments
RESULTS_DIR=${1:-""}
VERBOSE=${2:-"--verbose"}
DISABLE_TRANSCRIPTION=${3:-""}
USE_DOCKER=${4:-"true"}  # Set to "false" to run locally

if [ -z "$RESULTS_DIR" ]; then
    echo "Usage: $0 <results_dir> [--verbose] [--disable_transcription] [use_docker=true|false]"
    echo ""
    echo "Example (with Docker - recommended):"
    echo "  $0 /tmp/eval_experiment/s2s_demo.demo_20251124"
    echo ""
    echo "Example (local, requires torchaudio):"
    echo "  $0 /path/to/results --verbose \"\" false"
    echo ""
    echo "Arguments:"
    echo "  results_dir           Directory containing output.jsonl and audio/"
    echo "  --verbose             Print detailed segment information (default: enabled)"
    echo "  --disable_transcription  Disable ASR for user segments"
    echo "  use_docker            Use Docker container (default: true)"
    exit 1
fi

# Build extra args
EXTRA_ARGS=""
if [ -n "$VERBOSE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS $VERBOSE"
fi
if [ -n "$DISABLE_TRANSCRIPTION" ]; then
    EXTRA_ARGS="$EXTRA_ARGS $DISABLE_TRANSCRIPTION"
fi

# Create output log path
OUTPUT_LOG="${RESULTS_DIR}/eval_conv_v2.log"

echo "========================================"
echo "Conversation Behavior Evaluation (v2)"
echo "========================================"
echo "Results directory: $RESULTS_DIR"
echo "Output log: $OUTPUT_LOG"
echo "Parameters:"
echo "  - Barge-in threshold: ${BARGE_IN_THRESHOLD_SEC}s"
echo "  - TT latency threshold: ${TT_LATENCY_THRESHOLD_SEC}s"
echo "  - TT precision buffer: ${TT_PRECISION_BUFFER_SEC}s"
echo "  - TT recall buffer: ${TT_RECALL_BUFFER_SEC}s"
echo "  - VAD min silence: ${VAD_MIN_SILENCE_MS}ms"
echo "========================================"

if [ "$USE_DOCKER" = "true" ]; then
    echo "Using Docker container: $DOCKER_IMAGE"

    # Source HF token if available
    [ -f ~/.env ] && source ~/.env

    docker run --rm --gpus all --ipc=host \
        -e HF_TOKEN=${HF_READ_ONLY:-$HF_TOKEN} \
        -v "$(dirname $RESULTS_DIR):/data" \
        -v "$SCRIPT_DIR:/workspace/scripts" \
        -w /workspace/scripts \
        $DOCKER_IMAGE \
        python eval_conversation_behavior_v2.py \
            --results_dir "/data/$(basename $RESULTS_DIR)" \
            --barge_in_threshold_sec $BARGE_IN_THRESHOLD_SEC \
            --tt_latency_threshold_sec $TT_LATENCY_THRESHOLD_SEC \
            --tt_precision_buffer_sec $TT_PRECISION_BUFFER_SEC \
            --tt_recall_buffer_sec $TT_RECALL_BUFFER_SEC \
            --vad_min_silence_duration_ms $VAD_MIN_SILENCE_MS \
            --end_time "$END_TIME" \
            $EXTRA_ARGS 2>&1 | tee "$OUTPUT_LOG"
else
    echo "Running locally (requires torchaudio)"
    python "$SCRIPT_DIR/eval_conversation_behavior_v2.py" \
        --results_dir "$RESULTS_DIR" \
        --barge_in_threshold_sec $BARGE_IN_THRESHOLD_SEC \
        --tt_latency_threshold_sec $TT_LATENCY_THRESHOLD_SEC \
        --tt_precision_buffer_sec $TT_PRECISION_BUFFER_SEC \
        --tt_recall_buffer_sec $TT_RECALL_BUFFER_SEC \
        --vad_min_silence_duration_ms $VAD_MIN_SILENCE_MS \
        --end_time "$END_TIME" \
        $EXTRA_ARGS 2>&1 | tee "$OUTPUT_LOG"
fi

echo ""
echo "Evaluation complete. Log saved to: $OUTPUT_LOG"
