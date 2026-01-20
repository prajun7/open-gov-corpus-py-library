#!/usr/bin/env bash
# Bash-compatible environment settings for OpenGovCorpus
# Source this file in bash: source usage_examples/opengov_env.sh

# Use model-based generation (1 = enabled)
export OPENGOV_USE_LLM="1"

# Recommended model: a smaller Qwen chat variant for local runs; change to a HF name you have access to
export OPENGOV_MODEL_NAME="Qwen/Qwen-7B-Chat"

# Max tokens to generate per chunk (adjust if you see truncation)
export OPENGOV_MAX_NEW_TOKENS="512"

# Deterministic output for structured JSON
export OPENGOV_TEMPERATURE="0.0"
export OPENGOV_DO_SAMPLE="0"

# Pause between chunks to be polite to hosted endpoints (seconds)
export OPENGOV_SLEEP_SECONDS="0.5"

# Useful override if you prefer to disable LLM generation and use legacy fallback
# export OPENGOV_USE_LLM="0"
