#!/bin/bash
# =============================================================================
# willygpt_local_test.sh — validate the FULL pipeline on your M4 Air (MPS)
# =============================================================================
# Purpose: prove the repo installs and every stage runs + that WillyGPT identity
# injects correctly — BEFORE you rent a GPU. This trains a deliberately tiny,
# DUMB model (~30-45 min on Apple Silicon). Do NOT expect good answers; expect a
# model that completes the pipeline and can say its name is WillyGPT sometimes.
#
# PLACEMENT: nanochat repo root, with build_willygpt_identity.py in ./willygpt/.
# RUN:  bash willygpt_local_test.sh
#  (or copy-paste the commands one by one into your terminal to watch each stage)
# =============================================================================
set -euo pipefail

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- env (CPU/MPS build of torch) ----
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
WANDB_RUN="${WANDB_RUN:-dummy}"

# ---- tokenizer (~30s on M-series) ----
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# ---- tiny base model (~25-35 min on M3/M4) ----
# d6, short context, full-attention (MPS has no FA3/sliding-window).
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=$WANDB_RUN
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

# ---- install WillyGPT identity, then SFT (~10 min) ----
python "$SCRIPT_DIR/willygpt/build_willygpt_identity.py" \
    --out "$SCRIPT_DIR/willygpt/willygpt_identity.jsonl" --repeat 3
cp "$SCRIPT_DIR/willygpt/willygpt_identity.jsonl" \
   "$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
echo "Installed identity: $(wc -l < "$NANOCHAT_BASE_DIR/identity_conversations.jsonl") conversations"

python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=$WANDB_RUN

# ---- smoke-test the chat + identity ----
echo ""
echo "=== identity check (expect it to mention WillyGPT; may be rough at this size) ==="
python -m scripts.chat_cli -i sft -p "Who are you?"          || true
python -m scripts.chat_cli -i sft -p "Who made you?"         || true
python -m scripts.chat_cli -i sft -p "What is the capital of France?" || true

# ---- OPTIONAL: tiny RL smoke test (slow on MPS; uncomment to exercise the path) ----
# python -m scripts.chat_rl --num-epochs=1 --examples-per-step=4 --num-samples=4 \
#     --max-new-tokens=64 --eval-every=10 --eval-examples=16 --save-every=10 --run=$WANDB_RUN
# python -m scripts.chat_cli -i rl -p "What is 12 + 9?" || true

echo ""
echo "Local pipeline OK. If you saw WillyGPT mention its name, identity injection works."
echo "Now you're ready to run the real thing on 8xH100: willygpt_run.sh"
