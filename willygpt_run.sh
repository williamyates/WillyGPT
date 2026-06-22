#!/bin/bash
# =============================================================================
# willygpt_run.sh — full WillyGPT training pipeline for an 8xH100 node
# =============================================================================
# Config vs the stock speedrun (runs/speedrun.sh):
#   - depth 24 -> 26              (bigger; ~true GPT-2 grade)
#   - data:param ratio 8 -> 12    (properly trained, not undertrained)
#   - custom WillyGPT identity     (instead of karpathy's identity convos)
#   - ADD reinforcement-learning stage (GRPO on GSM8K) for math/reasoning
#   - checkpoint-safety notices after each stage (ephemeral box insurance)
#
# Estimated 8xH100 wall-clock: ~9h  (~6h pretrain + ~0.5h SFT + ~1.5h RL + eval)
# Estimated cost: ~$145 (RunPod ~$16/h) .. ~$216 (Lambda ~$24/h)
#
# PLACEMENT: put this file in the nanochat repo ROOT (next to runs/, scripts/).
#            put build_willygpt_identity.py in ./willygpt/ (next to this).
#
# LAUNCH (use a screen session — this runs for hours):
#   WANDB_RUN=willygpt screen -L -Logfile willygpt_run.log -S willy bash willygpt_run.sh
# or simplest:
#   bash willygpt_run.sh
# =============================================================================
set -euo pipefail

# ---- knobs you might tune -------------------------------------------------
DEPTH=26                 # model depth. 24=cheaper, 28=stronger+pricier
RATIO=12                 # data:param ratio. 8=undertrained, 12=compute-optimal, 16=Chinchilla-ish
DEVICE_BATCH_SIZE=8      # per-GPU micro-batch. If you OOM, drop to 4/2. If lots of free VRAM, try 12/16.
NUM_SHARDS=350           # pretraining shards to download (~33GB). d26@r12 needs ~285; 350 = margin.
RL_EPOCHS=1             # GRPO epochs on GSM8K. 1 is plenty for a first build.
NPROC=8                  # GPUs on the node
# ---------------------------------------------------------------------------

export OMP_NUM_THREADS=1
# Use RunPod's persistent volume (/workspace) if present so checkpoints survive a
# pod stop/restart; otherwise (Lambda etc.) use home. Override by exporting
# NANOCHAT_BASE_DIR before launch.
if [ -z "${NANOCHAT_BASE_DIR:-}" ]; then
  if [ -d /workspace ] && [ -w /workspace ]; then
    export NANOCHAT_BASE_DIR="/workspace/.cache/nanochat"
  else
    export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
  fi
fi
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
mkdir -p "$NANOCHAT_BASE_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$HOME/willygpt_artifacts"
mkdir -p "$ARTIFACT_DIR"

if [ -z "${WANDB_RUN:-}" ]; then WANDB_RUN=dummy; fi

# ---- helper: print a loud checkpoint-safety notice after each stage -------
# Does NOT auto-upload (we don't know your target), but gives you a copy-paste
# command. Run willygpt_save.sh from your LAPTOP to pull everything down.
checkpoint_notice () {
  local phase="$1"        # base | sft | rl
  local dir="$NANOCHAT_BASE_DIR/${phase}_checkpoints"
  echo ""
  echo "############################################################"
  echo "# CHECKPOINT SAVED: '$phase' stage complete."
  echo "# On disk at: $dir"
  echo "# BACK IT UP NOW (run on your LAPTOP, fill in user@host):"
  echo "#   rsync -avP USER@HOST:$dir  ./willygpt_backup/"
  echo "# (or use ./willygpt/willygpt_save.sh from your laptop)"
  echo "############################################################"
  echo ""
  # also snapshot the running report so far
  [ -f "$NANOCHAT_BASE_DIR/report/report.md" ] && \
    cp "$NANOCHAT_BASE_DIR/report/report.md" "$ARTIFACT_DIR/report_after_${phase}.md" 2>/dev/null || true
}

# ===========================================================================
# 0) Python env via uv
# ===========================================================================
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# fresh report (writes system info + start timestamp)
python -m nanochat.report reset

# ===========================================================================
# 1) Tokenizer
# ===========================================================================
# download first 8 shards (~2B chars) to train the tokenizer on
python -m nanochat.dataset -n 8
# kick off the rest of the download in the background while the tokenizer trains
python -m nanochat.dataset -n "$NUM_SHARDS" &
DL_PID=$!
# train + eval the 32768-vocab BPE tokenizer
python -m scripts.tok_train
python -m scripts.tok_eval

# ===========================================================================
# 2) Base model (pretraining)  ——  the expensive stage (~6h)
# ===========================================================================
echo "Waiting for dataset download to finish..."
wait $DL_PID

# --save-every gives resumable mid-run checkpoints (insurance against crashes).
# If pretrain dies, resume with: --resume-from-step=<last saved step>
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    --depth=$DEPTH \
    --target-param-data-ratio=$RATIO \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --fp8 \
    --save-every=2500 \
    --run=$WANDB_RUN
checkpoint_notice base

# evaluate base model: CORE score, train/val bpb, samples
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval -- \
    --device-batch-size=$DEVICE_BATCH_SIZE

# ===========================================================================
# 3) SFT  ——  teach chat format, tool use, MC, math; inject WillyGPT identity
# ===========================================================================
# Build WillyGPT's identity conversations and place them where SFT looks.
# (chat_sft.py loads $NANOCHAT_BASE_DIR/identity_conversations.jsonl, 2 epochs)
# Prefer the richer LLM-generated set if you scp'd it up; else build from template.
if [ -f "$SCRIPT_DIR/willygpt/willygpt_identity_full.jsonl" ]; then
    echo "Using LLM-generated identity set (willygpt_identity_full.jsonl)"
    cp "$SCRIPT_DIR/willygpt/willygpt_identity_full.jsonl" \
       "$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
else
    echo "No full set found; building templated identity (--repeat 3)"
    python "$SCRIPT_DIR/willygpt/build_willygpt_identity.py" \
        --out "$SCRIPT_DIR/willygpt/willygpt_identity.jsonl" --repeat 3
    cp "$SCRIPT_DIR/willygpt/willygpt_identity.jsonl" \
       "$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
fi
echo "Installed WillyGPT identity: $(wc -l < "$NANOCHAT_BASE_DIR/identity_conversations.jsonl") conversations"

torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --run=$WANDB_RUN
checkpoint_notice sft

torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- -i sft

# ===========================================================================
# 4) RL  ——  GRPO on GSM8K to sharpen math/reasoning (not in stock speedrun)
# ===========================================================================
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_rl -- \
    --num-epochs=$RL_EPOCHS \
    --run=$WANDB_RUN
checkpoint_notice rl

torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- -i rl

# ===========================================================================
# 5) Report
# ===========================================================================
python -m nanochat.report generate
cp "$NANOCHAT_BASE_DIR/report/report.md" "$ARTIFACT_DIR/willygpt_report_FINAL.md" 2>/dev/null || true
cp report.md "$ARTIFACT_DIR/willygpt_report_FINAL.md" 2>/dev/null || true

echo ""
echo "=================================================================="
echo " WillyGPT build complete."
echo " Final report: $ARTIFACT_DIR/willygpt_report_FINAL.md"
echo " Talk to it:   python -m scripts.chat_web   (then open the node IP:8000)"
echo "          or:  python -m scripts.chat_cli -i rl -p \"who are you?\""
echo " BACK UP checkpoints before you kill the box (see notices above)!"
echo "=================================================================="
