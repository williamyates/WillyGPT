#!/bin/bash
# =============================================================================
# willygpt_save.sh — pull WillyGPT artifacts OFF the cloud box (run on LAPTOP)
# =============================================================================
# The GPU box is ephemeral: when you terminate it, everything is gone. Run this
# to back up checkpoints + report to your Mac. Safe to run repeatedly (rsync is
# incremental) — back up after each stage's "CHECKPOINT SAVED" notice.
#
# Usage:
#   bash willygpt_save.sh USER@HOST [DEST_DIR] [SSH_KEY]
# Example (Lambda):
#   bash willygpt_save.sh ubuntu@209.20.xx.xx ./willygpt_backup ~/.ssh/lambda.pem
# =============================================================================
set -euo pipefail

HOST="${1:?usage: willygpt_save.sh USER@HOST [DEST_DIR] [SSH_KEY] [PORT] [REMOTE_BASE]}"
DEST="${2:-./willygpt_backup}"
KEY="${3:-}"
PORT="${4:-}"
# Lambda: ~/.cache/nanochat   |   RunPod: /workspace/.cache/nanochat
REMOTE_BASE="${5:-.cache/nanochat}"
SSHCMD="ssh"
[ -n "$KEY" ]  && SSHCMD="$SSHCMD -i $KEY"
[ -n "$PORT" ] && SSHCMD="$SSHCMD -p $PORT"
RSYNC=(rsync -avP --ignore-missing-args)
[ "$SSHCMD" != "ssh" ] && RSYNC+=(-e "$SSHCMD")

mkdir -p "$DEST"
echo "Backing up from $HOST:~/$REMOTE_BASE -> $DEST"

# Checkpoints for whichever stages exist (base/sft/rl), plus the report.
for sub in base_checkpoints chatsft_checkpoints chatrl_checkpoints report tokenizer; do
  echo "--- $sub ---"
  "${RSYNC[@]}" "$HOST:$REMOTE_BASE/$sub" "$DEST/" || echo "  (skip: $sub not present yet)"
done

echo ""
echo "Done. Backed up to: $DEST"
echo "Disk used:"
du -sh "$DEST" 2>/dev/null || true
