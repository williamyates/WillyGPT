#!/bin/bash
# =============================================================================
# willygpt_preflight.sh — run ON the GPU box, from the nanochat repo root,
# BEFORE you kick off willygpt_run.sh. Catches the cheap-to-fix problems that
# would otherwise waste hours of H100 time.
#
# Usage:
#   cd ~/nanochat
#   source .venv/bin/activate        # if you've already run `uv sync`; optional
#   bash willygpt_preflight.sh
#
# Exit code 0 = all critical checks passed. Non-zero = do NOT train yet.
# =============================================================================
PASS=0; FAIL=0; WARN=0
ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
warn() { echo "  [WARN] $1"; WARN=$((WARN+1)); }

echo "================ WillyGPT preflight ================"

# ---- 1. GPU: count + Hopper (needed for fp8 + FA3 + sliding-window) ----
echo "[1] GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  echo "      detected: ${NGPU}x ${NAME}"
  [ "$NGPU" -ge 8 ] && ok "8+ GPUs" || warn "expected 8 GPUs, found $NGPU (run still works, just slower / adjust --nproc)"
  echo "$NAME" | grep -qiE "H100|H200|GH200|B100|B200|Hopper|Blackwell" \
    && ok "Hopper/Blackwell class (fp8 + FA3 supported)" \
    || bad "NOT a Hopper GPU ($NAME) — fp8/FA3/sliding-window will fail. Use H100, or drop --fp8 and add --window-pattern L."
else
  bad "nvidia-smi not found — no GPU / drivers"
fi

# ---- 2. Disk: dataset (~33GB) + checkpoints (several GB each) ----
echo "[2] Disk"
# Match willygpt_run.sh: prefer RunPod's persistent /workspace volume if present.
if [ -n "${NANOCHAT_BASE_DIR:-}" ]; then
  BASE="$NANOCHAT_BASE_DIR"
elif [ -d /workspace ] && [ -w /workspace ]; then
  BASE="/workspace/.cache/nanochat"
else
  BASE="$HOME/.cache/nanochat"
fi
mkdir -p "$BASE"
AVAIL_GB=$(df -PB1G "$BASE" | awk 'NR==2{print $4}')
echo "      target: $BASE"
echo "      free here: ${AVAIL_GB} GB"
[ "${AVAIL_GB:-0}" -ge 200 ] && ok ">=200GB free" || warn "<200GB free at $BASE. On RunPod, raise the /workspace network volume; elsewhere set NANOCHAT_BASE_DIR to a bigger disk."

# ---- 3. Layout: run script in cwd, willygpt/ subfolder beside it ----
echo "[3] Layout"
[ -f "willygpt_run.sh" ] && ok "willygpt_run.sh in repo root" || bad "willygpt_run.sh missing from \$PWD ($(pwd)) — run preflight from the repo root"
[ -d "willygpt" ] && ok "willygpt/ subfolder present" || bad "willygpt/ subfolder missing"
[ -f "scripts/base_train.py" ] && ok "looks like the nanochat repo root" || bad "scripts/base_train.py not found — are you in the nanochat repo root?"

# ---- 4. Identity data: present + valid + correct facts ----
echo "[4] Identity data"
python3 - <<'PY'
import json, os, sys
full = "willygpt/willygpt_identity_full.jsonl"
tmpl = "willygpt/willygpt_identity.jsonl"
path = full if os.path.exists(full) else (tmpl if os.path.exists(tmpl) else None)
if not path:
    print("  [FAIL] no identity jsonl found (neither full nor template)"); sys.exit(3)
which = "full (LLM-generated)" if path==full else "template fallback"
convos=[json.loads(l) for l in open(path) if l.strip()]
bad=0
for msgs in convos:
    try:
        assert isinstance(msgs,list) and len(msgs)>=2
        for j,m in enumerate(msgs):
            assert m["role"]==("user" if j%2==0 else "assistant")
            assert isinstance(m["content"],str) and m["content"].strip()
    except Exception:
        bad+=1
txt=" ".join(m["content"] for c in convos for m in c if m["role"]=="assistant")
print(f"  [PASS] identity set = {which}: {len(convos)} convos" if bad==0
      else f"  [FAIL] {bad}/{len(convos)} convos invalid")
# facts consistency
try:
    import importlib.util
    spec=importlib.util.spec_from_file_location("b","willygpt/build_willygpt_identity.py")
    b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    builder=b.FACTS["builder"]
    print(f"  [PASS] FACTS.builder = '{builder}'")
    if builder in txt: print(f"  [PASS] dataset mentions builder '{builder}' ({txt.count(builder)}x)")
    else: print(f"  [WARN] dataset never mentions builder '{builder}' — regenerate if you changed FACTS")
    if "an independent developer" in txt:
        print("  [FAIL] dataset contains stale 'an independent developer' — regenerate with current FACTS")
except Exception as e:
    print(f"  [WARN] couldn't import build_willygpt_identity.py FACTS: {str(e)[:80]}")
PY

# ---- 5. Optional deps check (only meaningful if venv active) ----
echo "[5] Deps / framework (if venv active)"
if python3 -c "import torch" 2>/dev/null; then
  python3 - <<'PY'
import torch
print(f"  [PASS] torch {torch.__version__}")
print(f"  {'[PASS]' if torch.cuda.is_available() else '[FAIL]'} CUDA available: {torch.cuda.is_available()} ({torch.cuda.device_count()} devices)")
try:
    import torchao; print(f"  [PASS] torchao present (fp8 path)")
except Exception:
    print("  [WARN] torchao not importable yet — uv sync will install it during the run")
PY
else
  warn "torch not importable yet — that's fine, willygpt_run.sh runs 'uv sync' first. (Activate .venv to deep-check.)"
fi

echo "==================================================="
echo "PASS=$PASS  WARN=$WARN  FAIL=$FAIL"
if [ "$FAIL" -eq 0 ]; then
  echo ">>> READY TO TRAIN. Launch: WANDB_RUN=willygpt screen -L -Logfile willygpt_run.log -S willy bash willygpt_run.sh"
  exit 0
else
  echo ">>> NOT READY — fix the [FAIL] items above before training."
  exit 1
fi
