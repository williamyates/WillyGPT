# WillyGPT

A 1.68B-parameter chat model trained from scratch (pretraining, supervised finetuning (SFT), and GRPO reinforcement learning on the [nanochat](https://github.com/karpathy/nanochat) recipe (MIT). Built & trained by William Yates on a single 8×H100 node.

This repository contains the **training recipe, identity-data tooling, and ops scripts**. The **weights** are on [Hugging Face](https://huggingface.co/williamyates/WillyGPT). Base pretraining clears the GPT-2 CORE reference (0.3016 vs 0.2565); the model ships as two checkpoints (SFT and RL) whose trade-offs are characterized below.

---

## Model specification

| | |
|---|---|
| Architecture | GPT-style decoder transformer (nanochat `d26`) |
| Parameters | 1.68B (1,682M; 189 weight tensors) |
| Layers / heads | 26 layers · 13 attention heads · 13 KV heads · head dim 128 |
| Model dimension | 1664 |
| Context length | 2048 |
| Attention | Sliding-window (SSSL pattern, 3 short : 1 long), Flash Attention 3 |
| Precision | fp8 training (Hopper / H100 class) |
| Tokenizer | 32,768-token BPE, GPT-4 style |
| Checkpoint size | ~5.2 GB per stage |
| License / lineage | MIT · derived from nanochat (Karpathy) |

---

## Training pipeline

Three stages, run in sequence on one 8×H100 node (~9 h core wall-clock).

**1 — Pretraining** (~6.76 h). Next-token prediction on NVIDIA ClimbMix at a data:parameter ratio of 12 (compute-optimal; the stock nanochat speedrun uses 8).
`11.02B tokens · 10,510 iterations · train bpb 0.690 · val bpb 0.688 · CORE 0.3016`
Validation bpb at or below training bpb indicates model is not overfit.

**2 — Supervised finetuning** (~13 min). Chat format, multiple-choice, tool use, and math style on SmolTalk + MMLU/GSM8K/spelling, mixed with a 486-conversation identity set. The identity data was LLM-generated for diversity and constrained to keep the model honest about its scale and to refuse to fabricate a builder backstory.
`486 identity conversations · val bpb 0.2567 · final loss ~0.81`

**3 — GRPO reinforcement learning** (~1.5 h). Group-relative policy optimization rewarding correct GSM8K answers. Not part of the stock speedrun; added to test a targeted reasoning intervention.
`GSM8K · 467 steps · 1 epoch`

---

## Evaluation

Base model on the pretraining CORE benchmark; SFT and RL checkpoints on the nanochat chat suite. Values are accuracies (0–1) except the CORE/ChatCORE composites.

| Metric | Base | SFT | RL | Δ SFT→RL |
|---|---:|---:|---:|---:|
| CORE (pretrain) | 0.3016 | — | — | — |
| ChatCORE | — | 0.3982 | 0.2639 | −13.4 pt |
| GSM8K | — | 0.1099 | 0.1729 | +6.3 pt |
| SpellingBee | — | 0.9961 | 0.1836 | −81.3 pt |
| HumanEval | — | 0.1524 | 0.0915 | −6.1 pt |
| MMLU | — | 0.3831 | 0.3797 | −0.3 pt |
| ARC-Easy | — | 0.6843 | 0.6911 | +0.7 pt |
| ARC-Challenge | — | 0.5307 | 0.5307 | ±0.0 |

CORE 0.3016 exceeds the GPT-2 reference of 0.2565. **The SFT checkpoint is the default**; RL is retained as a math-specialized variant. The reasons are detailed below — the headline regressions are largely an output-discipline artifact rather than capability loss, but they are real in deployment.

---

## SFT vs. RL: regression for some specialization, mostly at the output layer

GRPO moved the single metric it optimized (GSM8K, +6.3 pt / +57% relative; reward climbed cleanly across all 467 steps) and left the rest flat or lower. This is the expected specialization tax of single-objective RL. A logit-level investigation locates some of the regression in when the model stops generating.

### SpellingBee regression

On raw spelling, SFT and RL are identical: 23/24, both missing only "mississippi" (a scale limit common to both, not a regression). What RL changed is termination. After the reasoning preamble, the RL checkpoint places near-certain probability on the end-of-turn token *before* emitting the `#### N` answer the grader requires:

```
P(<|assistant_end|>) at the answer position:  0.000 (SFT)  →  0.999 (RL)
```

The model quits before it answers. Suppressing `<|assistant_end|>` for the first ~80 decode steps (a `min_new_tokens` constraint, inference only, no weight changes) recovers the identical correct trace and restores SpellingBee from ~18% to **~95%**.

### HumanEval is mixed

Unlike SpellingBee, no single mechanism was identified to be the cause of the HumanEval drop. Hand-grading the RL failures splits roughly three ways: (a) correct code inside a malformed code fence — a harness extraction artifact, recoverable; (b) degenerate repetition loops (one token emitted hundreds of times) — real RL mode-collapse; (c) clean code with wrong logic — a real capability limit the SFT model shares. About half is a grading artifact and half is real; EOS-suppression does not help, because the model already writes complete functions on ordinary coding prompts.

### Identity robustness is thin, and RL thinned it further

On a cold, single-turn "Who are you?", both checkpoints answer correctly. Mid-conversation, after unrelated turns, the RL checkpoint lost the thread and confabulated a DeepMind/AlphaGo origin; the SFT checkpoint held its real identity in the same setting. Identity acquired through finetuning is fragile at this scale, and RL eroded what robustness existed — consistent with the regression pattern in the eval table. The recommended mitigation is an inference-time system prompt (zero retraining); a durable fix is multi-turn identity data where the question lands after a drifting context.

### Smear parameters (falsified hypothesis)

A weight-diff flagged the model's "smear" parameters as the largest relative movers (`smear_gate` +56%, `smear_lambda` 0.249 → 0.299), suggesting RL had blurred the letter-level signal. A dose-response sweep falsified this: spelling accuracy is flat as `smear_lambda` varies from 0 through 0.35 (past the RL value) and only degrades at 0.45. The weight movement is real but functionally inert. **Largest relative weight change ≠ functional importance**; every interpretability claim here is gated by a behavioral ablation.

---

## External comparison

WillyGPT's size-peers are GPT-2 XL (1.5B, 2019) and current ~1.5B models. Methodologies differ (WillyGPT/nanochat are 0-shot chat-format; the others are standard few-shot harness numbers), so these are directional.

| Model | CORE | MMLU | ARC-E | ARC-C | GSM8K |
|---|---:|---:|---:|---:|---:|
| WillyGPT · 1.68B · '26 | 0.30 | 38.3 | 68.4 | 53.1 | 11.0→17.3 |
| GPT-2 XL · 1.5B · '19 | ~0.26 | ~26 | — | ~30 | ~0 |
| nanochat d20 ($100) · 561M | 0.22 | 31.5 | 38.8 | 28.1 | 4.6→7.6 |
| Qwen2.5-1.5B · '24 | — | 59.8 | 79.1 | 53.4 | 68.5 |
| SmolLM2-1.7B · '24 | — | 51.9 | 77.8 | 50.3 | 47.7 |

At identical parameter count, WillyGPT is well ahead of GPT-2 XL (which is near-random on MMLU and ~0 on math/code) — the difference is six years of data quality and a finetuning pipeline GPT-2 never had. Against the same-framework nanochat d20 speedrun it wins every column, as a correctly-built d26 at ~3× the parameters should. The gap to 2024 1.5B models is specific rather than global: ARC-Challenge (53.1) is level with Qwen2.5-1.5B (53.4), and WillyGPT only falls off sharply on MMLU and GSM8K — knowledge breadth and multi-step math, the two axes that scale most directly with token budget. Qwen2.5 trained on ~18T tokens; WillyGPT on 11B (~1000×).

---

## Qualitative behavior

Sampled from the SFT checkpoint.

- **Multi-turn coherence and identity.** Holds context across turns, incorporates new user-supplied details, asks sensible clarifying questions, and answers "who are you?" correctly several turns deep with accurate origin and scale.
- **Documented weakness — confident hallucination.** Invents plausible-but-false specifics (e.g. a "Ghibli Museum" in Thailand; it is in Tokyo). The honesty about being unreliable is trained in; the unreliability itself is a consequence of parameter count. We've had LLMs for the better half of a decade, but **treat all specific factual claims as unverified**. Neat reminder of how severe hallucinations were, though.

---

## Reproducibility & provenance

- Both checkpoints were verified **bit-for-bit (SHA-256) between the GPU node and the laptop** before teardown: `model_000483.pt` (SFT), `model_000466.pt` (RL), `tokenizer.pkl`, `token_bytes.pt` — all matched.
- Both deserialize cleanly on CPU/MPS (189 tensors, ~1.68B params); the SFT meta's `val_bpb` 0.2567 matches the eval log.
- Confirmed to load and generate on a 32 GB M4 MacBook Air via MPS (`chat_cli` and `chat_web`), fully detached from the rented hardware.

**Cost / infrastructure:** 8×H100 (Hopper), RunPod Secure Cloud, AP-JP-1, ~$26.37/hr. Core pipeline ~9 h ≈ ~$240 of GPU time (real billed total was ~$300 with setup and debugging). 300 GB network volume; 76 GB pulled to laptop.

---

## Known issues & recommended fixes

**Inference-time (no retraining):**
- **Pin identity with a system prompt** (e.g. "You are WillyGPT, built by William Yates using nanochat; you are not made by DeepMind, OpenAI, or Google"). Highest-leverage fix for identity drift.
- **Default to the SFT checkpoint**; route to RL only when math accuracy is the explicit goal.
- **Min-length decoding for the RL checkpoint.** Suppressing `<|assistant_end|>` for the first ~80 generated tokens restores SpellingBee from ~18% to ~95% greedy.

**Next training run:**
- **Train identity in context**, with multi-turn conversations where "who are you?" follows unrelated chit-chat.
- **Constrain the RL objective** — a KL penalty anchoring the policy near the SFT model, and/or non-math rewards — so GSM8K gains do not tax spelling, code, and identity. The specific failure modes to penalize are premature end-of-turn termination and degenerate token repetition; both are RL mode-collapse, not lost knowledge.

---

## Running it

The weights are nanochat-architecture, not Hugging Face Transformers — they do not load with `AutoModel.from_pretrained`. Run them with the nanochat code:

```bash
# 1. get nanochat
git clone https://github.com/karpathy/nanochat.git && cd nanochat
uv sync --extra gpu        # or --extra cpu for a Mac (MPS), slow but functional

# 2. download the weights into nanochat's cache
export NANOCHAT_BASE_DIR=$PWD
hf download williamyates/WillyGPT --local-dir "$NANOCHAT_BASE_DIR/.cache/nanochat"

# 3. chat
python -m scripts.chat_cli -i sft -p "Who are you?"     # SFT = default
python -m scripts.chat_web
```

## Reproducing the build

```bash
git clone https://github.com/karpathy/nanochat.git
# copy this repo's files in:
#   willygpt_run.sh, willygpt_preflight.sh  -> nanochat/   (repo root)
#   willygpt/                                -> nanochat/willygpt/
cd nanochat
bash willygpt_preflight.sh         # checks GPU (8×H100), disk, identity data
WANDB_RUN=willygpt bash willygpt_run.sh
```

Configuration knobs (`DEPTH`, `RATIO`, `DEVICE_BATCH_SIZE`, `RL_EPOCHS`) are at the top of `willygpt_run.sh`.

## Repository contents

```
willygpt_run.sh                   end-to-end pipeline (pretrain → SFT+identity → RL → eval)
willygpt_preflight.sh             pre-launch checks for the GPU node
willygpt/
├── build_willygpt_identity.py    deterministic identity-data builder (FACTS defined here)
├── gen_willygpt_identity.py      LLM identity-data generator (multi-model, no fabricated builder bio)
├── willygpt_identity_full.jsonl  the 486-conversation identity set used for SFT
├── willygpt_identity.jsonl       67-conversation templated fallback
├── willygpt_local_test.sh        M4/MPS pipeline smoke test
└── willygpt_save.sh              pull checkpoints off an ephemeral cloud node
```

---

## Credits & license

MIT-licensed; a derivative of [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy (also MIT). The nanochat framework — architecture, training scripts, tokenizer, and eval harness — is Karpathy's work. This repository's contributions are the d26 configuration, the identity layer and its honesty constraints, the added GRPO stage, the mechanistic analysis above, and the build/ops tooling. Pretraining data: NVIDIA ClimbMix. SFT data: SmolTalk, MMLU, GSM8K, plus the custom identity set. See [LICENSE](LICENSE).
