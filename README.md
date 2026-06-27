# WillyGPT

A ~1.68-billion-parameter chat model trained from scratch — pretraining, supervised
finetuning, and reinforcement learning — on a rented 8×H100 node, built on Andrej Karpathy's
[nanochat](https://github.com/karpathy/nanochat) recipe (MIT).

The base model clears GPT-2 reference quality; the SFT stage produces a coherent multi-turn
assistant that reliably reports its own origin; the GRPO RL stage improves grade-school math at a
measurable cost to capabilities it did not reward. The build ships as two checkpoints. **SFT is
the recommended default**; RL is a math specialist with documented regressions.

- Model weights: [huggingface.co/williamyates/WillyGPT](https://huggingface.co/williamyates/WillyGPT)
- Full build report + mechanistic analysis: [WillyGPT Model Card](https://willygpt.netlify.app)

## Checkpoints

| | Stage | Path | Role |
|---|---|---|---|
| **default** | SFT — generalist | `chatsft_checkpoints/d26` · step 483 | Best all-rounder. Highest ChatCORE (0.398), near-perfect spelling, holds identity across multi-turn chat. **Deploy this.** |
| | RL — mathematician | `chatrl_checkpoints/d26` · step 466 | Best GSM8K (0.173, +57% over SFT) but regressed broadly. Use for math, not for talking. |

## Specification

| | |
|---|---|
| Architecture | GPT-style decoder transformer (nanochat d26) |
| Parameters | ~1.68 B (1,682 M; 189 weight tensors) |
| Layers / heads | 26 layers · 13 attention heads · 13 KV heads (head dim 128) |
| Model dimension | 1664 |
| Context length | 2048 tokens |
| Attention | Sliding-window, SSSL pattern (3 short : 1 long), Flash Attention 3 |
| Precision | fp8 training (Hopper / H100 class) |
| Tokenizer | 32,768-token BPE, GPT-4-style |
| Checkpoint size | ~5.2 GB per stage |
| License / lineage | MIT · derived from nanochat |

## Training

Three stages, run in sequence on one 8×H100 node:

1. **Pretraining** — ~6.8 h on ~11 B tokens of a FineWeb-EDU-style web corpus. Base CORE **0.3016**
   (beats the GPT-2 reference of 0.2565).
2. **SFT** — ~13 min, standard nanochat chat data plus a custom identity layer so the model
   reports its own origin.
3. **GRPO RL** — ~1.5 h on GSM8K grade-school math.

## Evaluation

Base model on the pretraining CORE benchmark; SFT and RL on the nanochat chat suite. Values are
accuracies (0–1) except the CORE/ChatCORE composites.

| Metric | Base | SFT | RL | Δ SFT→RL |
|---|---|---|---|---|
| CORE (pretrain) | 0.3016 | — | — | — |
| ARC-Easy | — | 0.6843 | 0.6911 | +0.7 pt |
| ARC-Challenge | — | 0.5307 | 0.5307 | ±0.0 |
| MMLU | — | 0.3831 | 0.3797 | −0.3 pt |
| GSM8K | — | 0.1099 | 0.1729 | **+6.3 pt** |
| HumanEval | — | 0.1524 | 0.0915 | −6.1 pt |
| SpellingBee | — | 0.9961 | 0.1836 | −81.3 pt |
| ChatCORE | — | 0.3982 | 0.2639 | −13.4 pt |

RL moved exactly one metric up — the one it optimized — and dragged the rest flat or down. That
is the whole argument for shipping SFT.

## The RL regression (and why the SpellingBee number is misleading)

The −81 pt SpellingBee drop is not lost capability. GRPO's narrow GSM8K reward induced a
**premature end-of-turn termination**: at the decode position right after its reasoning, the RL
model's probability of emitting the end token goes from ~0.000 (SFT) to ~0.999 (RL), so it stops
before producing the answer. The underlying spelling is intact — banning the end token for the
first few decode steps (or applying a small constant end-logit penalty) recovers the correct
output with no retraining.

A mechanistic analysis (direct logit attribution → activation patching → logit-lens) traces the
cause: the regression originates as a content-state feature at layer 9, is carried up the residual
stream, and is read out into the end-of-turn logit at the final MLP (whose weights barely moved).
It is recoverable at the output logit but not by dense residual edits. Full writeup on the
[model card](https://willygpt.netlify.app).

## Usage

WillyGPT is a nanochat checkpoint, loaded through the nanochat harness (not HF Transformers).

```bash
# in a nanochat checkout
export NANOCHAT_BASE_DIR=/path/to/willygpt        # contains chatsft_checkpoints/ , chatrl_checkpoints/

# chat (web UI) — SFT is the recommended default
python -m scripts.chat_web -i sft

# CLI
python -m scripts.chat_cli -i sft
```

If you run the RL checkpoint, apply the inference fix to avoid the early-stop collapse — either
`min_new_tokens` in the chat UI (already wired for `-i rl`) or a constant end-logit penalty:

```bash
WILLY_END_PENALTY=10 python -m scripts.chat_cli -i rl
```

## License

MIT, derived from [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy.
