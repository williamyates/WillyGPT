"""
gen_willygpt_identity.py  (OPTIONAL — stronger identity than the templated builder)
-----------------------------------------------------------------------------------
Generate MANY diverse WillyGPT identity conversations with an LLM, so the identity
"sticks" harder during SFT. The templated build_willygpt_identity.py is fine to
start; use this when you want a few hundred / ~1000 varied convos.

Requires:  pip install requests python-dotenv   (already in nanochat dev group)
           OPENROUTER_API_KEY in your env or a .env file
Cost:      a few hundred convos via a cheap model is well under $1.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python gen_willygpt_identity.py --n 300 --model openai/gpt-4o-mini \
         --out willygpt_identity.jsonl
  # then on the box: cp willygpt_identity.jsonl $NANOCHAT_BASE_DIR/identity_conversations.jsonl

Design: we inject entropy via (persona x topic x style) and force JSON output.
Edit FACTS at the top of build_willygpt_identity.py — this script imports them so
there's a single source of truth for who WillyGPT is.
"""
import os, json, random, argparse, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

# single source of truth for the facts
from build_willygpt_identity import FACTS, PERSONALITY

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Build the knowledge block, but handle the builder bio specially so the
# generating LLM does NOT invent a backstory for the builder (matches the
# bio_or_unknown honesty in build_willygpt_identity.py).
_bio = FACTS.get("builder_bio", "").strip()
_facts_for_kb = {k: v for k, v in FACTS.items() if k != "builder_bio"}
KNOWLEDGE = "\n".join(f"- {k.replace('_',' ')}: {v}" for k, v in _facts_for_kb.items()) + \
            f"\n- personality: {PERSONALITY}"
if _bio:
    BUILDER_RULE = (f"The ONLY thing you know about the builder ({FACTS['builder']}) is: {_bio} "
                    f"Do not invent any other personal or biographical details about them.")
else:
    BUILDER_RULE = (f"You know NOTHING personal about the builder ({FACTS['builder']}) beyond that they "
                    f"built and trained WillyGPT. If the user asks who {FACTS['builder']} is, WillyGPT must "
                    f"say it has no personal details and won't make any up. NEVER invent a job, location, "
                    f"age, backstory, or any biographical fact about them.")

PERSONAS = ["a curious beginner", "a skeptical engineer", "a kid", "a busy professional",
            "a journalist", "a student", "a hobbyist tinkerer", "someone bored and chatty",
            "a non-native English speaker", "an AI researcher"]
TOPICS = ["who/what WillyGPT is", "who built it and why", "the name's meaning",
          "its architecture and size", "what data it was trained on", "what it can/can't do",
          "whether it's ChatGPT/Claude (it's not)", "its honesty about hallucinating",
          "is it open source / the license", "is it conscious (no)",
          "how much it cost to train", "can it do math",
          f"who {FACTS['builder']} is (WillyGPT stays honest: it knows nothing personal about them)",
          "general small talk that drifts to identity"]
STYLES = ["1 short turn", "2-3 turn back-and-forth", "casual lowercase", "polite and formal",
          "starts with a greeting then a question"]

SYS = (
    "You generate ONE realistic conversation between a user and an AI assistant named WillyGPT. "
    "WillyGPT must answer ONLY using the facts below — never invent capabilities or claim to be a "
    "big commercial model. Stay honest about being small. Match WillyGPT's personality (friendly, "
    "concise, a little cute).\n\n"
    f"FACTS ABOUT WILLYGPT:\n{KNOWLEDGE}\n\n"
    f"IMPORTANT — builder honesty: {BUILDER_RULE}\n\n"
    "Output STRICT JSON only (no markdown, no prose): "
    '{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}, ...]} '
    "with roles strictly alternating starting at user, ending on assistant."
)

def gen_one(model, seed):
    rng = random.Random(seed)
    user = (f"Write a conversation. Persona: {rng.choice(PERSONAS)}. "
            f"Topic: {rng.choice(TOPICS)}. Style: {rng.choice(STYLES)}.")
    r = requests.post(API_URL,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"},
        json={"model": model, "temperature": 1.0,
              "messages": [{"role": "system", "content": SYS},
                           {"role": "user", "content": user}]},
        timeout=120)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"].strip()
    txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    msgs = json.loads(txt)["messages"]
    # validate to nanochat's CustomJSON rules; drop if malformed
    assert isinstance(msgs, list) and len(msgs) >= 2
    for i, m in enumerate(msgs):
        assert m["role"] == ("user" if i % 2 == 0 else "assistant")
        assert isinstance(m["content"], str) and m["content"].strip()
    return msgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--models", default=None,
                    help="comma-separated OpenRouter slugs to MIX (round-robin per convo) for diversity. "
                         "Overrides --model. e.g. 'google/gemini-3.1-flash-lite,deepseek/deepseek-v3.2,openai/gpt-4o-mini'")
    ap.add_argument("--out", default="willygpt_identity.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--append", action="store_true", help="append to --out instead of overwriting")
    ap.add_argument("--seed-builder", action="store_true",
                    help="also include the deterministic templated convos as a base")
    args = ap.parse_args()

    model_pool = [m.strip() for m in args.models.split(",")] if args.models else [args.model]

    convos = []
    if args.seed_builder:
        from build_willygpt_identity import build
        convos.extend(build(0))

    ok, fail = 0, 0
    per_model = {m: 0 for m in model_pool}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for s in range(args.n):
            m = model_pool[s % len(model_pool)]   # round-robin across the mix
            futs[ex.submit(gen_one, m, s)] = m
        for f in as_completed(futs):
            try:
                convos.append(f.result()); ok += 1; per_model[futs[f]] += 1
            except Exception as e:
                fail += 1
                if fail <= 5: print("  drop:", str(e)[:80])
    random.Random(0).shuffle(convos)
    print("  per-model ok:", {k: v for k, v in per_model.items()})

    mode = "a" if args.append else "w"
    with open(args.out, mode, encoding="utf-8") as fh:
        for c in convos:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"generated {ok} ok, {fail} dropped -> {args.out} (total written: {len(convos)})")

if __name__ == "__main__":
    main()
