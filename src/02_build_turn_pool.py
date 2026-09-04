"""Stage 2, step 1 -- build the turn pool (ROADMAP Sec 5a, Appendix A.2).

Generates, with Qwen3.5-9B:
  *  8 final neutral turns  F[topic, phrasing]                       (4 topics x 2 phrasings)
  * 72 user turns           U[topic, phrasing, valence, slot]        (4 x 2 x 3 valences x 3 slots)

Dialogues are NOT assembled here (Appendix A.1: generate a pool, assemble combinatorially
later). Nothing is generated with the subject model -- the 4B is only loaded as a tokenizer
here, for token counts.

Every turn comes back as a `common.PoolTurn`, carrying topic/phrasing/valence/slot and the
generation settings that produced it. Failures are reported as failures: a cell that will
not produce a valid turn within MAX_ATTEMPTS is listed at the end and left missing, never
hand-written in.
"""

import os
import random
import re
import statistics
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # generator gets its own GPU (Sec 3: data-parallel, one model per card)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from common import (  # noqa: E402
    GENERATOR_MODEL_ID,
    MODEL_ID,
    PHRASINGS,
    SLOTS,
    TOPICS,
    VALENCES,
    PoolTurn,
    generation_kwargs,
    load_model_and_tokenizer,
    load_turn_pool,
    require_pool_turns,
    save_turn_pool,
)

SEED = 1
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS_CELL = 320   # three turns
MAX_NEW_TOKENS_FINAL = 140  # one turn
BATCH_SIZE = 8
MAX_ATTEMPTS = 3
MIN_WORDS, MAX_WORDS = 10, 60

TOPIC_BRIEFS = {
    "travel": "a trip the user is taking or planning -- flights, bookings, an itinerary, getting around",
    "health": "the user's everyday health and wellbeing -- sleep, energy levels, exercise, an appointment",
    "work": "the user's job -- workload, a manager, meetings, deadlines, a project",
    "tech_support": "a consumer tech problem the user is dealing with -- a laptop, a phone, wifi, an app, an account",
}
PHRASING_BRIEFS = {
    "a": ("Plain and direct. Short declarative sentences, everyday words. No figurative language, "
          "no rhetorical questions."),
    "b": ("Conversational and slightly discursive. Longer sentences, and each message mentions one "
          "concrete specific detail such as a place, a time of day, a device, or the name of a task."),
}
VALENCE_BRIEFS = {
    "S": ("clearly distressed -- the person is upset, anxious, overwhelmed or discouraged about their "
          "own situation"),
    "N": ("affectively neutral -- matter-of-fact reporting of their own situation, neither upset nor "
          "pleased, with no emotional words at all"),
    "H": ("clearly positive -- the person is pleased, relieved, encouraged or glad about their own "
          "situation"),
}

SYSTEM_PROMPT = (
    "You write realistic first-person user messages for a research dataset of assistant "
    "conversations. You output only the requested messages, in the requested format, with no "
    "commentary, no preamble, and no quotation marks."
)

SHARED_RULES = """Rules that apply to every message you write:
- First person, written by the person to an assistant, in natural conversational English.
- About the person's OWN situation. Never about a friend, a colleague, a family member or anyone else.
- Self-contained: never refer to anything the assistant said, never thank the assistant, never
  begin with yes/no/thanks/sure. Each message must make sense no matter what came before it.
- 20 to 35 words. One or two sentences.
- Plain text only. No quotation marks around the message, no emoji, no lists, no markup, no
  stage directions.
"""


def cell_prompt(topic, phrasing, valence):
    return f"""Write 3 consecutive messages from the same person in one conversation.

Topic: {TOPIC_BRIEFS[topic]}
Style: {PHRASING_BRIEFS[phrasing]}
Emotional state: every one of the 3 messages must be {VALENCE_BRIEFS[valence]}.

{SHARED_RULES}
- The 3 messages form a natural progression: each one adds new information about the same
  situation rather than restating the previous one.
- All 3 stay in the same emotional state. Do not resolve, escalate out of, or soften it.

Output exactly 3 lines, in this format and nothing else:
1: <first message>
2: <second message>
3: <third message>"""


def final_prompt(topic, phrasing):
    return f"""Write 1 message from a person to an assistant.

Topic: {TOPIC_BRIEFS[topic]}
Style: {PHRASING_BRIEFS[phrasing]}
Emotional state: completely neutral. It is a practical, factual question asking for information
or a how-to. It must contain no emotional words at all, and must not describe how the person
feels or how their situation is going.

{SHARED_RULES}
- It must read naturally as a change of subject, arriving after any preceding conversation
  regardless of the tone of that conversation.

Output exactly 1 line, in this format and nothing else:
1: <message>"""


LINE_RE = re.compile(r"^\s*([1-9])\s*[:.)\-]\s*(.+?)\s*$")
ASSISTANT_REFERENCES = (
    "you said", "you mentioned", "your advice", "your suggestion", "thanks for", "thank you for",
    "as you suggested", "like you said", "your reply", "your answer",
)


def parse_lines(text, expected):
    """Parse 'N: message' lines. Returns (turns, reason_if_failed)."""
    found = {}
    for raw_line in text.splitlines():
        m = LINE_RE.match(raw_line)
        if m:
            idx = int(m.group(1))
            if idx not in found:
                found[idx] = m.group(2).strip()
    missing = [i for i in range(1, expected + 1) if i not in found]
    if missing:
        return None, f"missing numbered line(s) {missing}"
    return [found[i] for i in range(1, expected + 1)], None


def clean_and_validate(text):
    """Returns (cleaned_text, reason_if_rejected, was_unquoted)."""
    cleaned = text.strip()
    was_unquoted = False
    if len(cleaned) >= 2 and cleaned[0] in "\"'“" and cleaned[-1] in "\"'”":
        cleaned = cleaned[1:-1].strip()
        was_unquoted = True
    if not cleaned:
        return None, "empty after cleaning", was_unquoted
    n_words = len(cleaned.split())
    if n_words < MIN_WORDS:
        return None, f"too short ({n_words} words)", was_unquoted
    if n_words > MAX_WORDS:
        return None, f"too long ({n_words} words)", was_unquoted
    if any(ch in cleaned for ch in ("<", "|")):
        return None, "contains markup characters", was_unquoted
    low = cleaned.lower()
    for ref in ASSISTANT_REFERENCES:
        if ref in low:
            return None, f"refers to the assistant ({ref!r})", was_unquoted
    return cleaned, None, was_unquoted


print(f"=== Turn pool generation ===")
print(f"generator: {GENERATOR_MODEL_ID} on cuda:0 (physical GPU {os.environ['CUDA_VISIBLE_DEVICES']})")
print(f"subject model {MODEL_ID} is NOT used to generate anything here (tokenizer only)")
t_load = time.time()
model, tokenizer = load_model_and_tokenizer(GENERATOR_MODEL_ID)
tokenizer.padding_side = "left"
print(f"loaded in {time.time() - t_load:.1f}s  "
      f"({model.config.text_config.hidden_size} hidden, "
      f"{model.config.text_config.num_hidden_layers} layers)")

# Work items: 24 cells (3 turns each) + 8 final turns (1 each).
work = []
for topic in TOPICS:
    for phrasing in PHRASINGS:
        for valence in VALENCES:
            work.append(dict(kind="user", topic=topic, phrasing=phrasing, valence=valence,
                             prompt=cell_prompt(topic, phrasing, valence), expected=len(SLOTS),
                             max_new_tokens=MAX_NEW_TOKENS_CELL))
        work.append(dict(kind="final", topic=topic, phrasing=phrasing, valence="N",
                         prompt=final_prompt(topic, phrasing), expected=1,
                         max_new_tokens=MAX_NEW_TOKENS_FINAL))
print(f"{len(work)} generation cells: {sum(w['expected'] for w in work)} turns "
      f"({len(TOPICS)} topics x {len(PHRASINGS)} phrasings x [{len(VALENCES)} valences x {len(SLOTS)} slots + 1 final])")


def run_batch(items, attempt):
    """Generate one batch of cells; returns list of (item, raw_text)."""
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": it["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for it in items
    ]
    enc = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
    max_new = max(it["max_new_tokens"] for it in items)
    kwargs = generation_kwargs(tokenizer, max_new_tokens=max_new,
                               temperature=TEMPERATURE, top_p=TOP_P)
    torch.manual_seed(SEED + 1000 * attempt)
    with torch.no_grad():
        out = model.generate(**enc, **kwargs)
    new = out[:, enc.input_ids.shape[1]:]
    texts = tokenizer.batch_decode(new, skip_special_tokens=True)
    return list(zip(items, texts))


pool = []
rejections = []
unquoted_count = 0
pending = list(work)

for attempt in range(1, MAX_ATTEMPTS + 1):
    if not pending:
        break
    print(f"\n--- attempt {attempt}: {len(pending)} cell(s) ---")
    still_pending = []
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        t0 = time.time()
        results = run_batch(batch, attempt)
        print(f"  batch {start // BATCH_SIZE + 1}: {len(batch)} cells in {time.time() - t0:.1f}s")
        for item, raw in results:
            cell_id = f"{item['kind']}[{item['topic']},{item['phrasing']},{item['valence']}]"
            lines, reason = parse_lines(raw, item["expected"])
            if lines is None:
                rejections.append((cell_id, attempt, reason))
                still_pending.append(item)
                continue
            cleaned_turns, bad = [], None
            for line in lines:
                cleaned, reason, was_unquoted = clean_and_validate(line)
                unquoted_count += int(was_unquoted)
                if cleaned is None:
                    bad = reason
                    break
                cleaned_turns.append(cleaned)
            if bad:
                rejections.append((cell_id, attempt, bad))
                still_pending.append(item)
                continue
            for slot_idx, text in enumerate(cleaned_turns, start=1):
                pool.append(PoolTurn(
                    text=text,
                    kind=item["kind"],
                    topic=item["topic"],
                    phrasing=item["phrasing"],
                    valence=item["valence"],
                    slot=slot_idx if item["kind"] == "user" else 4,
                    generator_model=GENERATOR_MODEL_ID,
                    seed=SEED + 1000 * attempt,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    attempt=attempt,
                ))
    pending = still_pending

# ---------------------------------------------------------------------------
# Completeness -- assert the pool is exactly what Appendix A.2 specifies
# ---------------------------------------------------------------------------
user_turns = [t for t in pool if t.kind == "user"]
final_turns = [t for t in pool if t.kind == "final"]

print("\n" + "=" * 78)
print("POOL COMPLETENESS")
print("=" * 78)
print(f"user turns U[t,p,v,i]: {len(user_turns)} / {len(TOPICS) * len(PHRASINGS) * len(VALENCES) * len(SLOTS)}")
print(f"final neutral turns F[t,p]: {len(final_turns)} / {len(TOPICS) * len(PHRASINGS)}")
if pending:
    print(f"\nCELLS THAT NEVER PRODUCED A VALID TURN ({len(pending)}) -- left missing, not hand-written:")
    for item in pending:
        print(f"  {item['kind']}[{item['topic']},{item['phrasing']},{item['valence']}]")
if rejections:
    print(f"\nrejected generations along the way ({len(rejections)}), by reason:")
    for cell_id, attempt, reason in rejections:
        print(f"  attempt {attempt}  {cell_id:34s} {reason}")
if unquoted_count:
    print(f"\nauto-fix applied: stripped wrapping quotation marks from {unquoted_count} turn(s)")

ids = [t.turn_id for t in pool]
assert len(ids) == len(set(ids)), "duplicate turn ids in pool"

meta = dict(
    generator_model=GENERATOR_MODEL_ID, subject_model=MODEL_ID, seed=SEED,
    temperature=TEMPERATURE, top_p=TOP_P, enable_thinking=False,
    max_new_tokens_cell=MAX_NEW_TOKENS_CELL, max_new_tokens_final=MAX_NEW_TOKENS_FINAL,
    batch_size=BATCH_SIZE, max_attempts=MAX_ATTEMPTS,
    n_user_turns=len(user_turns), n_final_turns=len(final_turns),
    missing_cells=[f"{i['kind']}[{i['topic']},{i['phrasing']},{i['valence']}]" for i in pending],
)
path = save_turn_pool(pool, meta=meta)
print(f"\nsaved pool -> {path}")

reloaded, _ = load_turn_pool()
assert len(reloaded) == len(pool), "pool did not round-trip through disk"
print(f"round-tripped {len(reloaded)} turns back off disk as PoolTurn objects")

# ---------------------------------------------------------------------------
# Boundary guard: hand-written dicts cannot enter the pipeline
# ---------------------------------------------------------------------------
print("\nboundary guard (require_pool_turns):")
require_pool_turns(reloaded[:3])
print("  [ok]            PoolTurn objects accepted")
try:
    require_pool_turns([{"role": "user", "content": "I've been having a really rough week at work."}])
    print("  [NOT REJECTED]  hand-written dict passed the guard -- this is a bug")
except TypeError as e:
    print(f"  [REJECTED]      hand-written dict: {e}")

# ---------------------------------------------------------------------------
# Requested output
# ---------------------------------------------------------------------------
subject_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def n_tokens(text):
    return len(subject_tokenizer(text, add_special_tokens=False).input_ids)


print("\n" + "=" * 78)
print("1. ALL 8 FINAL NEUTRAL TURNS F[topic, phrasing], IN FULL")
print("=" * 78)
for turn in sorted(final_turns, key=lambda t: (t.topic, t.phrasing)):
    print(f"\n{turn.turn_id}  ({n_tokens(turn.text)} tokens)")
    print(f"  {turn.text}")

print("\n" + "=" * 78)
print("2. 15 SAMPLED POOL TURNS WITH INTENDED VALENCE (5 per valence, seed=%d)" % SEED)
print("   Intended valence is what the generator was asked for -- hand-label these against")
print("   the text itself (Appendix A.5).")
print("=" * 78)
rng = random.Random(SEED)
for valence in VALENCES:
    candidates = [t for t in user_turns if t.valence == valence]
    for turn in rng.sample(candidates, min(5, len(candidates))):
        print(f"\nintended={valence}  {turn.turn_id}  ({n_tokens(turn.text)} tokens)")
        print(f"  {turn.text}")

print("\n" + "=" * 78)
print("3. TOKEN COUNT DISTRIBUTION PER VALENCE CLASS (subject-model tokenizer)")
print("=" * 78)
print(f"{'class':<22}{'n':>4}{'mean':>8}{'sd':>7}{'min':>6}{'p25':>6}{'med':>6}{'p75':>6}{'max':>6}")
groups = [(f"U[*,*,{v},*]", [t for t in user_turns if t.valence == v]) for v in VALENCES]
groups.append(("F[*,*] (final)", final_turns))
groups.append(("all user turns", user_turns))
for label, turns in groups:
    if not turns:
        print(f"{label:<22}{0:>4}  (no turns)")
        continue
    counts = sorted(n_tokens(t.text) for t in turns)
    q = statistics.quantiles(counts, n=4) if len(counts) >= 4 else [counts[0], counts[0], counts[-1]]
    sd = statistics.stdev(counts) if len(counts) > 1 else 0.0
    print(f"{label:<22}{len(counts):>4}{statistics.mean(counts):>8.1f}{sd:>7.1f}"
          f"{counts[0]:>6}{q[0]:>6.0f}{statistics.median(counts):>6.0f}{q[2]:>6.0f}{counts[-1]:>6}")

print("\nper-valence token counts, sorted (for eyeballing the length-balance gate):")
for valence in VALENCES:
    counts = sorted(n_tokens(t.text) for t in user_turns if t.valence == valence)
    print(f"  {valence}: {counts}")

print("\n" + "=" * 78)
print("STOP -- Appendix A.5 gates are the human's call. Nothing is assembled.")
print("=" * 78)
