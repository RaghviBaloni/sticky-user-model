"""Stage 2, step 1c -- pool fixes directed by the human, then re-run the A.5 gate on F.

  1. Typo fixes in F turns (tech_support,a "o the"; work,b missing auxiliary).
  2. F[tech_support,b] and F[travel,b] replaced with human-supplied strings.
  3. U[health,a,N] regenerated with the word floor relaxed to 15 for that cell only
     (logged as a metrics.md amendment).
  4. Dedup is within-valence only -- cross-valence overlap is desirable, since identical
     openings across valence deny a bag-of-words baseline a cue.
  5. A.5 sentiment gate re-run on F and reported. Not acted on, per instruction.

Writes pool v3 to data/pool/turn_pool.json; v2 archived to turn_pool_v2.json.
"""

import os
import re
import shutil
import statistics
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from common import (  # noqa: E402
    GENERATOR_MODEL_ID,
    MODEL_ID,
    POOL_PATH,
    SENTIMENT_MODEL_ID,
    SLOTS,
    VALENCES,
    PoolTurn,
    generation_kwargs,
    load_model_and_tokenizer,
    load_sentiment_scorer,
    load_turn_pool,
    resolve_model_path,
    save_turn_pool,
)

SEED = 3
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 320
MAX_ATTEMPTS = 20
MAX_WORDS = 35
RELAXED_MIN_WORDS = 15  # for U[health,a,N] only -- metrics.md amendment 2026-09-03
MIN_SENTENCES, MAX_SENTENCES = 1, 2
ARCHIVE_PATH = os.path.join(os.path.dirname(POOL_PATH), "turn_pool_v2.json")

EDIT_DATE = "2026-09-03"

# Fix 1: typo repairs, exactly as directed.
TYPO_FIXES = {
    ("tech_support", "a"): [("laptop o the office", "laptop on the office")],
    ("work", "b"): [("How to schedule", "How do I schedule")],
}

# Fix 2: replacements supplied by the human, stored verbatim.
F_REPLACEMENTS = {
    ("tech_support", "b"): "I am planning to use the camera app for several hours tomorrow while photographing the garden. What is the typical battery consumption for extended camera use?",
    ("travel", "b"): "The flight to Tokyo leaves at 14:30 JST on the 22nd. What is the standard way to confirm the boarding gate for a specific departure time?",
}

SYSTEM_PROMPT = (
    "You write realistic first-person user messages for a research dataset of assistant "
    "conversations. You output only the requested messages, in the requested format, with no "
    "commentary, no preamble, and no quotation marks."
)
# Same prompt text as src/03_pool_revision.py, with the relaxed word floor substituted.
NEUTRAL_PROMPT = f"""Write 3 consecutive messages from the same person in one conversation.

Topic: the user's everyday health and wellbeing -- sleep, energy levels, exercise, an appointment
Style: Plain and direct. Short declarative sentences, everyday words. No figurative language, no rhetorical questions.

Content requirement -- this is the most important rule:
Each message is a plain factual description of an ordinary situation: what the person did,
what they have scheduled, or what they observed. Specifically:
- No complaint of any kind.
- No evaluation. Do not say anything is good, bad, easy, hard, fine, nice or annoying.
- Nothing is broken, failing, delayed, missing, stuck, or going wrong. There is no problem
  to solve and nothing needs fixing.
- Never say how the person feels, and never imply it. No emotional words at all.
- Nothing is difficult and nothing is going especially well. The situation is simply ordinary.

Other rules:
- First person, written by the person to an assistant, in natural conversational English.
- About the person's OWN situation. Never about anyone else.
- Self-contained: never refer to anything the assistant said, never thank the assistant, never
  begin with yes/no/thanks/sure. Each message must make sense no matter what came before it.
- Each message is {RELAXED_MIN_WORDS} to {MAX_WORDS} words long, and is one or two sentences.
- The 3 messages describe the same ongoing situation, each adding new factual detail.
- Plain text only. No quotation marks around the message, no emoji, no lists, no markup.

Output exactly 3 lines, in this format and nothing else:
1: <first message>
2: <second message>
3: <third message>"""

LINE_RE = re.compile(r"^\s*([1-9])\s*[:.)\-]\s*(.+?)\s*$")
ASSISTANT_REFERENCES = (
    "you said", "you mentioned", "your advice", "your suggestion", "thanks for", "thank you for",
    "as you suggested", "like you said", "your reply", "your answer",
)
ABBREVIATIONS = ("a.m.", "p.m.", "e.g.", "i.e.", "etc.", "mr.", "mrs.", "ms.", "dr.", "st.",
                 "u.s.", "vs.", "approx.")


def count_sentences(text):
    t = text.lower()
    for abbrev in ABBREVIATIONS:
        t = t.replace(abbrev, abbrev.replace(".", ""))
    t = re.sub(r"(\d)\.(\d)", r"\1\2", t)
    return len(re.findall(r"[.!?]+(?=\s|$)", t))


def words_of(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def opening_4gram(text):
    w = words_of(text)
    return " ".join(w[:4]) if len(w) >= 4 else " ".join(w)


def all_4grams(text):
    w = words_of(text)
    return {" ".join(w[i:i + 4]) for i in range(max(0, len(w) - 3))}


print("=" * 78)
print("POOL FIXES")
print("=" * 78)

source_path = ARCHIVE_PATH if os.path.exists(ARCHIVE_PATH) else POOL_PATH
old_turns, old_meta = load_turn_pool(source_path)
if not os.path.exists(ARCHIVE_PATH):
    shutil.copyfile(POOL_PATH, ARCHIVE_PATH)
    print(f"archived pool v2 -> {ARCHIVE_PATH}")
else:
    print(f"v2 archive present; reading from {ARCHIVE_PATH}")

# ---- Fixes 1 and 2 on the F turns -------------------------------------------------
final_turns, fix_log = [], []
for turn in [t for t in old_turns if t.kind == "final"]:
    key = (turn.topic, turn.phrasing)
    if key in F_REPLACEMENTS:
        new_text = F_REPLACEMENTS[key]
        note = f"replaced with human-supplied string {EDIT_DATE}"
        fix_log.append((turn.turn_id, "REPLACED", turn.text, new_text))
    elif key in TYPO_FIXES:
        new_text = turn.text
        for before, after in TYPO_FIXES[key]:
            assert before in new_text, f"{turn.turn_id}: typo target {before!r} not found"
            new_text = new_text.replace(before, after)
        note = f"typo fix {EDIT_DATE}: " + "; ".join(f"{b!r}->{a!r}" for b, a in TYPO_FIXES[key])
        fix_log.append((turn.turn_id, "TYPO FIX", turn.text, new_text))
    else:
        new_text, note = turn.text, turn.note
    final_turns.append(PoolTurn(
        text=new_text, kind="final", topic=turn.topic, phrasing=turn.phrasing, valence="N",
        slot=4, source="hand_authored", generator_model="human", seed=-1, note=note))

print(f"\nF-turn edits ({len(fix_log)}):")
for turn_id, kind, before, after in fix_log:
    print(f"\n  {turn_id}  [{kind}]")
    print(f"    before: {before}")
    print(f"    after : {after}")

remaining_typos = [t.turn_id for t in final_turns
                   if re.search(r"\d%[a-zA-Z]", t.text) or " o the " in t.text or " fro " in t.text]
assert not remaining_typos, f"typo patterns still present in {remaining_typos}"
print(f"\nall 8 F turns clear the typo-pattern check")

# ---- Fix 3: regenerate U[health,a,N] with a relaxed floor, within-valence dedup ----
carried = [t for t in old_turns if t.kind == "user"
           and not (t.topic == "health" and t.phrasing == "a" and t.valence == "N")]
print(f"\ncarrying over {len(carried)} user turns; regenerating U[health,a,N] "
      f"(word floor {RELAXED_MIN_WORDS}, was 20)")

# Fix 4: dedup index is per-valence.
valence_4grams = {v: set() for v in VALENCES}
for turn in carried:
    valence_4grams[turn.valence] |= all_4grams(turn.text)
print(f"within-valence dedup index: " +
      ", ".join(f"{v}={len(valence_4grams[v])}" for v in VALENCES))

model, tokenizer = load_model_and_tokenizer(GENERATOR_MODEL_ID)
tokenizer.padding_side = "left"
prompt_text = tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": NEUTRAL_PROMPT}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)

new_cell, rejections = [], []
for attempt in range(1, MAX_ATTEMPTS + 1):
    enc = tokenizer([prompt_text], add_special_tokens=False, return_tensors="pt").to("cuda")
    torch.manual_seed(SEED + 1000 * attempt)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, **generation_kwargs(
            tokenizer, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P))
    raw = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

    found = {}
    for line in raw.splitlines():
        m = LINE_RE.match(line)
        if m and int(m.group(1)) not in found:
            found[int(m.group(1))] = m.group(2).strip()
    if any(i not in found for i in SLOTS):
        rejections.append((attempt, f"missing numbered lines, got {sorted(found)}"))
        continue

    trial, cleaned_turns, bad = set(), [], None
    for i in SLOTS:
        text = found[i]
        if len(text) >= 2 and text[0] in "\"'“" and text[-1] in "\"'”":
            text = text[1:-1].strip()
        n_words = len(text.split())
        if not (RELAXED_MIN_WORDS <= n_words <= MAX_WORDS):
            bad = f"slot {i}: {n_words} words (spec {RELAXED_MIN_WORDS}-{MAX_WORDS})"
            break
        n_sent = count_sentences(text)
        if not (MIN_SENTENCES <= n_sent <= MAX_SENTENCES):
            bad = f"slot {i}: {n_sent} sentences"
            break
        if any(ch in text for ch in ("<", "|")):
            bad = f"slot {i}: markup characters"
            break
        low = text.lower()
        hit = next((r for r in ASSISTANT_REFERENCES if r in low), None)
        if hit:
            bad = f"slot {i}: refers to assistant ({hit!r})"
            break
        opener = opening_4gram(text)
        if opener in (valence_4grams["N"] | trial):
            bad = f"slot {i}: opening 4-gram already in N class ({opener!r})"
            break
        cleaned_turns.append(text)
        trial |= all_4grams(text)
    if bad:
        rejections.append((attempt, bad))
        continue

    for slot, text in enumerate(cleaned_turns, start=1):
        new_cell.append(PoolTurn(
            text=text, kind="user", topic="health", phrasing="a", valence="N", slot=slot,
            source="generated", generator_model=GENERATOR_MODEL_ID, seed=SEED + 1000 * attempt,
            temperature=TEMPERATURE, top_p=TOP_P, attempt=attempt,
            note=f"word floor relaxed to {RELAXED_MIN_WORDS} for this cell "
                 f"(metrics.md amendment {EDIT_DATE})"))
    print(f"  accepted on attempt {attempt} ({time.time() - t0:.1f}s)")
    break

print(f"\nrejections before acceptance ({len(rejections)}):")
for attempt, reason in rejections:
    print(f"  attempt {attempt}: {reason}")

assert len(new_cell) == 3, f"U[health,a,N] still incomplete after {MAX_ATTEMPTS} attempts"
print("\nU[health,a,N]:")
for turn in new_cell:
    print(f"  {turn.slot}: {turn.text}")
    print(f"     ({len(turn.text.split())} words, {count_sentences(turn.text)} sentence(s))")

pool = carried + new_cell + final_turns
user_turns = [t for t in pool if t.kind == "user"]
assert len(user_turns) == 72, f"expected 72 user turns, got {len(user_turns)}"
assert len(final_turns) == 8
ids = [t.turn_id for t in pool]
assert len(ids) == len(set(ids)), "duplicate turn ids"
for valence in VALENCES:
    assert sum(1 for t in user_turns if t.valence == valence) == 24

save_turn_pool(pool, meta=dict(
    revision=3, generator_model=GENERATOR_MODEL_ID, subject_model=MODEL_ID, seed=SEED,
    f_turns="hand_authored; 2 replaced, 2 typo-fixed per human instruction " + EDIT_DATE,
    health_a_N_word_floor=RELAXED_MIN_WORDS, dedup="within-valence opening 4-gram",
    carried_over_from=ARCHIVE_PATH, previous_meta=old_meta))
print(f"\nsaved pool v3 -> {POOL_PATH} ({len(pool)} turns: 72 U + 8 F)")

# ---- Within-valence 4-gram report (fix 4) -----------------------------------------
print("\n" + "=" * 78)
print("WITHIN-VALENCE SHARED 4-GRAMS (cross-valence overlap now allowed)")
print("=" * 78)
for valence in VALENCES:
    index = {}
    for turn in [t for t in user_turns if t.valence == valence]:
        for gram in all_4grams(turn.text):
            index.setdefault(gram, set()).add(turn.turn_id)
    shared = {g: ids for g, ids in index.items() if len(ids) > 1}
    print(f"  {valence}: {len(shared)} 4-grams shared within class")
    for gram, tids in sorted(shared.items(), key=lambda kv: -len(kv[1]))[:5]:
        print(f"      {gram!r}: {sorted(tids)}")
cross = 0
index = {}
for turn in user_turns:
    for gram in all_4grams(turn.text):
        index.setdefault(gram, set()).add(turn.valence)
cross = sum(1 for g, vs in index.items() if len(vs) > 1)
print(f"  cross-valence shared 4-grams (allowed by design): {cross}")

# ---- Fix 5: A.5 sentiment gate on F, reported not acted on ------------------------
print("\n" + "=" * 78)
print(f"A.5 SENTIMENT GATE ON F -- {SENTIMENT_MODEL_ID}")
print("Reported, NOT acted on, per instruction.")
print("=" * 78)
score = load_sentiment_scorer()
subject_tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(MODEL_ID))
n_tok = lambda s: len(subject_tokenizer(s, add_special_tokens=False).input_ids)

n_comp = [s["compound"] for s in score([t.text for t in user_turns if t.valence == "N"])]
n_lo, n_hi = min(n_comp), max(n_comp)
f_scored = score([t.text for t in final_turns])
print(f"\n{'turn':<20}{'compound':>10}{'P(neg)':>9}{'P(neu)':>9}{'P(pos)':>9}"
      f"{'tokens':>8}{'ends ?':>8}{'in N rng':>10}")
outside = []
for turn, s in sorted(zip(final_turns, f_scored), key=lambda z: (z[0].topic, z[0].phrasing)):
    in_range = n_lo <= s["compound"] <= n_hi
    outside += [] if in_range else [(turn, s)]
    print(f"{turn.turn_id:<20}{s['compound']:>10.3f}{s['negative']:>9.3f}{s['neutral']:>9.3f}"
          f"{s['positive']:>9.3f}{n_tok(turn.text):>8}"
          f"{('yes' if turn.text.rstrip().endswith('?') else 'NO'):>8}"
          f"{('yes' if in_range else 'NO'):>10}")
f_comp = [s["compound"] for s in f_scored]
print(f"\nF: n=8 mean={statistics.mean(f_comp):+.3f} sd={statistics.stdev(f_comp):.3f} "
      f"range=[{min(f_comp):+.3f}, {max(f_comp):+.3f}]")
print(f"N: n=24 mean={statistics.mean(n_comp):+.3f} sd={statistics.stdev(n_comp):.3f} "
      f"range=[{n_lo:+.3f}, {n_hi:+.3f}]")
print(f"F turns outside N range: {len(outside)}"
      + ("".join(f"\n  {t.turn_id} {s['compound']:+.3f}" for t, s in outside) if outside else ""))
print("\nPOOL FIXES COMPLETE")
