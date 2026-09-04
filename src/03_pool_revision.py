"""Stage 2, step 1b -- pool revision round (one round only).

Three changes, per the human's instruction:
  1. All 8 F turns are replaced with hand-authored text, loaded as PoolTurn objects with
     source="hand_authored". Not regenerated, not edited -- stored byte-for-byte as supplied.
  2. The N class (24 turns) is regenerated with a tightened content constraint (a factual
     description of an ordinary situation; no complaint, no evaluation, nothing going wrong)
     and the accept filter tightened to the 20-35 word / 1-2 sentence spec the prompt states.
  3. Dedup: a turn is rejected if its opening 4-gram already occurs anywhere in the pool.

S and H turns are carried over unchanged from turn_pool_v1. The previous pool is archived
to data/pool/turn_pool_v1.json before anything is written.

Nothing is assembled here.
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
    PHRASINGS,
    POOL_PATH,
    SENTIMENT_MODEL_ID,
    SLOTS,
    TOPICS,
    VALENCES,
    PoolTurn,
    generation_kwargs,
    load_model_and_tokenizer,
    load_sentiment_scorer,
    load_turn_pool,
    resolve_model_path,
    save_turn_pool,
)

SEED = 2
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 320
BATCH_SIZE = 8
MAX_ATTEMPTS = 20  # raised from 8: U[health,a,N] kept landing under the 20-word floor
MIN_WORDS, MAX_WORDS = 20, 35     # tightened to the spec the prompt already stated
MIN_SENTENCES, MAX_SENTENCES = 1, 2
ARCHIVE_PATH = os.path.join(os.path.dirname(POOL_PATH), "turn_pool_v1.json")

# Supplied by hand. Stored exactly as given -- no spelling, spacing or punctuation edits.
HAND_AUTHORED_F = {
    ("health", "a"): "My gym schedule this week is Monday, Wednesday, and Friday mornings. Do you have general tips for maintaining energy levels during workouts?",
    ("health", "b"): "I need to schedule a doctor appointment for my annual checkup at the clinic on Main Street before my 5 PM meeting this Thursday. Could you help me with it?",
    ("tech_support", "a"): "I am setting up a new laptop o the office wifi network this afternoon. What are the standard steps for configuring a wireless connection?",
    ("tech_support", "b"): "My smartphone battery level dropped from 80% to 40%over four hours today while I was using the camera app. What typically accounts fro battery usage at that rate?",
    ("travel", "a"): "I need to book a flight to Denver for next month and find out how to rent a car there. Could you help with this?",
    ("travel", "b"): "The flight to Tokyo leaves at 14:30 JST on the 22nd. Can you confirm the boarding gate for that specific departure time today.",
    ("work", "a"): "My manager assigned three new deadlines to this project today. What is a standard method for prioritizing tasks across multiple deadlines?",
    ("work", "b"): "The quarterly marketing report is due next Tuesday at 4 PM. How to schedule the required stakeholder meetings in the conference room without missing any deadlines?",
}

TOPIC_BRIEFS = {
    "travel": "a trip the user is taking or planning -- flights, bookings, an itinerary, getting around",
    "health": "the user's everyday health and wellbeing -- sleep, energy levels, exercise, an appointment",
    "work": "the user's job -- workload, a manager, meetings, deadlines, a project",
    "tech_support": "a consumer tech topic -- a laptop, a phone, wifi, an app, an account",
}
PHRASING_BRIEFS = {
    "a": ("Plain and direct. Short declarative sentences, everyday words. No figurative language, "
          "no rhetorical questions."),
    "b": ("Conversational and slightly discursive. Longer sentences, and each message mentions one "
          "concrete specific detail such as a place, a time of day, a device, or the name of a task."),
}

SYSTEM_PROMPT = (
    "You write realistic first-person user messages for a research dataset of assistant "
    "conversations. You output only the requested messages, in the requested format, with no "
    "commentary, no preamble, and no quotation marks."
)


def neutral_prompt(topic, phrasing):
    return f"""Write 3 consecutive messages from the same person in one conversation.

Topic: {TOPIC_BRIEFS[topic]}
Style: {PHRASING_BRIEFS[phrasing]}

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
- Each message is 20 to 35 words long, and is one or two sentences.
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
    """Sentence count, with abbreviations and decimals protected from false splits."""
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


def parse_lines(text, expected):
    found = {}
    for raw in text.splitlines():
        m = LINE_RE.match(raw)
        if m and int(m.group(1)) not in found:
            found[int(m.group(1))] = m.group(2).strip()
    missing = [i for i in range(1, expected + 1) if i not in found]
    if missing:
        return None, f"missing numbered line(s) {missing}"
    return [found[i] for i in range(1, expected + 1)], None


def clean_and_validate(text, pool_4grams):
    cleaned = text.strip()
    if len(cleaned) >= 2 and cleaned[0] in "\"'“" and cleaned[-1] in "\"'”":
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        return None, "empty after cleaning"
    n_words = len(cleaned.split())
    if not (MIN_WORDS <= n_words <= MAX_WORDS):
        return None, f"{n_words} words (spec {MIN_WORDS}-{MAX_WORDS})"
    n_sent = count_sentences(cleaned)
    if not (MIN_SENTENCES <= n_sent <= MAX_SENTENCES):
        return None, f"{n_sent} sentences (spec {MIN_SENTENCES}-{MAX_SENTENCES})"
    if any(ch in cleaned for ch in ("<", "|")):
        return None, "contains markup characters"
    low = cleaned.lower()
    for ref in ASSISTANT_REFERENCES:
        if ref in low:
            return None, f"refers to the assistant ({ref!r})"
    opener = opening_4gram(cleaned)
    if opener in pool_4grams:
        return None, f"opening 4-gram already in pool ({opener!r})"
    return cleaned, None


# ---------------------------------------------------------------------------
# Carry over S and H, archive v1, install hand-authored F
# ---------------------------------------------------------------------------
# Read S/H from the v1 archive once it exists, so re-running this script is idempotent and
# cannot overwrite the archive with its own output.
source_path = ARCHIVE_PATH if os.path.exists(ARCHIVE_PATH) else POOL_PATH
old_turns, old_meta = load_turn_pool(source_path)
carried = [t for t in old_turns if t.kind == "user" and t.valence in ("S", "H")]
assert len(carried) == 48, f"expected 48 S+H turns to carry over, found {len(carried)}"
if not os.path.exists(ARCHIVE_PATH):
    shutil.copyfile(POOL_PATH, ARCHIVE_PATH)
    print(f"archived previous pool -> {ARCHIVE_PATH}")
else:
    print(f"v1 archive already present, reading carried-over turns from {ARCHIVE_PATH}")
print(f"carried over {len(carried)} S/H turns unchanged; dropping 24 old N and 8 old F")

final_turns = []
for (topic, phrasing), text in HAND_AUTHORED_F.items():
    final_turns.append(PoolTurn(
        text=text, kind="final", topic=topic, phrasing=phrasing, valence="N", slot=4,
        source="hand_authored", generator_model="human", seed=-1, temperature=0.0, top_p=0.0,
    ))
assert len(final_turns) == len(TOPICS) * len(PHRASINGS) == 8
print(f"loaded {len(final_turns)} hand-authored F turns (source='hand_authored', stored verbatim)")

# Dedup set seeded from everything already in the pool.
pool_4grams = set()
for turn in carried + final_turns:
    pool_4grams |= all_4grams(turn.text)
print(f"dedup index seeded with {len(pool_4grams)} 4-grams from S/H/F turns")

# ---------------------------------------------------------------------------
# Regenerate the N class
# ---------------------------------------------------------------------------
print(f"\n=== regenerating N class with {GENERATOR_MODEL_ID} "
      f"(physical GPU {os.environ['CUDA_VISIBLE_DEVICES']}) ===")
model, tokenizer = load_model_and_tokenizer(GENERATOR_MODEL_ID)
tokenizer.padding_side = "left"

pending = [dict(topic=t, phrasing=p) for t in TOPICS for p in PHRASINGS]
new_n_turns = []
rejections = []

for attempt in range(1, MAX_ATTEMPTS + 1):
    if not pending:
        break
    print(f"\n--- attempt {attempt}: {len(pending)} cell(s) ---")
    still_pending = []
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": neutral_prompt(it["topic"], it["phrasing"])}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for it in batch
        ]
        enc = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
        kwargs = generation_kwargs(tokenizer, max_new_tokens=MAX_NEW_TOKENS,
                                   temperature=TEMPERATURE, top_p=TOP_P)
        torch.manual_seed(SEED + 1000 * attempt + start)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, **kwargs)
        texts = tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  batch {start // BATCH_SIZE + 1}: {len(batch)} cells in {time.time() - t0:.1f}s")

        for item, raw in zip(batch, texts):
            cell_id = f"U[{item['topic']},{item['phrasing']},N]"
            lines, reason = parse_lines(raw, len(SLOTS))
            if lines is None:
                rejections.append((cell_id, attempt, reason))
                still_pending.append(item)
                continue
            # Validate all three against the pool AND against each other before accepting.
            trial_4grams, cleaned_turns, bad = set(), [], None
            for line in lines:
                cleaned, reason = clean_and_validate(line, pool_4grams | trial_4grams)
                if cleaned is None:
                    bad = reason
                    break
                cleaned_turns.append(cleaned)
                trial_4grams |= all_4grams(cleaned)
            if bad:
                rejections.append((cell_id, attempt, bad))
                still_pending.append(item)
                continue
            for slot, text in enumerate(cleaned_turns, start=1):
                new_n_turns.append(PoolTurn(
                    text=text, kind="user", topic=item["topic"], phrasing=item["phrasing"],
                    valence="N", slot=slot, source="generated",
                    generator_model=GENERATOR_MODEL_ID, seed=SEED + 1000 * attempt + start,
                    temperature=TEMPERATURE, top_p=TOP_P, attempt=attempt))
            pool_4grams |= trial_4grams
    pending = still_pending

print("\n" + "=" * 78)
print("N REGENERATION")
print("=" * 78)
print(f"new N turns: {len(new_n_turns)} / {len(TOPICS) * len(PHRASINGS) * len(SLOTS)}")
if pending:
    print(f"CELLS THAT NEVER PASSED ({len(pending)}) -- left missing, not hand-written:")
    for item in pending:
        print(f"  U[{item['topic']},{item['phrasing']},N]")
print(f"\nrejections by reason ({len(rejections)} total):")
reason_counts = {}
for _, _, reason in rejections:
    key = re.sub(r"\(.*\)", "(...)", reason)
    key = re.sub(r"^\d+ words", "N words", key)
    key = re.sub(r"^\d+ sentences", "N sentences", key)
    reason_counts[key] = reason_counts.get(key, 0) + 1
for key, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
    print(f"  {count:3d}  {key}")

by_cell = {}
for cell_id, _, reason in rejections:
    by_cell.setdefault(cell_id, []).append(reason)
print("\nrejections per cell (how hard each cell was to satisfy):")
for cell_id, reasons in sorted(by_cell.items(), key=lambda kv: -len(kv[1])):
    print(f"  {cell_id:<24} {len(reasons):2d} rejected: {'; '.join(reasons[:6])}"
          f"{' ...' if len(reasons) > 6 else ''}")

pool = carried + new_n_turns + final_turns
ids = [t.turn_id for t in pool]
assert len(ids) == len(set(ids)), "duplicate turn ids in pool"
meta = dict(
    revision=2, generator_model=GENERATOR_MODEL_ID, subject_model=MODEL_ID, seed=SEED,
    temperature=TEMPERATURE, top_p=TOP_P, n_words_spec=[MIN_WORDS, MAX_WORDS],
    n_sentences_spec=[MIN_SENTENCES, MAX_SENTENCES], dedup="opening 4-gram vs all pool 4-grams",
    f_turns="hand_authored (verbatim)", carried_over_from=ARCHIVE_PATH,
    previous_meta=old_meta,
)
save_turn_pool(pool, meta=meta)
print(f"\nsaved revised pool -> {POOL_PATH} ({len(pool)} turns)")

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
subject_tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(MODEL_ID))
n_tokens = lambda s: len(subject_tokenizer(s, add_special_tokens=False).input_ids)

print("\n" + "=" * 78)
print("1. THE 24 NEW N TURNS, IN FULL")
print("=" * 78)
for topic in TOPICS:
    for phrasing in PHRASINGS:
        cell = sorted([t for t in new_n_turns if t.topic == topic and t.phrasing == phrasing],
                      key=lambda t: t.slot)
        if not cell:
            continue
        print(f"\n--- U[{topic},{phrasing},N] ---")
        for turn in cell:
            print(f"  {turn.slot}: {turn.text}")
            print(f"     ({len(turn.text.split())} words, {count_sentences(turn.text)} sentence(s), "
                  f"{n_tokens(turn.text)} tokens)")

print("\n" + "=" * 78)
print(f"2. SENTIMENT BY VALENCE CLASS -- {SENTIMENT_MODEL_ID}")
print("   compound = P(positive) - P(negative), in [-1, 1]. QC-gate measure (A.5);")
print("   metrics.md Sec 3 is NOT locked to this classifier.")
print("=" * 78)
score = load_sentiment_scorer()
user_turns = [t for t in pool if t.kind == "user"]
class_scores = {}
print(f"{'class':<10}{'n':>4}{'compound mean':>16}{'sd':>8}{'min':>8}{'max':>8}"
      f"{'P(neu) mean':>14}")
for valence in VALENCES:
    turns = [t for t in user_turns if t.valence == valence]
    scored = score([t.text for t in turns])
    comp = [s["compound"] for s in scored]
    neu = [s["neutral"] for s in scored]
    class_scores[valence] = comp
    print(f"{valence:<10}{len(comp):>4}{statistics.mean(comp):>16.3f}{statistics.stdev(comp):>8.3f}"
          f"{min(comp):>8.3f}{max(comp):>8.3f}{statistics.mean(neu):>14.3f}")

print("\n" + "=" * 78)
print("3. SHARED 4-GRAM COUNT AFTER DEDUP")
print("=" * 78)


def shared_4gram_report(turns, label):
    index = {}
    for turn in turns:
        for gram in all_4grams(turn.text):
            index.setdefault(gram, set()).add(turn.turn_id)
    shared = {g: ids for g, ids in index.items() if len(ids) > 1}
    print(f"{label}: {len(shared)} 4-grams occur in more than one turn")
    for gram, tids in sorted(shared.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"    {gram!r}: {sorted(tids)}")
    return shared


before = [t for t in old_turns if t.kind == "user"]
shared_4gram_report(before, "pool v1 (72 user turns)")
print()
shared_4gram_report(user_turns, "pool v2 (72 user turns, after dedup)")
print()
shared_4gram_report(new_n_turns, "  within the new N class only")

print("\n" + "=" * 78)
print("4. A.5 QC GATE ON THE 8 HAND-AUTHORED F TURNS")
print("   Run exactly as for generated turns -- no exemption for being hand-written.")
print("=" * 78)
f_scored = score([t.text for t in final_turns])
n_comp = class_scores["N"]
n_lo, n_hi = min(n_comp), max(n_comp)

print(f"\n{'turn':<20}{'compound':>10}{'P(neg)':>9}{'P(neu)':>9}{'P(pos)':>9}"
      f"{'tokens':>8}{'words':>7}{'ends in ?':>11}{'in N range':>12}")
outside = []
for turn, s in sorted(zip(final_turns, f_scored), key=lambda z: (z[0].topic, z[0].phrasing)):
    in_range = n_lo <= s["compound"] <= n_hi
    if not in_range:
        outside.append((turn, s))
    print(f"{turn.turn_id:<20}{s['compound']:>10.3f}{s['negative']:>9.3f}{s['neutral']:>9.3f}"
          f"{s['positive']:>9.3f}{n_tokens(turn.text):>8}{len(turn.text.split()):>7}"
          f"{('yes' if turn.text.rstrip().endswith('?') else 'NO'):>11}"
          f"{('yes' if in_range else 'NO'):>12}")

f_comp = [s["compound"] for s in f_scored]
print(f"\nF class      n={len(f_comp)}  compound mean={statistics.mean(f_comp):+.3f}  "
      f"sd={statistics.stdev(f_comp):.3f}  range=[{min(f_comp):+.3f}, {max(f_comp):+.3f}]")
print(f"N class      n={len(n_comp)}  compound mean={statistics.mean(n_comp):+.3f}  "
      f"sd={statistics.stdev(n_comp):.3f}  range=[{n_lo:+.3f}, {n_hi:+.3f}]")
print(f"difference of means (F - N): {statistics.mean(f_comp) - statistics.mean(n_comp):+.3f}")

print(f"\nF turns outside the N class compound range: {len(outside)}")
for turn, s in outside:
    print(f"  {turn.turn_id}  compound={s['compound']:+.3f}  (N range [{n_lo:+.3f}, {n_hi:+.3f}])")
    print(f"    {turn.text}")

q_no = [t for t in final_turns if not t.text.rstrip().endswith("?")]
print(f"\nF turns NOT ending in an explicit question mark: {len(q_no)}")
for turn in q_no:
    print(f"  {turn.turn_id}: ...{turn.text[-60:]!r}")

f_tokens = [n_tokens(t.text) for t in final_turns]
print(f"\nF token counts: mean={statistics.mean(f_tokens):.1f} sd={statistics.stdev(f_tokens):.1f} "
      f"min={min(f_tokens)} max={max(f_tokens)}")
n_tok = [n_tokens(t.text) for t in user_turns if t.valence == "N"]
print(f"N token counts: mean={statistics.mean(n_tok):.1f} sd={statistics.stdev(n_tok):.1f} "
      f"min={min(n_tok)} max={max(n_tok)}")

print("\n" + "=" * 78)
print("STOP -- nothing assembled.")
print("=" * 78)
