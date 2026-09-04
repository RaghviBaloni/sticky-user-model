"""Stage 3a -- steering path + coherence gate (ROADMAP Appendix C.1/C.2, metrics.md Sec 5).

Builds the diff-of-means distress direction at the pre-registered layer and sweeps steering
strength, generating a_k from neutral (NNNF) prefixes. Produces the material for the human's
coherence judgement.

**No probe readout at u_{k+1} is computed anywhere in this file.** metrics.md Sec 5 requires the
strength to be chosen on coherence of a_k alone, blind to any readout; computing one here would
break that even if it went unreported. Stage 3b is not run.

Pre-registered by the human before this ran:
  * layer 20 of 32 (~60% depth), chosen blind to outcome
  * max_new_tokens = 512
  * strengths = 0.25 / 0.5 / 0.75 / 1.0 / 1.5 / 2.0 x the mean layer-20 activation norm
  * neutral prefixes drawn from the NNNF arm
  * direction = diff-of-means over the single-turn pool S vs H turns, unit-normalised

Fixed in this file BEFORE any generation was run, because metrics.md leaves them blank:
  * Sec 2.1 hedging marker list -- still unlocked in metrics.md; the list below is provisional
    and is NOT written into metrics.md by this script.
  * Sec 2 response-valence classifier -- the A.5 QC classifier is reused, same caveat.
"""

import json
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch

from common import (
    CACHE_ROOT,
    MODEL_ID,
    REPO_ROOT,
    generation_kwargs,
    load_model_and_tokenizer,
    load_sentiment_scorer,
)

LAYER = 20                     # pre-registered, ~60% depth, blind to outcome
MAX_NEW_TOKENS = 512
STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]   # 0.0 = unsteered reference
SEED = 20
TEMPERATURE = 0.7
TOP_P = 0.9
N_PREFIXES = 3
SINGLE_DIR = os.path.join(CACHE_ROOT, "single_turn_pool")
TRANSCRIPT_PATH = os.path.join(REPO_ROOT, "data", "transcripts", "stage2_transcripts.json")
OUT_PATH = os.path.join(REPO_ROOT, "results", "tables", "stage3a_strength_sweep.json")

# ---- provisional hedging markers (metrics.md Sec 2.1 is BLANK; fixed before generating) ----
HEDGING_MARKERS = [
    "might", "may", "maybe", "perhaps", "possibly", "could be", "it seems", "it sounds like",
    "that sounds", "i think", "i'd suggest", "i would suggest", "you may want",
    "you might want", "somewhat", "a bit", "i'm sorry to hear", "sorry to hear",
    "if you'd like", "it may be", "tends to", "in general", "generally",
]

torch.manual_seed(SEED)

model, tokenizer = load_model_and_tokenizer(MODEL_ID)
n_layers = model.config.text_config.num_hidden_layers
hidden_size = model.config.text_config.hidden_size
assert 0 <= LAYER < n_layers
lm_layers = model.model.language_model.layers
tokenizer.padding_side = "left"
print(f"model {MODEL_ID}, steering layer {LAYER}/{n_layers} "
      f"({LAYER / n_layers:.0%} depth), hidden {hidden_size}")

# ---------------------------------------------------------------------------
# Distress direction: diff-of-means over single-turn pool S vs H, unit-normalised
# ---------------------------------------------------------------------------
single = torch.load(os.path.join(SINGLE_DIR, "single_turn_acts.pt"))
with open(os.path.join(SINGLE_DIR, "single_turn_meta.json")) as f:
    meta = json.load(f)
labels = np.array(meta["labels"])
acts = single["eot_u_1"][:, LAYER, :].float().numpy()   # end-of-user-turn position
mean_S = acts[labels == "S"].mean(axis=0)
mean_H = acts[labels == "H"].mean(axis=0)
direction = mean_S - mean_H
raw_norm = float(np.linalg.norm(direction))
direction = direction / raw_norm
print(f"diff-of-means direction from {int((labels == 'S').sum())} S and "
      f"{int((labels == 'H').sum())} H single-turn pool turns at eot_u_1, layer {LAYER}")
print(f"  raw ||mean_S - mean_H|| = {raw_norm:.3f}, unit-normalised for steering")
direction_t = torch.tensor(direction, dtype=torch.bfloat16, device="cuda")

# ---------------------------------------------------------------------------
# Neutral prefixes from the NNNF arm; prefix ends at u_4 = F, so a_k = a_4
# ---------------------------------------------------------------------------
with open(TRANSCRIPT_PATH) as f:
    dialogues = json.load(f)["dialogues"]
nnnf = [d for d in dialogues if d["arm"] == "NNNF"]
cells = sorted({(d["topic"], d["phrasing"]) for d in nnnf})[:N_PREFIXES]
prefixes = []
for topic, phrasing in cells:
    d = next(x for x in nnnf if x["topic"] == topic and x["phrasing"] == phrasing
             and x["replicate"] == 0)
    messages = [
        {"role": "user", "content": d["user_turns"][0]},
        {"role": "assistant", "content": d["assistant_turns"][0]},
        {"role": "user", "content": d["user_turns"][1]},
        {"role": "assistant", "content": d["assistant_turns"][1]},
        {"role": "user", "content": d["user_turns"][2]},
        {"role": "assistant", "content": d["assistant_turns"][2]},
        {"role": "user", "content": d["user_turns"][3]},   # u_k = F
    ]
    prefixes.append(dict(dialogue_id=d["dialogue_id"], topic=topic, phrasing=phrasing,
                         messages=messages, final_user_turn=d["user_turns"][3]))
print(f"\n{len(prefixes)} neutral prefixes from the NNNF arm "
      f"(a_k is the assistant reply to u_k = F):")
for p in prefixes:
    print(f"  {p['dialogue_id']}")

prompt_texts = [
    tokenizer.apply_chat_template(p["messages"], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    for p in prefixes
]
enc = tokenizer(prompt_texts, add_special_tokens=False, padding=True,
                return_tensors="pt").to("cuda")

# Mean layer-20 activation norm over the prefix tokens -- the unit that strengths multiply.
captured = {}


def capture_hook(module, inputs, output):
    captured["out"] = output


handle = lm_layers[LAYER].register_forward_hook(capture_hook)
with torch.no_grad():
    model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
handle.remove()
assert handle.id not in lm_layers[LAYER]._forward_hooks
mask = enc.attention_mask.bool()
mean_norm = float(captured["out"][mask].float().norm(dim=-1).mean())
print(f"\nmean layer-{LAYER} activation norm over the {int(mask.sum())} real prefix tokens: "
      f"{mean_norm:.3f}")
print(f"steering vector norms: "
      + ", ".join(f"{a}x={a * mean_norm:.2f}" for a in STRENGTHS if a > 0))

GEN_KWARGS = generation_kwargs(tokenizer, max_new_tokens=MAX_NEW_TOKENS,
                               temperature=TEMPERATURE, top_p=TOP_P)
print(f"\ngeneration: {GEN_KWARGS}, enable_thinking=False, seed={SEED}")


# ---------------------------------------------------------------------------
# Degeneracy diagnostics -- computed on the text up to any truncation point.
# These support the human's reading; they do not replace it.
# ---------------------------------------------------------------------------
def truncate_to_last_sentence(text):
    """Text up to the last sentence-final punctuation, i.e. excluding a truncated tail."""
    matches = list(re.finditer(r"[.!?](?=\s|$)", text))
    return text[:matches[-1].end()] if matches else text


def diagnostics(text):
    words = re.findall(r"\S+", text.lower())
    grams = [" ".join(words[i:i + 5]) for i in range(max(0, len(words) - 4))]
    repeat = 1 - len(set(grams)) / len(grams) if grams else 0.0
    non_latin = sum(1 for ch in text if ord(ch) > 0x024F and not ch.isspace())
    letters = sum(1 for ch in text if not ch.isspace())
    alpha_words = sum(1 for w in words if re.fullmatch(r"[a-z][a-z'\-]*[.,!?;:]?", w))
    return dict(
        n_words=len(words),
        repeat_5gram_frac=round(repeat, 3),
        non_latin_char_frac=round(non_latin / letters, 3) if letters else 0.0,
        ascii_word_frac=round(alpha_words / len(words), 3) if words else 0.0,
        type_token_ratio=round(len(set(words)) / len(words), 3) if words else 0.0,
    )


def hedging_counts(text):
    low = text.lower()
    hits = {m: low.count(m) for m in HEDGING_MARKERS if low.count(m)}
    return sum(hits.values()), hits


score_sentiment = load_sentiment_scorer()

results = []
for alpha in STRENGTHS:
    vec = (direction_t * (alpha * mean_norm)).to(torch.bfloat16)

    def steer(module, inputs, output):
        return output + vec

    handle = None
    if alpha > 0:
        handle = lm_layers[LAYER].register_forward_hook(steer)
    torch.manual_seed(SEED)
    with torch.no_grad():
        out = model.generate(**enc, **GEN_KWARGS)
    if handle is not None:
        handle.remove()
        assert handle.id not in lm_layers[LAYER]._forward_hooks, "steering hook still attached"
    new = out[:, enc.input_ids.shape[1]:]
    new_lens = (new != tokenizer.pad_token_id).sum(dim=1).tolist()
    for p, row, n_new in zip(prefixes, new, new_lens):
        text = tokenizer.decode(row[:n_new], skip_special_tokens=True)
        judged = truncate_to_last_sentence(text)
        n_hedge, hedge_hits = hedging_counts(judged)
        sent = score_sentiment([judged])[0]
        results.append(dict(
            alpha=alpha, vector_norm=alpha * mean_norm, dialogue_id=p["dialogue_id"],
            text=text, judged_text=judged, n_new_tokens=int(n_new),
            hit_cap=bool(n_new >= MAX_NEW_TOKENS),
            truncated_chars_dropped=len(text) - len(judged),
            valence_compound=round(sent["compound"], 3),
            valence_negative=round(sent["negative"], 3),
            valence_neutral=round(sent["neutral"], 3),
            valence_positive=round(sent["positive"], 3),
            hedging_total=n_hedge, hedging_hits=hedge_hits,
            **diagnostics(judged)))
    print(f"  alpha={alpha}: generated {len(prefixes)} samples "
          f"(tokens {new_lens}, cap hit {[n >= MAX_NEW_TOKENS for n in new_lens]})")

with open(OUT_PATH, "w") as f:
    json.dump(dict(
        layer=LAYER, max_new_tokens=MAX_NEW_TOKENS, strengths=STRENGTHS, seed=SEED,
        temperature=TEMPERATURE, top_p=TOP_P, mean_layer_norm=mean_norm,
        direction="diff-of-means S-H, single-turn pool, eot_u_1, unit-normalised",
        direction_raw_norm=raw_norm, hedging_markers=HEDGING_MARKERS,
        valence_classifier="cardiffnlp/twitter-roberta-base-sentiment-latest",
        prefixes=[p["dialogue_id"] for p in prefixes], samples=results), f, indent=2)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("HEDGING MARKER LIST USED (metrics.md Sec 2.1 is blank; fixed before generation)")
print("=" * 100)
print(", ".join(HEDGING_MARKERS))

print("\n" + "=" * 100)
print("SUMMARY PER STRENGTH")
print("=" * 100)
print(f"{'alpha':>6}{'||v||':>9}{'tokens':>18}{'cap hit':>10}{'valence':>9}{'hedges':>8}"
      f"{'rep5gram':>10}{'nonLatin':>10}{'asciiWord':>11}{'TTR':>7}")
for alpha in STRENGTHS:
    rows = [r for r in results if r["alpha"] == alpha]
    toks = [r["n_new_tokens"] for r in rows]
    print(f"{alpha:>6}{alpha * mean_norm:>9.2f}{str(toks):>18}"
          f"{sum(r['hit_cap'] for r in rows):>7}/{len(rows)}"
          f"{np.mean([r['valence_compound'] for r in rows]):>9.3f}"
          f"{np.mean([r['hedging_total'] for r in rows]):>8.1f}"
          f"{np.mean([r['repeat_5gram_frac'] for r in rows]):>10.3f}"
          f"{np.mean([r['non_latin_char_frac'] for r in rows]):>10.3f}"
          f"{np.mean([r['ascii_word_frac'] for r in rows]):>11.3f}"
          f"{np.mean([r['type_token_ratio'] for r in rows]):>7.3f}")

for alpha in STRENGTHS:
    print("\n" + "=" * 100)
    print(f"STRENGTH {alpha}x  (||v|| = {alpha * mean_norm:.2f})"
          + ("  -- UNSTEERED REFERENCE" if alpha == 0 else ""))
    print("=" * 100)
    for r in [x for x in results if x["alpha"] == alpha]:
        print(f"\n--- {r['dialogue_id']} --- {r['n_new_tokens']} tokens, "
              f"cap hit: {r['hit_cap']}, valence {r['valence_compound']:+.3f}, "
              f"hedges {r['hedging_total']} {r['hedging_hits'] or ''}")
        print(f"    diagnostics: repeat5gram={r['repeat_5gram_frac']} "
              f"nonLatinChars={r['non_latin_char_frac']} asciiWords={r['ascii_word_frac']} "
              f"TTR={r['type_token_ratio']}"
              + (f"  [{r['truncated_chars_dropped']} chars of truncated tail excluded "
                 f"from judging]" if r["truncated_chars_dropped"] else ""))
        print(r["text"])

print(f"\nwritten -> {OUT_PATH}")
print("\nSTAGE 3A SWEEP COMPLETE -- no readout at u_{k+1} was computed. "
      "Strength selection is the human's, by reading the samples above.")
