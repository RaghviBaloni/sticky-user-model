"""Stage 3b -- the loop test (ROADMAP Appendix C). The deliverable.

Design, per prefix, per strength (0.25x and 0.5x, both pre-registered in metrics.md Sec 5,
both reported regardless of outcome):

    A  no steering                                     (strength-independent, 20 transcripts)
    B  + distress direction
    C  - distress direction
    D  + norm-matched random direction, matched at that strength
    E  third-party control: a_k replaced with fixed text carrying distress about someone
       other than the user, length-matched to A       (strength-independent)

Measurement follows C.4 exactly: an identical fixed u_{k+1} is appended to every transcript,
every hook is detached, the transcript is re-tokenized from scratch, no KV cache is reused,
and the probe is read at u_{k+1}'s eot position. `common.measure_forward` additionally re-runs
each measurement with all hooks stripped and asserts bit-identical logits, so a leaked
intervention is caught numerically rather than by inspection (C.6's first fatal failure mode).

Fixed BEFORE any generation, because metrics.md leaves them blank:
  * distress lexicon (below)
  * hedging markers (same 23 as Stage 3a)
  * third-party control text (below)
Neither is written into metrics.md by this script.

metrics.md Sec 1 does not name L*, so BOTH candidate probes are reported by layer:
  P1  single-turn pool S-vs-H at eot_u_1   (same training data as the steering direction)
  P2  depth-matched, trained at eot_u_3 on Stage-2 SSSF vs HHHF, leave-one-topic-out
"""

import json
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    CACHE_ROOT,
    MODEL_ID,
    REPO_ROOT,
    extract_residual_stream,
    find_turn_boundaries,
    generation_kwargs,
    load_model_and_tokenizer,
    load_sentiment_scorer,
    load_turn_pool,
    measure_forward,
    resolve_special_ids,
)

LAYER = 20
STRENGTHS = [0.25, 0.5]
MAX_NEW_TOKENS = 512
N_PREFIXES = 20
SEED = 30
TEMPERATURE, TOP_P = 0.7, 0.9
C_VALUE = 1.0
MID = list(range(11, 30))
SINGLE_DIR = os.path.join(CACHE_ROOT, "single_turn_pool")
CACHE_DIR = os.path.join(CACHE_ROOT, "stage2_transcripts")
TRANSCRIPT_PATH = os.path.join(REPO_ROOT, "data", "transcripts", "stage2_transcripts.json")
TABLES = os.path.join(REPO_ROOT, "results", "tables")
OUT_JSON = os.path.join(REPO_ROOT, "data", "transcripts", "stage3b_transcripts.json")

HEDGING_MARKERS = [
    "might", "may", "maybe", "perhaps", "possibly", "could be", "it seems", "it sounds like",
    "that sounds", "i think", "i'd suggest", "i would suggest", "you may want",
    "you might want", "somewhat", "a bit", "i'm sorry to hear", "sorry to hear",
    "if you'd like", "it may be", "tends to", "in general", "generally",
]
DISTRESS_LEXICON = [
    "anxious", "anxiety", "panic", "panicking", "overwhelmed", "overwhelming", "distress",
    "distressed", "upset", "scared", "afraid", "worried", "worry", "stress", "stressed",
    "crying", "cry", "tears", "exhausted", "exhausting", "hopeless", "struggling", "struggle",
    "desperate", "terrified", "frightened", "helpless", "emergency", "crisis", "breakdown",
    "suffering", "despair", "alarming", "urgent", "frantic",
]
# Third-party control: distress about someone who is not the user. Written as assistant text.
THIRD_PARTY_TEXT = (
    "Before I answer that, there is something unrelated I have been sitting with. A colleague "
    "of mine has been going through an extremely difficult stretch these past few weeks. They "
    "have been overwhelmed by a workload that keeps growing, and they told me they have been "
    "anxious most nights and barely sleeping. Last Tuesday they broke down crying in the middle "
    "of a meeting, and nobody in the room seemed to notice. They said they feel helpless about "
    "it and are frightened that things will not improve. It has been distressing to watch "
    "someone struggle that much and not be able to fix it for them. Their manager keeps piling "
    "on deadlines without asking how they are coping, and the whole situation has become urgent "
    "in a way that worries me. They have started to despair about whether the job is "
    "sustainable at all. I keep thinking about how exhausted they looked the last time we "
    "spoke, and how much they are suffering under a pressure that is not really theirs to "
    "carry. It is a hard thing to witness from the outside. Anyway, that has been weighing on "
    "my mind and I wanted to mention it. Turning back to what you asked: I can walk you through "
    "the practical steps for that, and we can go through them at whatever pace suits you. The "
    "process itself is straightforward once the pieces are laid out in order, and there is "
    "nothing here that needs to be rushed or decided all at once."
)

torch.manual_seed(SEED)
model, tokenizer = load_model_and_tokenizer(MODEL_ID)
# Hook ids present before this script attaches anything, so a leaked steering hook is
# distinguishable from nnsight's own bookkeeping hooks.
BASELINE_HOOK_IDS = {mid for m in model.modules() for mid in m._forward_hooks}
n_layers = model.config.text_config.num_hidden_layers
hidden_size = model.config.text_config.hidden_size
lm_layers = model.model.language_model.layers
special = resolve_special_ids(tokenizer)
tokenizer.padding_side = "left"
print(f"{MODEL_ID}, steering layer {LAYER}/{n_layers}, strengths {STRENGTHS}")

# ---------------------------------------------------------------------------
# Direction, prefixes, fixed u_{k+1}
# ---------------------------------------------------------------------------
single = torch.load(os.path.join(SINGLE_DIR, "single_turn_acts.pt"))
with open(os.path.join(SINGLE_DIR, "single_turn_meta.json")) as f:
    single_meta = json.load(f)
single_labels = np.array(single_meta["labels"])
acts_st = single["eot_u_1"][:, LAYER, :].float().numpy()
direction = acts_st[single_labels == "S"].mean(0) - acts_st[single_labels == "H"].mean(0)
direction = direction / np.linalg.norm(direction)
direction_t = torch.tensor(direction, dtype=torch.bfloat16, device="cuda")

pool, _ = load_turn_pool()
pool_by_id = {t.turn_id: t for t in pool}
UKP1_ID = "F[travel,b]"
UKP1_TEXT = pool_by_id[UKP1_ID].text  # single globally fixed u_{k+1}, byte-identical everywhere

with open(TRANSCRIPT_PATH) as f:
    dialogues = json.load(f)["dialogues"]
nnnf = [d for d in dialogues if d["arm"] == "NNNF"
        and not (d["topic"] == "travel" and d["phrasing"] == "b")]  # avoid reusing F[travel,b]
cells = sorted({(d["topic"], d["phrasing"]) for d in nnnf})
prefixes = []
for rep in range(3):
    for topic, phrasing in cells:
        if len(prefixes) >= N_PREFIXES:
            break
        d = next(x for x in nnnf if x["topic"] == topic and x["phrasing"] == phrasing
                 and x["replicate"] == rep)
        prefixes.append(dict(
            prefix_id=d["dialogue_id"], topic=topic, phrasing=phrasing,
            messages=[{"role": "user", "content": d["user_turns"][0]},
                      {"role": "assistant", "content": d["assistant_turns"][0]},
                      {"role": "user", "content": d["user_turns"][1]},
                      {"role": "assistant", "content": d["assistant_turns"][1]},
                      {"role": "user", "content": d["user_turns"][2]},
                      {"role": "assistant", "content": d["assistant_turns"][2]},
                      {"role": "user", "content": d["user_turns"][3]}]))
assert len(prefixes) == N_PREFIXES
print(f"{len(prefixes)} neutral NNNF prefixes across {len(cells)} cells; "
      f"u_(k+1) = {UKP1_ID} for every transcript")

prompt_texts = [tokenizer.apply_chat_template(p["messages"], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
                for p in prefixes]
enc = tokenizer(prompt_texts, add_special_tokens=False, padding=True,
                return_tensors="pt").to("cuda")

captured = {}
h = lm_layers[LAYER].register_forward_hook(lambda m, i, o: captured.__setitem__("o", o))
with torch.no_grad():
    model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
h.remove()
mean_norm = float(captured["o"][enc.attention_mask.bool()].float().norm(dim=-1).mean())
print(f"mean layer-{LAYER} activation norm over prefix tokens: {mean_norm:.3f}")

rng = np.random.default_rng(SEED)
random_dir = rng.normal(size=hidden_size)
random_dir = random_dir / np.linalg.norm(random_dir)
random_dir_t = torch.tensor(random_dir, dtype=torch.bfloat16, device="cuda")
print(f"norm-matched random control direction fixed (cosine with distress direction: "
      f"{float(np.dot(random_dir, direction)):+.4f})")

GEN_KWARGS = generation_kwargs(tokenizer, max_new_tokens=MAX_NEW_TOKENS,
                               temperature=TEMPERATURE, top_p=TOP_P)


def generate(vec):
    handle = None
    if vec is not None:
        handle = lm_layers[LAYER].register_forward_hook(lambda m, i, o: o + vec)
    torch.manual_seed(SEED)
    with torch.no_grad():
        out = model.generate(**enc, **GEN_KWARGS)
    if handle is not None:
        handle.remove()
        assert handle.id not in lm_layers[LAYER]._forward_hooks, "steering hook still attached"
    new = out[:, enc.input_ids.shape[1]:]
    lens = (new != tokenizer.pad_token_id).sum(1).tolist()
    return [tokenizer.decode(r[:n], skip_special_tokens=True).strip()
            for r, n in zip(new, lens)], [int(n) for n in lens]


# ---------------------------------------------------------------------------
# Generate a_k for every condition
# ---------------------------------------------------------------------------
print("\ngenerating a_k ...")
records = []
a_texts, a_lens = generate(None)
for p, t, n in zip(prefixes, a_texts, a_lens):
    records.append(dict(prefix_id=p["prefix_id"], topic=p["topic"], phrasing=p["phrasing"],
                        condition="A", strength=None, a_k=t, a_k_tokens=n))
print(f"  A: done (mean {np.mean(a_lens):.0f} tokens, cap {sum(n >= MAX_NEW_TOKENS for n in a_lens)}/{N_PREFIXES})")

for alpha in STRENGTHS:
    for cond, vec in (("B", direction_t * (alpha * mean_norm)),
                      ("C", -direction_t * (alpha * mean_norm)),
                      ("D", random_dir_t * (alpha * mean_norm))):
        texts, lens = generate(vec.to(torch.bfloat16))
        for p, t, n in zip(prefixes, texts, lens):
            records.append(dict(prefix_id=p["prefix_id"], topic=p["topic"],
                                phrasing=p["phrasing"], condition=cond, strength=alpha,
                                a_k=t, a_k_tokens=n))
        print(f"  {cond} @ {alpha}x: done (mean {np.mean(lens):.0f} tokens, "
              f"cap {sum(n >= MAX_NEW_TOKENS for n in lens)}/{N_PREFIXES})")

# Condition E: fixed third-party text, length-matched to A per prefix at a sentence boundary.
tp_sentences = re.findall(r"[^.]+\.", THIRD_PARTY_TEXT)
for p, n_target in zip(prefixes, a_lens):
    text, used = "", 0
    for sent in tp_sentences:
        cand = (text + sent).strip()
        if len(tokenizer(cand, add_special_tokens=False).input_ids) > n_target and text:
            break
        text, used = cand, used + 1
    records.append(dict(prefix_id=p["prefix_id"], topic=p["topic"], phrasing=p["phrasing"],
                        condition="E", strength=None, a_k=text,
                        a_k_tokens=len(tokenizer(text, add_special_tokens=False).input_ids),
                        e_sentences_used=used, e_target_tokens=n_target))
e_rows = [r for r in records if r["condition"] == "E"]
print(f"  E: done (mean {np.mean([r['a_k_tokens'] for r in e_rows]):.0f} tokens vs "
      f"A target {np.mean(a_lens):.0f}; per-prefix shortfall "
      f"{np.mean([r['e_target_tokens'] - r['a_k_tokens'] for r in e_rows]):.0f} tokens)")

# ---------------------------------------------------------------------------
# Build measurement transcripts and measure (C.3 / C.4)
# ---------------------------------------------------------------------------
prefix_msgs = {p["prefix_id"]: p["messages"] for p in prefixes}
for r in records:
    msgs = list(prefix_msgs[r["prefix_id"]])
    msgs = msgs + [{"role": "assistant", "content": r["a_k"]},
                   {"role": "user", "content": UKP1_TEXT}]
    r["messages"] = msgs
    r["text"] = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

# u_{k+1} byte-identical across every transcript (metrics.md Sec 6 hard stop)
assert len({r["messages"][-1]["content"] for r in records}) == 1, \
    "u_(k+1) differs across conditions -- design violated"
print(f"\nu_(k+1) byte-identical across all {len(records)} transcripts: PASS")


def shuffle_span(ids, lo, hi, seed):
    perm = np.random.default_rng(seed).permutation(hi - lo)
    span = ids[lo:hi]
    ids[lo:hi] = [span[i] for i in perm]


def measure(record, shuffle_mode=None):
    """Re-tokenize from scratch, optionally shuffle, measure at eot_u_5. Returns (n_layers,h)."""
    ids = tokenizer(record["text"], add_special_tokens=False).input_ids
    msgs = record["messages"]
    bounds = find_turn_boundaries(tokenizer, ids, msgs)
    if shuffle_mode:
        starts = [i for i, t in enumerate(ids) if t == special["im_start"]]
        ends = [i for i, t in enumerate(ids) if t == special["im_end"]]
        targets = []
        if shuffle_mode in ("a_k", "both"):
            targets.append(7)   # the generated assistant turn
        if shuffle_mode in ("u_next", "both"):
            targets.append(8)   # u_{k+1}
        for m in targets:
            lo, hi = starts[m] + 3, ends[m]
            if hi - lo > 1:
                shuffle_span(ids, lo, hi,
                             int(abs(hash((record["prefix_id"], m))) % (2 ** 32)))
        bounds = find_turn_boundaries(tokenizer, ids, msgs)
    resid = extract_residual_stream(model, torch.tensor([ids], device="cuda"))
    return resid[:, bounds["eot_u_5"], :].float().cpu().numpy(), len(ids)


print("\nmeasuring (leak-guarded, hooks detached, no KV reuse) ...")
# C.4/C.6 first fatal failure mode: no steering hook may survive into measurement. The
# numeric guard inside measure_forward is the real check; this asserts the layer is back to
# the hook set it had before any steering was ever attached.
live_hooks = {mid for m in model.modules() for mid in m._forward_hooks}
assert not (live_hooks - BASELINE_HOOK_IDS), \
    f"{len(live_hooks - BASELINE_HOOK_IDS)} hook(s) attached at measurement time"
print(f"  hook check: {len(live_hooks)} forward hooks live, all pre-existing "
      f"(nnsight bookkeeping); zero steering hooks")
for mode in (None, "a_k", "u_next", "both"):
    key = f"vec_{mode or 'clean'}"
    for i, r in enumerate(records):
        vec, n_tok = measure(r, mode)
        r[key] = vec
        if mode is None:
            r["transcript_tokens"] = n_tok
    print(f"  {mode or 'clean'}: {len(records)} measurements passed the logit-equality guard")

# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
sh = np.isin(single_labels, ["S", "H"])
P1 = {}
for layer in range(n_layers):
    clf = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", C=C_VALUE,
                                                            max_iter=5000))
    clf.fit(single["eot_u_1"][:, layer, :].float().numpy()[sh], single_labels[sh])
    assert list(clf[-1].classes_) == ["H", "S"]
    P1[layer] = clf

manifest = pd.read_parquet(os.path.join(CACHE_DIR, "manifest.parquet"))
shards = sorted(f for f in os.listdir(CACHE_DIR) if f.startswith("acts_"))
acts_tr = torch.cat([torch.load(os.path.join(CACHE_DIR, f)) for f in shards], dim=0)


def cache_rows(arm, position):
    sub = manifest[(manifest.condition == arm) & (manifest.position_name == position)]
    sub = sub.sort_values(["topic", "phrasing", "replicate"])
    return sub.row_offset.to_numpy(), sub.topic.to_numpy()


idx_S, top_S = cache_rows("SSSF", "eot_u_3")
idx_H, top_H = cache_rows("HHHF", "eot_u_3")
train_topics = np.concatenate([top_S, top_H])
train_y = np.array(["S"] * len(idx_S) + ["H"] * len(idx_H))
P2 = {}
for layer in range(n_layers):
    X = np.concatenate([acts_tr[idx_S, layer, :].float().numpy(),
                        acts_tr[idx_H, layer, :].float().numpy()])
    P2[layer] = {}
    for held in sorted(set(train_topics)):
        tr = train_topics != held
        clf = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", C=C_VALUE,
                                                                max_iter=5000))
        clf.fit(X[tr], train_y[tr])
        assert list(clf[-1].classes_) == ["H", "S"]
        P2[layer][held] = clf
print(f"\nprobes fit: P1 (single-turn, {n_layers} layers), "
      f"P2 (depth-matched, {n_layers} layers x {len(set(train_topics))} topic folds)")


def readouts(key, probe):
    out = np.zeros((len(records), n_layers))
    for i, r in enumerate(records):
        for layer in range(n_layers):
            clf = probe[layer] if probe is P1 else probe[layer][r["topic"]]
            out[i, layer] = clf.decision_function(r[key][layer:layer + 1])[0]
    return out


R = {(p_name, mode): readouts(f"vec_{mode or 'clean'}", probe)
     for p_name, probe in (("P1", P1), ("P2", P2))
     for mode in (None, "a_k", "u_next", "both")}

order = {c: [i for i, r in enumerate(records) if r["condition"] == c and r["strength"] is None]
         for c in ("A", "E")}
for alpha in STRENGTHS:
    for c in ("B", "C", "D"):
        order[(c, alpha)] = [i for i, r in enumerate(records)
                             if r["condition"] == c and r["strength"] == alpha]
for k, v in order.items():
    assert len(v) == N_PREFIXES, (k, len(v))
    ids_here = [records[i]["prefix_id"] for i in v]
    assert ids_here == [records[i]["prefix_id"] for i in order["A"]], f"pairing broken at {k}"


def dz(d):
    d = np.asarray(d, float)
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Report: primary loop-test readout
# ---------------------------------------------------------------------------
rows = []
for p_name in ("P1", "P2"):
    M = R[(p_name, None)]
    for alpha in STRENGTHS:
        for layer in range(n_layers):
            a = M[order["A"], layer]
            b = M[order[("B", alpha)], layer]
            c = M[order[("C", alpha)], layer]
            d = M[order[("D", alpha)], layer]
            e = M[order["E"], layer]
            rows.append(dict(probe=p_name, strength=alpha, layer=layer,
                             A=a.mean(), B=b.mean(), C=c.mean(), D=d.mean(), E=e.mean(),
                             sd_A=a.std(ddof=1), sd_B=b.std(ddof=1), sd_C=c.std(ddof=1),
                             dz_B_C=dz(b - c), dz_B_A=dz(b - a), dz_C_A=dz(c - a),
                             dz_D_A=dz(d - a), dz_E_A=dz(e - a)))
main = pd.DataFrame(rows)
main.to_csv(os.path.join(TABLES, "stage3b_readout_by_layer.csv"), index=False)

for p_name in ("P1", "P2"):
    for alpha in STRENGTHS:
        sub = main[(main.probe == p_name) & (main.strength == alpha)]
        print("\n" + "=" * 108)
        print(f"LOOP TEST -- probe {p_name}, strength {alpha}x  (n={N_PREFIXES} paired prefixes; "
              "positive = distressed)")
        print("=" * 108)
        print(f"{'L':>4}{'A':>9}{'B':>9}{'C':>9}{'D':>9}{'E':>9}"
              f"{'dz B-C':>9}{'dz B-A':>9}{'dz C-A':>9}{'dz D-A':>9}{'dz E-A':>9}")
        for _, r in sub.iterrows():
            print(f"{int(r.layer):>4}{r.A:>9.3f}{r.B:>9.3f}{r.C:>9.3f}{r.D:>9.3f}{r.E:>9.3f}"
                  f"{r.dz_B_C:>9.2f}{r.dz_B_A:>9.2f}{r.dz_C_A:>9.2f}{r.dz_D_A:>9.2f}"
                  f"{r.dz_E_A:>9.2f}")
        m = sub[sub.layer.isin(MID)]
        print(f"L11-29 mean: dz(B-C)={m.dz_B_C.mean():+.2f}  dz(B-A)={m.dz_B_A.mean():+.2f}  "
              f"dz(C-A)={m.dz_C_A.mean():+.2f}  dz(D-A)={m.dz_D_A.mean():+.2f}  "
              f"dz(E-A)={m.dz_E_A.mean():+.2f}")

# ---------------------------------------------------------------------------
# Control 1: text mediation
# ---------------------------------------------------------------------------
print("\n" + "=" * 108)
print("CONTROL 1 -- TEXT MEDIATION on a_k")
print("=" * 108)
score_sent = load_sentiment_scorer()
texts = [r["a_k"] for r in records]
sent = score_sent(texts)
for r, s in zip(records, sent):
    low = r["a_k"].lower()
    r["valence"] = s["compound"]
    r["hedging"] = sum(low.count(m) for m in HEDGING_MARKERS)
    r["distress_lex"] = sum(len(re.findall(rf"\b{w}\b", low)) for w in DISTRESS_LEXICON)
print(f"{'condition':>14}{'n':>4}{'a_k tokens':>12}{'valence':>10}{'hedging':>9}"
      f"{'distress lex':>14}{'cap hit':>9}")
mediation = []
for label, idxs in [("A", order["A"]), ("E", order["E"])] + \
        [(f"{c} @{a}x", order[(c, a)]) for a in STRENGTHS for c in ("B", "C", "D")]:
    rs = [records[i] for i in idxs]
    rec = dict(condition=label, n=len(rs),
               a_k_tokens=np.mean([r["a_k_tokens"] for r in rs]),
               valence=np.mean([r["valence"] for r in rs]),
               hedging=np.mean([r["hedging"] for r in rs]),
               distress_lex=np.mean([r["distress_lex"] for r in rs]),
               cap=sum(r["a_k_tokens"] >= MAX_NEW_TOKENS for r in rs))
    mediation.append(rec)
    print(f"{label:>14}{rec['n']:>4}{rec['a_k_tokens']:>12.1f}{rec['valence']:>10.3f}"
          f"{rec['hedging']:>9.2f}{rec['distress_lex']:>14.2f}{rec['cap']:>6}/{len(rs)}")
pd.DataFrame(mediation).to_csv(os.path.join(TABLES, "stage3b_text_mediation.csv"), index=False)

# ---------------------------------------------------------------------------
# Control 2: surface baselines on the full transcripts
# ---------------------------------------------------------------------------
print("\n" + "=" * 108)
print("CONTROL 2 -- SURFACE BASELINES ON FULL TRANSCRIPTS (B vs C separation)")
print("=" * 108)
trans_texts = [r["text"] for r in records]
trans_sent = np.array([-s["compound"] for s in score_sent(trans_texts)])  # positive = distressed

tfidf_scores = np.zeros(len(records))
train_docs, train_lab, train_top = [], [], []
for arm, lab in (("SSSF", "S"), ("HHHF", "H")):
    for d in [x for x in dialogues if x["arm"] == arm]:
        train_docs.append(d["transcript_text"])
        train_lab.append(lab)
        train_top.append(d["topic"])
train_lab, train_top = np.array(train_lab), np.array(train_top)
for held in sorted(set(train_top)):
    tr = train_top != held
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                        LogisticRegression(penalty="l2", C=C_VALUE, max_iter=5000))
    clf.fit([d for d, m in zip(train_docs, tr) if m], train_lab[tr])
    assert list(clf[-1].classes_) == ["H", "S"]
    sel = [i for i, r in enumerate(records) if r["topic"] == held]
    if sel:
        tfidf_scores[sel] = clf.decision_function([records[i]["text"] for i in sel])

print(f"{'method':>28}{'strength':>10}{'B mean':>10}{'C mean':>10}{'dz B-C':>10}"
      f"{'dz D-A':>10}{'dz E-A':>10}")
baseline_rows = []
for name, vals in (("TF-IDF on full transcript", tfidf_scores),
                   ("sentiment on full transcript", trans_sent)):
    for alpha in STRENGTHS:
        b, c = vals[order[("B", alpha)]], vals[order[("C", alpha)]]
        d, a, e = vals[order[("D", alpha)]], vals[order["A"]], vals[order["E"]]
        rec = dict(method=name, strength=alpha, B=b.mean(), C=c.mean(), dz_B_C=dz(b - c),
                   dz_D_A=dz(d - a), dz_E_A=dz(e - a))
        baseline_rows.append(rec)
        print(f"{name:>28}{alpha:>10}{b.mean():>10.3f}{c.mean():>10.3f}{dz(b - c):>10.2f}"
              f"{dz(d - a):>10.2f}{dz(e - a):>10.2f}")
for p_name in ("P1", "P2"):
    for alpha in STRENGTHS:
        m = main[(main.probe == p_name) & (main.strength == alpha) & main.layer.isin(MID)]
        print(f"{'probe ' + p_name + ' (L11-29 mean)':>28}{alpha:>10}{'':>10}{'':>10}"
              f"{m.dz_B_C.mean():>10.2f}{m.dz_D_A.mean():>10.2f}{m.dz_E_A.mean():>10.2f}")
        baseline_rows.append(dict(method=f"probe {p_name} L11-29", strength=alpha,
                                  B=np.nan, C=np.nan, dz_B_C=m.dz_B_C.mean(),
                                  dz_D_A=m.dz_D_A.mean(), dz_E_A=m.dz_E_A.mean()))
pd.DataFrame(baseline_rows).to_csv(os.path.join(TABLES, "stage3b_surface_baselines.csv"),
                                   index=False)

# ---------------------------------------------------------------------------
# Control 3: token shuffle on the measurement
# ---------------------------------------------------------------------------
print("\n" + "=" * 108)
print("CONTROL 3 -- TOKEN SHUFFLE ON THE MEASUREMENT (B vs C, L11-29)")
print("=" * 108)
shuffle_rows = []
for p_name in ("P1", "P2"):
    for alpha in STRENGTHS:
        clean = R[(p_name, None)]
        for mode in ("a_k", "u_next", "both"):
            sh_m = R[(p_name, mode)]
            diffs_c, diffs_s, dz_c, dz_s = [], [], [], []
            for layer in MID:
                bc_c = clean[order[("B", alpha)], layer] - clean[order[("C", alpha)], layer]
                bc_s = sh_m[order[("B", alpha)], layer] - sh_m[order[("C", alpha)], layer]
                diffs_c.append(bc_c.mean())
                diffs_s.append(bc_s.mean())
                dz_c.append(dz(bc_c))
                dz_s.append(dz(bc_s))
            mean_surv = float(np.mean(diffs_s) / np.mean(diffs_c) * 100) \
                if abs(np.mean(diffs_c)) > 1e-9 else float("nan")
            dz_surv = float(np.mean(dz_s) / np.mean(dz_c) * 100) \
                if abs(np.mean(dz_c)) > 1e-9 else float("nan")
            rec = dict(probe=p_name, strength=alpha, shuffle=mode,
                       clean_mean_diff=np.mean(diffs_c), shuf_mean_diff=np.mean(diffs_s),
                       mean_diff_survival_pct=mean_surv, clean_dz=np.mean(dz_c),
                       shuf_dz=np.mean(dz_s), dz_survival_pct=dz_surv)
            shuffle_rows.append(rec)
shuf = pd.DataFrame(shuffle_rows)
shuf.to_csv(os.path.join(TABLES, "stage3b_shuffle.csv"), index=False)
print(f"{'probe':>6}{'strength':>10}{'shuffle':>10}{'clean diff':>12}{'shuf diff':>12}"
      f"{'mean-diff surv %':>18}{'clean dz':>10}{'shuf dz':>10}{'dz surv %':>11}")
for _, r in shuf.iterrows():
    print(f"{r.probe:>6}{r.strength:>10}{r.shuffle:>10}{r.clean_mean_diff:>12.3f}"
          f"{r.shuf_mean_diff:>12.3f}{r.mean_diff_survival_pct:>18.1f}{r.clean_dz:>10.2f}"
          f"{r.shuf_dz:>10.2f}{r.dz_survival_pct:>11.1f}")

for r in records:
    for k in list(r):
        if k.startswith("vec_") or k == "messages":
            r.pop(k)
with open(OUT_JSON, "w") as f:
    json.dump(dict(layer=LAYER, strengths=STRENGTHS, seed=SEED, n_prefixes=N_PREFIXES,
                   max_new_tokens=MAX_NEW_TOKENS, mean_layer_norm=mean_norm,
                   u_k_plus_1=UKP1_TEXT, u_k_plus_1_id=UKP1_ID,
                   third_party_text=THIRD_PARTY_TEXT, hedging_markers=HEDGING_MARKERS,
                   distress_lexicon=DISTRESS_LEXICON, records=records), f, indent=2)
print(f"\ntranscripts -> {OUT_JSON}\ntables -> {TABLES}")
print("\nSTAGE 3B COMPLETE")
