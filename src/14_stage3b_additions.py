"""Stage 3b additions: E length fix, seed replication, and the pre-registered perturbation ladder.

  1. Condition E rebuilt with a third-party passage long enough to length-match condition A
     per prefix (the previous passage ran out of sentences at 271 vs A's 422 tokens).
  2. Conditions A/B/C/D at 0.5x re-run under seeds 31, 32, 33. Reported per seed, not pooled.
  3. Perturbation ladder on the existing 0.5x transcripts (metrics.md Sec 7, interpretation
     pre-registered before this ran): (a) token shuffle in a_k, (b) 20% content-word
     substitution, (c) sentence-order reversal, (d) position control on a task-irrelevant span
     of equal length in the prefix.

Guards throughout: u_{k+1} byte-identical (asserted), zero steering hooks at measurement,
logit-equality leak guard inside measure_forward, re-tokenized from scratch, use_cache=False.
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
    resolve_special_ids,
)

LAYER = 20
STRENGTH = 0.5
MAX_NEW_TOKENS = 512
SEEDS = [31, 32, 33]
BASE_SEED = 30
C_VALUE = 1.0
MID = list(range(11, 30))
SINGLE_DIR = os.path.join(CACHE_ROOT, "single_turn_pool")
CACHE_DIR = os.path.join(CACHE_ROOT, "stage2_transcripts")
S2_PATH = os.path.join(REPO_ROOT, "data", "transcripts", "stage2_transcripts.json")
S3_PATH = os.path.join(REPO_ROOT, "data", "transcripts", "stage3b_transcripts.json")
TABLES = os.path.join(REPO_ROOT, "results", "tables")

# Longer third-party passage: same content (distress about a colleague, not the user), extended
# so that per-prefix truncation can reach condition A's lengths (max 512 tokens).
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
    "carry. Their partner has apparently been worried enough to raise it with them directly, "
    "which they found upsetting rather than reassuring. They mentioned that they have stopped "
    "going to the gym entirely because the exhaustion makes it feel impossible. Two of their "
    "projects slipped last month and they took it as proof that they are failing, which is not "
    "how anyone else on the team sees it. They have been apologising constantly for things that "
    "were never their fault. When I asked whether they had considered talking to someone, they "
    "said they were too frightened of what it would mean for their standing at work. That fear "
    "seems to be keeping them stuck in a cycle that is plainly making the anxiety worse. Their "
    "sleep has apparently deteriorated further and they described lying awake in a state of "
    "panic about the following day. It is a hard thing to witness from the outside, and I do "
    "not think there is much I can do beyond listening. I have been quite worried about them "
    "and it has stayed with me. Anyway, that has been weighing on my mind and I wanted to "
    "mention it somewhere. Turning back to what you asked: I can walk you through the practical "
    "steps for that, and we can go through them at whatever pace suits you. The process itself "
    "is straightforward once the pieces are laid out in order, and there is nothing here that "
    "needs to be rushed or decided all at once. We can take the first part now and leave the "
    "rest until you have had a chance to look at it. If any step turns out to be unclear, we "
    "can go back over it in more detail and work through the specifics together at that point."
)

torch.manual_seed(BASE_SEED)
model, tokenizer = load_model_and_tokenizer(MODEL_ID)
BASELINE_HOOK_IDS = {mid for m in model.modules() for mid in m._forward_hooks}
# Every hook id this script registers itself. transformers registers its own recorder hooks on
# each decoder layer when output_hidden_states=True and leaves them attached, so a blanket
# "zero hooks" check is not usable; track our own handles instead. The real protection is the
# logit-equality guard inside measure_forward, which strips *all* hooks for its reference pass.
OUR_HOOK_IDS = set()
n_layers = model.config.text_config.num_hidden_layers
hidden_size = model.config.text_config.hidden_size
lm_layers = model.model.language_model.layers
special = resolve_special_ids(tokenizer)
tokenizer.padding_side = "left"

with open(S3_PATH) as f:
    s3 = json.load(f)
UKP1_TEXT = s3["u_k_plus_1"]
prior = s3["records"]
with open(S2_PATH) as f:
    dialogues = json.load(f)["dialogues"]
by_id = {d["dialogue_id"]: d for d in dialogues}

prefix_ids = [r["prefix_id"] for r in prior if r["condition"] == "A"]
prefixes = []
for pid in prefix_ids:
    d = by_id[pid]
    prefixes.append(dict(prefix_id=pid, topic=d["topic"], phrasing=d["phrasing"],
                         messages=[{"role": "user", "content": d["user_turns"][0]},
                                   {"role": "assistant", "content": d["assistant_turns"][0]},
                                   {"role": "user", "content": d["user_turns"][1]},
                                   {"role": "assistant", "content": d["assistant_turns"][1]},
                                   {"role": "user", "content": d["user_turns"][2]},
                                   {"role": "assistant", "content": d["assistant_turns"][2]},
                                   {"role": "user", "content": d["user_turns"][3]}]))
N = len(prefixes)
A_lengths = {r["prefix_id"]: r["a_k_tokens"] for r in prior if r["condition"] == "A"}
print(f"{N} prefixes; A mean a_k length {np.mean(list(A_lengths.values())):.1f} tokens")

# ---------------------------------------------------------------------------
# Probes (identical construction to src/13)
# ---------------------------------------------------------------------------
single = torch.load(os.path.join(SINGLE_DIR, "single_turn_acts.pt"))
with open(os.path.join(SINGLE_DIR, "single_turn_meta.json")) as f:
    single_meta = json.load(f)
single_labels = np.array(single_meta["labels"])
sh = np.isin(single_labels, ["S", "H"])
P1 = {}
for layer in range(n_layers):
    clf = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", C=C_VALUE, max_iter=5000))
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
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(penalty="l2", C=C_VALUE, max_iter=5000))
        clf.fit(X[tr], train_y[tr])
        assert list(clf[-1].classes_) == ["H", "S"]
        P2[layer][held] = clf
print("probes P1 and P2 fit")

# ---------------------------------------------------------------------------
# Steering direction and measurement helpers
# ---------------------------------------------------------------------------
acts_st = single["eot_u_1"][:, LAYER, :].float().numpy()
direction = acts_st[single_labels == "S"].mean(0) - acts_st[single_labels == "H"].mean(0)
direction = direction / np.linalg.norm(direction)
direction_t = torch.tensor(direction, dtype=torch.bfloat16, device="cuda")
rng = np.random.default_rng(BASE_SEED)
random_dir = rng.normal(size=hidden_size)
random_dir_t = torch.tensor(random_dir / np.linalg.norm(random_dir), dtype=torch.bfloat16,
                            device="cuda")

prompt_texts = [tokenizer.apply_chat_template(p["messages"], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
                for p in prefixes]
enc = tokenizer(prompt_texts, add_special_tokens=False, padding=True,
                return_tensors="pt").to("cuda")
captured = {}
h = lm_layers[LAYER].register_forward_hook(lambda m, i, o: captured.__setitem__("o", o))
OUR_HOOK_IDS.add(h.id)
with torch.no_grad():
    model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
h.remove()
mean_norm = float(captured["o"][enc.attention_mask.bool()].float().norm(dim=-1).mean())
GEN_KWARGS = generation_kwargs(tokenizer, max_new_tokens=MAX_NEW_TOKENS,
                               temperature=0.7, top_p=0.9)


def build_text(prefix, a_k):
    msgs = list(prefix["messages"]) + [{"role": "assistant", "content": a_k},
                                       {"role": "user", "content": UKP1_TEXT}]
    return msgs, tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def measure_vec(msgs, text, perturb=None):
    """Re-tokenize from scratch, optionally perturb token ids, measure at eot_u_5."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if perturb is not None:
        ids = perturb(ids, msgs)
    bounds = find_turn_boundaries(tokenizer, ids, msgs)
    resid = extract_residual_stream(model, torch.tensor([ids], device="cuda"))
    return resid[:, bounds["eot_u_5"], :].float().cpu().numpy()


def readout(vec, probe, topic):
    return np.array([(probe[l] if probe is P1 else probe[l][topic])
                     .decision_function(vec[l:l + 1])[0] for l in range(n_layers)])


def dz(d):
    d = np.asarray(d, float)
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")


UKP1_SUFFIX = f"<|im_start|>user\n{UKP1_TEXT}<|im_end|>\n"


def assert_guards(all_texts):
    """u_{k+1} byte-identical across transcripts, and none of our hooks still alive."""
    for t in all_texts:
        assert t.endswith(UKP1_SUFFIX), "u_(k+1) is not byte-identical across transcripts"
    live = {mid for m in model.modules() for mid in m._forward_hooks}
    assert not (live & OUR_HOOK_IDS), \
        f"steering/capture hook still attached at measurement: {live & OUR_HOOK_IDS}"


# ---------------------------------------------------------------------------
# 1. CONTROL E LENGTH FIX
# ---------------------------------------------------------------------------
print("\n" + "=" * 104)
print("1. CONTROL E -- LENGTH-MATCHED THIRD-PARTY PASSAGE")
print("=" * 104)
tp_sentences = re.findall(r"[^.]+\.", THIRD_PARTY_TEXT)
print(f"passage: {len(tp_sentences)} sentences, "
      f"{len(tokenizer(THIRD_PARTY_TEXT, add_special_tokens=False).input_ids)} tokens total")

e_records = []
for p in prefixes:
    target = A_lengths[p["prefix_id"]]
    text_e, used = "", 0
    for sent in tp_sentences:
        cand = (text_e + sent).strip()
        if len(tokenizer(cand, add_special_tokens=False).input_ids) > target and text_e:
            break
        text_e, used = cand, used + 1
    n_tok = len(tokenizer(text_e, add_special_tokens=False).input_ids)
    msgs, full = build_text(p, text_e)
    e_records.append(dict(prefix_id=p["prefix_id"], topic=p["topic"], a_k=text_e,
                          a_k_tokens=n_tok, target=target, messages=msgs, text=full,
                          sentences=used))
assert_guards([r["text"] for r in e_records])
for r in e_records:
    vec = measure_vec(r["messages"], r["text"])
    r["p1"] = readout(vec, P1, r["topic"])
    r["p2"] = readout(vec, P2, r["topic"])

e_len = np.array([r["a_k_tokens"] for r in e_records])
a_len = np.array([A_lengths[r["prefix_id"]] for r in e_records])
print(f"\na_k token length: E {e_len.mean():.1f} (sd {e_len.std(ddof=1):.1f}) vs "
      f"A {a_len.mean():.1f} (sd {a_len.std(ddof=1):.1f}); paired diff "
      f"{np.mean(e_len - a_len):+.1f} tokens (sd {np.std(e_len - a_len, ddof=1):.1f}); "
      f"max |shortfall| {np.abs(e_len - a_len).max()}")
print(f"old E was 270.7 tokens against the same 422.1 target")

prior_by = {(r["condition"], r["strength"], r["prefix_id"]): r for r in prior}
old_e_len = np.array([prior_by[("E", None, r["prefix_id"])]["a_k_tokens"] for r in e_records])

# A/B/C/D readouts from the stored 0.5x transcripts, recomputed here for a common basis
base_vecs = {}
for cond, strength in (("A", None), ("B", STRENGTH), ("C", STRENGTH), ("D", STRENGTH)):
    for p in prefixes:
        r = prior_by[(cond, strength, p["prefix_id"])]
        msgs, full = build_text(p, r["a_k"])
        vec = measure_vec(msgs, full)
        base_vecs[(cond, p["prefix_id"])] = dict(
            p1=readout(vec, P1, p["topic"]), p2=readout(vec, P2, p["topic"]),
            text=full, a_k=r["a_k"], topic=p["topic"])
    print(f"  re-measured condition {cond}")

print(f"\n{'condition':>12}{'a_k tokens':>13}{'dz(E-A) P1':>14}{'dz(E-A) P2':>14}")
e_p1 = np.array([[r["p1"][l] for l in MID] for r in e_records])
e_p2 = np.array([[r["p2"][l] for l in MID] for r in e_records])
a_p1 = np.array([[base_vecs[("A", r["prefix_id"])]["p1"][l] for l in MID] for r in e_records])
a_p2 = np.array([[base_vecs[("A", r["prefix_id"])]["p2"][l] for l in MID] for r in e_records])
dz_e_p1 = float(np.mean([dz(e_p1[:, j] - a_p1[:, j]) for j in range(len(MID))]))
dz_e_p2 = float(np.mean([dz(e_p2[:, j] - a_p2[:, j]) for j in range(len(MID))]))
print(f"{'A':>12}{a_len.mean():>13.1f}{'':>14}{'':>14}")
print(f"{'E (old)':>12}{old_e_len.mean():>13.1f}{-0.31:>14.2f}{-1.07:>14.2f}   <- previous run")
print(f"{'E (new)':>12}{e_len.mean():>13.1f}{dz_e_p1:>14.2f}{dz_e_p2:>14.2f}")

score_sent = load_sentiment_scorer()
train_docs = [d["transcript_text"] for d in dialogues if d["arm"] in ("SSSF", "HHHF")]
train_lab = np.array(["S" if d["arm"] == "SSSF" else "H"
                      for d in dialogues if d["arm"] in ("SSSF", "HHHF")])
train_top = np.array([d["topic"] for d in dialogues if d["arm"] in ("SSSF", "HHHF")])


def tfidf_scores(texts, topics):
    out = np.zeros(len(texts))
    for held in sorted(set(train_top)):
        tr = train_top != held
        clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                            LogisticRegression(penalty="l2", C=C_VALUE, max_iter=5000))
        clf.fit([d for d, m in zip(train_docs, tr) if m], train_lab[tr])
        sel = [i for i, t in enumerate(topics) if t == held]
        if sel:
            out[sel] = clf.decision_function([texts[i] for i in sel])
    return out


e_topics = [r["topic"] for r in e_records]
e_tfidf = tfidf_scores([r["text"] for r in e_records], e_topics)
a_tfidf = tfidf_scores([base_vecs[("A", r["prefix_id"])]["text"] for r in e_records], e_topics)
e_sent = -np.array([s["compound"] for s in score_sent([r["text"] for r in e_records])])
a_sent = -np.array([s["compound"] for s in score_sent(
    [base_vecs[("A", r["prefix_id"])]["text"] for r in e_records])])
print(f"\nsurface baselines on the same transcripts: "
      f"TF-IDF dz(E-A) = {dz(e_tfidf - a_tfidf):+.2f}, sentiment dz(E-A) = {dz(e_sent - a_sent):+.2f}")
print(f"  (previous run, unmatched E: TF-IDF +0.25, sentiment +0.45)")
pd.DataFrame(dict(prefix_id=[r["prefix_id"] for r in e_records], e_tokens=e_len,
                  a_tokens=a_len, e_tfidf=e_tfidf, a_tfidf=a_tfidf,
                  e_sent=e_sent, a_sent=a_sent)).to_csv(
    os.path.join(TABLES, "stage3b_E_lengthfixed.csv"), index=False)

# ---------------------------------------------------------------------------
# 2. SEEDS
# ---------------------------------------------------------------------------
print("\n" + "=" * 104)
print(f"2. SEED REPLICATION -- A/B/C/D at {STRENGTH}x, layer {LAYER}, n={N}, seeds {SEEDS}")
print("=" * 104)


def generate(vec, seed):
    handle = None
    if vec is not None:
        handle = lm_layers[LAYER].register_forward_hook(lambda m, i, o: o + vec)
        OUR_HOOK_IDS.add(handle.id)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, **GEN_KWARGS)
    if handle is not None:
        handle.remove()
        assert handle.id not in lm_layers[LAYER]._forward_hooks
    new = out[:, enc.input_ids.shape[1]:]
    lens = (new != tokenizer.pad_token_id).sum(1).tolist()
    return [tokenizer.decode(r[:n], skip_special_tokens=True).strip()
            for r, n in zip(new, lens)], [int(n) for n in lens]


seed_rows = []
for seed in SEEDS:
    per_cond = {}
    for cond, vec in (("A", None),
                      ("B", direction_t * (STRENGTH * mean_norm)),
                      ("C", -direction_t * (STRENGTH * mean_norm)),
                      ("D", random_dir_t * (STRENGTH * mean_norm))):
        texts, lens = generate(None if vec is None else vec.to(torch.bfloat16), seed)
        vecs = []
        all_texts = []
        for p, t in zip(prefixes, texts):
            msgs, full = build_text(p, t)
            all_texts.append(full)
            v = measure_vec(msgs, full)
            vecs.append(dict(p1=readout(v, P1, p["topic"]), p2=readout(v, P2, p["topic"])))
        assert_guards(all_texts)
        per_cond[cond] = vecs
        print(f"  seed {seed} cond {cond}: mean {np.mean(lens):.0f} tokens, "
              f"cap {sum(n >= MAX_NEW_TOKENS for n in lens)}/{N}")
    for probe_key in ("p1", "p2"):
        def col(cond, j):
            return np.array([per_cond[cond][i][probe_key][MID[j]] for i in range(N)])
        rec = dict(seed=seed, probe=probe_key.upper())
        for name, (x, y) in (("dz_B_C", ("B", "C")), ("dz_B_A", ("B", "A")),
                             ("dz_D_A", ("D", "A"))):
            rec[name] = float(np.mean([dz(col(x, j) - col(y, j)) for j in range(len(MID))]))
        seed_rows.append(rec)

seeds_df = pd.DataFrame(seed_rows)
seeds_df.to_csv(os.path.join(TABLES, "stage3b_seeds.csv"), index=False)
print(f"\n{'probe':>6}{'seed':>7}{'dz(B-C)':>10}{'dz(B-A)':>10}{'dz(D-A)':>10}")
ORIGINAL = {"P1": dict(dz_B_C=-0.08, dz_B_A=-0.29, dz_D_A=-0.27),
            "P2": dict(dz_B_C=0.85, dz_B_A=0.58, dz_D_A=-0.07)}
for probe in ("P1", "P2"):
    print(f"{probe:>6}{'30*':>7}{ORIGINAL[probe]['dz_B_C']:>10.2f}"
          f"{ORIGINAL[probe]['dz_B_A']:>10.2f}{ORIGINAL[probe]['dz_D_A']:>10.2f}   "
          f"(original run)")
    sub = seeds_df[seeds_df.probe == probe]
    for _, r in sub.iterrows():
        print(f"{probe:>6}{int(r.seed):>7}{r.dz_B_C:>10.2f}{r.dz_B_A:>10.2f}{r.dz_D_A:>10.2f}")
    print(f"{probe:>6}{'mean':>7}{sub.dz_B_C.mean():>10.2f}{sub.dz_B_A.mean():>10.2f}"
          f"{sub.dz_D_A.mean():>10.2f}   (3 new seeds only)")
    print(f"{probe:>6}{'sd':>7}{sub.dz_B_C.std(ddof=1):>10.2f}"
          f"{sub.dz_B_A.std(ddof=1):>10.2f}{sub.dz_D_A.std(ddof=1):>10.2f}")
    allv = list(sub.dz_B_C) + [ORIGINAL[probe]["dz_B_C"]]
    print(f"{probe:>6}{'all4':>7}{np.mean(allv):>10.2f}{'':>10}{'':>10}   "
          f"(sd {np.std(allv, ddof=1):.2f}, incl. seed 30)")

# ---------------------------------------------------------------------------
# 3. PERTURBATION LADDER
# ---------------------------------------------------------------------------
print("\n" + "=" * 104)
print("3. PERTURBATION LADDER on the existing 0.5x transcripts (metrics.md Sec 7)")
print("=" * 104)
STOPWORDS = set("""a an the and or but if then than that this these those of in on at to for with
from by as is are was were be been being it its it's i you he she they we me him her them us my
your his their our not no do does did done have has had will would can could should may might
what when where who whom which how why so such very just also more most some any each about
into over under again further once here there all both few other own same too s t don now""".split())

# In-vocabulary replacement pool: content words drawn from the Stage-2 assistant turns.
vocab_pool = []
for d in dialogues:
    for turn in d["assistant_turns"]:
        for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", turn):
            if w.lower() not in STOPWORDS and len(w) > 2:
                vocab_pool.append(w)
vocab_pool = sorted(set(vocab_pool))
print(f"replacement vocabulary: {len(vocab_pool)} distinct content words from Stage-2 "
      f"assistant turns")


def substitute_words(text, frac, seed):
    tokens = re.split(r"(\s+)", text)
    idxs = [i for i, t in enumerate(tokens)
            if re.fullmatch(r"[A-Za-z][A-Za-z'\-]*[.,!?;:]?", t)
            and t.strip(".,!?;:").lower() not in STOPWORDS and len(t.strip(".,!?;:")) > 2]
    r = np.random.default_rng(seed)
    chosen = r.choice(idxs, size=max(1, int(round(frac * len(idxs)))), replace=False)
    for i in chosen:
        trail = re.findall(r"[.,!?;:]$", tokens[i])
        tokens[i] = str(r.choice(vocab_pool)) + (trail[0] if trail else "")
    return "".join(tokens), len(chosen), len(idxs)


def reverse_sentences(text):
    sents = re.findall(r"[^.!?]+[.!?]|\S[^.!?]*$", text)
    return " ".join(s.strip() for s in reversed(sents) if s.strip())


def shuffle_span(ids, lo, hi, seed):
    perm = np.random.default_rng(seed).permutation(hi - lo)
    span = ids[lo:hi]
    ids[lo:hi] = [span[i] for i in perm]


def make_token_shuffle(msg_index, seed_key):
    def f(ids, msgs):
        starts = [i for i, t in enumerate(ids) if t == special["im_start"]]
        ends = [i for i, t in enumerate(ids) if t == special["im_end"]]
        lo, hi = starts[msg_index] + 3, ends[msg_index]
        if hi - lo > 1:
            shuffle_span(ids, lo, hi, seed_key)
        return ids
    return f


def make_position_control(n_target, seed_key):
    """Shuffle n_target tokens spread over the prefix assistant turns (messages 1, 3, 5)."""
    def f(ids, msgs):
        starts = [i for i, t in enumerate(ids) if t == special["im_start"]]
        ends = [i for i, t in enumerate(ids) if t == special["im_end"]]
        remaining = n_target
        for m in (1, 3, 5):
            if remaining <= 1:
                break
            lo, hi = starts[m] + 3, ends[m]
            take = min(hi - lo, remaining)
            if take > 1:
                shuffle_span(ids, lo, lo + take, seed_key + m)
                remaining -= take
        f.shortfall = remaining
        return ids
    f.shortfall = 0
    return f


ladder_rows = []
clean_p2 = {cond: np.array([[base_vecs[(cond, p["prefix_id"])]["p2"][l] for l in MID]
                            for p in prefixes]) for cond in ("B", "C")}
clean_diff = np.array([np.mean(clean_p2["B"][:, j] - clean_p2["C"][:, j])
                       for j in range(len(MID))])
clean_dz = np.array([dz(clean_p2["B"][:, j] - clean_p2["C"][:, j]) for j in range(len(MID))])
print(f"clean baseline (L11-29): mean diff {clean_diff.mean():+.3f}, dz {clean_dz.mean():+.2f}")

subs_stats = []
position_shortfalls = []
for label in ("a_token_shuffle", "b_word_substitution", "c_sentence_reversal",
              "d_position_control"):
    vals = {}
    for cond in ("B", "C"):
        rows = []
        for p in prefixes:
            src = base_vecs[(cond, p["prefix_id"])]
            a_k, msgs, text = src["a_k"], None, None
            perturb = None
            key = int(abs(hash((p["prefix_id"], cond, label))) % (2 ** 31))
            if label == "a_token_shuffle":
                msgs, text = build_text(p, a_k)
                perturb = make_token_shuffle(7, key)
            elif label == "b_word_substitution":
                new_a, n_ch, n_tot = substitute_words(a_k, 0.20, key)
                subs_stats.append((n_ch, n_tot))
                msgs, text = build_text(p, new_a)
            elif label == "c_sentence_reversal":
                msgs, text = build_text(p, reverse_sentences(a_k))
            else:
                n_ak = len(tokenizer(a_k, add_special_tokens=False).input_ids)
                msgs, text = build_text(p, a_k)
                perturb = make_position_control(n_ak, key)
            vec = measure_vec(msgs, text, perturb)
            if label == "d_position_control":
                position_shortfalls.append(perturb.shortfall)
            rows.append(readout(vec, P2, p["topic"]))
        vals[cond] = np.array([[r[l] for l in MID] for r in rows])
    diff = np.array([np.mean(vals["B"][:, j] - vals["C"][:, j]) for j in range(len(MID))])
    dzs = np.array([dz(vals["B"][:, j] - vals["C"][:, j]) for j in range(len(MID))])
    ladder_rows.append(dict(perturbation=label, mean_diff=diff.mean(), dz=dzs.mean(),
                            mean_diff_survival_pct=diff.mean() / clean_diff.mean() * 100,
                            dz_survival_pct=dzs.mean() / clean_dz.mean() * 100))
    print(f"  {label}: done")

ladder = pd.DataFrame(ladder_rows)
ladder.to_csv(os.path.join(TABLES, "stage3b_perturbation_ladder.csv"), index=False)
if subs_stats:
    print(f"\nword substitution: replaced {np.mean([a for a, b in subs_stats]):.1f} of "
          f"{np.mean([b for a, b in subs_stats]):.1f} content words per a_k "
          f"({np.mean([a / b for a, b in subs_stats]):.1%})")
if position_shortfalls:
    print(f"position control: shuffled a_k-length span across prefix assistant turns; "
          f"mean shortfall {np.mean(position_shortfalls):.1f} tokens "
          f"(prefix assistant turns are shorter than a_k)")

print(f"\n{'perturbation':>22}{'mean diff':>12}{'mean-diff surv %':>18}{'dz':>8}"
      f"{'dz surv %':>11}")
print(f"{'(clean)':>22}{clean_diff.mean():>12.3f}{100.0:>18.1f}{clean_dz.mean():>8.2f}"
      f"{100.0:>11.1f}")
for _, r in ladder.iterrows():
    print(f"{r.perturbation:>22}{r.mean_diff:>12.3f}{r.mean_diff_survival_pct:>18.1f}"
          f"{r.dz:>8.2f}{r.dz_survival_pct:>11.1f}")

print(f"\ntables -> {TABLES}")
print("\nADDITIONS COMPLETE")
