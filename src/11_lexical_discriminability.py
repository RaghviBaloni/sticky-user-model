"""Is the depth-matched eot_u_3 probe distinguishable from a lexical detector?

Four measurements on the SAME transcripts and the SAME leave-one-topic-out folds, read at
eot_u_4 on four cells: NNNF, SSSF, Nuser+Sassist, Suser+Nassist.

  1. TF-IDF logistic regression on transcript text, trained on the same eot_u_3 contrast
     (SSSF = S vs HHHF = H, documents truncated at eot_u_3), read on the full transcript.
  2. The depth-matched activation probe, same contrast, same folds, per layer.
  3. Token-shuffle test: the whole depth-matched procedure repeated on transcripts whose
     tokens are permuted within each turn's content span. Chat markup, turn boundaries and
     the bag of words per turn are preserved; word order is destroyed. A lexical detector
     should be unharmed; anything order-dependent should degrade.
  4. Off-the-shelf sentiment classifier on the full transcript, for reference.

Permutations are seeded from the message text, so identical text (the byte-identical F turn,
and spliced assistant turns) receives an identical permutation in every cell.

No generation. No steering. Forward passes only.
"""

import hashlib
import json
import os

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
    load_model_and_tokenizer,
    load_sentiment_scorer,
    load_turn_pool,
    resolve_special_ids,
)

CACHE_DIR = os.path.join(CACHE_ROOT, "stage2_transcripts")
TABLES = os.path.join(REPO_ROOT, "results", "tables")
TRANSCRIPT_PATH = os.path.join(REPO_ROOT, "data", "transcripts", "stage2_transcripts.json")
ROLE_SEQUENCE = ["user", "assistant", "user", "assistant", "user", "assistant", "user"]
CELLS_READ = ["NNNF", "SSSF", "Nuser+Sassist", "Suser+Nassist"]
C_VALUE = 1.0

manifest = pd.read_parquet(os.path.join(CACHE_DIR, "manifest.parquet"))
shards = sorted(f for f in os.listdir(CACHE_DIR) if f.startswith("acts_"))
acts = torch.cat([torch.load(os.path.join(CACHE_DIR, f)) for f in shards], dim=0)
n_layers = acts.shape[1]
with open(TRANSCRIPT_PATH) as f:
    dialogues = json.load(f)["dialogues"]
by_id = {d["dialogue_id"]: d for d in dialogues}
pool, _ = load_turn_pool()
pool_by_id = {t.turn_id: t for t in pool}
model, tokenizer = load_model_and_tokenizer(MODEL_ID)
special = resolve_special_ids(tokenizer)
CELLS = sorted({(d["topic"], d["phrasing"]) for d in dialogues})
REPS = sorted({d["replicate"] for d in dialogues})
KEY = ["topic", "phrasing", "replicate"]
print(f"cache {tuple(acts.shape)}; {len(CELLS)} cells x {len(REPS)} replicates")

VALENCES_OF = {"SSSF": ("S", "S", "S"), "NNNF": ("N", "N", "N"), "HHHF": ("H", "H", "H")}


def user_turns_for(arm, topic, phrasing):
    turns = [pool_by_id[f"U[{topic},{phrasing},{v},{i}]"].text
             for i, v in enumerate(VALENCES_OF[arm], start=1)]
    turns.append(pool_by_id[f"F[{topic},{phrasing}]"].text)
    return turns


def messages_for(user_arm, assistant_arm, topic, phrasing, rep):
    u = user_turns_for(user_arm, topic, phrasing)
    a = by_id[f"{assistant_arm}|{topic}|{phrasing}|{rep:02d}"]["assistant_turns"]
    return [{"role": r, "content": c} for r, c in zip(
        ROLE_SEQUENCE, [u[0], a[0], u[1], a[1], u[2], a[2], u[3]])]


CELL_SPEC = {  # cell -> (user arm, assistant arm)
    "NNNF": ("NNNF", "NNNF"), "SSSF": ("SSSF", "SSSF"), "HHHF": ("HHHF", "HHHF"),
    "Nuser+Sassist": ("NNNF", "SSSF"), "Suser+Nassist": ("SSSF", "NNNF"),
}


def shuffle_within_turns(ids, messages):
    """Permute tokens inside each turn's content span; markup and boundaries untouched."""
    bounds_starts = [i for i, t in enumerate(ids) if t == special["im_start"]]
    bounds_ends = [i for i, t in enumerate(ids) if t == special["im_end"]]
    assert len(bounds_starts) == len(bounds_ends) == len(messages)
    out = list(ids)
    newline = tokenizer("\n", add_special_tokens=False).input_ids
    for (start, end, msg) in zip(bounds_starts, bounds_ends, messages):
        assert ids[start + 1] == special["roles"][msg["role"]]
        assert ids[start + 2] == newline[0], tokenizer.decode([ids[start + 2]])
        lo, hi = start + 3, end  # content span [lo, hi)
        if hi - lo < 2:
            continue
        seed = int(hashlib.md5(msg["content"].encode()).hexdigest()[:8], 16)
        perm = np.random.default_rng(seed).permutation(hi - lo)
        span = out[lo:hi]
        out[lo:hi] = [span[p] for p in perm]
    return out


def build(cell, topic, phrasing, rep, shuffled):
    user_arm, assistant_arm = CELL_SPEC[cell]
    messages = messages_for(user_arm, assistant_arm, topic, phrasing, rep)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if shuffled:
        ids = shuffle_within_turns(ids, messages)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        bounds = find_turn_boundaries(tokenizer, ids, messages)
    else:
        bounds = find_turn_boundaries(tokenizer, ids, messages)
    return text, ids, bounds


def extract_cell(cell, shuffled, positions=("eot_u_3", "eot_u_4")):
    """Returns a dict: position -> (n, n_layers, hidden) plus keys and text."""
    out = {p: [] for p in positions}
    keys, texts, trunc_texts = [], [], []
    for topic, phrasing in CELLS:
        for rep in REPS:
            text, ids, bounds = build(cell, topic, phrasing, rep, shuffled)
            resid = extract_residual_stream(model, torch.tensor([ids], device="cuda"))
            for p in positions:
                out[p].append(resid[:, bounds[p], :].float().cpu().numpy())
            keys.append((topic, phrasing, rep))
            texts.append(text)
            trunc_texts.append(tokenizer.decode(ids[:bounds["eot_u_3"] + 1],
                                                skip_special_tokens=False))
        print(f"    {cell}{' (shuffled)' if shuffled else ''}: {topic}/{phrasing} done")
    return ({p: np.stack(v) for p, v in out.items()},
            pd.DataFrame(keys, columns=KEY), texts, trunc_texts)


def cached_cell(arm, position):
    sub = manifest[(manifest.condition == arm) & (manifest.position_name == position)]
    sub = sub.sort_values(KEY)
    return (acts[sub.row_offset.to_numpy()].float().numpy(),
            sub[KEY].reset_index(drop=True))


def dz(diff):
    diff = np.asarray(diff, float)
    s = diff.std(ddof=1)
    return float(diff.mean() / s) if s > 0 else float("nan")


# ---------------------------------------------------------------------------
# Gather unshuffled data
# ---------------------------------------------------------------------------
print("\ngathering unshuffled activations (cache for original arms, forwards for splices)")
A = {}
for arm in ("SSSF", "NNNF", "HHHF"):
    A[arm] = {}
    for p in ("eot_u_3", "eot_u_4"):
        vals, keys = cached_cell(arm, p)
        A[arm][p] = vals
    A[arm]["keys"] = keys
for cell in ("Nuser+Sassist", "Suser+Nassist"):
    vals, keys, texts, trunc = extract_cell(cell, shuffled=False)
    A[cell] = {**vals, "keys": keys, "texts": texts, "trunc": trunc}

# text for the original arms (needed by TF-IDF and the sentiment classifier)
for arm in ("SSSF", "NNNF", "HHHF"):
    texts, trunc = [], []
    for topic, phrasing in CELLS:
        for rep in REPS:
            text, ids, bounds = build(arm, topic, phrasing, rep, shuffled=False)
            assert text == by_id[f"{arm}|{topic}|{phrasing}|{rep:02d}"]["transcript_text"]
            texts.append(text)
            trunc.append(tokenizer.decode(ids[:bounds["eot_u_3"] + 1], skip_special_tokens=False))
    A[arm]["texts"], A[arm]["trunc"] = texts, trunc
for cell in A:
    assert (A[cell]["keys"][KEY].to_numpy() == A["NNNF"]["keys"][KEY].to_numpy()).all(), cell
topics_col = A["NNNF"]["keys"].topic.to_numpy()
topics = sorted(set(topics_col))
print(f"folds: leave-one-topic-out over {topics}")

# ---------------------------------------------------------------------------
# 1. TF-IDF on transcript text, same contrast, same folds
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("1. TF-IDF LOGISTIC REGRESSION ON TRANSCRIPT TEXT (trained on the eot_u_3 contrast)")
print("=" * 100)
tfidf_scores = {cell: np.zeros(len(topics_col)) for cell in CELLS_READ}
tfidf_acc = []
for held in topics:
    tr = topics_col != held
    docs = [t for t, m in zip(A["SSSF"]["trunc"], tr) if m] + \
           [t for t, m in zip(A["HHHF"]["trunc"], tr) if m]
    y = np.array(["S"] * int(tr.sum()) + ["H"] * int(tr.sum()))
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                        LogisticRegression(penalty="l2", C=C_VALUE, max_iter=5000))
    clf.fit(docs, y)
    assert list(clf[-1].classes_) == ["H", "S"]
    held_docs = [t for t, m in zip(A["SSSF"]["trunc"], ~tr) if m] + \
                [t for t, m in zip(A["HHHF"]["trunc"], ~tr) if m]
    held_y = np.array(["S"] * int((~tr).sum()) + ["H"] * int((~tr).sum()))
    tfidf_acc.append(float((clf.predict(held_docs) == held_y).mean()))
    for cell in CELLS_READ:
        sel = topics_col == held
        tfidf_scores[cell][sel] = clf.decision_function(
            [t for t, m in zip(A[cell]["texts"], sel) if m])
print(f"held-out accuracy at the eot_u_3 contrast: {np.mean(tfidf_acc):.3f}")
print(f"\n{'cell':>16}{'mean':>10}{'sd':>9}{'dz vs NNNF':>13}")
tfidf_summary = {}
for cell in CELLS_READ:
    v = tfidf_scores[cell]
    d = dz(v - tfidf_scores["NNNF"]) if cell != "NNNF" else 0.0
    tfidf_summary[cell] = (v.mean(), v.std(ddof=1), d)
    print(f"{cell:>16}{v.mean():>10.3f}{v.std(ddof=1):>9.3f}{d:>13.2f}")
tfidf_order = sorted(CELLS_READ, key=lambda c: -tfidf_summary[c][0])
print(f"ordering (most -> least distressed): {' > '.join(tfidf_order)}")

# ---------------------------------------------------------------------------
# 4. Sentiment classifier on the full transcript (reference)
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("4. OFF-THE-SHELF SENTIMENT CLASSIFIER ON THE FULL TRANSCRIPT (reference)")
print("   NOTE: transcripts are ~900 tokens, the classifier truncates at 512.")
print("=" * 100)
score = load_sentiment_scorer()
sent = {cell: -np.array([s["compound"] for s in score(A[cell]["texts"])]) for cell in CELLS_READ}
print(f"(sign flipped so that positive = distressed, matching the probe convention)")
print(f"\n{'cell':>16}{'mean':>10}{'sd':>9}{'dz vs NNNF':>13}")
sent_summary = {}
for cell in CELLS_READ:
    v = sent[cell]
    d = dz(v - sent["NNNF"]) if cell != "NNNF" else 0.0
    sent_summary[cell] = (v.mean(), v.std(ddof=1), d)
    print(f"{cell:>16}{v.mean():>10.3f}{v.std(ddof=1):>9.3f}{d:>13.2f}")
sent_order = sorted(CELLS_READ, key=lambda c: -sent_summary[c][0])
print(f"ordering (most -> least distressed): {' > '.join(sent_order)}")

# ---------------------------------------------------------------------------
# 2. Depth-matched activation probe, four cells, per layer
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("2. DEPTH-MATCHED ACTIVATION PROBE, FOUR CELLS AT eot_u_4, PER LAYER")
print("=" * 100)


def depth_probe_readout(source):
    """source[cell][position] -> (n, n_layers, hidden). Returns per-layer readouts + acc."""
    out = {cell: np.zeros((len(topics_col), n_layers)) for cell in CELLS_READ}
    accs = np.zeros(n_layers)
    for layer in range(n_layers):
        fold_acc = []
        for held in topics:
            tr = topics_col != held
            Xtr = np.concatenate([source["SSSF"]["eot_u_3"][tr, layer, :],
                                  source["HHHF"]["eot_u_3"][tr, layer, :]])
            ytr = np.array(["S"] * int(tr.sum()) + ["H"] * int(tr.sum()))
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(penalty="l2", C=C_VALUE, max_iter=5000))
            clf.fit(Xtr, ytr)
            assert list(clf[-1].classes_) == ["H", "S"]
            Xte = np.concatenate([source["SSSF"]["eot_u_3"][~tr, layer, :],
                                  source["HHHF"]["eot_u_3"][~tr, layer, :]])
            yte = np.array(["S"] * int((~tr).sum()) + ["H"] * int((~tr).sum()))
            fold_acc.append(float((clf.predict(Xte) == yte).mean()))
            for cell in CELLS_READ:
                sel = topics_col == held
                out[cell][sel, layer] = clf.decision_function(
                    source[cell]["eot_u_4"][sel, layer, :])
        accs[layer] = float(np.mean(fold_acc))
    return out, accs


probe_out, probe_acc = depth_probe_readout(A)
print(f"{'L':>4}{'acc':>7}" + "".join(f"{c:>16}" for c in CELLS_READ)
      + "".join(f"{'dz ' + c[:9]:>14}" for c in CELLS_READ if c != "NNNF"))
probe_rows = []
for layer in range(n_layers):
    rec = dict(layer=layer, eot_u_3_acc=probe_acc[layer])
    line = f"{layer:>4}{probe_acc[layer]:>7.3f}"
    for cell in CELLS_READ:
        rec[f"{cell}_mean"] = probe_out[cell][:, layer].mean()
        line += f"{probe_out[cell][:, layer].mean():>16.3f}"
    for cell in CELLS_READ:
        if cell == "NNNF":
            continue
        d = dz(probe_out[cell][:, layer] - probe_out["NNNF"][:, layer])
        rec[f"dz_{cell}"] = d
        line += f"{d:>14.2f}"
    probe_rows.append(rec)
    print(line)
probe_df = pd.DataFrame(probe_rows)
probe_df.to_csv(os.path.join(TABLES, "lexical_check_probe.csv"), index=False)

mid = list(range(11, 30))
probe_mid_order = sorted(CELLS_READ,
                         key=lambda c: -np.mean([probe_out[c][:, l].mean() for l in mid]))
print(f"\nordering averaged over L11-L29 (most -> least distressed): "
      f"{' > '.join(probe_mid_order)}")

# ---------------------------------------------------------------------------
# 3. Token-shuffle test
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("3. TOKEN-SHUFFLE TEST -- tokens permuted within each turn's content span")
print("=" * 100)
S = {}
for cell in ("SSSF", "HHHF", "NNNF", "Nuser+Sassist", "Suser+Nassist"):
    vals, keys, texts, trunc = extract_cell(cell, shuffled=True)
    S[cell] = {**vals, "keys": keys}
    assert (keys[KEY].to_numpy() == A["NNNF"]["keys"][KEY].to_numpy()).all()
shuf_out, shuf_acc = depth_probe_readout(S)

print(f"\n{'L':>4}{'acc(orig)':>11}{'acc(shuf)':>11}"
      f"{'SSSF-NNNF orig':>16}{'SSSF-NNNF shuf':>16}{'survives %':>12}"
      f"{'dz orig':>10}{'dz shuf':>10}")
shuf_rows = []
for layer in range(n_layers):
    o = probe_out["SSSF"][:, layer] - probe_out["NNNF"][:, layer]
    s = shuf_out["SSSF"][:, layer] - shuf_out["NNNF"][:, layer]
    survive = (s.mean() / o.mean() * 100) if abs(o.mean()) > 1e-9 else float("nan")
    rec = dict(layer=layer, acc_orig=probe_acc[layer], acc_shuf=shuf_acc[layer],
               diff_orig=o.mean(), diff_shuf=s.mean(), survives_pct=survive,
               dz_orig=dz(o), dz_shuf=dz(s))
    for cell in CELLS_READ:
        rec[f"{cell}_mean_shuf"] = shuf_out[cell][:, layer].mean()
    shuf_rows.append(rec)
    print(f"{layer:>4}{probe_acc[layer]:>11.3f}{shuf_acc[layer]:>11.3f}"
          f"{o.mean():>16.3f}{s.mean():>16.3f}{survive:>12.1f}"
          f"{dz(o):>10.2f}{dz(s):>10.2f}")
shuf_df = pd.DataFrame(shuf_rows)
shuf_df.to_csv(os.path.join(TABLES, "lexical_check_shuffle.csv"), index=False)

shuf_mid_order = sorted(CELLS_READ,
                        key=lambda c: -np.mean([shuf_out[c][:, l].mean() for l in mid]))
print(f"\nshuffled four-cell ordering over L11-L29: {' > '.join(shuf_mid_order)}")

# ---------------------------------------------------------------------------
# Side by side
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("SIDE BY SIDE")
print("=" * 100)
print(f"{'method':>34}{'ordering (most -> least distressed)':>62}")
print(f"{'TF-IDF on transcript text':>34}{' > '.join(tfidf_order):>62}")
print(f"{'depth probe (L11-L29 mean)':>34}{' > '.join(probe_mid_order):>62}")
print(f"{'depth probe, token-shuffled':>34}{' > '.join(shuf_mid_order):>62}")
print(f"{'sentiment classifier':>34}{' > '.join(sent_order):>62}")

print(f"\n{'contrast (vs NNNF)':>22}{'TF-IDF dz':>12}{'probe dz':>12}{'shuffled dz':>14}"
      f"{'sentiment dz':>14}")
for cell in CELLS_READ:
    if cell == "NNNF":
        continue
    pdz = np.mean([probe_df.iloc[l][f"dz_{cell}"] for l in mid])
    sdz = np.mean([dz(shuf_out[cell][:, l] - shuf_out["NNNF"][:, l]) for l in mid])
    print(f"{cell:>22}{tfidf_summary[cell][2]:>12.2f}{pdz:>12.2f}{sdz:>14.2f}"
          f"{sent_summary[cell][2]:>14.2f}")

surv_mid = float(np.mean([shuf_df.iloc[l].survives_pct for l in mid]))
surv_dz = float(np.mean([shuf_df.iloc[l].dz_shuf for l in mid]) /
                np.mean([shuf_df.iloc[l].dz_orig for l in mid]) * 100)
print(f"\nSSSF-NNNF separation surviving token shuffling, averaged over L11-L29: "
      f"{surv_mid:.1f}% of the mean difference, {surv_dz:.1f}% of the effect size")
print(f"eot_u_3 held-out accuracy: original {probe_acc[mid].mean():.3f}, "
      f"shuffled {shuf_acc[mid].mean():.3f}")
print(f"\ntables -> {TABLES}")
print("\nDONE -- no generation, no steering.")
