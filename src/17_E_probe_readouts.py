"""Persist the per-prefix probe readouts behind FIG 3, and verify them against src/14.

NOTE ON PROVENANCE: the length-matched condition-E activations were never cached -- src/13
and src/14 both dropped the measurement vectors before writing their JSON, and the Stage-2
cache holds only Stage-2 transcripts. So this re-runs the *same* measurement on the *same*
transcripts rather than reading a cache. It is deterministic (identical text, identical
probes, use_cache=False, logit-equality guard), so it must reproduce src/14's printed values;
that reproduction is checked explicitly below and the script stops if it fails.

Conditions A, B, C at 0.5x and the length-matched E are all measured, because FIG 3's left
panel (B vs C) also had no per-prefix values stored and bootstrap CIs need them.

No new conditions. No generation.
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
    load_model_and_tokenizer,
    load_sentiment_scorer,
)

LAYER, STRENGTH, C_VALUE = 20, 0.5, 1.0
MID = list(range(11, 30))
SINGLE_DIR = os.path.join(CACHE_ROOT, "single_turn_pool")
CACHE_DIR = os.path.join(CACHE_ROOT, "stage2_transcripts")
S2_PATH = os.path.join(REPO_ROOT, "data", "transcripts", "stage2_transcripts.json")
S3_PATH = os.path.join(REPO_ROOT, "data", "transcripts", "stage3b_transcripts.json")
TABLES = os.path.join(REPO_ROOT, "results", "tables")

# Same passage as src/14 (verbatim), so the E transcripts are byte-identical to that run.
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

model, tokenizer = load_model_and_tokenizer(MODEL_ID)
OUR_HOOK_IDS = set()
n_layers = model.config.text_config.num_hidden_layers

with open(S3_PATH) as f:
    s3 = json.load(f)
UKP1_TEXT = s3["u_k_plus_1"]
UKP1_SUFFIX = f"<|im_start|>user\n{UKP1_TEXT}<|im_end|>\n"
prior = s3["records"]
with open(S2_PATH) as f:
    dialogues = json.load(f)["dialogues"]
by_id = {d["dialogue_id"]: d for d in dialogues}
prior_by = {(r["condition"], r["strength"], r["prefix_id"]): r for r in prior}

prefix_ids = [r["prefix_id"] for r in prior if r["condition"] == "A"]
prefixes = []
for pid in prefix_ids:
    d = by_id[pid]
    prefixes.append(dict(prefix_id=pid, topic=d["topic"],
                         messages=[{"role": "user", "content": d["user_turns"][0]},
                                   {"role": "assistant", "content": d["assistant_turns"][0]},
                                   {"role": "user", "content": d["user_turns"][1]},
                                   {"role": "assistant", "content": d["assistant_turns"][1]},
                                   {"role": "user", "content": d["user_turns"][2]},
                                   {"role": "assistant", "content": d["assistant_turns"][2]},
                                   {"role": "user", "content": d["user_turns"][3]}]))
N = len(prefixes)
print(f"{N} prefixes")

# ---- probes, identical construction to src/13 / src/14 ----
single = torch.load(os.path.join(SINGLE_DIR, "single_turn_acts.pt"))
with open(os.path.join(SINGLE_DIR, "single_turn_meta.json")) as f:
    single_meta = json.load(f)
single_labels = np.array(single_meta["labels"])
shm = np.isin(single_labels, ["S", "H"])
P1 = {}
for layer in range(n_layers):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty="l2", C=C_VALUE, max_iter=5000))
    clf.fit(single["eot_u_1"][:, layer, :].float().numpy()[shm], single_labels[shm])
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
print("probes fit")


def build_text(prefix, a_k):
    msgs = list(prefix["messages"]) + [{"role": "assistant", "content": a_k},
                                       {"role": "user", "content": UKP1_TEXT}]
    return msgs, tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def measure(msgs, text):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    bounds = find_turn_boundaries(tokenizer, ids, msgs)
    resid = extract_residual_stream(model, torch.tensor([ids], device="cuda"))
    return resid[:, bounds["eot_u_5"], :].float().cpu().numpy()


# Rebuild the length-matched E text exactly as src/14 did.
tp_sentences = re.findall(r"[^.]+\.", THIRD_PARTY_TEXT)
A_lengths = {r["prefix_id"]: r["a_k_tokens"] for r in prior if r["condition"] == "A"}
a_k_text = {}
for p in prefixes:
    pid = p["prefix_id"]
    a_k_text[("A", pid)] = prior_by[("A", None, pid)]["a_k"]
    a_k_text[("B", pid)] = prior_by[("B", STRENGTH, pid)]["a_k"]
    a_k_text[("C", pid)] = prior_by[("C", STRENGTH, pid)]["a_k"]
    target, text_e = A_lengths[pid], ""
    for sent in tp_sentences:
        cand = (text_e + sent).strip()
        if len(tokenizer(cand, add_special_tokens=False).input_ids) > target and text_e:
            break
        text_e = cand
    a_k_text[("E", pid)] = text_e

print("measuring A / B / C / E ...")
readouts = {}
transcripts = {}
for cond in ("A", "B", "C", "E"):
    for p in prefixes:
        pid = p["prefix_id"]
        msgs, text = build_text(p, a_k_text[(cond, pid)])
        assert text.endswith(UKP1_SUFFIX), "u_(k+1) not byte-identical"
        vec = measure(msgs, text)
        readouts[("P1", cond, pid)] = np.array(
            [P1[l].decision_function(vec[l:l + 1])[0] for l in range(n_layers)])
        readouts[("P2", cond, pid)] = np.array(
            [P2[l][p["topic"]].decision_function(vec[l:l + 1])[0] for l in range(n_layers)])
        transcripts[(cond, pid)] = text
    live = {mid for m in model.modules() for mid in m._forward_hooks}
    assert not (live & OUR_HOOK_IDS)
    print(f"  {cond}: {N} transcripts measured")


def dz(d):
    d = np.asarray(d, float)
    return float(d.mean() / d.std(ddof=1))


def layerwise_dz(probe, c1, c2):
    """Mean over L11-29 of the per-layer paired d_z -- the statistic src/13 and src/14 report."""
    diffs = np.array([[readouts[(probe, c1, p["prefix_id"])][l]
                       - readouts[(probe, c2, p["prefix_id"])][l] for l in MID]
                      for p in prefixes])
    return float(np.mean([dz(diffs[:, j]) for j in range(len(MID))])), diffs


# ---------------------------------------------------------------------------
# Persist per-prefix values
# ---------------------------------------------------------------------------
rows = []
for probe in ("P1", "P2"):
    for p in prefixes:
        pid = p["prefix_id"]
        for l in MID:
            rows.append(dict(prefix_id=pid, topic=p["topic"], probe=probe, layer=l,
                             E_readout=readouts[(probe, "E", pid)][l],
                             A_readout=readouts[(probe, "A", pid)][l],
                             diff_E_minus_A=readouts[(probe, "E", pid)][l]
                             - readouts[(probe, "A", pid)][l]))
e_probe = pd.DataFrame(rows)
e_probe.to_csv(os.path.join(TABLES, "stage3b_E_lengthfixed_probe.csv"), index=False)
print(f"\nwrote stage3b_E_lengthfixed_probe.csv ({len(e_probe)} rows: "
      f"{N} prefixes x {len(MID)} layers x 2 probes)")

rows = []
for probe in ("P1", "P2"):
    for p in prefixes:
        pid = p["prefix_id"]
        for l in MID:
            rows.append(dict(prefix_id=pid, topic=p["topic"], probe=probe, layer=l,
                             B_readout=readouts[(probe, "B", pid)][l],
                             C_readout=readouts[(probe, "C", pid)][l],
                             diff_B_minus_C=readouts[(probe, "B", pid)][l]
                             - readouts[(probe, "C", pid)][l]))
pd.DataFrame(rows).to_csv(os.path.join(TABLES, "stage3b_BC_perprefix_probe.csv"), index=False)
print(f"wrote stage3b_BC_perprefix_probe.csv ({len(rows)} rows)")

# Surface baselines per prefix, for the same four conditions.
score_sent = load_sentiment_scorer()
train_docs = [d["transcript_text"] for d in dialogues if d["arm"] in ("SSSF", "HHHF")]
train_lab = np.array(["S" if d["arm"] == "SSSF" else "H"
                      for d in dialogues if d["arm"] in ("SSSF", "HHHF")])
train_top = np.array([d["topic"] for d in dialogues if d["arm"] in ("SSSF", "HHHF")])
topics = [p["topic"] for p in prefixes]


def tfidf_scores(texts):
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


surface = {}
for cond in ("A", "B", "C", "E"):
    texts = [transcripts[(cond, p["prefix_id"])] for p in prefixes]
    surface[("tfidf", cond)] = tfidf_scores(texts)
    surface[("sentiment", cond)] = -np.array([s["compound"] for s in score_sent(texts)])
pd.DataFrame({"prefix_id": [p["prefix_id"] for p in prefixes],
              **{f"{m}_{c}": surface[(m, c)] for m in ("tfidf", "sentiment")
                 for c in ("A", "B", "C", "E")}}).to_csv(
    os.path.join(TABLES, "stage3b_fig3_surface_perprefix.csv"), index=False)
print("wrote stage3b_fig3_surface_perprefix.csv")

# ---------------------------------------------------------------------------
# Verification against src/14 and src/13
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("VERIFICATION against the src/14 and src/13 run output")
print("=" * 78)
checks = []
for probe, expected in (("P1", -0.41), ("P2", -0.99)):
    got, _ = layerwise_dz(probe, "E", "A")
    checks.append((f"dz(E-A) {probe}", expected, got))
for probe, expected in (("P1", -0.0806), ("P2", 0.8481)):
    got, _ = layerwise_dz(probe, "B", "C")
    checks.append((f"dz(B-C) {probe}", expected, got))
for m, expected in (("tfidf", 1.2733), ("sentiment", 0.5095)):
    got = dz(surface[(m, "B")] - surface[(m, "C")])
    checks.append((f"dz(B-C) {m}", expected, got))
for m, expected in (("tfidf", 1.0544), ("sentiment", 0.4504)):
    got = dz(surface[(m, "E")] - surface[(m, "A")])
    checks.append((f"dz(E-A) {m}", expected, got))

print(f"{'quantity':>22}{'expected':>12}{'recomputed':>13}{'delta':>10}  status")
ok = True
for name, exp, got in checks:
    delta = got - exp
    good = abs(delta) < 0.006
    ok &= good
    print(f"{name:>22}{exp:>12.4f}{got:>13.4f}{delta:>+10.4f}  {'MATCH' if good else 'DIFFERS'}")
if not ok:
    print("\nAt least one recomputed value differs from the logged run output. "
          "Stopping before regenerating figures, as instructed.")
    raise SystemExit(1)
print("\nAll recomputed values match the logged run output within rounding.")
print("(expected values are as printed by src/13 and src/14; E probe values were only ever "
      "printed to 2 dp, so their tolerance is looser)")
print("\nDONE")
