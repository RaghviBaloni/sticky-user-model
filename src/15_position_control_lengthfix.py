"""Length-matched position control (perturbation d), and the full ladder re-reported.

The previous position control shuffled only the prefix assistant turns (messages 1/3/5),
which total ~322 tokens against a_k's ~422 -- a 99.9-token shortfall, ~76% of the intended
span. Here the span is grown by walking contiguous prefix content spans in conversation order
until the target (that transcript's own a_k token count) is reached:

    message 0 (u1) -> 1 (a1) -> 2 (u2) -> 3 (a2) -> 4 (u3) -> 5 (a3) -> 6 (u4 = F, last resort)

u_{k+1} is never touched. Each contributing span is permuted within itself, which is the same
kind of perturbation as (a) -- within-turn token scrambling -- just relocated, so the two rows
are comparable. Exact per-message token usage and any residual shortfall are reported.

(a), (b) and (c) are recomputed here too so all four rows sit on one common basis.
No new conditions. No item 4.
"""

import json
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import torch
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
    resolve_special_ids,
)

LAYER, STRENGTH, C_VALUE = 20, 0.5, 1.0
MID = list(range(11, 30))
SINGLE_DIR = os.path.join(CACHE_ROOT, "single_turn_pool")
CACHE_DIR = os.path.join(CACHE_ROOT, "stage2_transcripts")
S2_PATH = os.path.join(REPO_ROOT, "data", "transcripts", "stage2_transcripts.json")
S3_PATH = os.path.join(REPO_ROOT, "data", "transcripts", "stage3b_transcripts.json")
TABLES = os.path.join(REPO_ROOT, "results", "tables")

model, tokenizer = load_model_and_tokenizer(MODEL_ID)
OUR_HOOK_IDS = set()
n_layers = model.config.text_config.num_hidden_layers
special = resolve_special_ids(tokenizer)

with open(S3_PATH) as f:
    s3 = json.load(f)
UKP1_TEXT = s3["u_k_plus_1"]
UKP1_SUFFIX = f"<|im_start|>user\n{UKP1_TEXT}<|im_end|>\n"
prior = s3["records"]
with open(S2_PATH) as f:
    dialogues = json.load(f)["dialogues"]
by_id = {d["dialogue_id"]: d for d in dialogues}

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
prior_by = {(r["condition"], r["strength"], r["prefix_id"]): r for r in prior}
N = len(prefixes)
print(f"{N} prefixes, strength {STRENGTH}x, layer {LAYER}")

# ---- P2 probe (depth-matched), identical construction to src/13 and src/14 ----
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
print("P2 fit")

STOPWORDS = set("""a an the and or but if then than that this these those of in on at to for with
from by as is are was were be been being it its it's i you he she they we me him her them us my
your his their our not no do does did done have has had will would can could should may might
what when where who whom which how why so such very just also more most some any each about
into over under again further once here there all both few other own same too s t don now""".split())
vocab_pool = sorted({w for d in dialogues for turn in d["assistant_turns"]
                     for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", turn)
                     if w.lower() not in STOPWORDS and len(w) > 2})


def build_text(prefix, a_k):
    msgs = list(prefix["messages"]) + [{"role": "assistant", "content": a_k},
                                       {"role": "user", "content": UKP1_TEXT}]
    return msgs, tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def shuffle_span(ids, lo, hi, seed):
    perm = np.random.default_rng(seed).permutation(hi - lo)
    span = ids[lo:hi]
    ids[lo:hi] = [span[i] for i in perm]


def measure_vec(msgs, text, perturb=None):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if perturb is not None:
        ids = perturb(ids, msgs)
    bounds = find_turn_boundaries(tokenizer, ids, msgs)
    resid = extract_residual_stream(model, torch.tensor([ids], device="cuda"))
    return resid[:, bounds["eot_u_5"], :].float().cpu().numpy()


def readout(vec, topic):
    return np.array([P2[l][topic].decision_function(vec[l:l + 1])[0] for l in range(n_layers)])


def dz(d):
    d = np.asarray(d, float)
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")


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
    return "".join(tokens)


def reverse_sentences(text):
    sents = re.findall(r"[^.!?]+[.!?]|\S[^.!?]*$", text)
    return " ".join(s.strip() for s in reversed(sents) if s.strip())


# Message order walked to accumulate the position-control span. u4 (index 6) is a last resort;
# u_{k+1} (index 8) and a_k (index 7) are never touched.
SPAN_ORDER = [0, 1, 2, 3, 4, 5, 6]


def make_position_control(n_target, seed_key, log):
    def f(ids, msgs):
        starts = [i for i, t in enumerate(ids) if t == special["im_start"]]
        ends = [i for i, t in enumerate(ids) if t == special["im_end"]]
        remaining, used = n_target, {}
        for m in SPAN_ORDER:
            if remaining <= 1:
                break
            lo, hi = starts[m] + 3, ends[m]
            take = min(hi - lo, remaining)
            if take > 1:
                shuffle_span(ids, lo, lo + take, seed_key + m)
                used[m] = take
                remaining -= take
        log.append(dict(target=n_target, achieved=n_target - remaining,
                        shortfall=remaining, used=used))
        return ids
    return f


# ---------------------------------------------------------------------------
# Clean baseline + the four perturbations, all on one basis
# ---------------------------------------------------------------------------
print("\nmeasuring ...")
results = {}
pos_log = []
for label in ("clean", "a_token_shuffle", "b_word_substitution", "c_sentence_reversal",
              "d_position_control"):
    vals = {}
    for cond in ("B", "C"):
        rows, texts = [], []
        for p in prefixes:
            a_k = prior_by[(cond, STRENGTH, p["prefix_id"])]["a_k"]
            n_ak = len(tokenizer(a_k, add_special_tokens=False).input_ids)
            key = int(abs(hash((p["prefix_id"], cond, label))) % (2 ** 31))
            perturb = None
            if label == "b_word_substitution":
                msgs, text = build_text(p, substitute_words(a_k, 0.20, key))
            elif label == "c_sentence_reversal":
                msgs, text = build_text(p, reverse_sentences(a_k))
            else:
                msgs, text = build_text(p, a_k)
                if label == "a_token_shuffle":
                    def perturb(ids, m, k=key):
                        st = [i for i, t in enumerate(ids) if t == special["im_start"]]
                        en = [i for i, t in enumerate(ids) if t == special["im_end"]]
                        shuffle_span(ids, st[7] + 3, en[7], k)
                        return ids
                elif label == "d_position_control":
                    perturb = make_position_control(n_ak, key, pos_log)
            texts.append(text)
            rows.append(readout(measure_vec(msgs, text, perturb), p["topic"]))
        for t in texts:
            assert t.endswith(UKP1_SUFFIX), "u_(k+1) not byte-identical"
        live = {mid for m in model.modules() for mid in m._forward_hooks}
        assert not (live & OUR_HOOK_IDS), "our hook still attached at measurement"
        vals[cond] = np.array([[r[l] for l in MID] for r in rows])
    diff = np.array([np.mean(vals["B"][:, j] - vals["C"][:, j]) for j in range(len(MID))])
    dzs = np.array([dz(vals["B"][:, j] - vals["C"][:, j]) for j in range(len(MID))])
    results[label] = dict(mean_diff=diff.mean(), dz=dzs.mean())
    print(f"  {label}: mean diff {diff.mean():+.3f}, dz {dzs.mean():+.2f}")

# ---------------------------------------------------------------------------
# Span accounting
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("LENGTH-MATCHED POSITION CONTROL -- SPAN ACCOUNTING")
print("=" * 100)
targets = np.array([e["target"] for e in pos_log])
achieved = np.array([e["achieved"] for e in pos_log])
short = np.array([e["shortfall"] for e in pos_log])
print(f"n = {len(pos_log)} transcripts (B and C at {STRENGTH}x)")
print(f"a_k length (target)      : mean {targets.mean():.1f} (sd {targets.std(ddof=1):.1f}), "
      f"range [{targets.min()}, {targets.max()}]")
print(f"shuffled span (achieved) : mean {achieved.mean():.1f} (sd {achieved.std(ddof=1):.1f}), "
      f"range [{achieved.min()}, {achieved.max()}]")
print(f"paired diff (achieved - target): mean {np.mean(achieved - targets):+.1f} tokens "
      f"(sd {np.std(achieved - targets, ddof=1):.1f})")
print(f"residual shortfall: mean {short.mean():.1f} tokens, "
      f"{int((short > 1).sum())}/{len(short)} transcripts fall short "
      f"(max {short.max()})")
print(f"previous version: 99.9-token mean shortfall, ~76% of intended span")

MSG_NAME = {0: "u1", 1: "a1", 2: "u2", 3: "a2", 4: "u3", 5: "a3", 6: "u4 (F)"}
print(f"\ntokens taken per prefix message (mean over {len(pos_log)} transcripts, "
      f"conversation order; a_k and u_(k+1) never touched):")
for m in SPAN_ORDER:
    taken = np.array([e["used"].get(m, 0) for e in pos_log])
    n_used = int((taken > 0).sum())
    print(f"  message {m} = {MSG_NAME[m]:<8} mean {taken.mean():6.1f} tokens, "
          f"used in {n_used}/{len(pos_log)} transcripts")
full_prefix = np.array([sum(e["used"].values()) for e in pos_log])
print(f"total prefix content available is the binding constraint: achieved = "
      f"{achieved.mean():.1f} tokens against a_k's {targets.mean():.1f}")

# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------
clean = results["clean"]
print("\n" + "=" * 100)
print("FULL PERTURBATION LADDER (P2, L11-29, corrected (d))")
print("=" * 100)
print(f"{'perturbation':>24}{'mean diff':>12}{'mean-diff surv %':>18}{'dz':>8}{'dz surv %':>11}")
print(f"{'(clean)':>24}{clean['mean_diff']:>12.3f}{100.0:>18.1f}{clean['dz']:>8.2f}"
      f"{100.0:>11.1f}")
rows = []
for label in ("a_token_shuffle", "b_word_substitution", "c_sentence_reversal",
              "d_position_control"):
    r = results[label]
    ms = r["mean_diff"] / clean["mean_diff"] * 100
    ds = r["dz"] / clean["dz"] * 100
    rows.append(dict(perturbation=label, mean_diff=r["mean_diff"], dz=r["dz"],
                     mean_diff_survival_pct=ms, dz_survival_pct=ds))
    print(f"{label:>24}{r['mean_diff']:>12.3f}{ms:>18.1f}{r['dz']:>8.2f}{ds:>11.1f}")
pd.DataFrame(rows).to_csv(os.path.join(TABLES, "stage3b_ladder_corrected.csv"), index=False)
pd.DataFrame(pos_log).to_csv(os.path.join(TABLES, "stage3b_position_control_spans.csv"),
                             index=False)

d_surv = rows[-1]["mean_diff_survival_pct"]
print(f"\n(d) mean-difference survival after proper length matching: {d_surv:.1f}%")
if d_surv > 100:
    print("(d) still exceeds 100%. Reported as unexplained, per instruction. Not investigated "
          "further; no additional perturbation variants run.")
print(f"\ntables -> {TABLES}")
print("\nDONE")
