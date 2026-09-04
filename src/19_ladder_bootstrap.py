"""Bootstrap 95% CIs on the perturbation-ladder survival ratios.

WHY THIS RE-MEASURES RATHER THAN READING A TABLE: src/15 persisted only the aggregate ladder
rows, not the per-prefix values a bootstrap needs. Worse, (a), (b) and (d) were seeded from
Python's per-process salted `hash()`, so their exact permutations from that run cannot be
reconstructed. Only (c), which uses no RNG, is exactly reproducible.

So this is a THIRD DRAW, seeded from hashlib (stable, reproducible from now on). Its point
estimates for (a), (b) and (d) will differ from src/15's by roughly the +/-20pp run-to-run
spread already documented; (c) should reproduce src/15 exactly, which is the check below.
The bootstrap CIs describe sampling uncertainty over the 20 paired prefixes WITHIN this draw.
They do not capture the across-draw RNG spread, which is a separate and comparable source of
variation.

Clean baseline is read from stage3b_BC_perprefix_probe.csv (stored by src/17) -- not remeasured.
No new conditions, no generation.
"""

import hashlib
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
N_BOOT, BOOT_SEED = 10000, 7
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
prior_by = {(r["condition"], r["strength"], r["prefix_id"]): r for r in s3["records"]}
with open(S2_PATH) as f:
    dialogues = json.load(f)["dialogues"]
by_id = {d["dialogue_id"]: d for d in dialogues}

prefix_ids = [r["prefix_id"] for r in s3["records"] if r["condition"] == "A"]
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
print(f"{N} prefixes, strength {STRENGTH}x, layer {LAYER}")

# ---- P2 probe, identical construction to src/13-17 ----
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


def stable_seed(*parts):
    """Reproducible across processes, unlike Python's salted hash()."""
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def build_text(prefix, a_k):
    msgs = list(prefix["messages"]) + [{"role": "assistant", "content": a_k},
                                       {"role": "user", "content": UKP1_TEXT}]
    return msgs, tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def shuffle_span(ids, lo, hi, seed):
    perm = np.random.default_rng(seed).permutation(hi - lo)
    span = ids[lo:hi]
    ids[lo:hi] = [span[i] for i in perm]


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


SPAN_ORDER = [0, 1, 2, 3, 4, 5, 6]


def make_position_control(n_target, seed_key):
    def f(ids, msgs):
        starts = [i for i, t in enumerate(ids) if t == special["im_start"]]
        ends = [i for i, t in enumerate(ids) if t == special["im_end"]]
        remaining = n_target
        for m in SPAN_ORDER:
            if remaining <= 1:
                break
            lo, hi = starts[m] + 3, ends[m]
            take = min(hi - lo, remaining)
            if take > 1:
                shuffle_span(ids, lo, lo + take, seed_key + m)
                remaining -= take
        return ids
    return f


def measure_readout(msgs, text, topic, perturb=None):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if perturb is not None:
        ids = perturb(ids, msgs)
    bounds = find_turn_boundaries(tokenizer, ids, msgs)
    resid = extract_residual_stream(model, torch.tensor([ids], device="cuda"))
    vec = resid[:, bounds["eot_u_5"], :].float().cpu().numpy()
    return np.array([P2[l][topic].decision_function(vec[l:l + 1])[0] for l in range(n_layers)])


# ---------------------------------------------------------------------------
# Clean baseline from stored per-prefix table
# ---------------------------------------------------------------------------
bc = pd.read_csv(os.path.join(TABLES, "stage3b_BC_perprefix_probe.csv"))
bc = bc[bc.probe == "P2"]
clean_mat = bc.pivot(index="prefix_id", columns="layer", values="diff_B_minus_C")
clean_mat = clean_mat.loc[[p["prefix_id"] for p in prefixes]].to_numpy()  # (N, len(MID))
assert clean_mat.shape == (N, len(MID)), clean_mat.shape
print(f"clean baseline from stored table: mean diff over L11-29 = {clean_mat.mean(0).mean():+.4f}")

# ---------------------------------------------------------------------------
# Re-measure the four perturbations with stable seeds
# ---------------------------------------------------------------------------
LABELS = ["a_token_shuffle", "b_word_substitution", "c_sentence_reversal", "d_position_control"]
pert_mat = {}
rows = []
for label in LABELS:
    per_cond = {}
    for cond in ("B", "C"):
        out = []
        for p in prefixes:
            pid = p["prefix_id"]
            a_k = prior_by[(cond, STRENGTH, pid)]["a_k"]
            n_ak = len(tokenizer(a_k, add_special_tokens=False).input_ids)
            key = stable_seed(pid, cond, label)
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
                else:
                    perturb = make_position_control(n_ak, key)
            assert text.endswith(UKP1_SUFFIX), "u_(k+1) not byte-identical"
            out.append(measure_readout(msgs, text, p["topic"], perturb))
        live = {mid for m in model.modules() for mid in m._forward_hooks}
        assert not (live & OUR_HOOK_IDS)
        per_cond[cond] = np.array([[r[l] for l in MID] for r in out])
    pert_mat[label] = per_cond["B"] - per_cond["C"]
    for i, p in enumerate(prefixes):
        for j, l in enumerate(MID):
            rows.append(dict(perturbation=label, prefix_id=p["prefix_id"], layer=l,
                             diff_B_minus_C=pert_mat[label][i, j],
                             clean_diff_B_minus_C=clean_mat[i, j]))
    print(f"  {label}: mean diff {pert_mat[label].mean(0).mean():+.4f}, "
          f"survival {pert_mat[label].mean(0).mean() / clean_mat.mean(0).mean() * 100:.1f}%")

pd.DataFrame(rows).to_csv(os.path.join(TABLES, "stage3b_ladder_perprefix_draw3.csv"),
                          index=False)
print(f"\nwrote stage3b_ladder_perprefix_draw3.csv ({len(rows)} rows)")

# ---------------------------------------------------------------------------
# Bootstrap the survival ratio over prefixes
# ---------------------------------------------------------------------------
rng = np.random.default_rng(BOOT_SEED)


def survival(pert, clean):
    """mean over layers of the per-layer mean paired difference, as a % of clean."""
    return pert.mean(0).mean() / clean.mean(0).mean() * 100


boot_idx = rng.integers(0, N, (N_BOOT, N))
results = []
for label in LABELS:
    point = survival(pert_mat[label], clean_mat)
    draws = np.array([survival(pert_mat[label][i], clean_mat[i]) for i in boot_idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    results.append(dict(perturbation=label, survival_pct=point, ci_lo=lo, ci_hi=hi,
                        contains_100=bool(lo <= 100 <= hi), n_boot=N_BOOT, n_prefixes=N))
    print(f"  {label}: {point:.1f}% [{lo:.1f}, {hi:.1f}]")
res = pd.DataFrame(results)
res.to_csv(os.path.join(TABLES, "stage3b_ladder_bootstrap_ci.csv"), index=False)

print("\n" + "=" * 92)
print("LADDER SURVIVAL WITH BOOTSTRAP 95% CIs "
      f"({N_BOOT:,} resamples over {N} paired prefixes, draw 3, stable seeds)")
print("=" * 92)
print(f"{'perturbation':>24}{'survival':>11}{'95% CI':>22}{'contains 100%?':>17}"
      f"{'src/15 draw 2':>15}")
lad2 = pd.read_csv(os.path.join(TABLES, "stage3b_ladder_corrected.csv")).set_index("perturbation")
for r in results:
    prev = lad2.loc[r["perturbation"], "mean_diff_survival_pct"]
    ci_txt = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"
    print(f"{r['perturbation']:>24}{r['survival_pct']:>10.1f}%{ci_txt:>22}"
          f"{('YES' if r['contains_100'] else 'no'):>17}{prev:>14.1f}%")

a, c = results[0], results[2]
overlap = (a["ci_lo"] <= c["ci_hi"]) and (c["ci_lo"] <= a["ci_hi"])
print(f"\n(a) CI = [{a['ci_lo']:.1f}, {a['ci_hi']:.1f}]   "
      f"(c) CI = [{c['ci_lo']:.1f}, {c['ci_hi']:.1f}]")
print(f"(a) and (c) CIs overlap: {'YES' if overlap else 'NO'}")

c_delta = abs(c["survival_pct"] - lad2.loc["c_sentence_reversal", "mean_diff_survival_pct"])
print(f"\nreproducibility check -- (c) uses no RNG, so it must match src/15 exactly: "
      f"draw 3 {c['survival_pct']:.2f}% vs src/15 "
      f"{lad2.loc['c_sentence_reversal', 'mean_diff_survival_pct']:.2f}%, "
      f"delta {c_delta:.4f}pp -> {'MATCH' if c_delta < 0.05 else 'DIFFERS'}")
print("\nDONE")
