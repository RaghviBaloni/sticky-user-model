"""Splice controls and a depth-matched probe. No generation, no steering.

  1. Scripted-assistant control at full n: for every cell and every replicate i, pair
     (SSSF user turns + NNNF|cell|i assistant text) against the original NNNF|cell|i.
     Both members of a pair carry byte-identical assistant text. n = 160 pairs.
  2. New arm, assembly only: NNNF user turns + assistant text spliced verbatim from
     SSSF|cell|i, same F. Compared against NNNF, this isolates what the assistant turns
     alone carry, with the user turns held fixed.
  3. Depth-matched probe: trained at eot_u_3 on the assembled transcripts (turn-3 valence is
     S in SSSF and H in HHHF), leave-one-topic-out, read at eot_u_4. Tests whether the sign
     inversion is an artifact of probing depth-850 activations with a single-turn probe.

All assistant and user text is spliced verbatim from existing transcripts and the existing
pool. Forward passes are required because these are new token sequences, but nothing is
generated and no steering hook is ever attached.
"""

import json
import os

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
    load_turn_pool,
)

CACHE_DIR = os.path.join(CACHE_ROOT, "stage2_transcripts")
SINGLE_DIR = os.path.join(CACHE_ROOT, "single_turn_pool")
TABLES = os.path.join(REPO_ROOT, "results", "tables")
TRANSCRIPT_PATH = os.path.join(REPO_ROOT, "data", "transcripts", "stage2_transcripts.json")
SPLICE_CACHE = os.path.join(CACHE_ROOT, "splice_controls")
ROLE_SEQUENCE = ["user", "assistant", "user", "assistant", "user", "assistant", "user"]
ARMS = ["SSSF", "NNNF", "HHHF", "SSHF"]
C_VALUE = 1.0
os.makedirs(SPLICE_CACHE, exist_ok=True)

manifest = pd.read_parquet(os.path.join(CACHE_DIR, "manifest.parquet"))
shards = sorted(f for f in os.listdir(CACHE_DIR) if f.startswith("acts_"))
acts = torch.cat([torch.load(os.path.join(CACHE_DIR, f)) for f in shards], dim=0)
n_layers = acts.shape[1]
with open(TRANSCRIPT_PATH) as f:
    dialogues = json.load(f)["dialogues"]
by_id = {d["dialogue_id"]: d for d in dialogues}
pool, _ = load_turn_pool()
pool_by_id = {t.turn_id: t for t in pool}
print(f"cache {tuple(acts.shape)}, {len(dialogues)} transcripts, {n_layers} layers")

single = torch.load(os.path.join(SINGLE_DIR, "single_turn_acts.pt"))
with open(os.path.join(SINGLE_DIR, "single_turn_meta.json")) as f:
    single_meta = json.load(f)
single_labels = np.array(single_meta["labels"])
sh = np.isin(single_labels, ["S", "H"])


def fit_single_turn_probe(role, layer):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty="l2", C=C_VALUE, max_iter=5000))
    clf.fit(single[role][:, layer, :].float().numpy()[sh], single_labels[sh])
    assert list(clf[-1].classes_) == ["H", "S"]
    return clf


st_probe = {layer: fit_single_turn_probe("eot_u_1", layer) for layer in range(n_layers)}
print(f"fit {len(st_probe)} single-turn eot_u probes (positive = distressed)")

model, tokenizer = load_model_and_tokenizer(MODEL_ID)
CELLS = sorted({(d["topic"], d["phrasing"]) for d in dialogues})
REPLICATES = sorted({d["replicate"] for d in dialogues})
print(f"{len(CELLS)} cells x {len(REPLICATES)} replicates")


def build_and_extract(user_arm, assistant_source_arm, tag):
    """Splice user turns from one arm with assistant text from another; extract positions."""
    records = []
    for topic, phrasing in CELLS:
        valences = {"SSSF": ("S", "S", "S"), "NNNF": ("N", "N", "N"),
                    "HHHF": ("H", "H", "H")}[user_arm]
        user_turns = [pool_by_id[f"U[{topic},{phrasing},{v},{i}]"].text
                      for i, v in enumerate(valences, start=1)]
        user_turns.append(pool_by_id[f"F[{topic},{phrasing}]"].text)
        for rep in REPLICATES:
            donor = by_id[f"{assistant_source_arm}|{topic}|{phrasing}|{rep:02d}"]
            assistant = donor["assistant_turns"]
            messages = [{"role": r, "content": c} for r, c in zip(
                ROLE_SEQUENCE,
                [user_turns[0], assistant[0], user_turns[1], assistant[1],
                 user_turns[2], assistant[2], user_turns[3]])]
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=False)
            ids = tokenizer(text, add_special_tokens=False).input_ids
            bounds = find_turn_boundaries(tokenizer, ids, messages)
            resid = extract_residual_stream(model, torch.tensor([ids], device="cuda"))
            rec = dict(tag=tag, topic=topic, phrasing=phrasing, replicate=rep,
                       n_tokens=len(ids), assistant_donor=donor["dialogue_id"])
            for position in ("eot_u_3", "eot_u_4"):
                vec = resid[:, bounds[position], :].float().cpu().numpy()
                rec[f"{position}_vec"] = vec
            records.append(rec)
        print(f"  {tag}: {topic}/{phrasing} done ({len(records)} total)")
    return records


def cached_readout(arm, position):
    """Readout of an original arm straight from the transcript cache."""
    sub = manifest[(manifest.condition == arm) & (manifest.position_name == position)]
    sub = sub.sort_values(["topic", "phrasing", "replicate"])
    idx = sub.row_offset.to_numpy()
    out = pd.DataFrame(dict(topic=sub.topic.to_numpy(), phrasing=sub.phrasing.to_numpy(),
                            replicate=sub.replicate.to_numpy(),
                            n_tokens=sub.n_tokens.to_numpy()))
    for layer in range(n_layers):
        out[f"L{layer}"] = st_probe[layer].decision_function(acts[idx, layer, :].float().numpy())
    return out.sort_values(["topic", "phrasing", "replicate"]).reset_index(drop=True)


def records_to_frame(records, position):
    rows = []
    for rec in records:
        row = dict(topic=rec["topic"], phrasing=rec["phrasing"], replicate=rec["replicate"],
                   n_tokens=rec["n_tokens"])
        vec = rec[f"{position}_vec"]
        for layer in range(n_layers):
            row[f"L{layer}"] = float(
                st_probe[layer].decision_function(vec[layer:layer + 1])[0])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["topic", "phrasing", "replicate"]).reset_index(drop=True)


def dz(diff):
    diff = np.asarray(diff, float)
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan")


# Sanity: reconstructing NNNF from its own pieces must reproduce the stored transcript.
topic, phrasing = CELLS[0]
donor = by_id[f"NNNF|{topic}|{phrasing}|00"]
valences = ("N", "N", "N")
u = [pool_by_id[f"U[{topic},{phrasing},{v},{i}]"].text for i, v in enumerate(valences, start=1)]
u.append(pool_by_id[f"F[{topic},{phrasing}]"].text)
msgs = [{"role": r, "content": c} for r, c in zip(
    ROLE_SEQUENCE, [u[0], donor["assistant_turns"][0], u[1], donor["assistant_turns"][1],
                    u[2], donor["assistant_turns"][2], u[3]])]
assert tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) \
    == donor["transcript_text"], "reconstruction does not reproduce the stored NNNF transcript"
print("reconstruction check: NNNF rebuilt from pool + its own assistant turns is byte-identical "
      "to the stored transcript -- cached NNNF readouts are reusable as the pair partner")

# ---------------------------------------------------------------------------
# 1. Scripted-assistant control at full n
# ---------------------------------------------------------------------------
print("\nbuilding control 1: SSSF user turns + NNNF|cell|i assistant text")
ctrl1 = build_and_extract("SSSF", "NNNF", "SSSF_user__NNNF_assist")
frame1 = records_to_frame(ctrl1, "eot_u_4")
nnnf = cached_readout("NNNF", "eot_u_4")
assert (frame1[["topic", "phrasing", "replicate"]].to_numpy()
        == nnnf[["topic", "phrasing", "replicate"]].to_numpy()).all(), "pairing misaligned"
n_pairs = len(frame1)

print("\n" + "=" * 104)
print(f"1. SCRIPTED-ASSISTANT CONTROL AT FULL n  (n={n_pairs} paired, "
      "assistant text byte-identical within each pair)")
print("=" * 104)
len_a, len_b = frame1.n_tokens.to_numpy(), nnnf.n_tokens.to_numpy()
print(f"length equalisation: SSSF-user {len_a.mean():.1f} (sd {len_a.std(ddof=1):.1f}) vs "
      f"NNNF-user {len_b.mean():.1f} (sd {len_b.std(ddof=1):.1f}); "
      f"paired diff {np.mean(len_a - len_b):+.1f} tokens "
      f"(sd {np.std(len_a - len_b, ddof=1):.1f}) -- assistant text identical, so this is the "
      "user-turn length difference alone")
print(f"\n{'L':>4}{'SSSFuser+Nassist':>19}{'sd':>8}{'NNNF':>10}{'sd':>8}{'diff':>9}{'dz':>8}"
      f"{'orig diff':>11}{'orig dz':>9}")
orig = pd.read_csv(os.path.join(TABLES, "readout_eot_u_4_by_layer.csv"))
rows1 = []
for layer in range(n_layers):
    a, b = frame1[f"L{layer}"].to_numpy(), nnnf[f"L{layer}"].to_numpy()
    o = orig.iloc[layer]
    rec = dict(layer=layer, spliced_mean=a.mean(), spliced_sd=a.std(ddof=1),
               nnnf_mean=b.mean(), nnnf_sd=b.std(ddof=1), diff=a.mean() - b.mean(),
               dz=dz(a - b), original_diff=o.SSSF_mean - o.NNNF_mean,
               original_dz=o.dz_SSSF_NNNF)
    rows1.append(rec)
    print(f"{layer:>4}{a.mean():>19.3f}{a.std(ddof=1):>8.3f}{b.mean():>10.3f}"
          f"{b.std(ddof=1):>8.3f}{rec['diff']:>9.3f}{rec['dz']:>8.2f}"
          f"{rec['original_diff']:>11.3f}{rec['original_dz']:>9.2f}")
pd.DataFrame(rows1).to_csv(os.path.join(TABLES, "control1_scripted_assistant_fulln.csv"),
                           index=False)

# ---------------------------------------------------------------------------
# 2. NNNF user turns + SSSF assistant text
# ---------------------------------------------------------------------------
print("\nbuilding control 2: NNNF user turns + SSSF|cell|i assistant text")
ctrl2 = build_and_extract("NNNF", "SSSF", "NNNF_user__SSSF_assist")
frame2 = records_to_frame(ctrl2, "eot_u_4")
assert (frame2[["topic", "phrasing", "replicate"]].to_numpy()
        == nnnf[["topic", "phrasing", "replicate"]].to_numpy()).all(), "pairing misaligned"

print("\n" + "=" * 104)
print(f"2. ASSISTANT TURNS ALONE: NNNF user turns + SSSF assistant text, vs NNNF (n={n_pairs})")
print("=" * 104)
la, lb = frame2.n_tokens.to_numpy(), nnnf.n_tokens.to_numpy()
print(f"length: spliced {la.mean():.1f} vs NNNF {lb.mean():.1f}, paired diff "
      f"{np.mean(la - lb):+.1f} (sd {np.std(la - lb, ddof=1):.1f}) -- user turns identical, so "
      "this is the assistant-turn length difference alone")
print(f"\n{'L':>4}{'Nuser+Sassist':>16}{'sd':>8}{'NNNF':>10}{'sd':>8}{'diff':>9}{'dz':>8}"
      f"{'  | full SSSF diff':>19}{'dz':>8}")
rows2 = []
for layer in range(n_layers):
    a, b = frame2[f"L{layer}"].to_numpy(), nnnf[f"L{layer}"].to_numpy()
    o = orig.iloc[layer]
    rec = dict(layer=layer, spliced_mean=a.mean(), spliced_sd=a.std(ddof=1),
               nnnf_mean=b.mean(), nnnf_sd=b.std(ddof=1), diff=a.mean() - b.mean(),
               dz=dz(a - b), full_sssf_diff=o.SSSF_mean - o.NNNF_mean,
               full_sssf_dz=o.dz_SSSF_NNNF)
    rows2.append(rec)
    print(f"{layer:>4}{a.mean():>16.3f}{a.std(ddof=1):>8.3f}{b.mean():>10.3f}"
          f"{b.std(ddof=1):>8.3f}{rec['diff']:>9.3f}{rec['dz']:>8.2f}"
          f"{rec['full_sssf_diff']:>19.3f}{rec['full_sssf_dz']:>8.2f}")
pd.DataFrame(rows2).to_csv(os.path.join(TABLES, "control2_assistant_only.csv"), index=False)

# ---------------------------------------------------------------------------
# 3. Depth-matched probe: train at eot_u_3, leave-one-topic-out, read at eot_u_4
# ---------------------------------------------------------------------------
print("\n" + "=" * 104)
print("3. DEPTH-MATCHED PROBE -- trained at eot_u_3 (SSSF turn-3 = S vs HHHF turn-3 = H),")
print("   leave-one-topic-out, read at eot_u_4")
print("=" * 104)


def cache_rows(arm, position):
    sub = manifest[(manifest.condition == arm) & (manifest.position_name == position)]
    sub = sub.sort_values(["topic", "phrasing", "replicate"])
    return sub.row_offset.to_numpy(), sub.topic.to_numpy()


train_idx_S, train_topic_S = cache_rows("SSSF", "eot_u_3")
train_idx_H, train_topic_H = cache_rows("HHHF", "eot_u_3")
train_topics = np.concatenate([train_topic_S, train_topic_H])
train_y = np.array(["S"] * len(train_idx_S) + ["H"] * len(train_idx_H))
topics = sorted(set(train_topics))
print(f"training set at eot_u_3: {len(train_idx_S)} S (SSSF) + {len(train_idx_H)} H (HHHF), "
      f"{len(topics)} topic folds")

read_idx, read_topic, read_arm = {}, {}, {}
for arm in ARMS:
    idx, tp = cache_rows(arm, "eot_u_4")
    read_idx[arm], read_topic[arm] = idx, tp

rows3 = []
for layer in range(n_layers):
    Xtr = np.concatenate([acts[train_idx_S, layer, :].float().numpy(),
                          acts[train_idx_H, layer, :].float().numpy()], axis=0)
    fold_acc, readouts = [], {arm: np.zeros(len(read_idx[arm])) for arm in ARMS}
    spliced_readouts = {}
    for held in topics:
        tr = train_topics != held
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(penalty="l2", C=C_VALUE, max_iter=5000))
        clf.fit(Xtr[tr], train_y[tr])
        assert list(clf[-1].classes_) == ["H", "S"]
        fold_acc.append(float((clf.predict(Xtr[~tr]) == train_y[~tr]).mean()))
        for arm in ARMS:
            sel = read_topic[arm] == held
            readouts[arm][sel] = clf.decision_function(
                acts[read_idx[arm][sel], layer, :].float().numpy())
        for tag, records in (("SSSF_user__NNNF_assist", ctrl1),
                             ("NNNF_user__SSSF_assist", ctrl2)):
            vecs = np.stack([r[f"eot_u_4_vec"][layer] for r in records
                             if r["topic"] == held])
            spliced_readouts.setdefault(tag, {})[held] = clf.decision_function(vecs)
    rec = dict(layer=layer, eot_u_3_heldout_acc=float(np.mean(fold_acc)))
    for arm in ARMS:
        rec[f"{arm}_mean"] = readouts[arm].mean()
        rec[f"{arm}_sd"] = readouts[arm].std(ddof=1)
    rec["dz_SSSF_NNNF"] = dz(readouts["SSSF"] - readouts["NNNF"])
    rec["dz_HHHF_NNNF"] = dz(readouts["HHHF"] - readouts["NNNF"])
    rec["dz_SSHF_NNNF"] = dz(readouts["SSHF"] - readouts["NNNF"])
    for tag in spliced_readouts:
        allv = np.concatenate([spliced_readouts[tag][t] for t in topics])
        rec[f"{tag}_mean"] = allv.mean()
    rows3.append(rec)

depth = pd.DataFrame(rows3)
depth.to_csv(os.path.join(TABLES, "depth_matched_probe.csv"), index=False)
print(f"\n{'L':>4}{'eot_u3 acc':>12}{'SSSF':>9}{'NNNF':>9}{'HHHF':>9}{'SSHF':>9}"
      f"{'dz S-N':>9}{'dz H-N':>9}{'  Suser+Nass':>14}{'Nuser+Sass':>12}")
for layer in range(n_layers):
    r = depth.iloc[layer]
    print(f"{layer:>4}{r.eot_u_3_heldout_acc:>12.3f}{r.SSSF_mean:>9.3f}{r.NNNF_mean:>9.3f}"
          f"{r.HHHF_mean:>9.3f}{r.SSHF_mean:>9.3f}{r.dz_SSSF_NNNF:>9.2f}{r.dz_HHHF_NNNF:>9.2f}"
          f"{r['SSSF_user__NNNF_assist_mean']:>14.3f}"
          f"{r['NNNF_user__SSSF_assist_mean']:>12.3f}")

neg_single = sum(1 for layer in range(11, 30) if orig.iloc[layer].dz_SSSF_NNNF < 0)
neg_depth = sum(1 for layer in range(11, 30) if depth.iloc[layer].dz_SSSF_NNNF < 0)
print(f"\nsign inversion across L11-L29 (19 layers):")
print(f"  single-turn probe : dz(SSSF-NNNF) < 0 in {neg_single}/19 layers")
print(f"  depth-matched probe: dz(SSSF-NNNF) < 0 in {neg_depth}/19 layers")
print(f"\ntables -> {TABLES}")
print("\nDONE -- no generation, no steering, no layer selected.")
