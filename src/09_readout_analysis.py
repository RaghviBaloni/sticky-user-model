"""Stage 2 follow-up -- probe readout from the existing cache, all layers, no L* selection.

  1. Signed-distance readout at u_4 / eot_u_4 by arm, every layer, paired effect sizes.
  2. Same readout at eot_u_1..eot_u_4 -- the Figure 1 trajectory data.
  3. Readout at eot_u_4 regressed on transcript token length, within arm and pooled.
  4. Scripted-assistant control arms (SSSF vs NNNF with byte-identical assistant text).

Nothing is regenerated: transcript activations come from data/cache, and the probes are fit
on the single-turn pool activations cached by 07_single_turn_probes.py. No layer is selected;
every table spans all 32 layers.

Readout definition: signed distance w.h + b from a per-layer L2 logistic probe trained on the
single-turn pool S vs H turns only (48 turns), standardised on that training set.
**Positive = distressed (S).** Trained at u_1 and read at u_4; trained at eot_u_1 and read at
eot_u_2..4, so probe and readout always share a token role. metrics.md Sec 1 is still unlocked;
this is a readout for inspection, not a locked metric.
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
ARMS = ["SSSF", "NNNF", "HHHF", "SSHF"]
C_VALUE = 1.0
os.makedirs(TABLES, exist_ok=True)

# ---------------------------------------------------------------------------
# Load caches
# ---------------------------------------------------------------------------
manifest = pd.read_parquet(os.path.join(CACHE_DIR, "manifest.parquet"))
shards = sorted(f for f in os.listdir(CACHE_DIR) if f.startswith("acts_"))
acts = torch.cat([torch.load(os.path.join(CACHE_DIR, f)) for f in shards], dim=0)
assert acts.shape[0] == len(manifest), (acts.shape, len(manifest))
n_layers = acts.shape[1]
print(f"loaded transcript cache {tuple(acts.shape)} and manifest ({len(manifest)} rows)")

single = torch.load(os.path.join(SINGLE_DIR, "single_turn_acts.pt"))
with open(os.path.join(SINGLE_DIR, "single_turn_meta.json")) as f:
    single_meta = json.load(f)
single_labels = np.array(single_meta["labels"])
sh_mask = np.isin(single_labels, ["S", "H"])
print(f"single-turn probe training set: {int(sh_mask.sum())} turns (S vs H) "
      f"at positions {single_meta['positions']}")


def fit_probe(position_role, layer):
    """L2 logistic S-vs-H probe on single-turn pool data. Positive score = S."""
    X = single[position_role][:, layer, :].float().numpy()[sh_mask]
    y = single_labels[sh_mask]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty="l2", C=C_VALUE, max_iter=5000))
    clf.fit(X, y)
    assert list(clf[-1].classes_) == ["H", "S"], clf[-1].classes_
    return clf


PROBE_FOR = {"u_4": "u_1", "eot_u_1": "eot_u_1", "eot_u_2": "eot_u_1",
             "eot_u_3": "eot_u_1", "eot_u_4": "eot_u_1"}
probes = {}
for position, train_pos in PROBE_FOR.items():
    for layer in range(n_layers):
        probes.setdefault(train_pos, {})
        if layer not in probes[train_pos]:
            probes[train_pos][layer] = fit_probe(train_pos, layer)
print(f"fit {sum(len(v) for v in probes.values())} probes "
      f"({len(probes)} token roles x {n_layers} layers)")


def readout_frame(position_name, rows_index=None):
    """Signed distance at `position_name` for every dialogue, every layer."""
    sub = manifest[manifest.position_name == position_name].sort_values("row_offset")
    idx = sub.row_offset.to_numpy()
    train_pos = PROBE_FOR[position_name]
    out = pd.DataFrame(dict(dialogue_id=sub.dialogue_id.to_numpy(),
                            arm=sub.condition.to_numpy(), topic=sub.topic.to_numpy(),
                            phrasing=sub.phrasing.to_numpy(),
                            replicate=sub.replicate.to_numpy(),
                            n_tokens=sub.n_tokens.to_numpy()))
    for layer in range(n_layers):
        X = acts[idx, layer, :].float().numpy()
        out[f"L{layer}"] = probes[train_pos][layer].decision_function(X)
    return out


def cohens_dz(diff):
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def paired_table(frame, layer_cols):
    """Wide table paired on (topic, phrasing, replicate) across arms."""
    key = ["topic", "phrasing", "replicate"]
    wide = {}
    for arm in ARMS:
        wide[arm] = frame[frame.arm == arm].set_index(key)[layer_cols].sort_index()
    common = wide[ARMS[0]].index
    for arm in ARMS[1:]:
        common = common.intersection(wide[arm].index)
    return {arm: wide[arm].loc[common] for arm in ARMS}, len(common)


# ---------------------------------------------------------------------------
# 1. Readout at u_4 and eot_u_4, all layers, by arm + paired effect sizes
# ---------------------------------------------------------------------------
layer_cols = [f"L{i}" for i in range(n_layers)]
CONTRASTS = [("SSSF", "NNNF"), ("HHHF", "NNNF"), ("SSHF", "NNNF"), ("SSHF", "SSSF")]

for position in ("eot_u_4", "u_4"):
    frame = readout_frame(position)
    frame.to_csv(os.path.join(TABLES, f"readout_{position}.csv"), index=False)
    paired, n_pairs = paired_table(frame, layer_cols)
    print("\n" + "=" * 110)
    print(f"1. READOUT AT {position} BY ARM, ALL LAYERS  (positive = distressed; "
          f"{n_pairs} paired (topic,phrasing,replicate) tuples per arm)")
    print("=" * 110)
    header = (f"{'layer':>5}" + "".join(f"{a + ' mean':>13}{'sd':>8}" for a in ARMS)
              + "".join(f"{f'dz {a}-{b}':>13}" for a, b in CONTRASTS))
    print(header)
    rows = []
    for layer, col in enumerate(layer_cols):
        line = f"{layer:>5}"
        rec = dict(layer=layer, position=position, n_pairs=n_pairs)
        for arm in ARMS:
            vals = paired[arm][col].to_numpy()
            line += f"{vals.mean():>13.3f}{vals.std(ddof=1):>8.3f}"
            rec[f"{arm}_mean"] = vals.mean()
            rec[f"{arm}_sd"] = vals.std(ddof=1)
        for a, b in CONTRASTS:
            dz = cohens_dz(paired[a][col].to_numpy() - paired[b][col].to_numpy())
            line += f"{dz:>13.2f}"
            rec[f"dz_{a}_{b}"] = dz
        rows.append(rec)
        print(line)
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, f"readout_{position}_by_layer.csv"),
                              index=False)

# ---------------------------------------------------------------------------
# 2. Trajectory: eot_u_1 .. eot_u_4
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("2. TRAJECTORY -- readout at eot_u_1..eot_u_4 by arm (Figure 1 data)")
print("=" * 110)
traj_rows = []
frames = {p: readout_frame(p) for p in ("eot_u_1", "eot_u_2", "eot_u_3", "eot_u_4")}
for position, frame in frames.items():
    paired, n_pairs = paired_table(frame, layer_cols)
    for layer, col in enumerate(layer_cols):
        rec = dict(position=position, turn=int(position[-1]), layer=layer, n_pairs=n_pairs)
        for arm in ARMS:
            vals = paired[arm][col].to_numpy()
            rec[f"{arm}_mean"] = vals.mean()
            rec[f"{arm}_sd"] = vals.std(ddof=1)
        for a, b in CONTRASTS:
            rec[f"dz_{a}_{b}"] = cohens_dz(paired[a][col].to_numpy() - paired[b][col].to_numpy())
        traj_rows.append(rec)
traj = pd.DataFrame(traj_rows)
traj.to_csv(os.path.join(TABLES, "trajectory_eot_u.csv"), index=False)

for layer in [0, 4, 8, 12, 16, 20, 24, 28, 31]:
    print(f"\n  layer {layer}: mean readout by turn boundary")
    print(f"{'':>10}" + "".join(f"{a:>10}" for a in ARMS))
    for turn in (1, 2, 3, 4):
        r = traj[(traj.layer == layer) & (traj.turn == turn)].iloc[0]
        print(f"  eot_u_{turn}" + "".join(f"{r[f'{a}_mean']:>10.3f}" for a in ARMS))
print(f"\nfull trajectory (4 turns x {n_layers} layers x 4 arms) -> "
      f"{os.path.join(TABLES, 'trajectory_eot_u.csv')}")

# ---------------------------------------------------------------------------
# 3. Readout at eot_u_4 vs transcript length
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("3. READOUT AT eot_u_4 REGRESSED ON TRANSCRIPT TOKEN LENGTH")
print("=" * 110)
frame4 = readout_frame("eot_u_4")


def ols(X, y):
    """Returns (beta, se, t). X includes its own intercept column."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return beta, se, beta / se


print("\nlength overlap across arms (transcript tokens):")
for arm in ARMS:
    lengths = frame4[frame4.arm == arm].n_tokens.to_numpy()
    print(f"  {arm}: n={len(lengths)} mean={lengths.mean():7.1f} "
          f"range=[{lengths.min()}, {lengths.max()}]")
lo = max(frame4[frame4.arm == a].n_tokens.min() for a in ARMS)
hi = min(frame4[frame4.arm == a].n_tokens.max() for a in ARMS)
common = frame4[(frame4.n_tokens >= lo) & (frame4.n_tokens <= hi)]
print(f"  common support = [{lo}, {hi}] tokens, "
      f"{len(common)}/{len(frame4)} dialogues ("
      + ", ".join(f"{a}:{int((common.arm == a).sum())}" for a in ARMS) + ")")

reg_rows = []
for layer in range(n_layers):
    col = f"L{layer}"
    rec = dict(layer=layer)
    # (a) within-arm slope of readout on length
    for arm in ARMS:
        sub = frame4[frame4.arm == arm]
        X = np.column_stack([np.ones(len(sub)), sub.n_tokens.to_numpy()])
        beta, se, t = ols(X, sub[col].to_numpy())
        rec[f"slope_{arm}"] = beta[1]
        rec[f"t_{arm}"] = t[1]
    # (b) pooled arm effect, with and without length as a covariate
    d = frame4
    dummies = np.column_stack([(d.arm == a).to_numpy(float) for a in ARMS if a != "NNNF"])
    X0 = np.column_stack([np.ones(len(d)), dummies])
    X1 = np.column_stack([X0, d.n_tokens.to_numpy()])
    y = d[col].to_numpy()
    b0, _, t0 = ols(X0, y)
    b1, _, t1 = ols(X1, y)
    for i, arm in enumerate([a for a in ARMS if a != "NNNF"]):
        rec[f"beta_{arm}_nolen"] = b0[i + 1]
        rec[f"beta_{arm}_withlen"] = b1[i + 1]
        rec[f"t_{arm}_withlen"] = t1[i + 1]
    rec["beta_length"] = b1[-1]
    rec["t_length"] = t1[-1]
    # (c) same, restricted to common length support
    dc = common
    dumc = np.column_stack([(dc.arm == a).to_numpy(float) for a in ARMS if a != "NNNF"])
    Xc = np.column_stack([np.ones(len(dc)), dumc, dc.n_tokens.to_numpy()])
    bc, _, tc = ols(Xc, dc[col].to_numpy())
    for i, arm in enumerate([a for a in ARMS if a != "NNNF"]):
        rec[f"beta_{arm}_common"] = bc[i + 1]
        rec[f"t_{arm}_common"] = tc[i + 1]
    reg_rows.append(rec)
reg = pd.DataFrame(reg_rows)
reg.to_csv(os.path.join(TABLES, "readout_length_regression.csv"), index=False)

print(f"\n{'layer':>5}{'SSSF b(no len)':>16}{'SSSF b(+len)':>14}{'t':>8}"
      f"{'HHHF b(no len)':>16}{'HHHF b(+len)':>14}{'t':>8}{'b_length':>10}{'t_len':>8}")
for layer in range(n_layers):
    r = reg.iloc[layer]
    print(f"{layer:>5}{r.beta_SSSF_nolen:>16.3f}{r.beta_SSSF_withlen:>14.3f}"
          f"{r.t_SSSF_withlen:>8.1f}{r.beta_HHHF_nolen:>16.3f}{r.beta_HHHF_withlen:>14.3f}"
          f"{r.t_HHHF_withlen:>8.1f}{r.beta_length:>10.4f}{r.t_length:>8.1f}")

print(f"\nwithin-arm slope of readout on length (per 100 tokens), and common-support arm effect:")
print(f"{'layer':>5}" + "".join(f"{'slope ' + a:>14}" for a in ARMS)
      + f"{'SSSF b common':>15}{'t':>7}")
for layer in range(n_layers):
    r = reg.iloc[layer]
    print(f"{layer:>5}" + "".join(f"{r[f'slope_{a}'] * 100:>14.3f}" for a in ARMS)
          + f"{r.beta_SSSF_common:>15.3f}{r.t_SSSF_common:>7.1f}")

# ---------------------------------------------------------------------------
# 4. Scripted-assistant control arms
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("4. SCRIPTED-ASSISTANT CONTROL: SSSF vs NNNF with byte-identical assistant text")
print("=" * 110)
print("No generation. The three assistant turns for a (topic, phrasing) cell are taken verbatim")
print("from the already-generated NNNF|topic|phrasing|00 transcript and reused in BOTH arms, so")
print("assistant text is byte-identical across arms within a cell and only the user turns differ.")
print("Replicates are dropped: with fixed user turns and fixed assistant text every replicate")
print("would be byte-identical, so n=1 per (arm, topic, phrasing) = 16 transcripts.")

with open(TRANSCRIPT_PATH) as f:
    dialogues = json.load(f)["dialogues"]
by_id = {d["dialogue_id"]: d for d in dialogues}
pool, _ = load_turn_pool()
pool_by_id = {t.turn_id: t for t in pool}

model, tokenizer = load_model_and_tokenizer(MODEL_ID)
ROLE_SEQUENCE = ["user", "assistant", "user", "assistant", "user", "assistant", "user"]
control_rows = []
scripted_check = {}

for topic in sorted({d["topic"] for d in dialogues}):
    for phrasing in sorted({d["phrasing"] for d in dialogues}):
        scripted = by_id[f"NNNF|{topic}|{phrasing}|00"]["assistant_turns"]
        scripted_check[(topic, phrasing)] = scripted
        for arm, valences in (("SSSF", ("S", "S", "S")), ("NNNF", ("N", "N", "N"))):
            user_turns = [pool_by_id[f"U[{topic},{phrasing},{v},{i}]"].text
                          for i, v in enumerate(valences, start=1)]
            user_turns.append(pool_by_id[f"F[{topic},{phrasing}]"].text)
            messages = [{"role": r, "content": c} for r, c in zip(
                ROLE_SEQUENCE,
                [user_turns[0], scripted[0], user_turns[1], scripted[1],
                 user_turns[2], scripted[2], user_turns[3]])]
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=False)
            ids = tokenizer(text, add_special_tokens=False).input_ids
            boundaries = find_turn_boundaries(tokenizer, ids, messages)
            resid = extract_residual_stream(model, torch.tensor([ids], device="cuda"))
            rec = dict(arm=arm, topic=topic, phrasing=phrasing, n_tokens=len(ids))
            for position in ("eot_u_4", "u_4"):
                vec = resid[:, boundaries[position], :].float().cpu().numpy()
                train_pos = PROBE_FOR[position]
                for layer in range(n_layers):
                    rec[f"{position}_L{layer}"] = float(
                        probes[train_pos][layer].decision_function(vec[layer:layer + 1])[0])
            control_rows.append(rec)

# assistant text identical across the two arms within each cell
for (topic, phrasing), scripted in scripted_check.items():
    arms_here = [r for r in control_rows if r["topic"] == topic and r["phrasing"] == phrasing]
    assert len(arms_here) == 2
print(f"\nbuilt and extracted {len(control_rows)} scripted-assistant transcripts")
control = pd.DataFrame(control_rows)
control.to_csv(os.path.join(TABLES, "scripted_assistant_control.csv"), index=False)

print(f"\ntranscript length: "
      + ", ".join(f"{a}={control[control.arm == a].n_tokens.mean():.1f}" for a in ("SSSF", "NNNF"))
      + "  (identical assistant text, so length differs only by user-turn length)")

print(f"\n{'layer':>5}{'SSSF mean':>12}{'sd':>8}{'NNNF mean':>12}{'sd':>8}"
      f"{'diff':>9}{'dz':>8}   |{'  u_4 SSSF':>12}{'u_4 NNNF':>11}{'u_4 diff':>10}{'dz':>7}")
ctrl_rows = []
for layer in range(n_layers):
    s = control[control.arm == "SSSF"].sort_values(["topic", "phrasing"])
    n = control[control.arm == "NNNF"].sort_values(["topic", "phrasing"])
    line = f"{layer:>5}"
    rec = dict(layer=layer)
    for position, tail in (("eot_u_4", False), ("u_4", True)):
        col = f"{position}_L{layer}"
        sv, nv = s[col].to_numpy(), n[col].to_numpy()
        dz = cohens_dz(sv - nv)
        rec.update({f"{position}_SSSF_mean": sv.mean(), f"{position}_NNNF_mean": nv.mean(),
                    f"{position}_diff": sv.mean() - nv.mean(), f"{position}_dz": dz})
        if not tail:
            line += (f"{sv.mean():>12.3f}{sv.std(ddof=1):>8.3f}{nv.mean():>12.3f}"
                     f"{nv.std(ddof=1):>8.3f}{sv.mean() - nv.mean():>9.3f}{dz:>8.2f}   |")
        else:
            line += (f"{sv.mean():>12.3f}{nv.mean():>11.3f}"
                     f"{sv.mean() - nv.mean():>10.3f}{dz:>7.2f}")
    ctrl_rows.append(rec)
    print(line)
pd.DataFrame(ctrl_rows).to_csv(os.path.join(TABLES, "scripted_control_by_layer.csv"), index=False)

# Comparison against the generated-assistant arms, same contrast.
print("\ncomparison -- SSSF minus NNNF at eot_u_4, generated vs scripted assistant text:")
gen = pd.read_csv(os.path.join(TABLES, "readout_eot_u_4_by_layer.csv"))
ctrl = pd.DataFrame(ctrl_rows)
print(f"{'layer':>5}{'generated diff':>16}{'scripted diff':>15}{'retained %':>12}"
      f"{'generated dz':>14}{'scripted dz':>13}")
for layer in range(n_layers):
    g = gen.iloc[layer]
    c = ctrl.iloc[layer]
    gdiff = g.SSSF_mean - g.NNNF_mean
    cdiff = c.eot_u_4_diff
    retained = (cdiff / gdiff * 100) if abs(gdiff) > 1e-9 else float("nan")
    print(f"{layer:>5}{gdiff:>16.3f}{cdiff:>15.3f}{retained:>12.1f}"
          f"{g.dz_SSSF_NNNF:>14.2f}{c.eot_u_4_dz:>13.2f}")

print(f"\ntables written to {TABLES}")
print("\nANALYSIS COMPLETE -- no layer selected, nothing regenerated.")
