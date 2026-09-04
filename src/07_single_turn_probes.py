"""Stage 2, step 9 -- per-layer, per-position probes on single-turn pool data only.

Training data is the 72 pool user turns, each rendered as a one-turn conversation. No
transcript, no assistant text, no multi-turn context -- so the probe cannot learn anything
about dialogue position or the assistant's contribution. Positions: u_1 and eot_u_1.

Held-out protocol: leave-one-(topic, phrasing)-out, 8 folds. Grouping matters here -- the
three slot turns inside a cell come from one generation call and share style, so a random
split would put near-siblings in train and test and inflate accuracy. Each fold therefore
tests generalisation to an unseen topic+phrasing combination: 63 train / 9 test, and the
test fold is always balanced 3/3/3 across valence.

Probe: L2 logistic regression, C=1.0, standardised with the scaler fit on the training fold
only. metrics.md Sec 1's C and layer are still blank -- selecting L* is what this script is
for; it does not write metrics.md.
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import (  # noqa: E402
    CACHE_ROOT,
    MODEL_ID,
    REPO_ROOT,
    VALENCES,
    extract_residual_stream,
    find_turn_boundaries,
    load_model_and_tokenizer,
    load_turn_pool,
)

C_VALUE = 1.0
GATE = 0.75
POSITIONS = ("u_1", "eot_u_1")
CACHE_DIR = os.path.join(CACHE_ROOT, "single_turn_pool")
RESULTS_PATH = os.path.join(REPO_ROOT, "results", "tables", "stage2_probe_sweep.json")

pool, _ = load_turn_pool()
user_turns = [t for t in pool if t.kind == "user"]
assert len(user_turns) == 72, len(user_turns)
print(f"single-turn pool data: {len(user_turns)} user turns "
      f"({', '.join(f'{v}={sum(1 for t in user_turns if t.valence == v)}' for v in VALENCES)})")
print("F turns are excluded from probe training (they are the readout position, not training data)")

model, tokenizer = load_model_and_tokenizer(MODEL_ID)
n_layers = model.config.text_config.num_hidden_layers
hidden_size = model.config.text_config.hidden_size

# ---------------------------------------------------------------------------
# Extract u_1 / eot_u_1 for each single-turn render
# ---------------------------------------------------------------------------
print(f"\nextracting {len(POSITIONS)} positions x {n_layers} layers per turn "
      f"via the leak-guarded measurement path")
features = {pos: np.zeros((len(user_turns), n_layers, hidden_size), dtype=np.float32)
            for pos in POSITIONS}
labels, groups, turn_ids = [], [], []
decode_log = []

for i, turn in enumerate(user_turns):
    messages = [turn.as_message()]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    boundaries = find_turn_boundaries(tokenizer, ids, messages)
    if i < 3:
        decode_log.append((turn.turn_id, [(name, idx, tokenizer.decode([ids[idx]]))
                                          for name, idx in boundaries.items()]))
    resid = extract_residual_stream(model, torch.tensor([ids], device="cuda"))
    for pos in POSITIONS:
        features[pos][i] = resid[:, boundaries[pos], :].float().cpu().numpy()
    labels.append(turn.valence)
    groups.append(f"{turn.topic}|{turn.phrasing}")
    turn_ids.append(turn.turn_id)

labels = np.array(labels)
groups = np.array(groups)
unique_groups = sorted(set(groups))
print(f"extracted. groups (CV folds) = {len(unique_groups)}: {unique_groups}")

os.makedirs(CACHE_DIR, exist_ok=True)
torch.save({pos: torch.tensor(features[pos], dtype=torch.float16) for pos in POSITIONS},
           os.path.join(CACHE_DIR, "single_turn_acts.pt"))
with open(os.path.join(CACHE_DIR, "single_turn_meta.json"), "w") as f:
    json.dump(dict(turn_ids=turn_ids, labels=labels.tolist(), groups=groups.tolist(),
                   positions=list(POSITIONS), model_id=MODEL_ID), f, indent=2)
print(f"cached single-turn activations -> {CACHE_DIR}")

print("\nB.3 decode check on the first 3 single-turn renders:")
for turn_id, entries in decode_log:
    print(f"  {turn_id}")
    for name, idx, tok in entries:
        print(f"    {name:12s} idx={idx:4d} tok={tok!r}")


# ---------------------------------------------------------------------------
# Layer sweep
# ---------------------------------------------------------------------------
def grouped_cv_accuracy(X, y, g):
    """Leave-one-group-out accuracy, scaler fit on train folds only."""
    correct = total = 0
    per_fold = []
    for group in sorted(set(g)):
        test = g == group
        train = ~test
        if len(set(y[train])) < 2 or not test.any():
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(penalty="l2", C=C_VALUE, max_iter=5000))
        clf.fit(X[train], y[train])
        pred = clf.predict(X[test])
        hits = int((pred == y[test]).sum())
        correct += hits
        total += int(test.sum())
        per_fold.append(hits / int(test.sum()))
    return correct / total, per_fold


binary_mask = np.isin(labels, ["S", "H"])
sweep = {pos: {"three_way": [], "binary_SH": []} for pos in POSITIONS}
print(f"\nsweeping {n_layers} layers x {len(POSITIONS)} positions "
      f"({len(unique_groups)}-fold grouped CV, L2 logistic C={C_VALUE})")
for pos in POSITIONS:
    for layer in range(n_layers):
        X = features[pos][:, layer, :]
        acc3, _ = grouped_cv_accuracy(X, labels, groups)
        sweep[pos]["three_way"].append(acc3)
        acc2, _ = grouped_cv_accuracy(X[binary_mask], labels[binary_mask], groups[binary_mask])
        sweep[pos]["binary_SH"].append(acc2)
    print(f"  {pos}: done")

print("\n" + "=" * 78)
print("ACCURACY vs LAYER (leave-one-topic+phrasing-out CV)")
print("=" * 78)
print(f"{'layer':>6}{'u_1 3-way':>12}{'u_1 S vs H':>13}{'eot_u_1 3-way':>16}{'eot_u_1 S vs H':>17}")
for layer in range(n_layers):
    print(f"{layer:>6}{sweep['u_1']['three_way'][layer]:>12.3f}"
          f"{sweep['u_1']['binary_SH'][layer]:>13.3f}"
          f"{sweep['eot_u_1']['three_way'][layer]:>16.3f}"
          f"{sweep['eot_u_1']['binary_SH'][layer]:>17.3f}")

best = {}
for task in ("three_way", "binary_SH"):
    candidates = [(sweep[pos][task][layer], pos, layer)
                  for pos in POSITIONS for layer in range(n_layers)]
    acc, pos, layer = max(candidates)
    best[task] = dict(accuracy=acc, position=pos, layer=layer)

print("\n" + "=" * 78)
print("BEST CONFIGURATIONS")
print("=" * 78)
chance3 = 1 / 3
for task, label, chance in (("three_way", "3-way (S/N/H)", chance3),
                            ("binary_SH", "binary (S vs H)", 0.5)):
    b = best[task]
    print(f"{label:<18} best acc={b['accuracy']:.3f}  layer={b['layer']}  "
          f"position={b['position']}  (chance {chance:.3f})")

for pos in POSITIONS:
    arr = sweep[pos]["three_way"]
    print(f"  {pos:<9} 3-way: max={max(arr):.3f} @L{int(np.argmax(arr))}  "
          f"mean over layers={np.mean(arr):.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump(dict(model_id=MODEL_ID, C=C_VALUE, n_layers=n_layers,
                   cv="leave-one-(topic,phrasing)-out", n_folds=len(unique_groups),
                   n_samples=len(labels), positions=list(POSITIONS),
                   sweep=sweep, best=best), f, indent=2)
print(f"\nsweep written -> {RESULTS_PATH}")

gate_acc = best["three_way"]["accuracy"]
print("\n" + "=" * 78)
print(f"STAGE 2 GATE: held-out 3-way accuracy {gate_acc:.3f} vs threshold {GATE:.2f}")
if gate_acc > GATE:
    print("GATE PASSED")
else:
    print("GATE FAILED -- coarsen to binary per ROADMAP Sec 6 / metrics.md Sec 6.")
    print(f"  binary (S vs H) best = {best['binary_SH']['accuracy']:.3f} "
          f"@ layer {best['binary_SH']['layer']} / {best['binary_SH']['position']}")
if gate_acc > 0.97:
    print("FLAG: accuracy above 0.97 -- CLAUDE.md says suspect a leak, not a finding.")
print("=" * 78)
print("\nSTEP 9 COMPLETE")
