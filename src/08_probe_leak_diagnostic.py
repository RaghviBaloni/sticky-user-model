"""Diagnostic for the flagged probe accuracy (CLAUDE.md: >97% is usually a leak).

The layer sweep returned 1.000 held-out 3-way accuracy. Three questions this answers:

  1. Is the single-turn data lexically separable on its own? -> TF-IDF logistic regression on
     the same 72 turns with the identical leave-one-(topic,phrasing)-out folds. If a
     bag-of-words model also scores ~1.0, the probe's number says the *data* is trivially
     separable, not that the model holds a rich user representation.
  2. Does an off-the-shelf sentiment classifier already solve it, with no training at all?
  3. How early does the signal appear? Accuracy at layer 0 bounds how much computation the
     model needs; a lexical cue is readable almost immediately.

This is a diagnostic for a flagged anomaly. It changes no pre-registered metric, and it is
not the Stage 3b surface-sentiment baseline (metrics.md Sec 3, still unlocked).
"""

import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common import CACHE_ROOT, REPO_ROOT, VALENCES, load_sentiment_scorer, load_turn_pool

SWEEP_PATH = os.path.join(REPO_ROOT, "results", "tables", "stage2_probe_sweep.json")
OUT_PATH = os.path.join(REPO_ROOT, "results", "tables", "stage2_leak_diagnostic.json")

pool, _ = load_turn_pool()
user_turns = [t for t in pool if t.kind == "user"]
texts = [t.text for t in user_turns]
labels = np.array([t.valence for t in user_turns])
groups = np.array([f"{t.topic}|{t.phrasing}" for t in user_turns])
unique_groups = sorted(set(groups))
print(f"{len(texts)} single-turn pool turns, {len(unique_groups)} CV groups "
      "(identical folds to the probe sweep)")


def grouped_cv_text(build_model, X_text, y, g):
    correct = total = 0
    for group in sorted(set(g)):
        test = g == group
        train = ~test
        model = build_model()
        model.fit([X_text[i] for i in np.where(train)[0]], y[train])
        pred = model.predict([X_text[i] for i in np.where(test)[0]])
        correct += int((pred == y[test]).sum())
        total += int(test.sum())
    return correct / total


def tfidf_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        LogisticRegression(penalty="l2", C=1.0, max_iter=5000))


print("\n" + "=" * 78)
print("1. TF-IDF BAG-OF-WORDS BASELINE (same data, same folds)")
print("=" * 78)
binary = np.isin(labels, ["S", "H"])
tfidf_3 = grouped_cv_text(tfidf_model, texts, labels, groups)
tfidf_2 = grouped_cv_text(tfidf_model, [t for t, m in zip(texts, binary) if m],
                          labels[binary], groups[binary])
print(f"TF-IDF (1-2 grams) 3-way   : {tfidf_3:.3f}   (chance 0.333)")
print(f"TF-IDF (1-2 grams) S vs H  : {tfidf_2:.3f}   (chance 0.500)")

print("\n" + "=" * 78)
print("2. OFF-THE-SHELF SENTIMENT CLASSIFIER, ZERO TRAINING")
print("=" * 78)
score = load_sentiment_scorer()
scored = score(texts)
label_map = {"negative": "S", "neutral": "N", "positive": "H"}
pred = np.array([label_map[max(("negative", "neutral", "positive"), key=lambda k: s[k])]
                 for s in scored])
acc3 = float((pred == labels).mean())
acc2 = float((pred[binary] == labels[binary]).mean())
print(f"argmax(neg/neu/pos) -> S/N/H, 3-way : {acc3:.3f}")
print(f"                       S vs H only  : {acc2:.3f}")
print("\nconfusion (rows = intended, cols = classifier):")
print(f"{'':>6}" + "".join(f"{v:>6}" for v in VALENCES))
for true_v in VALENCES:
    row = [int(((labels == true_v) & (pred == p)).sum()) for p in VALENCES]
    print(f"{true_v:>6}" + "".join(f"{n:>6}" for n in row))

print("\n" + "=" * 78)
print("3. HOW EARLY DOES THE SIGNAL APPEAR?")
print("=" * 78)
with open(SWEEP_PATH) as f:
    sweep = json.load(f)
for pos in sweep["positions"]:
    arr = sweep["sweep"][pos]["three_way"]
    first_above = next((i for i, a in enumerate(arr) if a >= 0.95), None)
    print(f"  {pos:<9} L0={arr[0]:.3f}  L1={arr[1]:.3f}  L2={arr[2]:.3f}  "
          f"first layer >=0.95: {first_above}  max={max(arr):.3f}")

probe_best = sweep["best"]["three_way"]["accuracy"]
print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
print(f"probe (best layer/position) 3-way : {probe_best:.3f}")
print(f"TF-IDF bag-of-words 3-way         : {tfidf_3:.3f}")
print(f"sentiment classifier, untrained   : {acc3:.3f}")
gap = probe_best - tfidf_3
print(f"probe - TF-IDF gap                : {gap:+.3f}")
if tfidf_3 >= 0.95:
    print("\nA bag-of-words model solves this task about as well as the probe does. The 1.000\n"
          "is therefore a property of the SINGLE-TURN POOL DATA (generated to be lexically\n"
          "distinct per valence), not evidence that the model holds a rich user representation.\n"
          "The Stage 2 gate is passed but UNINFORMATIVE about representation quality, exactly\n"
          "the weakness ROADMAP Sec 4 anticipated. Consequence for Stage 3: the diff-of-means\n"
          "direction learned here may be largely a sentiment-word direction, so the loop test's\n"
          "third-party-sentiment control and the surface baselines carry the whole argument.")
else:
    print("\nThe probe beats bag-of-words by a clear margin, so the accuracy is not purely lexical.")

with open(OUT_PATH, "w") as f:
    json.dump(dict(tfidf_3way=tfidf_3, tfidf_SH=tfidf_2, sentiment_3way=acc3,
                   sentiment_SH=acc2, probe_best_3way=probe_best, gap=gap,
                   cv="leave-one-(topic,phrasing)-out", n=len(texts)), f, indent=2)
print(f"\nwritten -> {OUT_PATH}")
