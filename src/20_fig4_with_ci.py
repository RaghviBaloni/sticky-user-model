"""Regenerate FIG 4 with bootstrap CIs on the survival ratio.

Three draws are shown per perturbation. Error bars belong to draw 3 only -- it is the only
draw whose per-prefix values were persisted (src/19), and the only one seeded reproducibly.
The CI covers sampling uncertainty over the 20 paired prefixes within that draw; the spread
*between* the three bars is a separate source of variation (the salted-hash seeding bug), and
for (a) it is larger than the CI itself.

Plotting only. Same palette as src/18.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(REPO, "results", "tables")
FIGS = os.path.join(REPO, "results", "figures")
DPI = 200
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.autolayout": False})
SAGE, MUSTARD, DEEP_ORANGE, CRIMSON = "#7D9471", "#D6A419", "#C1571A", "#A31621"

names = ["a_token_shuffle", "b_word_substitution", "c_sentence_reversal", "d_position_control"]
pretty = ["(a) token shuffle\nwithin $a_k$", "(b) 20% content-word\nsubstitution",
          "(c) sentence-order\nreversal", "(d) position control\n(equal-length span)"]
lad1 = pd.read_csv(os.path.join(TABLES, "stage3b_perturbation_ladder.csv")).set_index("perturbation")
lad2 = pd.read_csv(os.path.join(TABLES, "stage3b_ladder_corrected.csv")).set_index("perturbation")
boot = pd.read_csv(os.path.join(TABLES, "stage3b_ladder_bootstrap_ci.csv")).set_index("perturbation")

r1 = [lad1.loc[n, "mean_diff_survival_pct"] for n in names]
r2 = [lad2.loc[n, "mean_diff_survival_pct"] for n in names]
r3 = [boot.loc[n, "survival_pct"] for n in names]
lo = [boot.loc[n, "ci_lo"] for n in names]
hi = [boot.loc[n, "ci_hi"] for n in names]
err = np.array([[p - l for p, l in zip(r3, lo)], [h - p for p, h in zip(r3, hi)]])

x = np.arange(len(names))
w = 0.26
fig, ax = plt.subplots(figsize=(8.0, 5.0))
ax.bar(x - w, r1, width=w, color=MUSTARD, label="draw 1 (src/14; (d) length-short)")
ax.bar(x, r2, width=w, color=SAGE, label="draw 2 (src/15; (d) length-matched)")
ax.bar(x + w, r3, width=w, color=DEEP_ORANGE,
       yerr=err, capsize=4, error_kw=dict(ecolor=CRIMSON, lw=1.5),
       label="draw 3 (src/19; stable seed) with bootstrap 95% CI")
for xi, (a_, b_, c_) in enumerate(zip(r1, r2, r3)):
    ax.text(xi - w, a_ + 4, f"{a_:.0f}", ha="center", fontsize=7.5)
    ax.text(xi, b_ + 4, f"{b_:.0f}", ha="center", fontsize=7.5)
    ax.text(xi + w, hi[xi] + 5, f"{c_:.0f}", ha="center", fontsize=7.5, fontweight="bold")
ax.axhline(100, ls="--", lw=1.2, color=CRIMSON)
ax.axhline(0, ls=":", lw=1.0, color="black")
ax.set_xticks(x)
ax.set_xticklabels(pretty, fontsize=8.5)
ax.set_ylabel("% of clean effect retained  (mean-difference survival)")
ax.set_ylim(-75, 215)
ax.set_title("Perturbation ladder: only shuffling tokens within $a_k$ is distinguishable\n"
             "from full retention", fontsize=10)
ax.text(0.0, -0.255,
        "Probe P2, layers 11–29, strength 0.5×, n=20 paired prefixes. Error bars: bootstrap "
        "95% CI, 10,000 resamples over prefixes,\ndraw 3 only — it is the only draw with "
        "per-prefix values persisted and a reproducible seed. Only (a)'s CI excludes 100%, and\n"
        "only (a)'s CI fails to overlap (c)'s. The spread *between* draws is a separate source "
        "of variation (per-process salted-hash\nseeding in draws 1–2); for (a) it is larger "
        "than the CI. (c) uses no RNG and reproduced across draws exactly.",
        transform=ax.transAxes, fontsize=7.3, va="top")
ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
fig.subplots_adjust(bottom=0.34, top=0.89)
fig.savefig(os.path.join(FIGS, "perturbation_ladder.png"), dpi=DPI)
plt.close(fig)
print("FIG 4 rewritten with bootstrap CIs")
for n, p, l, h in zip(names, r3, lo, hi):
    print(f"  {n:>24}: {p:6.1f}%  [{l:6.1f}, {h:6.1f}]  contains 100%: "
          f"{'YES' if l <= 100 <= h else 'no'}")
