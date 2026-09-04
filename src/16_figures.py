"""Figures for the writeup. Plotting only -- every number is read from results/tables.

Nothing is recomputed: where a summary statistic is not stored as a summary (e.g. an L11-29
mean), it is aggregated arithmetically from the per-layer or per-prefix rows already saved.
No model is loaded and no measurement is repeated.

TWO PROVENANCE GAPS, marked in the figures rather than papered over:
  * the length-matched condition-E probe readouts (P1 -0.41, P2 -0.99) were printed by
    src/14 but never written to results/tables. They are taken from that run's log and
    annotated as such in FIG 3.
  * valence / hedging / distress-lexicon were never computed for the length-matched E, so
    those cells are left empty in text_mediation_summary.csv.
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
os.makedirs(FIGS, exist_ok=True)
DPI = 200
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.autolayout": False})

C_P2, C_P1, C_CTRL = "#1f4e79", "#c0392b", "#7f8c8d"
C_RUN1, C_RUN2 = "#95a5a6", "#1f4e79"


def dz(d):
    d = np.asarray(d, float)
    return float(d.mean() / d.std(ddof=1))


# ---------------------------------------------------------------------------
# FIG 1 -- token-shuffle survival, Stage 2 readout vs Stage 3b loop test
# ---------------------------------------------------------------------------
s2 = pd.read_csv(os.path.join(TABLES, "lexical_check_shuffle.csv"))
s2_surv = s2[s2.layer.between(11, 29)].survives_pct.mean()
lad1 = pd.read_csv(os.path.join(TABLES, "stage3b_perturbation_ladder.csv")).set_index("perturbation")
lad2 = pd.read_csv(os.path.join(TABLES, "stage3b_ladder_corrected.csv")).set_index("perturbation")
s3_draw1 = lad1.loc["a_token_shuffle", "mean_diff_survival_pct"]
s3_draw2 = lad2.loc["a_token_shuffle", "mean_diff_survival_pct"]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
labels = ["Stage 2 readout\n(SSSF vs NNNF at eot_u_4)",
          "Stage 3b loop test\n(B vs C at eot_u_5)"]
ax.bar([0], [s2_surv], width=0.42, color=C_P2, label="Stage 2 (single draw)")
ax.bar([0.92], [s3_draw1], width=0.34, color=C_RUN1, label="Stage 3b, draw 1 (src/14)")
ax.bar([1.30], [s3_draw2], width=0.34, color=C_RUN2, label="Stage 3b, draw 2 (src/15)")
for x, v in ((0, s2_surv), (0.92, s3_draw1), (1.30, s3_draw2)):
    ax.text(x, v + 3, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.axhline(100, ls="--", lw=1.2, color="black")
ax.text(0.63, 103, "100% = effect fully retained", fontsize=8, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
ax.set_xticks([0, 1.11])
ax.set_xticklabels(labels)
ax.set_ylabel("% of clean effect retained after token shuffle")
ax.set_ylim(0, 130)
ax.set_title("Token-shuffle survival: the Stage 2 readout is order-invariant,\n"
             "the Stage 3b loop-test effect is not", fontsize=10)
ax.text(0.0, -0.20, "Mean-difference survival averaged over layers 11–29. Stage 2: n=160 "
        "paired dialogues per arm.\nStage 3b: n=20 paired prefixes; two draws shown because "
        "the shuffle RNG was seeded per process\n(salted hash), so the permutation differed "
        "between runs.", transform=ax.transAxes, fontsize=7.5, va="top")
ax.legend(loc="upper center", bbox_to_anchor=(0.52, 0.72), fontsize=8, framealpha=0.95)
fig.subplots_adjust(bottom=0.30, top=0.87)
fig.savefig(os.path.join(FIGS, "shuffle_survival.png"), dpi=DPI)
plt.close(fig)
print(f"FIG 1: stage2={s2_surv:.1f}%, stage3b draws {s3_draw1:.1f}% / {s3_draw2:.1f}%")

# ---------------------------------------------------------------------------
# FIG 2 -- probe dependence across seeds
# ---------------------------------------------------------------------------
seeds_df = pd.read_csv(os.path.join(TABLES, "stage3b_seeds.csv"))
readout = pd.read_csv(os.path.join(TABLES, "stage3b_readout_by_layer.csv"))
mid = readout.layer.between(11, 29)


def seed30(probe, col):
    sub = readout[(readout.probe == probe) & (readout.strength == 0.5) & mid]
    return sub[col].mean()


seeds = [30, 31, 32, 33]
series = {
    "P2  d_z(B−C)": ([seed30("P2", "dz_B_C")] +
                     [seeds_df[(seeds_df.seed == s) & (seeds_df.probe == "P2")].dz_B_C.iloc[0]
                      for s in seeds[1:]], C_P2, "o", "-"),
    "P1  d_z(B−C)": ([seed30("P1", "dz_B_C")] +
                     [seeds_df[(seeds_df.seed == s) & (seeds_df.probe == "P1")].dz_B_C.iloc[0]
                      for s in seeds[1:]], C_P1, "s", "-"),
    "P2  d_z(D−A)  random-direction control": (
        [seed30("P2", "dz_D_A")] +
        [seeds_df[(seeds_df.seed == s) & (seeds_df.probe == "P2")].dz_D_A.iloc[0]
         for s in seeds[1:]], C_CTRL, "^", "--"),
}
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for name, (vals, colour, marker, ls) in series.items():
    ax.plot(seeds, vals, marker=marker, ls=ls, color=colour, label=name, lw=1.8, ms=6)
    yoff = {"o": 0, "s": -14, "^": 14}[marker]
    ax.annotate(f"mean {np.mean(vals):+.2f}\nsd {np.std(vals, ddof=1):.2f}",
                xy=(seeds[-1], vals[-1]), xytext=(8, yoff), textcoords="offset points",
                fontsize=7.5, color=colour, va="center")
ax.axhline(0, ls=":", lw=1.2, color="black")
ax.set_xticks(seeds)
ax.set_xlabel("generation seed")
ax.set_ylabel("paired effect size  $d_z$")
ax.set_xlim(29.6, 34.4)
ax.set_title("The loop-test effect replicates across seeds — and so does the\n"
             "disagreement between the two probes", fontsize=10)
ax.text(0.0, -0.20, "Layer 20, strength 0.5×, n=20 paired prefixes per seed; each point is the "
        "mean over layers 11–29.\nSeed 30 is the original Stage 3b run. Positive = distressed.",
        transform=ax.transAxes, fontsize=7.5, va="top")
ax.legend(loc="center left", fontsize=8, framealpha=0.95)
fig.subplots_adjust(bottom=0.26, top=0.87, right=0.86)
fig.savefig(os.path.join(FIGS, "probe_dependence.png"), dpi=DPI)
plt.close(fig)
print("FIG 2: written")

# ---------------------------------------------------------------------------
# FIG 3 -- effect size vs validity
# ---------------------------------------------------------------------------
base = pd.read_csv(os.path.join(TABLES, "stage3b_surface_baselines.csv"))
bc = {
    "TF-IDF": base[(base.method == "TF-IDF on full transcript") & (base.strength == 0.5)].dz_B_C.iloc[0],
    "probe P2": base[(base.method == "probe P2 L11-29") & (base.strength == 0.5)].dz_B_C.iloc[0],
    "sentiment": base[(base.method == "sentiment on full transcript") & (base.strength == 0.5)].dz_B_C.iloc[0],
}
e = pd.read_csv(os.path.join(TABLES, "stage3b_E_lengthfixed.csv"))
ea = {
    "TF-IDF": dz(e.e_tfidf - e.a_tfidf),
    "probe P2": -0.99,   # from the src/14 run log; not persisted to results/tables
    "sentiment": dz(e.e_sent - e.a_sent),
}
E_FROM_LOG = True

fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.3), sharey=True)
order = ["TF-IDF", "probe P2", "sentiment"]
colours = ["#7f8c8d", C_P2, "#b8912f"]
for ax, data, title in ((axes[0], bc, "B vs C\n(steering contrast)"),
                        (axes[1], ea, "E vs A\n(third-party control, length-matched)")):
    vals = [data[k] for k in order]
    bars = ax.bar(order, vals, color=colours, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.06 if v >= 0 else -0.06),
                f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.axhline(0, ls="--", lw=1.3, color="black")
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelrotation=12)
    ax.set_ylim(-1.45, 1.6)
axes[0].set_ylabel("paired effect size  $d_z$   (positive = distressed)")
axes[1].annotate("probe and surface\nbaselines disagree\nin sign", xy=(0.66, -0.70),
                 xytext=(-0.46, -1.36), fontsize=8, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", lw=1.1, color="black",
                                 connectionstyle="arc3,rad=0.15"))
PROV = ("  probe P2 bar in the right panel: value taken from the src/14 run log — "
        "it was never persisted to results/tables." if E_FROM_LOG else "")
fig.suptitle("Bigger effect size is not the same as better validity", fontsize=11, y=0.98)
fig.text(0.015, 0.015,
         "Layer 20, strength 0.5×, n=20 paired prefixes; probe values are means over layers 11–29.\n"
         "Left: TF-IDF separates B from C better than the probe does. Right: distress about a third party\n"
         "moves the surface baselines up and the probe down. No confidence intervals are stored."
         + "\n" + PROV, fontsize=7.5, va="bottom")
fig.subplots_adjust(bottom=0.33, top=0.82)
fig.savefig(os.path.join(FIGS, "effect_vs_validity.png"), dpi=DPI)
plt.close(fig)
print(f"FIG 3: B-C {bc}, E-A {ea}")

# ---------------------------------------------------------------------------
# FIG 4 -- perturbation ladder, both runs
# ---------------------------------------------------------------------------
names = ["a_token_shuffle", "b_word_substitution", "c_sentence_reversal", "d_position_control"]
pretty = ["(a) token shuffle\nwithin $a_k$", "(b) 20% content-word\nsubstitution",
          "(c) sentence-order\nreversal", "(d) position control\n(equal-length span)"]
r1 = [lad1.loc[n, "mean_diff_survival_pct"] for n in names]
r2 = [lad2.loc[n, "mean_diff_survival_pct"] for n in names]
d1 = [lad1.loc[n, "dz_survival_pct"] for n in names]
d2 = [lad2.loc[n, "dz_survival_pct"] for n in names]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.bar(x - 0.19, r1, width=0.36, color=C_RUN1, label="draw 1 (src/14; (d) length-short)")
ax.bar(x + 0.19, r2, width=0.36, color=C_RUN2, label="draw 2 (src/15; (d) length-matched)")
ax.plot(x - 0.19, d1, "o", color="#2c3e50", ms=5, label="$d_z$ survival, draw 1")
ax.plot(x + 0.19, d2, "D", color="#e67e22", ms=5, label="$d_z$ survival, draw 2")
for xi, (a, b) in enumerate(zip(r1, r2)):
    ax.text(xi - 0.19, a + 3, f"{a:.0f}%", ha="center", fontsize=8)
    ax.text(xi + 0.19, b + 3, f"{b:.0f}%", ha="center", fontsize=8)
ax.axhline(100, ls="--", lw=1.2, color="black")
ax.set_xticks(x)
ax.set_xticklabels(pretty, fontsize=8.5)
ax.set_ylabel("% of clean effect retained  (mean-difference survival)")
ax.set_ylim(0, 175)
ax.set_title("Perturbation ladder: only destroying local word order in $a_k$ collapses\n"
             "the loop-test effect", fontsize=10)
ax.text(0.0, -0.26, "Probe P2, layers 11–29, strength 0.5×, n=20 paired prefixes. Bars are "
        "mean-difference survival; markers are $d_z$ survival.\nBoth draws are shown because "
        "(a), (b) and (d) were seeded from a per-process salted hash, so their permutations\n"
        "differ between runs; (c) uses no RNG and reproduced exactly. (d) also changed because "
        "draw 2 length-matched it.", transform=ax.transAxes, fontsize=7.5, va="top")
ax.legend(loc="upper left", fontsize=8, framealpha=0.95, ncol=2)
fig.subplots_adjust(bottom=0.32, top=0.88)
fig.savefig(os.path.join(FIGS, "perturbation_ladder.png"), dpi=DPI)
plt.close(fig)
print("FIG 4: written")

# ---------------------------------------------------------------------------
# Table -- text mediation summary
# ---------------------------------------------------------------------------
tm = pd.read_csv(os.path.join(TABLES, "stage3b_text_mediation.csv")).set_index("condition")
rows = []
for label, key in (("A", "A"), ("B @0.25x", "B @0.25x"), ("B @0.5x", "B @0.5x"),
                   ("C @0.5x", "C @0.5x"), ("D @0.5x", "D @0.5x")):
    r = tm.loc[key]
    rows.append(dict(condition=label, n=int(r["n"]), a_k_tokens=r["a_k_tokens"],
                     valence=r["valence"], hedging_count=r["hedging"],
                     distress_lexicon_count=r["distress_lex"], source="stage3b_text_mediation.csv"))
rows.append(dict(condition="E (length-matched)", n=len(e), a_k_tokens=round(e.e_tokens.mean(), 2),
                 valence=np.nan, hedging_count=np.nan, distress_lexicon_count=np.nan,
                 source="tokens from stage3b_E_lengthfixed.csv; valence/hedging/distress "
                        "NOT COMPUTED for the length-matched passage"))
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(TABLES, "text_mediation_summary.csv"), index=False)
print("\ntext_mediation_summary.csv:")
print(summary.drop(columns="source").to_string(index=False))
print(f"\nfigures -> {FIGS}")
