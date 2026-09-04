"""Figures v2: sage / mustard / deep-orange / crimson palette, and FIG 3 fully traceable.

Changes from src/16:
  * new colour scheme across all four figures
  * FIG 3 is built from stored per-prefix tables (stage3b_E_lengthfixed_probe.csv,
    stage3b_BC_perprefix_probe.csv, stage3b_fig3_surface_perprefix.csv), so every bar traces
    to stored data -- the run-log provenance note is gone
  * FIG 3 carries bootstrap 95% CIs resampled over the 20 paired prefixes

src/16_figures.py is left as the record of the first figure set.
Plotting only; no model is loaded and nothing is measured.
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
N_BOOT = 10000
BOOT_SEED = 7
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.autolayout": False})

SAGE = "#7D9471"
MUSTARD = "#D6A419"
DEEP_ORANGE = "#C1571A"
CRIMSON = "#A31621"


def dz(d):
    d = np.asarray(d, float)
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 0 else np.nan


# ---------------------------------------------------------------------------
# FIG 1 -- token-shuffle survival
# ---------------------------------------------------------------------------
s2 = pd.read_csv(os.path.join(TABLES, "lexical_check_shuffle.csv"))
s2_surv = s2[s2.layer.between(11, 29)].survives_pct.mean()
lad1 = pd.read_csv(os.path.join(TABLES, "stage3b_perturbation_ladder.csv")).set_index("perturbation")
lad2 = pd.read_csv(os.path.join(TABLES, "stage3b_ladder_corrected.csv")).set_index("perturbation")
s3_draw1 = lad1.loc["a_token_shuffle", "mean_diff_survival_pct"]
s3_draw2 = lad2.loc["a_token_shuffle", "mean_diff_survival_pct"]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.bar([0], [s2_surv], width=0.42, color=SAGE, label="Stage 2 (single draw)")
ax.bar([0.92], [s3_draw1], width=0.34, color=MUSTARD, label="Stage 3b, draw 1 (src/14)")
ax.bar([1.30], [s3_draw2], width=0.34, color=DEEP_ORANGE, label="Stage 3b, draw 2 (src/15)")
for x, v in ((0, s2_surv), (0.92, s3_draw1), (1.30, s3_draw2)):
    ax.text(x, v + 3, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.axhline(100, ls="--", lw=1.2, color=CRIMSON)
ax.text(0.63, 103, "100% = effect fully retained", fontsize=8, ha="center", va="bottom",
        color=CRIMSON, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
ax.set_xticks([0, 1.11])
ax.set_xticklabels(["Stage 2 readout\n(SSSF vs NNNF at eot_u_4)",
                    "Stage 3b loop test\n(B vs C at eot_u_5)"])
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
print("FIG 1 written")

# ---------------------------------------------------------------------------
# FIG 2 -- probe dependence across seeds
# ---------------------------------------------------------------------------
seeds_df = pd.read_csv(os.path.join(TABLES, "stage3b_seeds.csv"))
readout = pd.read_csv(os.path.join(TABLES, "stage3b_readout_by_layer.csv"))
mid_mask = readout.layer.between(11, 29)


def seed30(probe, col):
    return readout[(readout.probe == probe) & (readout.strength == 0.5) & mid_mask][col].mean()


seeds = [30, 31, 32, 33]
series = {
    "P2  $d_z$(B−C)": ([seed30("P2", "dz_B_C")] +
                       [seeds_df[(seeds_df.seed == s) & (seeds_df.probe == "P2")].dz_B_C.iloc[0]
                        for s in seeds[1:]], SAGE, "o", "-"),
    "P1  $d_z$(B−C)": ([seed30("P1", "dz_B_C")] +
                       [seeds_df[(seeds_df.seed == s) & (seeds_df.probe == "P1")].dz_B_C.iloc[0]
                        for s in seeds[1:]], CRIMSON, "s", "-"),
    "P2  $d_z$(D−A)  random-direction control": (
        [seed30("P2", "dz_D_A")] +
        [seeds_df[(seeds_df.seed == s) & (seeds_df.probe == "P2")].dz_D_A.iloc[0]
         for s in seeds[1:]], MUSTARD, "^", "--"),
}
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for name, (vals, colour, marker, ls) in series.items():
    ax.plot(seeds, vals, marker=marker, ls=ls, color=colour, label=name, lw=1.9, ms=6.5)
    yoff = {"o": 0, "s": -14, "^": 14}[marker]
    ax.annotate(f"mean {np.mean(vals):+.2f}\nsd {np.std(vals, ddof=1):.2f}",
                xy=(seeds[-1], vals[-1]), xytext=(8, yoff), textcoords="offset points",
                fontsize=7.5, color=colour, va="center", fontweight="bold")
ax.axhline(0, ls=":", lw=1.3, color="black")
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
print("FIG 2 written")

# ---------------------------------------------------------------------------
# FIG 3 -- effect size vs validity, from stored per-prefix tables, with bootstrap CIs
# ---------------------------------------------------------------------------
e_probe = pd.read_csv(os.path.join(TABLES, "stage3b_E_lengthfixed_probe.csv"))
bc_probe = pd.read_csv(os.path.join(TABLES, "stage3b_BC_perprefix_probe.csv"))
surf = pd.read_csv(os.path.join(TABLES, "stage3b_fig3_surface_perprefix.csv"))
rng = np.random.default_rng(BOOT_SEED)


def probe_matrix(df, probe, col):
    """(n_prefixes, n_layers) matrix of per-prefix paired differences."""
    sub = df[df.probe == probe]
    return sub.pivot(index="prefix_id", columns="layer", values=col).to_numpy()


def boot_ci_layered(mat):
    """Bootstrap the layer-averaged paired d_z, resampling prefixes."""
    point = float(np.mean([dz(mat[:, j]) for j in range(mat.shape[1])]))
    n = mat.shape[0]
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = mat[idx]
        draws[b] = np.mean([dz(m[:, j]) for j in range(m.shape[1])])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, lo, hi


def boot_ci_scalar(diffs):
    point = dz(diffs)
    n = len(diffs)
    idx = rng.integers(0, n, (N_BOOT, n))
    draws = np.array([dz(diffs[i]) for i in idx])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, lo, hi


bc = {
    "TF-IDF": boot_ci_scalar((surf.tfidf_B - surf.tfidf_C).to_numpy()),
    "probe P2": boot_ci_layered(probe_matrix(bc_probe, "P2", "diff_B_minus_C")),
    "sentiment": boot_ci_scalar((surf.sentiment_B - surf.sentiment_C).to_numpy()),
}
ea = {
    "TF-IDF": boot_ci_scalar((surf.tfidf_E - surf.tfidf_A).to_numpy()),
    "probe P2": boot_ci_layered(probe_matrix(e_probe, "P2", "diff_E_minus_A")),
    "sentiment": boot_ci_scalar((surf.sentiment_E - surf.sentiment_A).to_numpy()),
}

fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.5), sharey=True)
order = ["TF-IDF", "probe P2", "sentiment"]
colours = [MUSTARD, SAGE, DEEP_ORANGE]
for ax, data, title in ((axes[0], bc, "B vs C\n(steering contrast)"),
                        (axes[1], ea, "E vs A\n(third-party control, length-matched)")):
    vals = [data[k][0] for k in order]
    err = np.array([[v - data[k][1] for k, v in zip(order, vals)],
                    [data[k][2] - v for k, v in zip(order, vals)]])
    ax.bar(order, vals, color=colours, width=0.62,
           yerr=err, capsize=5, error_kw=dict(ecolor=CRIMSON, lw=1.4))
    for k, v in zip(order, vals):
        ax.text(order.index(k), v / 2, f"{v:+.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
    ax.axhline(0, ls="--", lw=1.3, color="black")
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelrotation=12)
    ax.set_ylim(-2.05, 2.55)
axes[0].set_ylabel("paired effect size  $d_z$   (positive = distressed)")
axes[1].annotate("probe and surface\nbaselines disagree\nin sign", xy=(0.66, -0.80),
                 xytext=(-0.46, -1.98), fontsize=8, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", lw=1.1, color="black",
                                 connectionstyle="arc3,rad=0.15"))
fig.suptitle("Bigger effect size is not the same as better validity", fontsize=11, y=0.98)
fig.text(0.015, 0.015,
         "Layer 20, strength 0.5×, n=20 paired prefixes; probe values are the mean over layers "
         "11–29 of the per-layer paired $d_z$.\nError bars are bootstrap 95% CIs "
         f"({N_BOOT:,} resamples over prefixes). Left: TF-IDF separates B from C better than "
         "the probe.\nRight: distress about a third party moves the surface baselines up and "
         "the probe down. All bars from stored per-prefix tables.",
         fontsize=7.5, va="bottom")
fig.subplots_adjust(bottom=0.33, top=0.82)
fig.savefig(os.path.join(FIGS, "effect_vs_validity.png"), dpi=DPI)
plt.close(fig)
print("FIG 3 written")
print("  B vs C : " + ", ".join(f"{k} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
                                for k, v in bc.items()))
print("  E vs A : " + ", ".join(f"{k} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
                                for k, v in ea.items()))
pd.DataFrame([dict(panel=p, method=k, dz=v[0], ci_lo=v[1], ci_hi=v[2])
              for p, d in (("B_vs_C", bc), ("E_vs_A", ea)) for k, v in d.items()]).to_csv(
    os.path.join(TABLES, "stage3b_fig3_bootstrap_ci.csv"), index=False)

# ---------------------------------------------------------------------------
# FIG 4 -- perturbation ladder
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
ax.bar(x - 0.19, r1, width=0.36, color=MUSTARD, label="draw 1 (src/14; (d) length-short)")
ax.bar(x + 0.19, r2, width=0.36, color=SAGE, label="draw 2 (src/15; (d) length-matched)")
ax.plot(x - 0.19, d1, "o", color=DEEP_ORANGE, ms=6, label="$d_z$ survival, draw 1")
ax.plot(x + 0.19, d2, "D", color=CRIMSON, ms=5.5, label="$d_z$ survival, draw 2")
for xi, (a, b) in enumerate(zip(r1, r2)):
    ax.text(xi - 0.19, a + 3, f"{a:.0f}%", ha="center", fontsize=8)
    ax.text(xi + 0.19, b + 3, f"{b:.0f}%", ha="center", fontsize=8)
ax.axhline(100, ls="--", lw=1.2, color=CRIMSON)
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
print("FIG 4 written")
print(f"\nfigures -> {FIGS}")
