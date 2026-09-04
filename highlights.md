# Highlights

Running log of results. One entry per experiment. **Include null and confusing results** — those are
usually the informative ones, and this file is what makes the writeup fast.

Do not write this chronologically into the final doc. This is raw material; the writeup is restructured
around the finding.

---

## Template

### [YYYY-MM-DD HH:MM] Short title

- **Question:**
- **Ran:** (script, config, n, seeds)
- **Result:**
- **Reading:** (what it means — and what it does *not* mean)
- **Surprise / anomaly:**
- **Next:**

---

## Entries

### [2026-09-04 01:15] Position control length-matched — and the ladder's RNG-dependent rows turn out to vary ~20pp run to run

- **Question:** Re-run perturbation (d) properly length-matched, and re-report the full ladder.
- **Ran:** `src/15_position_control_lengthfix.py`. Span grown by walking prefix content spans in
  conversation order u1 -> a1 -> u2 -> a2 -> u3 -> a3 -> u4(F), each contributing span permuted
  within itself (same perturbation kind as (a), relocated). a_k and u_{k+1} never touched;
  u_{k+1} byte-identity asserted; logit-equality guard on every measurement.
- **Result:**
  - **Span accounting (n=40 transcripts, B and C at 0.5x):** target (a_k length) mean 432.3
    tokens; achieved 398.4; **paired diff −34.0 tokens (sd 45.3)**; 20/40 still fall short, max
    shortfall 181. Improvement on the previous 99.9-token shortfall, but **not exact** — total
    prefix content is the binding constraint even after consuming all seven prefix turns.
    Per-message usage: u1 24.4, a1 86.1, u2 23.8, a2 131.1, u3 20.9 (35/40), a3 96.1 (35/40),
    u4 16.0 (26/40).
  - **Corrected ladder (P2, L11–29, clean mean diff +0.225 / dz +0.85):**
    | perturbation | mean-diff survival | dz survival |
    |---|---|---|
    | (a) token shuffle in a_k | 48.8% | 43.1% |
    | (b) 20% content-word substitution | 86.7% | 75.0% |
    | (c) sentence-order reversal | 99.1% | 72.1% |
    | (d) position control, length-matched | **105.3%** | 66.3% |
  - **(d) still exceeds 100% (105.3%) after proper length matching. Reported as unexplained,
    per instruction; not investigated further and no additional variants run.** It is much
    closer to a no-op than the previous 155.7%.
- **Reading:** The pre-registered Sec 7 verdict is unchanged and now rests on a properly
  length-matched control: (a) collapses the effect most, (c) does not collapse it, (d) does not
  collapse it. **Row 1 of Sec 7 — Reading A: locally order-sensitive, structurally different
  from the Stage 2 lexical readout** (which survived the same token shuffle at 113%).
- **Surprise / anomaly — a reproducibility bug in my own code, found by comparing the two runs:**
  1. **The perturbation RNG was seeded from Python's `hash()` of a tuple containing strings,
     which is salted per process.** So (a), (b) and (d) drew *different permutations* in this run
     than in `src/14`, and their survival numbers moved accordingly: (a) 27.3% -> 48.8%,
     (b) 74.1% -> 86.7%, (d) 155.7% -> 105.3%. (c), which uses no RNG, reproduced exactly
     (99.1% both runs).
  2. **Consequence: the ladder's RNG-dependent rows carry roughly +/-20 percentage points of
     run-to-run noise**, and differences of that size between (a), (b) and (d) should not be
     over-read. The qualitative ordering that Sec 7 turns on — (a) lowest, (c) near 100%, (d)
     at or above 100% — holds in both runs, so the pre-registered verdict is unaffected. But
     the point estimates are softer than they look, and the earlier "27.3%" should be read as
     "roughly 30–50%".
  3. Fixing this properly means seeding from a stable hash (e.g. hashlib) rather than `hash()`.
     Not done here: the instruction was to stop after this fix, and re-running would change the
     numbers again without resolving which draw is "right".
- **Next:** Nothing. No further experiments run.

### [2026-09-04 00:30] Stage 3b additions — E holds after length matching, effect replicates across seeds, and the pre-registered ladder returns Reading A

- **Question:** (1) Does condition E survive proper length matching? (2) Does the 0.5x effect
  replicate across seeds? (3) What does the metrics.md Sec 7 perturbation ladder say?
- **Ran:** `src/14_stage3b_additions.py`. Guards as before; u_{k+1} byte-identical asserted,
  logit-equality leak guard on every measurement, re-tokenized from scratch, use_cache=False.
- **Result:**
  - **(1) E length fix.** New third-party passage (23 sentences, 509 tokens) truncates per
    prefix to **415.3 tokens vs A's 422.1** — paired diff −6.8 (sd 7.1), max shortfall 27,
    against the old 270.7. **The result barely moves: dz(E−A) = −0.99 (P2) and −0.41 (P1),
    versus −1.07 and −0.31 before.** So E was never a length artifact.
    On the same transcripts the surface baselines go the *other way*, and the properly
    length-matched passage makes the divergence much larger: **TF-IDF dz(E−A) = +1.05** (was
    +0.25 with the short passage), sentiment +0.45. Distress vocabulary about a third party
    pushes bag-of-words strongly up and the probe down — a ~2.0 opposite-sign gap.
  - **(2) Seeds (31/32/33, not pooled).** Both probes are highly reproducible, including the
    disagreement between them:
    | probe | seed 30 | 31 | 32 | 33 | mean (3 new) | sd | all 4 |
    |---|---|---|---|---|---|---|---|
    | P2 dz(B−C) | 0.85 | 0.67 | 0.77 | 0.73 | **0.72** | 0.05 | 0.76 (sd 0.08) |
    | P2 dz(B−A) | 0.58 | 0.58 | 0.47 | 0.59 | 0.54 | 0.07 | — |
    | P2 dz(D−A) | −0.07 | 0.03 | −0.07 | −0.01 | **−0.02** | 0.05 | — |
    | P1 dz(B−C) | −0.08 | −0.04 | −0.17 | −0.13 | **−0.11** | 0.07 | −0.10 (sd 0.06) |
    The random-direction control stays null across every seed. The probe-dependence is *not*
    seed noise — P2 reliably lands near +0.75 and P1 reliably near −0.10.
  - **(3) Perturbation ladder (P2, L11–29, clean baseline mean diff +0.225 / dz +0.85):**
    | perturbation | mean-diff survival | dz survival |
    |---|---|---|
    | (a) token shuffle in a_k | **27.3%** | 25.9% |
    | (b) 20% content-word substitution | 74.1% | 64.6% |
    | (c) sentence-order reversal | **99.1%** | 72.1% |
    | (d) position control (equal-length span in prefix) | **155.7%** | 122.5% |
    **This is row 1 of the pre-registered Sec 7 table: (a) collapses it, (c) does not, and (d)
    does not collapse it either. Reading A survives — the signal is locally order-sensitive and
    structurally different from the Stage 2 lexical readout**, which survived the same token
    shuffle at 113%.
- **Reading:** Three of the weaknesses in the previous entry are now closed. E is not a length
  artifact and is the sharpest evidence in the project that the probe is not a surface-lexical
  detector: on the same transcripts TF-IDF says +1.05 and the probe says −0.99. The effect
  replicates tightly across four seeds with a null random-direction control. And the ladder
  discriminates in the direction the pre-registration assigned to Reading A. What is *not*
  closed: TF-IDF still beats the probe on B-vs-C (1.27 vs 0.85), so the Sec 3 decision rule still
  fires on the headline contrast, and the whole positive result still rests on P2 rather than P1.
- **Surprise / anomaly:**
  1. **Correction to my previous report.** I claimed "zero forward hooks live at measurement" in
     Stage 3b. That check was run *before* the first measurement and was true at that instant,
     but transformers registers its own recorder hooks on every decoder layer whenever
     `output_hidden_states=True` and leaves them attached, so it would not have held afterwards.
     The guard is now keyed to our own handle ids. The substantive protection was always the
     logit-equality check inside `measure_forward` — it strips *all* hooks for the reference pass
     and has never fired across 640+ Stage-3b measurements — so no number changes.
  2. **(d) exceeds 100%** (155.7% mean-diff survival): scrambling an equal-length task-irrelevant
     span in the prefix *increases* the B−C separation. Unexplained. It is on the safe side of
     the pre-registered reading, but it means the position control is not a clean no-op.
  3. **(d) is length-short**: prefix assistant turns total ~322 tokens against a_k's ~422, so the
     shuffled span was 99.9 tokens shy of a_k's length on average. It shuffled ~76% of the
     intended span.
  4. (b) at 74%/65% sits between (a) and (c) — degrading vocabulary while keeping order hurts
     less than destroying order, which is consistent with Reading A rather than a pure
     bag-of-words account.
- **Next:** Human's call. Item 4 not started, as instructed.

### [2026-09-03 22:40] Stage 3b — the loop test. Pattern present and dose-dependent under one probe, absent under the other, and beaten by TF-IDF.

- **Question:** Does steering the distress direction during generation of a_k change the probe
  readout at an identical fixed u_{k+1}? (ROADMAP Appendix C, the deliverable.)
- **Ran:** `src/13_stage3b_loop_test.py`. Layer 20, strengths 0.25x and 0.5x (both pre-registered
  in metrics.md Sec 5, both reported). n=20 neutral NNNF prefixes, paired across all conditions.
  max_new_tokens=512, seed 30, temp 0.7, top_p 0.9. Conditions A/B/C/D/E; u_{k+1} = F[travel,b],
  byte-identical across all 160 transcripts (asserted). Random control direction cosine with the
  distress direction: +0.019.
  **Guards: 0 steering hooks live at measurement; all 640 measurements (160 x 4 shuffle modes)
  passed the logit-equality leak guard; every transcript re-tokenized from scratch, use_cache=False.**
  metrics.md Sec 1 never fixed L*/probe, so both candidates are reported.
- **Result:**
  - **P2 (depth-matched probe), 0.5x: dz(B−C)=+0.85, dz(B−A)=+0.58, dz(C−A)=−0.24,
    dz(D−A)=−0.07.** That is the Appendix C.5 pattern — B > A > C with the norm-matched random
    control null — and it is dose-dependent: dz(B−C) goes +0.29 at 0.25x to +0.85 at 0.5x.
    Peaks at L19 (dz(B−C)=1.44).
  - **P1 (single-turn probe): nothing.** dz(B−C)=+0.28 at 0.25x and **−0.08** at 0.5x — no
    consistent sign, no dose response. **The headline result depends entirely on which probe
    would have been pre-registered, and metrics.md Sec 1 was never filled in.**
  - **Control 1 (text mediation): the text did change**, so the readout is not moving without a
    textual pathway (C.7's bug check passes). B@0.5x carries 3.15 distress-lexicon hits vs C 0.55
    and A 1.05, valence 0.043 vs C 0.475. Lengths are well matched (A 422, B 439, C 426, D 424
    tokens; cap 10/13/9/10 of 20).
  - **Control 2 (surface baselines): TF-IDF beats the probe.** dz(B−C) = **1.27** for TF-IDF at
    0.5x versus **0.85** for probe P2 and −0.08 for P1; at 0.25x, 0.68 vs 0.29/0.28. Sentiment
    classifier 0.51 / 0.19. metrics.md Sec 3's decision rule ("if either baseline separates B
    from C with an effect size within ___ of the probe's, the probe adds nothing beyond surface
    sentiment and this is reported as the result") has its threshold blank, but the baseline does
    not merely match the probe — it exceeds it. On the pre-registered rule this is the result.
  - **Third-party control E is the one place the probe and the surface baselines disagree, and
    the probe wins.** E carries 10.0 distress-lexicon hits (vs A's 1.05) about someone who is not
    the user. TF-IDF moves *up* (dz +0.25) and sentiment moves up (dz +0.45), as lexical detectors
    should. **The probe moves down: dz(E−A) = −1.07 (P2), −0.31 (P1).** So the probe is not a
    distress-word counter. **But E is badly length-mismatched** — 271 tokens against A's 422, a
    151-token shortfall, because my fixed third-party passage ran out of sentences before
    reaching A's length. The E−A contrast is therefore confounded with length and should not be
    leaned on.
  - **Control 3 (token shuffle) destroys the effect.** Under P2 at 0.5x, shuffling a_k leaves
    −22.5% of the mean difference (sign flips), shuffling u_{k+1} leaves 6.7%, shuffling both
    −24.7%. So unlike the Stage-2 readout — which survived shuffling intact — the loop-test
    difference requires word order in both turns.
- **Reading:** Read strictly against Appendix C.5, the loop test **passes under P2**: B > A > C,
  D ≈ A, dose-dependent, with the text-mediation and hook/KV guards clean. Read against the
  pre-registered decision rule in metrics.md Sec 3, it **fails**: a bag-of-words model on the same
  transcripts separates B from C better than the probe does. Both statements are true and the
  second is the one the pre-registration says to report. The honest summary is that steering a_k
  changes a_k's text, that changed text is visible at u_{k+1}, and nothing here shows the model
  carrying anything beyond that text — which is the trivial reading the roadmap's Sec 3b warned
  about, not a demonstrated loop.
- **Surprise / anomaly:**
  1. **Probe-dependence is the biggest threat to the result.** P1 and P2 disagree in sign at 0.5x.
     Since L* was never locked, a reader can reasonably ask whether P2 was chosen post hoc; it was
     not (both were pre-declared in this script), but the pre-registration cannot back that up.
  2. **Shuffling u_{k+1} alone destroys the effect (6.7% survival)** even though u_{k+1} is
     byte-identical across conditions. The carried difference only projects onto the probe
     direction when the final turn is well-formed.
  3. Shuffle-survival percentages are unstable wherever the clean difference is near zero (P1 at
     0.5x: clean diff −0.030 gives −293.8% survival). Those cells are noise ratios, not results.
  4. E's length shortfall is my error: the fixed passage should have been long enough to match a
     422-token mean. Reported as-is rather than regenerated.
- **Next:** Stage 4 (seeds/robustness) and writeup are the human's call. No metric was changed
  after seeing results; metrics.md Sec 1/3/4 remain unlocked and I did not fill them.

### [2026-09-03 21:10] Stage 3a — steering works; coherence breaks between 0.5x and 0.75x

- **Question:** At the pre-registered layer 20, what is the largest steering strength at which
  a_k still reads as fluent? (ROADMAP Appendix C.2, metrics.md Sec 5.)
- **Ran:** `src/12_stage3a_steering_sweep.py`. Layer 20/32 (62% depth, pre-registered blind to
  outcome). Direction = diff-of-means over the 24 S vs 24 H single-turn pool turns at eot_u_1,
  layer 20, unit-normalised (raw ||mean_S − mean_H|| = 4.761). 3 neutral NNNF prefixes, a_k is
  the reply to u_k = F. max_new_tokens=512, temp 0.7, top_p 0.9, seed 20. Mean layer-20
  activation norm over 1045 prefix tokens = 18.419, so strengths are 4.60 / 9.21 / 13.81 /
  18.42 / 27.63 / 36.84 in vector norm. **No readout at u_{k+1} was computed anywhere.**
- **Result:** The steering path works and the degeneracy boundary is sharp.
  | alpha | cap hit | valence | hedges | repeat-5gram | ascii-word | TTR | verdict |
  |---|---|---|---|---|---|---|---|
  | 0.0 | 1/3 | +0.320 | 1.3 | 0.000 | 0.862 | 0.662 | fluent (reference) |
  | 0.25 | 0/3 | +0.382 | 1.0 | 0.000 | 0.884 | 0.624 | fluent |
  | 0.5 | 1/3 | +0.263 | 0.7 | 0.003 | 0.861 | 0.645 | fluent, content shifted |
  | 0.75 | 3/3 | −0.102 | 0.3 | 0.088 | 0.851 | 0.493 | **degenerate** |
  | 1.0 | 3/3 | −0.664 | 0.0 | 0.878 | 0.971 | 0.080 | degenerate |
  | 1.5 | 3/3 | −0.008 | 0.0 | 0.941 | 0.026 | 0.048 | token salad |
  | 2.0 | 3/3 | +0.081 | 0.0 | 0.000 | 0.000 | 1.000 | 512 tokens of "*" |
  - 0.25x and 0.5x are fluent and on-task. At 0.5x the *content* has clearly moved — the model
    starts advising the user to claim "a medical emergency or urgent situation" and references
    "the stress of your run" — while the prose stays grammatical and unrepetitive.
  - 0.75x is the first degenerate strength, and it is degeneracy rather than truncation:
    repetition loops on the pre-truncation text (`Please call 999/911 immediately.` five times,
    repeat-5gram 0.185, TTR 0.413), plus a self-contradicting sentence — "I am not in the middle
    of a panic attack; I am in the middle of a panic attack."
  - 1.0x adds **role confusion**: the assistant starts speaking as the user ("I'm feeling
    incredibly stressed right now... I'm panicking. I'm panicking." x20).
  - 2.0x emits 512 "*" characters for all three prefixes, identically.
- **Reading:** The largest strength at which a_k still reads fluent is **0.5x**, with 0.25x
  comfortably safe. The human selects; nothing is written into metrics.md Sec 5 by me.
- **Surprise / anomaly:**
  1. **Valence is non-monotonic in strength and useless as a coherence proxy**: −0.664 at 1.0x
     but −0.008 at 1.5x and +0.081 at 2.0x, because salad carries no sentiment. Anyone using
     response valence to pick a strength would be misled at the top of the range.
  2. **Cap-hit rate is confounded with strength** (0/3 at 0.25x, 3/3 at every strength >= 0.75x),
     so token count cannot be used as a coherence signal either. Both diagnostics had to be read
     on the pre-truncation text, as instructed.
  3. **Hedging falls monotonically to zero** (1.3 -> 1.0 -> 0.7 -> 0.3 -> 0) across the
     coherent-to-degenerate range, which is at least consistent with the direction doing
     something to the response style before it destroys it.
  4. **metrics.md Sec 2.1 (hedging markers) and Sec 2 (response-valence classifier) are both
     blank.** I fixed a 23-marker list and reused the A.5 sentiment classifier *before*
     generating, and did not write either into metrics.md. Both still need locking.
- **Next:** Human selects the strength by reading the samples, then records it in metrics.md
  Sec 5 with the date, before any Stage 3b run. Stage 3b not started. Note that 3b additionally
  requires the norm-matched random-direction control (CLAUDE.md rule 3), which was not part of
  this sweep.

### [2026-09-03 20:15] Lexical control — the probe is distinguishable from bag-of-words, but in the unhelpful direction

- **Question:** Is the depth-matched eot_u_3 probe distinguishable from a lexical detector?
- **Ran:** `src/11_lexical_discriminability.py`. Four methods, same transcripts, same
  leave-one-topic-out folds, read at eot_u_4 on four cells. Token shuffle permutes tokens within
  each turn's content span, seeded from the message text so identical text (the byte-identical F
  turn, spliced assistant turns) gets an identical permutation everywhere. ~1120 forward passes,
  no generation, no steering.
- **Result:**
  - **Orderings diverge on exactly one cell.** TF-IDF and the sentiment classifier both give
    SSSF > Nuser+Sassist > **Suser+Nassist** > NNNF. The activation probe (and its shuffled
    version) gives SSSF > Nuser+Sassist > **NNNF** > Suser+Nassist. The disputed cell is
    Suser+Nassist — distressed user turns with neutral assistant text: dz vs NNNF is **+2.49
    (TF-IDF)**, **+3.05 (sentiment)**, **−0.11 (probe)**.
  - **Effect sizes are not comparable — the probe is the weakest.** dz vs NNNF, L11–L29 mean for
    the probe: SSSF 4.16 / 1.60 / 6.85 and Nuser+Sassist 3.91 / 1.76 / 3.92 for
    TF-IDF / probe / sentiment. Bag-of-words separates these cells 2–4x better than the
    activation probe does.
  - **Token shuffling does not remove the separation.** SSSF−NNNF at eot_u_4 retains **113.1% of
    the mean difference** and **43.5% of the effect size** averaged over L11–L29, and the
    four-cell ordering is unchanged. eot_u_3 held-out accuracy falls 0.996 → 0.815, so shuffling
    damages the training contrast somewhat but leaves it far above chance.
- **Reading:** The probe *is* distinguishable from a lexical detector, but the distinction runs
  the wrong way for a "user model" interpretation. It is **less** sensitive to distressed
  user-turn wording than bag-of-words is (blind to Suser+Nassist, which TF-IDF and sentiment both
  rank clearly above NNNF), and what it does track is order-invariant — destroying word order
  within turns leaves the raw separation intact. Taken with the previous entry, the readout at
  eot_u_4 behaves like an order-insensitive detector of assistant-turn lexical content, not like a
  representation of the user's state. Given that the two surface baselines beat it on their shared
  contrasts, metrics.md Sec 3's decision rule — "if either baseline separates as well as the
  probe, the probe adds nothing and this is reported as the result" — points at reporting exactly
  that.
- **Surprise / anomaly:**
  1. "Survives shuffling" exceeds 100% of the mean difference at 13 of 19 mid layers — shuffled
     separation is sometimes *larger* than unshuffled. Only the effect size shrinks (variance
     grows), which is what an order-invariant signal plus added noise looks like.
  2. The probe ranking Suser+Nassist marginally *below* NNNF (dz −0.11) is the same wrong-signed
     residue seen with the single-turn probe, now isolated to the cell where user distress is
     present but assistant distress is absent.
- **Next:** Stage 3, per instruction, regardless of this outcome. No further readout variants.
  Tables: `lexical_check_probe.csv`, `lexical_check_shuffle.csv`.

### [2026-09-03 19:20] Sign inversion was a probe-depth artifact — and under a depth-matched probe the readout is assistant-driven

- **Question:** (1) Scripted-assistant control at full n. (2) A new splice arm isolating the
  assistant turns. (3) Does the sign inversion survive a depth-matched probe?
- **Ran:** `src/10_splice_controls.py`. No generation, no steering. 320 new forward passes over
  transcripts spliced verbatim from existing text. Verified first that rebuilding NNNF from pool
  + its own assistant turns reproduces the stored transcript byte-for-byte, so cached NNNF
  readouts are valid pair partners.
- **Result:**
  - **(1) Scripted-assistant control, n=160 paired, length equalised to +1.9 tokens** (sd 29.9;
    assistant text byte-identical within each pair, so only user-turn length differs). Under the
    single-turn probe the SSSF−NNNF gap at eot_u_4 is retained at roughly 60–100% of the
    generated-text value across L11–L29 (L16 −0.975 vs −1.510 original, dz −2.14 vs −1.83;
    L28 −0.895 vs −0.936). Confirms the n=8 pilot at full n and with length controlled.
  - **(2) Assistant-only splice** (NNNF user turns + SSSF assistant text) gives a *negative*
    difference vs NNNF of similar magnitude to control 1 under the single-turn probe (L16 −0.929,
    dz −1.16; L13 −0.866, dz −1.74). **But it is not length-matched**: swapping in SSSF assistant
    turns adds +408.4 tokens (sd 171.5), because SSSF assistant turns are cap-saturated at 256
    and NNNF's average ~120. That confound is inherent to the splice and cannot be removed by
    construction.
  - **(3) The sign inversion is a probe-depth artifact and it disappears completely.** A probe
    trained at eot_u_3 on the assembled transcripts (SSSF turn-3 = S vs HHHF turn-3 = H),
    leave-one-topic-out, read at eot_u_4: **dz(SSSF−NNNF) is positive in 19/19 layers L11–L29
    (+0.80 to +2.93), where the single-turn probe was negative in 19/19.** The distressed prefix
    now reads *more* distressed at the byte-identical final turn, which is the intuitive
    direction. Held-out accuracy at eot_u_3 is 0.97–1.00 across layers.
  - Under that depth-matched probe the splices reverse the story of the previous entry. At L27:
    SSSF +2.08, NNNF −1.63, SSSF-user+NNNF-assist **−2.04** (i.e. at NNNF's level), NNNF-user+
    SSSF-assist **+1.47** (i.e. near SSSF's level). Read literally, the assistant text carries
    most of the eot_u_4 readout and the user turns carry little.
- **Reading:** Result (3) is the solid one and it invalidates the sign-inversion puzzle from the
  previous entry: that inversion was an artifact of applying a probe fit on ~20-token contexts to
  activations ~850 tokens deep, not a property of the model. **The previous entry's conclusion —
  "the residue is carried by the user turns, not the assistant" — does not survive.** Under the
  better-specified probe the attribution flips.
- **Surprise / anomaly (why I would not yet call the assistant-driven result a finding):**
  1. **The depth-matched probe's training contrast is confounded.** SSSF and HHHF transcripts
     differ at eot_u_3 in *both* user-turn valence and assistant-turn valence, and the assistant
     text is ~750 of the ~850 tokens of context. So the probe may have learned "is there
     distressed assistant text here", in which case the splice results follow tautologically. A
     clean attribution needs a probe whose training contrast varies only one of the two.
  2. **Control 2 is length-confounded (+408 tokens)** while control 1 is not (+1.9). The two
     controls are therefore not on equal evidential footing, and the cap saturation is the cause.
  3. eot_u_3 held-out accuracy is again at ceiling (0.97–1.00), so the depth probe inherits the
     same "trivially separable data" caveat as the Stage 2 gate.
- **Next:** Human's call. Tables: `control1_scripted_assistant_fulln.csv`,
  `control2_assistant_only.csv`, `depth_matched_probe.csv`. No steering, no layer selected.

### [2026-09-03 18:05] Readout from cache — prefix valence barely survives to the identical final turn, and what survives points the wrong way

- **Question:** Read the probe out of the existing cache at u_4/eot_u_4 and every turn boundary,
  by arm, across all 32 layers. Does the arm effect survive length adjustment, and does it
  survive holding the assistant text byte-identical?
- **Ran:** `src/09_readout_analysis.py`. No regeneration, no L\* selection. Readout = signed
  distance from a per-layer L2 logistic S-vs-H probe fit on the 48 single-turn pool S/H renders
  (positive = distressed); trained at `u_1` and read at `u_4`, trained at `eot_u_1` and read at
  `eot_u_2..4`. 160 paired (topic, phrasing, replicate) tuples per arm.
- **Result:**
  - **Trajectory (Figure 1 data).** At eot_u_1..3 the readout tracks the *current* turn's valence
    almost perfectly: at L16, SSSF +8.13/+7.98/+8.20, NNNF −4.68/−3.58/−3.50, HHHF
    −7.54/−4.42/−3.71, and SSHF follows SSSF at turns 1–2 (+8.13/+7.96) then flips to −7.10 at
    turn 3 exactly where its valence flips. **At eot_u_4 — the byte-identical F turn — the
    separation collapses**: all four arms land within ~1.6 of each other (L16: −0.02, 1.49, 1.62,
    0.74), against a turn-1 spread of ~15.
  - **What remains at eot_u_4 is small and sign-inverted.** From L11–L29, `dz(SSSF−NNNF)` is
    **negative** (−1.96 L11, −2.22 L13, −1.90 L15, −1.83 L16, −1.93 L29) and `dz(HHHF−NNNF)` is
    positive (+0.2…+1.0). The distressed prefix reads *less* distressed at F than the neutral
    prefix does. Only at L1–L4 is the sign in the intuitive direction (+2.31 at L1).
  - **Position disagreement.** At `u_4` the late layers (L17–L31) give *positive* dz(SSSF−NNNF)
    (+0.3…+1.2), the opposite sign to `eot_u_4` at mid layers. The readout's sign depends on
    which of the two tokens you read.
  - **Length.** Early-layer effects are pure length: at L1 the SSSF coefficient flips +4.11 →
    −1.74 once transcript length enters, and length itself is massively significant there
    (t=18.0). At mid layers length is weak (|t|<2 for most layers ≥11) and **the arm effect
    survives essentially unchanged** (L13 −1.337 → −1.552; L15 −1.540 → −1.600; L16 −1.510 →
    −1.874).
  - **Scripted-assistant control (the informative one).** Rebuilding SSSF and NNNF with
    byte-identical assistant text (taken verbatim from the existing NNNF|cell|00 transcripts, no
    generation) equalises length to 480.6 vs 478.8 tokens. The SSSF−NNNF difference at eot_u_4 is
    **retained at 55–125% of its generated-text magnitude across L11–L29**, with comparable or
    larger effect sizes (L12 dz −2.83 scripted vs −1.41 generated; L16 −1.86 vs −1.83).
- **Reading:** Two things follow, and neither is the loop test. (1) The probe readout at a
  byte-identical neutral turn is dominated by that turn's own text; prefix valence leaves only a
  small residue. (2) That residue does **not** require the assistant's generated text — it
  survives at full strength when the assistant turns are held byte-identical, so it is carried by
  the user turns in context, not by the assistant echoing valence back. On its face that is
  evidence for user-evidence integration (H1) rather than assistant feedback (H2) at this
  position — but see the caveats, which are serious enough that I would not state it as a finding.
- **Surprise / anomaly:**
  1. **The trajectory is partly in-sample.** The probe is trained on single-turn renders of the
     same pool turns that fill slots 1–3, so eot_u_1..3 readouts are essentially in-sample; only
     the F turn is unseen text. The decline from turn 1 to turn 4 therefore conflates context
     growth with an in-sample → out-of-sample transition. Figure 1 must not be drawn from this
     without addressing that.
  2. **Distribution shift.** The probe is fit on ~20–40-token contexts and applied ~850 tokens
     deep. Readout scale compresses from ±8 at turn 1 to ±2 at turn 4, which could produce the
     "collapse" on its own.
  3. **The sign inversion is unexplained.** A distressed prefix reading *less* distressed at F,
     consistently across L11–L29 and reproduced in the length-controlled scripted control, is not
     what any of H0/H1/H2 predicts. It is stable, so it is not noise — but I have no account of it.
  4. Common-support length matching is badly unbalanced (SSSF 100, SSHF 115 vs NNNF 19, HHHF 14),
     so the common-support column is weak evidence either way.
  5. Scripted control is n=8 pairs, and I chose NNNF-arm assistant text as the "neutral scripted"
     content — a defensible but consequential choice that the human did not specify.
- **Next:** Human's call on all of the above. Nothing from Stage 3 started; no layer selected.
  Tables: `results/tables/readout_{u_4,eot_u_4}_by_layer.csv`, `trajectory_eot_u.csv`,
  `readout_length_regression.csv`, `scripted_assistant_control.csv`, `scripted_control_by_layer.csv`.

### [2026-09-03 17:10] Stage 2 complete — gate "passes" at 1.000, which is the problem; and the token cap is confounded with arm

- **Question:** Assemble the four arms, generate transcripts, cache activations, and select L\*
  by held-out single-turn probe accuracy. Gate: >75%.
- **Ran:** `src/05_assemble_and_generate.py` (640 dialogues, 1920 generations, 1458 s, GPU 0,
  seed=10, temp=0.7, top_p=0.9, max_new_tokens=256, batch 16), `src/06_extract_activations.py`
  (7040 rows, 308 s, 1.15 GB cache + manifest.parquet), `src/07_single_turn_probes.py`
  (32 layers x 2 positions, 8-fold leave-one-(topic,phrasing)-out CV, L2 logistic C=1.0),
  `src/08_probe_leak_diagnostic.py`.
- **Result:**
  - Assembly: 640 dialogues = 4 arms x 4 topics x 2 phrasings x n=20. **F byte-identity across
    all four arms asserted in all 8 cells, and re-checked on all 640 transcripts: 0 cells with
    more than one distinct F text.** Zero asserts fired anywhere in steps 6–8.
  - **Probe gate: 1.000 held-out 3-way accuracy (layer 22, `u_1`); binary S-vs-H also 1.000.**
    `eot_u_1` peaks at 1.000 (L8). Mean over all layers: `u_1` 0.973, `eot_u_1` 0.963.
  - **The gate number is uninformative, and I do not think it should be treated as a pass in
    substance.** Diagnostics on identical data and folds: TF-IDF bag-of-words 0.819, but an
    **off-the-shelf sentiment classifier with zero training gets 0.972** (S/N/H confusion: 2
    errors in 72). Probe 1.000 vs sentiment 0.972 is a two-sample difference at n=72. Accuracy
    is already **0.778 at layer 0** and >=0.95 **by layer 2** — almost no computation is needed,
    which is the signature of a lexical cue, not a constructed user model.
  - **Assistant-turn length is confounded with arm, via the cap.** Mean assistant tokens:
    SSSF 254.7, SSHF 242.1, NNNF 119.4, HHHF 106.9 — a 2.4x spread. Cap-binding rate: SSSF
    **96.9%** (turns 2 and 3 are 100% at cap, sd = 0.0), SSHF 75.0%, NNNF 8.8%, HHHF 5.0%;
    overall 46.4%. Transcript length follows: SSSF 902 tokens mean vs HHHF 456.
- **Reading:** Stage 2's infrastructure is sound and the pipeline is clean (byte-identity holds,
  no tokenization drift, no hook leaks, cache + manifest written). But two things must be settled
  before Stage 3, and both are the human's call:
  1. The probe's ceiling accuracy means L\* selection is essentially arbitrary — any layer from 2
     to 31 scores >=0.90, so "best layer" is noise, not a maximum. Picking L\*=22 because it hit
     1.000 first is selecting on 72 samples.
  2. The distress arms are truncated mid-sentence in nearly every assistant turn. This is not a
     cosmetic issue for H2b: Appendix C.7 requires length matching as a control, and here length
     differs by construction *before* any steering is applied. A readout difference between arms
     could be a truncation artifact rather than a user-state effect.
- **Surprise / anomaly:**
  1. **`a_t` readout positions sit on truncation boundaries for 46.4% of assistant turns**
     (891/1920). The B.3 decode output shows it directly: `a_1`/`a_2`/`a_3` in the SSSF arm decode
     to `'️'` (an emoji fragment), `'*'`, `' *'`, `' can'`, `' and'`, `'ing'` — mid-word and
     mid-markdown tokens, because "last content token of the turn" is wherever the 256-token cap
     fell. For SSSF turns 2 and 3 that is 100% of turns. The user positions are clean
     (`u_t` -> `'.'`, `eot_u_t` -> `'<|im_end|>'`, `u_4` -> `'?'`).
  2. My own diagnostic script's auto-verdict line ("the probe beats bag-of-words by a clear
     margin, so the accuracy is not purely lexical") is **too generous** — it thresholds on
     TF-IDF only and ignores the 0.972 sentiment-classifier result, which is the damning
     comparison. Reported the stronger reading rather than the script's line.
  3. `conda run` buffers child stdout even with `python -u`, so long runs were invisible until
     exit. Call `envs/eval/bin/python` directly for anything long-running.
- **Next:** Human decides on (a) L\* selection given a ceiling-accuracy sweep, (b) what to do
  about the cap/length confound before Stage 3a. Nothing from Stage 3 has been started.

### [2026-09-03 13:55] Pool fixes applied (v3) — F sentiment outlier resolved, U[health,a,N] filled

- **Question:** Do the human-directed pool fixes clear the problems flagged in the v2 round?
- **Ran:** `src/04_pool_fixes.py`. Generator Qwen3.5-9B (GPU 1), seed=3. v2 archived to
  `data/pool/turn_pool_v2.json`; v3 at `data/pool/turn_pool.json` (72 U + 8 F).
- **Result:**
  - **F turns (4 edits).** Typo fixes: `F[tech_support,a]` "laptop **o** the office" → "on the";
    `F[work,b]` "How to schedule" → "How do I schedule". Replacements supplied by the human and
    stored verbatim: `F[tech_support,b]`, `F[travel,b]`. All 8 now pass a typo-pattern assert and
    all 8 end in an explicit question mark (was 7/8).
  - **`U[health,a,N]` filled on attempt 1** with the floor relaxed to 15 words — and came back at
    24–25 words, i.e. the relaxed floor was never the binding constraint, the retry lottery was.
    Pool is complete at 24/24 N.
  - **A.5 sentiment gate on F (reported, not acted on, per instruction):** F mean **+0.029**
    (sd 0.110), range [−0.144, +0.186] — versus v2's mean −0.071, sd 0.274. The −0.707 outlier is
    gone: `F[tech_support,b]` now reads +0.136. N class is +0.169 (sd 0.122), range
    [+0.027, +0.528]. **3 of 8 F turns still sit just below the N floor** (`F[travel,b]` −0.002,
    `F[work,a]` −0.144, `F[work,b]` −0.097).
  - **Within-valence dedup (rule narrowed):** 7 shared 4-grams within S, 1 within N, 5 within H;
    4 cross-valence, now allowed by design.
- **Reading:** The F turns are in much better shape — the systematic negative offset in the probe
  position is largely gone. The 3 remaining sub-floor turns are mild (worst −0.144 against an N
  floor of +0.027) and cluster in `work`, whose neutral content is intrinsically deadline-shaped.
  Since F is byte-identical across all four arms, any residual offset is common-mode and cancels
  in the paired arm contrasts; it would only matter for absolute readout values.
- **Surprise / anomaly:** The N class's own sentiment range is entirely *positive* (+0.027 to
  +0.528), so "inside the N range" is a stricter test than "neutral" — a genuinely 0.000-compound
  turn fails it. That is why `F[travel,b]` at −0.002 is flagged. Worth keeping in mind before
  treating "outside N range" as a defect rather than a boundary artifact.
- **Next:** metrics.md amended with 4 dated entries (relaxed floor + reason, dedup narrowing,
  `[H,H,H,H]` arm dropped in favour of `[S,S,H,F]`, QC-gate classifier choice — noting §3's
  Stage 3b baseline classifier is still unlocked). Steps 6–9 running.

### [2026-09-03 13:20] Pool revision round — N regenerated (21/24), hand-authored F turns fail the A.5 gate 4/8

- **Question:** Does the revised pool (hand-authored F, regenerated N, dedup) pass A.5?
- **Ran:** `src/03_pool_revision.py`. Generator Qwen3.5-9B (GPU 1), seed=2, temp=0.9,
  top_p=0.95. Accept filter tightened to 20–35 words AND 1–2 sentences. Dedup: reject a turn
  whose opening 4-gram occurs anywhere in the pool. S/H carried over unchanged; v1 archived to
  `data/pool/turn_pool_v1.json`. Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`,
  compound = P(pos) − P(neg).
- **Result:**
  - **N class incomplete: 21/24.** `U[health,a,N]` never passed in 20 attempts. Cause is
    specific and not random: 19 of its 20 rejections were the **20-word floor** (candidates
    came back at 13–19 words). Phrasing *a* is defined as "plain and direct, short declarative
    sentences", and neutral health content is intrinsically terse ("I have a doctor appointment
    at two p.m." = 9 words). The style brief and the word floor are in direct tension, and
    acceptance is per-cell (all 3 slots must pass together), so one short slot kills the cell.
    Left missing rather than hand-written or filter-relaxed — that's a spec decision.
  - Sentiment by class: S −0.844 (sd 0.111), N +0.168 (sd 0.124), H +0.938 (sd 0.059).
    N's P(neutral) mean is 0.819, so the regenerated N class does read as genuinely neutral —
    the old N class's "problem report" confound is gone.
  - Shared 4-grams: 19 → 16 across the pool; **0 within the new N class**. The 16 that remain
    are all S↔S, H↔H or S↔H pairs inside the carried-over S/H turns, which this round did not
    touch.
  - **A.5 gate on the 8 hand-authored F turns: 4 of 8 fall outside the N class compound range.**
    `F[tech_support,b]` is the serious one at −0.707 (P(neg)=0.724) — "battery level dropped
    from 80% to 40%" reads as a fault report. The other three are marginal (+0.015, −0.050,
    −0.144 against an N floor of +0.027). `F[travel,b]` ends in a period, not a question mark.
    F turns are also longer than N (29.8 vs 23.7 tokens mean).
- **Reading:** The N regeneration worked and fixed the main v1 problem. The F turns did not
  clear the gate as supplied, and F is the byte-identical probe position present in every
  dialogue of every arm — so a negative-reading F contaminates all four arms equally rather
  than one, which is a systematic offset in the readout, not noise.
- **Surprise / anomaly:**
  1. Three of the supplied F turns contain typos that will appear verbatim in every assembled
     transcript: `F[tech_support,a]` "a new laptop **o** the office wifi", `F[tech_support,b]`
     "40%**over** four hours" and "accounts **fro** battery usage". `F[work,b]` reads
     "How to schedule..." without an auxiliary. Stored verbatim as instructed; not edited.
  2. The dedup rule rejected a health,a,N candidate opening "my doctor appointment is" because
     `U[health,a,S,2]` already opens that way. That overlap is arguably *desirable* for the
     design (identical openings across valence deny a bag-of-words baseline a cue), so the
     rule as written can reject benign — even helpful — turns. Worth a second look.
  3. Fixed an idempotency bug in the script mid-round: re-running it would have overwritten
     the v1 archive with v2. It now reads carried-over turns from the archive when present.
- **Next:** Human decides (a) how to fill `U[health,a,N]`, and (b) what to do about the 4 F
  turns outside the N range. Assembly spec is recorded — four arms `[S,S,S,F]`, `[N,N,N,F]`,
  `[H,H,H,F]`, `[S,S,H,F]`, all sharing byte-identical F; the old `[H,H,H,H]` arm is dropped.
  `data/cache` is now a symlink to `/media/data/baloni/sticky-user-model/cache` (269 GB free).
  Nothing assembled.

### [2026-09-03 10:40] Stage 2 step 1 — turn pool built (80/80 cells), several A.5 concerns to adjudicate

- **Question:** Does the Appendix A.2 turn pool generate cleanly with the 9B, and does it pass
  the A.5 gates?
- **Ran:** `src/02_build_turn_pool.py`. Generator Qwen3.5-9B (bf16, GPU 1), subject 4B used
  as tokenizer only. seed=1, temperature=0.9, top_p=0.95, enable_thinking=False, batch 8,
  32 cells (24 x 3 slot turns + 8 finals), 3 attempts max. ~34 s of generation total.
  Output: `data/pool/turn_pool.json`, 72 U + 8 F as typed `common.PoolTurn` objects.
- **Result:** All 80 turns present, no missing cells. 4 generations rejected and retried
  (3x too short, 1x unparseable); no cell needed all 3 attempts. Pool round-trips off disk
  as typed objects; `require_pool_turns()` rejects hand-written dicts by type.
  Length: S 24.8 / N 22.7 / H 24.6 mean tokens — means balanced within ~2 tokens.
- **Reading:** Infrastructure works and the pool is nominally complete, but I do **not**
  think it passes A.5 as generated. Content problems below are for the human to adjudicate;
  nothing was edited or regenerated.
- **Surprise / anomaly (all flagged, none fixed):**
  1. **F[health,a] is not neutral.** "My sleep schedule is irregular last night and I feel
     too tired for the gym today. Do you have specific tips for improving energy levels when
     sleep is poor?" — explicit negative state, and ungrammatical. This is the *probe turn*,
     the one position where affect must be absent. 3 more F turns are problem reports with
     mild negative framing (`F[tech_support,a]` "still fails every time",
     `F[tech_support,b]` "draining unusually fast", `F[work,a]` "assigned three new
     deadlines"). Only 4/8 F turns read as clean neutral requests.
  2. **The N class is not affect-free for 2 of 4 topics.** Neutral tech_support and work
     turns are *problem reports* ("screen flickers", "data still missing", and
     `U[work,b,N,3]` "I will likely miss the deadline"). So N vs S differs in intensity of
     the same negative situation rather than in valence. This directly weakens the
     probe's S/N contrast and is the kind of thing that shows up later as a mediocre
     three-way accuracy with no obvious cause.
  3. **Length balance holds only in the marginal.** Split by phrasing it does not:
     S is 17.3 tokens under phrasing a but 32.2 under phrasing b (N: 20.9/24.5,
     H: 23.6/25.7). Valence and length are balanced overall; valence x phrasing is not.
  4. **Generator tics repeat across cells**: 18 shared 4-grams, e.g. "I am absolutely
     terrified that" opens both `U[travel,b,S,1]` and `U[tech_support,b,S,1]`; "everything
     is falling apart" opens `U[health,a,S,3]` and `U[travel,b,S,3]`. Combinatorial
     assembly will multiply these, giving a bag-of-words baseline exact phrases to latch
     onto.
  5. **Topic drift in slot 3**: `U[health,a,S,3]` ("Everything is falling apart and I do
     not know what to do.") carries no health content at all.
  6. **My accept filter was looser than the prompt.** Prompt asked for 20-35 words and 1-2
     sentences; the validator only enforced 10-60 words, so 26/72 turns are under 20 words
     and 13/72 run to 3+ sentences. Tightening the filter is the obvious lever if turns get
     regenerated.
  7. **Spec gap for the reversal family (blocks assembly).** Sec 5a defines reversal as
     `[S,S,S,H]` vs `[H,H,H,H]`, but A.2 provides only F[t,p] (neutral) as a shared final
     turn, and slots i in {1,2,3}. There is no byte-identical *positive* final turn, and
     `[H,H,H,H]` needs four H turns from a pool holding three. Carryover is fine
     (`[S,S,S,F]` vs `[N,N,N,F]`). Needs a decision before any assembly.
- **Infra note (not scientific):** the home quota (110 GB hard) was full — the 9B download
  died at 5 GB with "Disk quota exceeded". Removed my own failed partial (freed 5 GB, back
  to 105 GB) and loaded the 9B read-only from the complete shared copy at
  `/media/data/.cache/huggingface` (shard sizes verified against the hub). `common.py` now
  resolves models from extra cache roots and refuses to resolve an incomplete snapshot.
  **Home is still ~5 GB under a hard cap; Stage 2 activation caches will not fit there.**
  `data/cache/` needs to point at `/media/data/baloni/` before extraction runs.
- **Next:** Human adjudicates the A.5 gates and the reversal-family spec gap. Nothing is
  assembled.

### [2026-09-02 22:30] Two guards added to a shared utility + Stage 2 throughput measured

- **Question:** (1) Can the Appendix C.4 "zero hooks" check be replaced by something that
  actually holds? (2) Can the boundary scanner catch turn/role misalignment rather than
  assuming alternation? (3) How slow is Stage 2 generation on the gated-deltanet torch
  fallback — do we need the kernels, or a smaller n?
- **Ran:** `src/common.py` (new shared utility), verified by `src/01_patch_checks.py`.
  Qwen3.5-4B bf16, GPU 0. Throughput at Stage 2 settings: batch 16, max_new_tokens=100,
  enable_thinking=False, temperature=0.7, top_p=0.9, 3 sequential assistant turns.
- **Result:**
  - **Patch 1 (`measure_forward`)**: measurement pass is re-run with every forward hook on
    the model detached, and the two logit tensors must be `torch.equal`. Verified: passes
    clean; fires on a leaked steering hook at 4.0x, 0.05x **and 0.001x** the mean activation
    norm at the hooked layer. So the guard is sensitive down to essentially the bf16 noise
    floor, not just to gross leaks. Two identical forward passes on this model are
    bit-identical, so no tolerance is needed. Cost: exactly 2.00x one forward pass
    (290 tok: 191 ms -> 384 ms), ~62 s for 160 transcripts measured singly.
  - **Patch 2 (`find_turn_boundaries`)**: keeps the `<|im_end|>` scan, adds (a) one
    `<|im_start|>`/`<|im_end|>` per declared message, (b) no interleaving of markers,
    (c) the role token after each `<|im_start|>` must equal the role declared in
    `messages`, (d) non-empty content span. Roles come from the message list, never from
    position parity. Fires on: a dropped assistant turn (transcript `[u,a,u,u]` scanned
    against declared alternation — the exact case where parity labelling would silently
    call the third user turn `a_2`); user content containing literal `<|im_start|>` markup;
    and a swapped role declaration.
  - **Throughput (torch fallback, no flash-linear-attention/causal-conv1d):**
    ~5.0 s per batch of 16 generations, near-flat across turn depth (4.88 / 5.06 / 5.28 s
    at mean prompt lengths 29 / 139 / 256 tokens). ~300 new tokens/s aggregate.
    **Whole 480-generation counted run projects to ~152 s (2.5 min), 11.5 GB peak.**
- **Reading:** Generation is not the bottleneck and the kernel install is not needed for
  sizing — even a hypothetical 5x fallback penalty puts the full run near 13 min, against a
  16 h budget. n is compute-free; if n moves it should move for statistical-power reasons,
  not throughput ones. (Human decision — not making it here.)
- **Surprise / anomaly:**
  1. **No `generation_config.json` in the Qwen3.5-4B repo.** The model's default
     `eos_token_id` is `<|endoftext|>` (248044), *not* the chat turn terminator `<|im_end|>`
     (248046), and `pad_token_id` is `None`. Unset, every generated turn runs to the token
     cap and never stops at end-of-turn. `common.generation_kwargs()` now sets both so no
     caller can forget.
  2. **Most turns hit the 100-token cap anyway**: 11/16, 13/16, 16/16 across the three turn
     depths, i.e. assistant turns are being truncated mid-sentence rather than ending
     naturally. Appendix A.4 does say to cap turn length, so this is by design — but the cap
     value is a live decision, and at 100 it is binding on nearly every turn. Flagging, not
     changing.
  3. NNsight keeps 40 bookkeeping forward hooks live on the model, which is why the
     `len(_forward_hooks) == 0` form of the C.4 check could never have held.
- **Next:** Stage 2, pending human go-ahead. `src/00_infra_gate.py` is left frozen as the
  Stage 0 run record; its inline boundary scanner is superseded by `common.find_turn_boundaries`
  and stage scripts should import from `common.py`, not copy from it.

### [2026-09-02 21:50] Stage 0 infra gate — all four checks pass, two load-bearing gotchas found

- **Question:** Does Qwen3.5-4B load on one GPU via HF/nnsight, format a real dialogue, extract
  activations at the right positions, respond to a steering hook during generation, and survive a
  tokenization round-trip? (ROADMAP.md Stage 0 checklist.)
- **Ran:** `src/00_infra_gate.py`. Qwen/Qwen3.5-4B, bf16, GPU 0 (idle at run time; GPU 3 is another
  user's job — did not touch it). Hand-written 4-turn dialogue (S,S,S,N carryover shape, not
  pool-generated). Dummy random-direction steering hook at layer 16/32, norm = 4x mean activation
  norm at that layer. seed=0, temperature=0.7, top_p=0.9, max_new_tokens=80.
- **Result:** All four Stage 0 gate items pass.
  1. Model loads (~5s, ~9.3 GB peak on one L40S).
  2. Chat template formats the 8-message dialogue correctly.
  3. All-layer residual stream extracted via `nnsight.trace`; every one of the 12 boundary
     positions (u_t/eot_u_t/a_t, t=1..4) decodes to the expected token on eyeball (B.3).
  4. Dummy steering hook changes generation: no-hook turn is a coherent, on-topic reply; with-hook
     turn (same seed) is incoherent multilingual token salad. Output changed = True.
  5. Round-trip check passes exactly: `saved_gen_ids == tokenizer(transcript_text).input_ids`
     (329 tokens, zero drift).
- **Reading:** Infra is sound for Stage 2. Neither gotcha below required a design change — both
  were caught by asserts before they could produce a silent indexing bug, exactly per rule 6.
- **Surprise / anomaly (two, both load-bearing for Stage 2 code):**
  1. **Qwen3.5 is not the architecture the roadmap assumed.** It's a natively multimodal
     (image-text-to-text) hybrid linear/full-attention model (`Qwen3_5ForConditionalGeneration`,
     `layer_types` = 3x linear_attention : 1x full_attention, fla-org gated-deltanet, falls back to
     a torch implementation since `flash-linear-attention`/`causal-conv1d` aren't installed — slower
     but correct). Text decoder lives at `model.model.language_model.layers`, 32 layers,
     hidden=2560. Every decoder layer's `forward()` returns a bare `Tensor`, not the
     `(hidden_states, ...)` tuple Appendix B.1's sample code assumes — so extraction code must use
     `.output` directly (nnsight) or `output` directly (raw hook), not `.output[0]`. It's also a
     hybrid-thinking model: `add_generation_prompt=True` opens a `<think>` block by default; Stage 2
     generation code should decide explicitly whether `enable_thinking=False` is wanted (used here
     to keep step 5/6 output short) since thinking traces will otherwise land inside `a_t` capture
     windows and consume the token budget.
  2. **The chat template is not prefix-stable.** It wraps a message in `<think>...</think>` iff that
     message is the *last* assistant turn in the list being rendered (`loop.index0 >
     ns.last_query_index`, computed per-call). So `DIALOGUE[:2]` renders `a_1` with a think-wrapper
     that the full 8-message dialogue does not. The natural first implementation —
     incrementally-truncate the message list, tokenize each prefix, and match it against the full
     sequence to locate turn boundaries — silently gives the wrong index for every assistant turn
     except the last one. Caught immediately by a prefix-equality assert (`full_ids[:len(prefix)]
     == prefix`), which failed loudly at turn 1 instead of producing a plausible-looking wrong
     activation. **Fix used:** tokenize the full dialogue once, scan for `<|im_end|>` token-id
     occurrences directly (one per message, in order) instead of incremental retokenization.
  3. **Smaller nnsight gotcha:** a list comprehension inside `with nn_model.trace():` does not
     propagate its assignment to the enclosing scope (`NameError` on the target name right after the
     `with` block exits) — nnsight's tracer only reliably captures explicit statement-level
     `.append()` calls in a real `for` loop. Also: `NNsight(model)` registers its own bookkeeping
     forward hooks on every module, so "assert zero active hooks" (Appendix C.4, for Stage 3) must
     check the specific hook handle, not `len(module._forward_hooks) == 0`.
- **Next:** Proceed to Stage 1 (literature) / Stage 2 (pipeline + probes) once the human confirms.
  Stage 2's activation-extraction and turn-pool-assembly code should reuse the
  scan-for-`<|im_end|>` boundary method and the bare-tensor `.output` convention from this run.

