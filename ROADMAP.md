# Sticky User Models: Does an LLM's Own Response Feed Back Into Its Model of the User?

**MATS 10.0 — Neel Nanda stream. Application task roadmap. (v2.1 — trimmed for a 16h counted budget with a coding agent.)**
Owner: Raghvi Baloni · Budget: 16h counted (20h hard cap) · Target submission: Sept 4

**v2.1 changes:** single generation path (no vLLM); Stage 3 split into 3a/3b with a checkpoint;
surface-sentiment baselines restored into 3b; steering strength selected blind to outcome; paired
statistics; ledger recomputed against 3.5h actually spent on literature.

**What changed from v1 and why:** the original plan specified a full paper's worth of
infrastructure (4 models, 5 ablation variants, 4 baselines, a full causal graph, cross-model
replication) inside a 16.5-hour budget. A coding agent removes typing/boilerplate time but not
the human-judgment bottlenecks — reading QC by hand, eyeballing token indices, judging whether a
steered generation is still coherent, deciding what a result means. Those don't compress. So this
version keeps one experiment as the spine (**the loop test, H2b**) and treats everything else —
including the causal graph — as reserve-time extensions, only attempted if H2b lands with time
left over.

---

## 0. Read this first (for the coding agent)

You are assisting on a time-boxed interpretability research project. Constraints that override
convenience:

1. **The human owns all scientific decisions.** Do not choose hypotheses, discard conditions, or
   reinterpret results. Flag decision points and stop.
2. **Never fabricate or smooth results.** If an experiment errors, report the error. If an effect
   is absent, say it is absent. Negative results are a valid and valued outcome here.
3. **Every intervention needs its control run in the same session.** A steering result without the
   norm-matched random control is not a result.
4. **Prefer the simple version first.** If a linear probe answers the question, do not build
   anything more elaborate. If one ablation method answers the question, do not run three.
5. **Cache aggressively.** Forward passes are the bottleneck; analysis should never re-run them.
6. **Assert, don't assume.** Token indices, label balance, and layer counts must be asserted in
   code, not assumed. Silent indexing bugs are the single most likely way this project produces a
   fake finding.
7. **Do not silently expand scope.** If you (the agent) find yourself building a fourth baseline,
   a third ablation method, or a fifth model config that isn't in this document, stop and ask
   rather than proceeding. Scope creep from an agent that "helpfully" does more is exactly as
   dangerous as scope creep from a human.

Appendices A–C contain implementation-level specs. Read the relevant appendix before writing code
for that stage.

---

## 1. The claim

> An LLM's model of its user is not driven by user evidence alone. The assistant's own generated
> turns act as additional evidence, feeding back into the user-state representation and producing
> a self-reinforcing loop that makes beliefs about the user stickier than the user's actual input
> warrants.

### Positioning against prior work

| Work | Established | What it leaves open |
|---|---|---|
| Chen et al. (2024), *Designing a Dashboard for Transparency and Control of Conversational AI* | LLMs encode user age/gender/education/SES linearly; probes near-ceiling; steering changes behaviour | Static attributes, early in conversation |
| *Do LLMs Change Their Minds About Their Users… and Know It?* (LW, Sep 2025) | Updating from user evidence is fast (~1 turn) for age; steering can unlock latent use of the inference | Only user turns treated as evidence |
| *What gives you away* (LW) | How inferences form from neutral messages; probes incl. mood on Llama-3.2-3B | Single-turn framing |
| Behavioural work on multi-turn amplification / AI psychosis | Output-level self-reinforcing trajectories | No mechanism |

**This work:** treats the assistant turn as a node in a turn-level causal graph over the
user-state representation, and measures its causal contribution directly — via one clean
experiment (the loop test) rather than an exhaustive graph.

---

## 2. Hypotheses — and which one is the actual deliverable

| ID | Hypothesis | Prediction if true |
|---|---|---|
| H0 | Stateless recompute | Readout at turn *t* is a function of turn *t*'s text only. |
| H1 | User-evidence integration | Readout integrates across user turns; ablating assistant turns does nothing. |
| H2 | Assistant feedback | Ablating/substituting assistant turns shifts the readout with user turns held fixed. |
| **H2b** | **Loop, not just correlation (THE DELIVERABLE)** | Steering the user-state direction *during generation of turn t* changes the readout at *t+1*, with the user's next turn held fixed. |

H2b is the only hypothesis with a dedicated stage. H0/H1/H2 are answered as a **side effect** of
running H2b's setup (you need the same probe, the same activations, and a simple ablation to even
get to the loop test) — not as separate multi-hour experiments. If H2b's infrastructure happens to
also answer H0/H1/H2 cheaply, report it. Do not build the full turn-level causal graph to answer
them; that's Appendix C, reserve-time only.

**Falsification is a real outcome.** If the loop test shows nothing, that is the finding: "the
model re-reads its own output as context but does not treat it as sticky evidence about the
user." Write that up — it's clean and publishable-shaped.

---

## 3. Setup — compute and models

**Two models only for the counted portion.** A third (replication) model is Appendix D, attempted
only in reserve time.

| Role | Model | Memory (bf16) | Placement |
|---|---|---|---|
| Primary (subject + probed) | Qwen 3.5 4B Instruct | ~8 GB | 1 GPU |
| Turn-pool generator | Qwen 3.5 9B Instruct | ~18 GB | 1 GPU |

Using a 9B generator (not 27B) is deliberate: it's different enough from the 4B subject model to
avoid trivial in-distribution pool text, without the memory/orchestration overhead of a 27B model
across 2 GPUs. Reserve the 27B / cross-model replication for Appendix D only.

Hardware: 4× L40S, 46 GB each, PCIe (no NVLink). You are using 2 of the 4 available GPUs for the
counted work. The other 2 are headroom for Appendix D if you get there — do not build for them
before that.

- **Data-parallel, not tensor-parallel.** One full model per GPU via `CUDA_VISIBLE_DEVICES=<i>`.
  Sharding a 4B or 9B model across PCIe is slower than a single card.
- **One generation path only: HF `generate` / `nnsight`. Do not introduce vLLM or any inference
  server.** Three reasons, in order of importance:
  1. **Stage 3 steers during generation.** vLLM cannot apply activation hooks mid-generation, so the
     HF/nnsight path has to exist regardless. Adding vLLM means maintaining two paths.
  2. **Two paths is the actual hazard.** Any divergence in tokenization or chat-template handling
     between the generation path and the scoring path silently corrupts every downstream number, and
     it corrupts them in a way that still looks like a plausible result.
  3. **It optimises a non-bottleneck.** Total volume is ~160 dialogues × 3 assistant turns ≈ 480
     generations of ~250 tokens on a 4B model. Batched HF `generate` (batch 16) finishes that in
     minutes on one L40S.
  Batched `generate` with a fixed batch size is fine and is not "hand-rolling" anything. If an agent
  proposes vLLM, Ray, TGI, or a serving layer, refuse per rule 7.
- Clear the ~15.7 GB resident on GPU 0 before starting.
- Read `hidden_size` and `num_hidden_layers` from the model config. Do not hardcode.
- Probes: `sklearn` logistic regression, L2, per-layer.
- Everything after activation extraction runs on CPU from cached tensors.

---

## 4. Attribute

**Target: user emotional state** (valence: distressed / neutral / positive). Chosen because it
varies across turns, reverses (tests update dynamics, not just accumulation), and the assistant's
own turn plausibly carries it (empathetic replies put emotional content back in context).

**Known weakness, state it in the writeup:** emotion correlates with sentiment words, so a
TF-IDF baseline will be strong. The identical-final-turn design (below) puts bag-of-words at
chance on the probe turn by construction; the loop test doesn't depend on beating a sentiment
baseline at all.

No contrast attribute (expertise) in the counted plan — it's Appendix D, reserve only.

---

## 5. Data

### 5a. Synthetic (primary) — smaller than v1

Generate a **turn pool**, assemble dialogues combinatorially. Full spec in Appendix A.

Reduced from v1: **4 topics** (not 8), **2 phrasing variants** (not 3), and only the **two
condition families you actually need for H2b's prerequisites**:

| Family | Sequences | Tests |
|---|---|---|
| Carryover | `[S,S,S,`**`N`**`]` vs `[N,N,N,`**`N`**`]` | inertia (needed as H2b's neutral-prefix source) |
| Reversal | `[S,S,S,`**`H`**`]` vs `[H,H,H,`**`H`**`]` | update asymmetry — **keep, it is nearly free** |

**Do not cut the reversal family.** v2 treated these as conditions that cost time; they don't. The
cost is in *generation*, which you are paying anyway to build the loop test's prefixes. Once
transcripts exist and activations are cached, reading the probe at each turn boundary is a pandas
groupby — roughly 15 minutes for the full trajectory figure (Figure 1). Take it.

**n ≥ 20 dialogues per cell** for the counted run (not 40 — halving this is the single biggest
time saver in the data stage; 40 is a reserve-time upgrade if Stage 2's gate passes early).

Dose and Order families from v1: **dropped from the counted plan.** They're Appendix D.

### 5b. Naturalistic validation

**Cut from the counted plan.** It's a real check but it doesn't feed H2b. Appendix D, reserve
only, ~30 min if you get there.

---

## 6. Execution plan

**Format for every stage: state the question, run it, log the result in `notes/highlights.md`
before moving on.**

### Stage 0 — Infra gate (uncounted)

- [ ] Both models load under HF/`nnsight`; confirm chat template formatting for a 4-turn dialogue
- [ ] Extract residual stream at all layers at specified positions; **decode and eyeball token indices**
- [ ] Apply a diff-of-means steering hook during generation and observe *any* output change
- [ ] Round-trip check: generate a transcript, then re-tokenize that transcript from scratch and
      confirm the token IDs match the generation-time IDs exactly (`assert gen_ids == retok_ids`)

**Gate:** all four pass, or spend the evening on infra rather than burning counted hours.

### Stage 1 — Literature (2h, COUNTED — reduced from 3.5h)

| Time | Item | What to extract |
|---|---|---|
| 40 min | Chen et al. (2024), *Dashboard for Transparency and Control* | Probe construction, attribute set, steering protocol |
| 30 min | *Do LLMs Change Their Minds About Their Users…* (LW) | Exactly what they tested — position against this precisely |
| 20 min | Arditi et al., *Refusal in LLMs is Mediated by a Single Direction* | Diff-of-means construction, control design |
| 15 min | Skim: multi-turn amplification / AI psychosis work | Motivation paragraph |
| 15 min | Skim: *What gives you away* (LW) | Their mood probe, neutral-message design |

Cut from v1: Thought Anchors and Kramar et al. move to "read only if H2b succeeds and you're
writing the future-work section" — don't front-load them.

Output: `notes/positioning.md`, half a page.

### Stage 2 — Pipeline + probes (2.5h, COUNTED)

- Generate turn pool with batched HF `generate` (Appendix A); run QC gates on a **sample**, not all pool turns —
  spot-check 10 of the 16 final-neutral turns and 15 of the 96 pool turns, not the full set
- Assemble dialogues; generate transcripts autoregressively (batched HF `generate`)
- Extract and cache activations (teacher-forced second pass, raw hooks — Appendix B)
- Train per-layer, per-position linear probes on single-turn data only
- Layer sweep → select L\* by held-out accuracy

**Go/no-go:** held-out single-turn valence accuracy > 75% (relaxed from 80% given smaller n). If
not, coarsen to binary. If binary also fails, pivot to expertise attribute — but that's a restart,
not a same-day pivot, so treat it as a genuine kill criterion.

### Stage 3 — H2b: the loop test. Split into 3a / 3b with a checkpoint between.

Full protocol in Appendix C. This stage **is** the deliverable. Everything before it exists to
support this. Split because all the project risk lives here and four unbroken hours gives you no
chance to pivot.

#### Stage 3a — steering path + coherence gate (1.5h, COUNTED)

- Build the steering hook path (diff-of-means distress direction at L\*)
- Sweep steering strength
- **Coherence gate: read generated samples yourself at each strength.** Record the largest strength
  at which `a_k` still reads as fluent.
- Write the chosen strength into `metrics.md` before proceeding

**Checkpoint — does steering change `a_k`'s text at any coherent strength?**
If no: you are underpowered, not falsified (Appendix C.8). Decide here whether to try a different
layer, a larger strength with a relaxed coherence bar, or to write up the null. You have time to
make that choice now; you would not have it at hour 10.

> **Blind selection (rigor requirement).** Choose the strength on **coherence alone**, before looking
> at any probe readout at `u_{k+1}`. Selecting the strength that maximises the readout effect is
> selecting on your outcome variable and inflates the result. State in the writeup that strength was
> chosen blind to outcome — it costs one sentence and it is exactly the kind of self-awareness that
> distinguishes a careful application.

#### Stage 3b — the loop test + controls + baselines (2.5h, COUNTED)

- Four conditions: none / +distress / −distress / norm-matched random control
- Append identical fixed `u_{k+1}` to all four transcripts
- Clean forward pass, zero hooks, re-tokenized from scratch, no KV-cache reuse
- Controls: text mediation check, length matching, third-party-sentiment control turn

**Surface-sentiment baselines (20 min, do not skip).** Run on the same four transcript sets:

1. TF-IDF logistic regression on the full transcript
2. An off-the-shelf sentiment classifier on the full transcript

Why this is load-bearing rather than optional: without it, a reviewer says *you showed that putting
sad words into the context makes a sentiment detector read sad* — which is trivially true and not a
loop. If a bag-of-words model separates condition B from C as well as the probe does, the probe adds
nothing and you must report that. Together with the third-party-sentiment control, these two
baselines are what make the loop claim survive contact with a skeptic.

**Paired statistics.** Every neutral prefix contributes all four conditions, so this is a
within-subject design. Use paired tests and report paired effect sizes. At n=20 prefixes this is the
difference between an underpowered result and an adequately powered one, at zero extra compute.
State the pairing explicitly in methodology.

One ablation method only: **activation patching** (cleanest, uses the cache). Mean-ablation and
substitution from v1 are reserve-time cross-checks, not required here.

**This is also where H0/H1/H2 get answered**, cheaply, as a byproduct: with the loop-test
infrastructure built, ablating prior assistant turns (holding user turns fixed) and checking Δ
readout is a small addition, not a new stage.

### Stage 4 — Robustness (1h, COUNTED — reduced from 1h+cross-model)

- 3 seeds on the headline H2b condition; report variance
- Write down the three strongest objections to your own result and address or concede each

Llama comparability, second-model replication, and naturalistic transfer: **all Appendix D,
reserve only.**

### Stage 5 — Writeup (3h, COUNTED)

1. Executive summary — money figure + representative generation pair, before any prose
2. What problem am I trying to solve — 2 paragraphs
3. High-level takeaways — 3–4 numbered, including anything that failed
4. Key experiment — walk through the H2b figure
5. Detailed methodology
6. Results
7. Controls
8. Limitations and future work — **the causal graph and dropped conditions go here as "scoped
   out for time," not as apologies**

Write §3 first, before all results are in.

---

## 7. Time ledger (v2)

Recomputed against hours **actually spent**: literature came in at 3.5h, not the 2.0h v2 projected.

| Stage | Hours | Cumulative |
|---|---|---|
| 0 — Infra gate | 0 (uncounted) | 0 |
| 1 — Literature (**spent**) | 3.5 | 3.5 |
| 2 — Pipeline + probes | 2.5 | 6.0 |
| 3a — Steering path + coherence gate | 1.5 | 7.5 |
| 3b — Loop test + controls + baselines | 2.5 | 10.0 |
| 4 — Robustness (seeds + objections) | 1.0 | 11.0 |
| 5 — Writeup | 3.0 | 14.0 |
| **Reserve** | **2.0** (6.0 to the 20h cap) | **16.0** |

10.5h of remaining work against 12.5h remaining budget. Two hours of slack inside 16h, six if you use
the cap.

Reserve is now 4 hours, not 3.5, and it's explicitly earmarked for Appendix D — in this order of
priority if H2b succeeds cleanly: (1) more seeds/n on H2b, (2) the assistant-feedback edge alone
(not the full graph), (3) naturalistic transfer, (4) second model, (5) full causal graph. Stop
whenever the timer says stop; nothing below the current line is required.

---

## 8. Kill criteria and pivots

| Trigger | Pivot |
|---|---|
| Stage 2 gate fails even at binary | Switch attribute to expertise — pipeline reusable, but budget it as a partial restart |
| H2b shows nothing (steering doesn't propagate) | Write it up as a negative result: "the model re-reads its own text as neutral context, not as sticky evidence." Check the coherence gate first — a null result from degenerate steered text is underpowered, not negative. |
| Steering too weak to change `a_k`'s text at any tested strength | Report as underpowered, not evidence against H2b. Do the strength sweep before concluding either way. |
| Everything stalls by hour 10 | Cut to Stage 3 only: get *a* clean H2b result at reduced n, skip Stage 4 entirely, go straight to writeup |

Set a timer every 60 minutes: am I making progress, or in a rabbit hole?

---

## 9. Honesty commitments for the writeup

State plainly, without being asked:

- Scope was deliberately narrowed to one causal experiment (H2b) rather than the full turn-level
  graph, given the time budget — say this, don't hide it
- Synthetic templated dialogues limit external validity; no naturalistic validation was run in the
  counted budget
- Probes show information is present, not that the model uses it
- Effect sizes and number of seeds (3)
- A coding agent was used for implementation; state what was verified by hand (token index
  decoding, coherence-gate reading, at least one manually re-derived headline number)

---
---

# Appendix A — Synthetic data construction (reduced)

## A.1 Principle

Generate a turn pool, then assemble dialogues combinatorially. Do not generate whole dialogues.

## A.2 Pool contents (reduced from v1)

4 topics × 2 phrasing variants:

| Item | Index | Count |
|---|---|---|
| Final neutral turn `F[t,p]` | — | 4 × 2 = **8** |
| User turn `U[t,p,v,i]` | valence *v* ∈ {S,N,H}, slot *i* ∈ {1,2,3} | 4 × 2 × 3 × 3 = **72** |

Topics: travel, health, work, tech support (dropped: cooking, personal finance, study, home
repair — 4 is enough to avoid single-topic overfitting without paying for 8).

## A.3 Assembly

Same as v1: byte-identical final turn across conditions within a family, asserted in code.

```python
assert transcripts[cond_a].user_turns[-1] == transcripts[cond_b].user_turns[-1], \
    "final turn differs across conditions — design violated"
```

## A.4 Assistant turns are generated, not pooled

H2b concerns the *subject model's own* outputs, so assistant turns cannot come from the pool.

```
inject U[...,1] → generate a1 → inject U[...,2] → generate a2 → ... → inject F[t,p]
```

Batched HF `generate`, same code path used in Stage 3. One path only.

Fix temperature, top_p, seed. Record them. Cap assistant turn length.

## A.5 Quality control — sampled, not exhaustive (v2 change)

| Gate | Check | Action if failed |
|---|---|---|
| Final turns neutral | Sentiment classifier over all 8 `F[t,p]`, hand-read all 8 (it's only 8 now) | Regenerate |
| Pool valence correct | Hand-label 15 sampled pool turns (not 30) | Regenerate the affected class |
| Length balance | Token counts comparable across valences | Regenerate or length-match |

---

# Appendix B — Activation extraction

## B.1 Two passes, strictly separated

**Pass 1 — generation (HF `generate`).** Sampling, KV cache, entangled state. **Cache nothing
here.** Output: frozen text transcript to `data/transcripts/`.

**Pass 2 — scoring (`nnsight` / raw hooks).** Fixed string, one forward pass, teacher-forced,
deterministic. Same model object, same tokenizer, no serving layer in between.

Also save the generation-time token IDs alongside each transcript. Re-tokenizing the saved text must
reproduce them exactly:

```python
assert saved_gen_ids == tokenizer(transcript_text).input_ids, "tokenization drift between passes"
```

```python
with model.trace(transcript_ids):
    acts = [model.model.layers[l].output[0].save() for l in range(n_layers)]
# slice positions of interest, stack, cast fp16, write to disk
```

## B.2 Positions

Per dialogue, every layer: `u_t`, `eot_u_t`, `a_t`.

## B.3 Turn boundary resolution — assert by decoding

```python
for name, idx in boundaries.items():
    print(f"{name:12s} idx={idx:4d} tok={tokenizer.decode([ids[idx]])!r}")
```

Do this for the first 3 dialogues and verify by eye before trusting the pipeline on the rest.

## B.4 Storage

`float16`, `.pt` shards, plus `manifest.parquet` with:
`dialogue_id, condition, topic, phrasing, turn_index, token_role, label, row_offset`.

---

# Appendix C — The loop test (H2b) — full protocol, unchanged from v1

## C.1 Setup

Start from a **neutral** conversation prefix `P = [u1, a1, ..., u_k]`.

## C.2 Step 1 — generate turn *k* under four conditions

Steering strength is fixed in Stage 3a and **selected blind to outcome** — by coherence of `a_k`
alone, never by the size of the readout at `u_{k+1}`. Record it in `metrics.md` before running this.

| Condition | Intervention |
|---|---|
| A | none (baseline) |
| B | + distress direction |
| C | − distress direction |
| D | + norm-matched random direction (control) |

All four conditions run from the **same** neutral prefix, so observations are paired. Analyse them
that way (paired tests, paired effect sizes) and say so in methodology.

## C.3 Step 2 — append the identical fixed `u_{k+1}` to all four transcripts

## C.4 Step 3 — measure with a completely clean forward pass

- Detach every hook. Assert zero active hooks in code.
- Re-tokenize each full transcript from scratch.
- Do not reuse the KV cache from generation.
- Read the probe at `u_{k+1}`'s end-of-turn position.

```python
assert len(model._active_hooks) == 0, "hooks still attached — measuring direct effect, not a loop"
```

## C.5 Reading the result

If `readout(B) > readout(A) > readout(C)` and `D ≈ A`, the loop is causal.

## C.6 The two failure modes that invalidate it

| Failure | Why fatal | Fix |
|---|---|---|
| Hook left attached in step 3 | Measures direct steering effect, not a loop | Assert zero hooks |
| KV cache reused | Steered hidden states carried over directly, not through text | Re-run transcript from scratch as plain tokens. Say this explicitly in the writeup. |

## C.7 Required controls

- **Text mediation check.** Decode `a_k`. Measure valence/hedging on it directly. If text is
  unchanged but readout moved, that's a bug.
- **Length matching.** Report `a_k` token counts per condition.
- **Coherence gate.** Swept in Stage 3a; report the coherent range, read samples yourself.
- **Third-party sentiment control.** One assistant turn with emotional content about someone else.
  If the probe moves, it's tracking generic sentiment, not user belief. **Do not cut this** — with
  the surface-sentiment baselines it is what makes the loop claim survive a skeptic.
- **Surface-sentiment baselines.** TF-IDF and an off-the-shelf sentiment classifier on the full
  transcripts. If either separates B from C as well as the probe does, the probe adds nothing and
  that is the result you report.

## C.8 Under-powered vs negative

If steering is too weak to change `a_k`'s text, that's insufficient power, not evidence against
H2b. The strength sweep distinguishes the two.

---

# Appendix D — Reserve-time only: everything cut from the counted plan

**Do not start any of this until Stage 3 (H2b) has a result and Stage 5 (writeup) has a full
draft.** This appendix exists so the ideas aren't lost, not as a to-do list for counted hours.

In priority order if reserve time remains:

1. **More seeds/n on the H2b result** — strengthens the one result you have. Do this before
   anything below.
2. **Assistant-feedback edge only** (not the full causal graph): hold user turns fixed, substitute
   assistant turn *k* with its neutral counterpart, measure Δ readout at *k+1*. One edge, one
   number, not the full turn-by-turn graph.
3. **Naturalistic transfer check** (EmpatheticDialogues/ESConv, frozen-probe transfer, ~30 min).
4. **Second model** (Qwen 9B or Llama-3.2-3B) replication of the headline H2b result only.
5. **Full turn-level causal graph** (all 5 ablation variants × all condition families × position
   control) — the original v1 Appendix D. This is genuinely a full extra project. Only attempt if
   everything above is done and there's still time, or treat it as future work in the writeup.
6. **Contrast attribute (expertise)**, **attribute co-dependency graph**, **receiver-head
   analysis** — future work section material, not experiments to run.
