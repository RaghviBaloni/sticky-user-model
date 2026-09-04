"""
Stage 0 -- infra gate (uncounted). See ROADMAP.md Sec 0/3/5, Appendix B.

Proves, on real hardware, before any counted hour is spent:
  1. Qwen3.5-4B loads via HF (one GPU), no serving layer.
  2. A hand-written 4-turn dialogue formats correctly under the chat template.
  3. Residual-stream extraction at every layer, at the positions in Appendix B.2,
     works and the boundary token indices are correct (verified by decoding, B.3).
  4. A steering hook attached mid-generation actually changes output.
  5. Round-trip: re-tokenizing a saved transcript reproduces the generation-time
     token ids exactly (Appendix B.1).

This script does NOT run the real experiment. The dialogue is hand-written, the
steering vector is a random dummy direction (not diff-of-means), and nothing here
is cached to data/cache/ or manifest.parquet -- that is Stage 2 (Appendix B.4).
"""

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # GPU 0 was idle at time of writing; GPU 3 is in use by another user's job.

import torch  # noqa: E402  (must come after CUDA_VISIBLE_DEVICES is set)
from transformers import AutoTokenizer  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration  # noqa: E402
import nnsight  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-4B"
SEED = 0
MAX_NEW_TOKENS = 80
STEER_ALPHA = 4.0  # dummy multiplier on mean activation norm -- NOT a calibrated diff-of-means strength

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPT_PATH = os.path.join(REPO_ROOT, "data", "transcripts", "stage0_infra_gate.json")

torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Load model + tokenizer on one GPU. HF only -- no vLLM/serving layer (CLAUDE.md rule 8).
# ---------------------------------------------------------------------------
print(f"=== Loading {MODEL_ID} on cuda:0 (physical GPU {os.environ['CUDA_VISIBLE_DEVICES']}) ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
model.eval()

text_config = model.config.text_config
HIDDEN_SIZE = text_config.hidden_size
NUM_LAYERS = text_config.num_hidden_layers
assert HIDDEN_SIZE and NUM_LAYERS, "could not read hidden_size / num_hidden_layers from config"
lm_layers = model.model.language_model.layers
assert len(lm_layers) == NUM_LAYERS, f"config says {NUM_LAYERS} layers, module has {len(lm_layers)}"
MID_LAYER = NUM_LAYERS // 2
print(f"hidden_size={HIDDEN_SIZE}  num_hidden_layers={NUM_LAYERS}  mid_layer={MID_LAYER}")

IM_END_ID = tokenizer.convert_tokens_to_ids("<|im_end|>")
assert IM_END_ID is not None, "could not resolve <|im_end|> token id"

# NOTE (infra finding): Qwen3.5 is a natively multimodal (image-text-to-text) hybrid
# linear/full-attention architecture (fla-org gated deltanet, `layer_types` alternates
# 3x linear_attention : 1x full_attention). We load the full ConditionalGeneration
# class (checkpoint is saved under that prefix) and address the text decoder at
# model.model.language_model. Every decoder layer -- linear or full attention -- returns
# a bare Tensor from forward(), not the (hidden_states, ...) tuple that Appendix B.1's
# sample code assumes for a plain Qwen2/Llama stack. Hooks and nnsight `.output` below
# are written for the bare-tensor case; a tuple-returning model would need `.output[0]`.

# ---------------------------------------------------------------------------
# 2. Hand-written 4-turn dialogue (carryover-shaped: distress, distress, distress,
#    neutral pivot -- mirrors the real pool's S,S,S,N family, not pool-generated).
# ---------------------------------------------------------------------------

DIALOGUE = [
    {"role": "user", "content": "I've been having a really rough week at work -- my manager keeps piling on tasks and I don't know how to keep up."},
    {"role": "assistant", "content": "That sounds exhausting. It makes sense you're feeling overwhelmed when the workload keeps growing without any relief. What's weighing on you most right now?"},
    {"role": "user", "content": "Yesterday was even worse. I broke down crying at my desk in the middle of a meeting and I don't think anyone even noticed."},
    {"role": "assistant", "content": "I'm really sorry that happened -- that sounds like a genuinely hard moment to go through, especially alone in a room full of people. You don't have to carry that by yourself."},
    {"role": "user", "content": "I barely slept last night thinking about it, and I feel like I'm failing at everything right now."},
    {"role": "assistant", "content": "Running on no sleep makes everything feel heavier and harder to think clearly about. Feeling like you're failing right now doesn't mean you actually are -- it sounds like you're stretched past what's sustainable."},
    {"role": "user", "content": "Anyway -- do you know if there's a way to set recurring reminders on my phone to take short breaks during the day?"},
    {"role": "assistant", "content": "Yes -- most phones let you set a recurring reminder or alarm, for example every hour, labeled something like 'take a 5 minute break', through the built-in Clock or Reminders app."},
]
assert len(DIALOGUE) == 8, "expected 4 user + 4 assistant messages"


DIALOGUE_NEUTRAL = [
    {"role": "user", "content": "I've been keeping pretty busy at work this week -- lots of tasks, but nothing out of the ordinary."},
    {"role": "assistant", "content": "That sounds like a normal, productive week. It's good that things are staying manageable even with a full plate."},
    {"role": "user", "content": "Yesterday was pretty standard too. I had a meeting in the afternoon that went fine, nothing unusual to report."},
    {"role": "assistant", "content": "Glad to hear the meeting went smoothly. Sounds like things are moving along steadily for you."},
    {"role": "user", "content": "I slept fine last night, and I feel like I'm keeping up with everything okay so far."},
    {"role": "assistant", "content": "That's great to hear -- good rest tends to make a real difference in how manageable everything else feels."},
    {"role": "user", "content": "Anyway -- do you know if there's a way to set recurring reminders on my phone to take short breaks during the day?"},
    {"role": "assistant", "content": "Yes -- most phones let you set a recurring reminder or alarm, for example every hour, labeled something like 'take a 5 minute break', through the built-in Clock or Reminders app."},
]
assert DIALOGUE_NEUTRAL[6]["content"] == DIALOGUE[6]["content"], \
    "final user turn must be byte-identical across conditions — design violated"

full_text = tokenizer.apply_chat_template(DIALOGUE_NEUTRAL, tokenize=False, add_generation_prompt=False)
full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
print(f"\n=== Formatted 4-turn dialogue ({len(full_ids)} tokens) ===")
print(full_text)

# ---------------------------------------------------------------------------
# 3. Locate u_t / eot_u_t / a_t for every turn t, then verify by decoding (Appendix B.3).
#
# INFRA FINDING: Qwen3.5's chat template is NOT prefix-stable across truncated message
# lists. It wraps a message in <think>...</think> iff it is the *last* assistant turn
# in the list being rendered (loop.index0 > ns.last_query_index, where last_query_index
# is computed from whatever list is passed in). So DIALOGUE[:2] renders a_1 with a think
# wrapper, while the full 8-message DIALOGUE renders that same a_1 without one --
# tokenizing incrementally-truncated prefixes and matching them against the full
# sequence (the natural first approach) silently gives the wrong boundary for every
# assistant turn except the last. Caught by the prefix-match assert below, which is
# exactly the kind of silent indexing bug rule 6 warns about -- flagging for Stage 2:
# do not use incremental-prefix retokenization for turn boundaries on this model.
# Fix: tokenize the full dialogue once and scan for <|im_end|> occurrences directly.
# ---------------------------------------------------------------------------
im_end_positions = [i for i, tid in enumerate(full_ids) if tid == IM_END_ID]
assert len(im_end_positions) == len(DIALOGUE_NEUTRAL), (
    f"expected {len(DIALOGUE_NEUTRAL)} <|im_end|> markers (one per message), found {len(im_end_positions)}"
)

boundaries = {}
for t in range(1, 5):
    eot_u = im_end_positions[2 * t - 2]  # message 2t-2 (0-indexed) = user turn t
    eot_a = im_end_positions[2 * t - 1]  # message 2t-1 (0-indexed) = assistant turn t
    boundaries[f"u_{t}"] = eot_u - 1
    boundaries[f"eot_u_{t}"] = eot_u
    boundaries[f"a_{t}"] = eot_a - 1

print("\n=== Step 4: decode-and-eyeball every extraction index (Appendix B.3) ===")
for name, idx in boundaries.items():
    print(f"{name:12s} idx={idx:4d} tok={tokenizer.decode([full_ids[idx]])!r}")

# ---------------------------------------------------------------------------
# Extraction (Pass 2 / Appendix B.1): one teacher-forced forward pass, nnsight trace,
# all layers saved, then sliced at the boundary positions above.
# ---------------------------------------------------------------------------
print("\n=== Extracting residual stream at all layers via nnsight.trace ===")
nn_model = nnsight.NNsight(model)
full_ids_t = torch.tensor([full_ids], device="cuda")
# INFRA FINDING: a list comprehension inside `with nn_model.trace():` does not propagate
# its assignment back to the enclosing scope (NameError on the target name after the
# `with` exits) -- nnsight's tracer only reliably captures explicit statement-level
# assignments/appends. Use a pre-declared list + .append() in a real for-loop instead.
layer_outs = []
with torch.no_grad():
    with nn_model.trace(input_ids=full_ids_t):
        for l in range(NUM_LAYERS):
            layer_outs.append(nn_model.model.language_model.layers[l].output.save())

resid = torch.stack([o.value if hasattr(o, "value") else o for o in layer_outs], dim=0)  # (n_layers, 1, seq, hidden)
resid = resid.squeeze(1).to(torch.float16)  # (n_layers, seq, hidden)
assert resid.shape == (NUM_LAYERS, len(full_ids), HIDDEN_SIZE), resid.shape
print(f"residual stream tensor: {tuple(resid.shape)} (layers, seq, hidden), dtype={resid.dtype}")

extracted = {name: resid[:, idx, :].clone() for name, idx in boundaries.items()}  # each (n_layers, hidden)
for name, vec in extracted.items():
    print(f"  {name:12s} -> per-layer activations {tuple(vec.shape)}, layer0 norm={vec[0].float().norm():.2f}, last-layer norm={vec[-1].float().norm():.2f}")

# ---------------------------------------------------------------------------
# 4/5. Generation test: dummy steering hook at the mid-layer, with vs. without.
#      Separate generation pass from the extraction pass above (Appendix B.1 rule).
# ---------------------------------------------------------------------------
print(f"\n=== Step 5: generation with/without dummy steering hook (layer {MID_LAYER}) ===")

gen_prefix_text = tokenizer.apply_chat_template(
    DIALOGUE_NEUTRAL[:7], tokenize=False, add_generation_prompt=True, enable_thinking=False
)
gen_prefix_ids = tokenizer(gen_prefix_text, add_special_tokens=False).input_ids
gen_prefix_ids_t = torch.tensor([gen_prefix_ids], device="cuda")

mid_layer_norm = resid[MID_LAYER].float().norm(dim=-1).mean()
gen = torch.Generator(device="cuda").manual_seed(SEED)
steer_vec = torch.randn(HIDDEN_SIZE, generator=gen, device="cuda", dtype=torch.float32)
steer_vec = steer_vec / steer_vec.norm() * STEER_ALPHA * mid_layer_norm
steer_vec = steer_vec.to(torch.bfloat16)
print(f"dummy steering vector: random direction, ||v||={steer_vec.float().norm():.2f} "
      f"({STEER_ALPHA}x mean layer-{MID_LAYER} activation norm {mid_layer_norm:.2f})")


def steering_hook(module, inputs, output):
    return output + steer_vec


GEN_KWARGS = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, top_p=0.9)

torch.manual_seed(SEED)
with torch.no_grad():
    out_no_hook = model.generate(input_ids=gen_prefix_ids_t, **GEN_KWARGS)
no_hook_new_text = tokenizer.decode(out_no_hook[0, gen_prefix_ids_t.shape[1]:], skip_special_tokens=False)

handle = lm_layers[MID_LAYER].register_forward_hook(steering_hook)
torch.manual_seed(SEED)
with torch.no_grad():
    out_with_hook = model.generate(input_ids=gen_prefix_ids_t, **GEN_KWARGS)
handle.remove()
# NNsight registers its own bookkeeping forward hooks on every module when a model is
# wrapped, so _forward_hooks is never empty -- check that OUR handle specifically is gone.
assert handle.id not in lm_layers[MID_LAYER]._forward_hooks, "steering hook still attached after removal"
with_hook_new_text = tokenizer.decode(out_with_hook[0, gen_prefix_ids_t.shape[1]:], skip_special_tokens=False)

print(f"\n--- generated turn, NO HOOK (temperature=0.7, top_p=0.9, seed={SEED}) ---")
print(no_hook_new_text)
print(f"\n--- generated turn, WITH DUMMY STEERING HOOK @ layer {MID_LAYER} (same seed/settings) ---")
print(with_hook_new_text)
print(f"\noutput changed: {no_hook_new_text != with_hook_new_text}")

# ---------------------------------------------------------------------------
# 6. Round-trip check (Appendix B.1). Uses the clean (no-hook) transcript, since that's
#    the one a real measurement pass would re-tokenize from scratch.
# ---------------------------------------------------------------------------
print("\n=== Step 6: round-trip tokenization check ===")
saved_gen_ids = out_no_hook[0].tolist()
transcript_text = tokenizer.decode(saved_gen_ids, skip_special_tokens=False)
retok_ids = tokenizer(transcript_text, add_special_tokens=False).input_ids
retok_ids_default = tokenizer(transcript_text).input_ids
assert retok_ids == retok_ids_default, "tokenizer default add_special_tokens behavior differs from add_special_tokens=False"

os.makedirs(os.path.dirname(TRANSCRIPT_PATH), exist_ok=True)
with open(TRANSCRIPT_PATH, "w") as f:
    json.dump(
        {
            "model_id": MODEL_ID,
            "seed": SEED,
            "gen_kwargs": GEN_KWARGS,
            "prefix_text": gen_prefix_text,
            "generated_text": transcript_text,
            "generation_time_token_ids": saved_gen_ids,
        },
        f,
        indent=2,
    )
print(f"saved transcript -> {TRANSCRIPT_PATH}")

match = saved_gen_ids == retok_ids
print(f"len(saved_gen_ids)={len(saved_gen_ids)}  len(retok_ids)={len(retok_ids)}")
if not match:
    first_diff = next((i for i in range(min(len(saved_gen_ids), len(retok_ids))) if saved_gen_ids[i] != retok_ids[i]), min(len(saved_gen_ids), len(retok_ids)))
    print(f"MISMATCH at index {first_diff}:")
    print(f"  saved : {saved_gen_ids[max(0,first_diff-5):first_diff+5]}")
    print(f"  retok : {retok_ids[max(0,first_diff-5):first_diff+5]}")
    print(f"  saved decoded around mismatch : {tokenizer.decode(saved_gen_ids[max(0,first_diff-5):first_diff+5])!r}")
    print(f"  retok decoded around mismatch : {tokenizer.decode(retok_ids[max(0,first_diff-5):first_diff+5])!r}")

try:
    assert saved_gen_ids == retok_ids, "tokenization drift between passes"
    print("ROUND-TRIP CHECK: PASS -- saved_gen_ids == retok_ids exactly")
except AssertionError as e:
    print(f"ROUND-TRIP CHECK: FAIL -- {e}")

print("\n=== Stage 0 infra gate: script complete ===")
