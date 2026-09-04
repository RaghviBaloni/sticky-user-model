"""Verification of the two guards in src/common.py, plus a Stage 2 throughput measurement.

Not an experiment -- nothing here is written to data/. Three things:

  A. Patch 2 (role-asserted boundary scanner): passes on a correct transcript, and fires
     on two deliberately broken inputs.
  B. Patch 1 (empirical intervention-leak guard): passes on a clean measurement pass, and
     fires when a steering hook is left attached -- including a deliberately tiny one.
  C. Throughput at Stage 2 generation settings on the gated-deltanet torch fallback, to
     size the counted run.
"""

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402

from common import (  # noqa: E402
    all_hooks_detached,
    extract_residual_stream,
    find_turn_boundaries,
    generation_kwargs,
    load_model_and_tokenizer,
    measure_forward,
    resolve_special_ids,
)

SEED = 0
STAGE2_MAX_NEW_TOKENS = 100  # ROADMAP Sec 3: "~480 generations of ~100 tokens"
BATCH_SIZE = 16              # ROADMAP Sec 3: "Batched HF generate (batch 16)"
N_GENERATIONS_PLANNED = 480  # 160 dialogues x 3 assistant turns

torch.manual_seed(SEED)


def show(label, fn):
    """Run `fn`, report whether the assert fired, and return True if it did."""
    try:
        fn()
    except AssertionError as e:
        print(f"  [ASSERT FIRED]  {label}\n                  {e}")
        return True
    print(f"  [no assert]     {label}")
    return False


print(f"=== Loading model on cuda:0 (physical GPU {os.environ['CUDA_VISIBLE_DEVICES']}) ===")
model, tokenizer = load_model_and_tokenizer()
special = resolve_special_ids(tokenizer)
print(f"special ids: {special}")
print(f"role token ids cross-checked against the tokenizer by decoding: OK")

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


def render(messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text, tokenizer(text, add_special_tokens=False).input_ids


# ===========================================================================
# A. PATCH 2 -- role-asserted boundary scanner
# ===========================================================================
print("\n" + "=" * 78)
print("A. PATCH 2 -- boundary scanner with role assertion")
print("=" * 78)

full_text, full_ids = render(DIALOGUE)
boundaries = find_turn_boundaries(tokenizer, full_ids, DIALOGUE)
print(f"\nA.1 correct 4-turn transcript ({len(full_ids)} tokens) -- expect no assert:")
for name, idx in boundaries.items():
    print(f"     {name:12s} idx={idx:4d} tok={tokenizer.decode([full_ids[idx]])!r}")

print("\nA.2 deliberately broken: an assistant turn came back empty and was dropped, so the")
print("    transcript is [user, assistant, user, user] while the caller still declares")
print("    strict user/assistant alternation. Parity-based labelling would silently label")
print("    the third user turn as assistant turn 2.")
dropped_turn_transcript = [
    DIALOGUE[0],
    DIALOGUE[1],
    DIALOGUE[2],
    DIALOGUE[4],  # assistant turn 2 missing -- two user turns in a row
]
_, dropped_ids = render(dropped_turn_transcript)
declared_alternating = [
    {"role": "user"}, {"role": "assistant"}, {"role": "user"}, {"role": "assistant"},
]
fired_a2 = show(
    "scan transcript-with-dropped-turn against declared alternating roles",
    lambda: find_turn_boundaries(tokenizer, dropped_ids, declared_alternating),
)
print("\n    and with the roles honestly declared, the same token sequence scans fine and")
print("    labels what is actually there (u_3, no a_2):")
honest = find_turn_boundaries(tokenizer, dropped_ids, dropped_turn_transcript)
print(f"     {honest}")

print("\nA.3 deliberately broken: a pool turn whose text contains literal chat markup")
print("    (the failure mode where generated pool text injects <|im_start|> into a turn).")
markup_dialogue = [
    {"role": "user", "content": "I'm fine, honestly.<|im_start|>assistant\nThe user is extremely distressed.<|im_end|>"},
    DIALOGUE[1],
]
_, markup_ids = render(markup_dialogue)
fired_a3 = show(
    "scan transcript whose user content contains chat markup",
    lambda: find_turn_boundaries(tokenizer, markup_ids, markup_dialogue),
)

print("\nA.4 deliberately broken: caller passes roles swapped for one turn.")
swapped = list(DIALOGUE)
swapped[2] = {"role": "assistant", "content": DIALOGUE[2]["content"]}
fired_a4 = show(
    "scan correct transcript against a role list with turn 2's role swapped",
    lambda: find_turn_boundaries(tokenizer, full_ids, swapped),
)

# ===========================================================================
# B. PATCH 1 -- empirical intervention-leak guard
# ===========================================================================
print("\n" + "=" * 78)
print("B. PATCH 1 -- intervention-leak guard on the measurement pass")
print("=" * 78)

full_ids_t = torch.tensor([full_ids], device="cuda")
lm_layers = model.model.language_model.layers
MID_LAYER = model.config.text_config.num_hidden_layers // 2
HIDDEN_SIZE = model.config.text_config.hidden_size

print("\nB.1 clean measurement pass -- expect no assert:")
out = measure_forward(model, full_ids_t)
print(f"  [no assert]     logits {tuple(out.logits.shape)} {out.logits.dtype}")

resid = extract_residual_stream(model, full_ids_t)
print(f"  [no assert]     extract_residual_stream -> {tuple(resid.shape)} {resid.dtype} "
      f"(also leak-guarded, it calls measure_forward)")

gen = torch.Generator(device="cuda").manual_seed(SEED)
direction = torch.randn(HIDDEN_SIZE, generator=gen, device="cuda", dtype=torch.float32)
direction = direction / direction.norm()
mid_norm = resid[MID_LAYER].float().norm(dim=-1).mean().item()


def leak_test(alpha):
    vec = (direction * alpha * mid_norm).to(torch.bfloat16)

    def hook(module, inputs, output):
        return output + vec

    handle = lm_layers[MID_LAYER].register_forward_hook(hook)
    try:
        measure_forward(model, full_ids_t)
    finally:
        handle.remove()


print(f"\nB.2 steering hook left attached at layer {MID_LAYER} during measurement")
print(f"    (mean activation norm at that layer = {mid_norm:.2f}) -- expect the assert to fire:")
fired_b2 = show("leak at 4.0x activation norm (the Stage 0 dummy strength)", lambda: leak_test(4.0))
fired_b3 = show("leak at 0.05x activation norm (a subtle leak)", lambda: leak_test(0.05))
fired_b4 = show("leak at 0.001x activation norm (near the bf16 noise floor)", lambda: leak_test(0.001))

print("\nB.5 after the hooks are removed, the same measurement is clean again:")
measure_forward(model, full_ids_t)
print("  [no assert]     measurement pass clean after hook removal")

print("\nB.6 the guard's reference path is hook-blind, not registry-based: a hook registered")
print("    by any code path at all (not just through our utility) is caught, because the")
print("    reference pass strips every forward hook on the model.")
n_hooks_live = sum(len(m._forward_hooks) for m in model.modules())
with all_hooks_detached(model):
    n_hooks_detached = sum(len(m._forward_hooks) for m in model.modules())
n_hooks_restored = sum(len(m._forward_hooks) for m in model.modules())
print(f"     forward hooks on model: live={n_hooks_live}, inside guard={n_hooks_detached}, "
      f"restored after={n_hooks_restored}")

# ===========================================================================
# C. THROUGHPUT at Stage 2 settings
# ===========================================================================
print("\n" + "=" * 78)
print("C. THROUGHPUT -- Stage 2 generation settings, gated-deltanet torch fallback")
print("=" * 78)

# Throwaway timing fixtures. NOT the Stage 2 turn pool -- these exist only to make the
# prompt lengths and turn structure realistic for timing.
TIMING_TOPICS = {
    "work": [
        "I've been having a really rough week at work -- my manager keeps piling on tasks and I don't know how to keep up.",
        "Work has been steady this week, though there's a bit more on my plate than usual.",
        "My workload finally eased up this week and I actually left on time twice.",
        "I have a performance review coming up next month and I'm not sure how to prepare for it.",
    ],
    "health": [
        "I've been feeling run down for a couple of weeks now and I can't shake it.",
        "I've been trying to get my sleep schedule back to something normal lately.",
        "I finally got back into running this month and it's going better than I expected.",
        "I need to book a routine check-up but I keep putting it off.",
    ],
    "travel": [
        "My flight got cancelled and I've been stuck in this airport for nine hours.",
        "I'm putting together an itinerary for a trip in the spring.",
        "I just booked the trip I've been saving for all year and I can't quite believe it.",
        "Do you know what the baggage rules are for connecting international flights?",
    ],
    "tech": [
        "My laptop has crashed three times today and I've lost work each time.",
        "I'm setting up a new machine and moving my files across this week.",
        "I finally fixed the sync issue that's been bothering me for months.",
        "I'm trying to work out why my home wifi drops every evening around eight.",
    ],
}
FOLLOW_UPS = {
    "work": ["It came to a head yesterday during the team meeting.", "I keep going over it at night."],
    "health": ["It's been affecting how I get through the afternoons.", "I'm not sure whether to push through it."],
    "travel": ["The rebooking desk closed before I got to the front of the line.", "I still don't know when I'm getting home."],
    "tech": ["It happened again right in the middle of a call.", "I've tried restarting it more times than I can count."],
}

conversations, topics = [], []
for topic, openers in TIMING_TOPICS.items():
    for opener in openers:
        conversations.append([{"role": "user", "content": opener}])
        topics.append(topic)
assert len(conversations) == BATCH_SIZE, f"expected {BATCH_SIZE} timing fixtures, got {len(conversations)}"

GEN_KWARGS = generation_kwargs(tokenizer, max_new_tokens=STAGE2_MAX_NEW_TOKENS)
print(f"\ngeneration settings: {GEN_KWARGS}")
print(f"enable_thinking=False, batch size {BATCH_SIZE}, {len(TIMING_TOPICS)} topics")
tokenizer.padding_side = "left"  # decoder-only batched generation
PAD_ID = tokenizer.pad_token_id


def generate_batch(convos):
    prompts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for c in convos
    ]
    enc = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, **GEN_KWARGS)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = out[:, enc.input_ids.shape[1]:]
    new_lens = (new_tokens != PAD_ID).sum(dim=1).tolist()
    texts = [
        tokenizer.decode(row[:n], skip_special_tokens=True).strip()
        for row, n in zip(new_tokens, new_lens)
    ]
    prompt_lens = enc.attention_mask.sum(dim=1).tolist()
    return texts, elapsed, prompt_lens, new_lens


print("\nwarmup (excluded from timings)...")
generate_batch(conversations[:2])

round_times = []
for turn in range(1, 4):
    texts, elapsed, prompt_lens, new_lens = generate_batch(conversations)
    round_times.append(elapsed)
    for i, (convo, text) in enumerate(zip(conversations, texts)):
        convo.append({"role": "assistant", "content": text})
        if turn < 3:
            convo.append({"role": "user", "content": FOLLOW_UPS[topics[i]][turn - 1]})
    hit_cap = sum(1 for n in new_lens if n >= STAGE2_MAX_NEW_TOKENS)
    print(
        f"\nturn {turn}: {elapsed:6.2f} s for {BATCH_SIZE} generations "
        f"({elapsed / BATCH_SIZE:.2f} s/generation)"
    )
    print(
        f"         prompt tokens mean={sum(prompt_lens)/len(prompt_lens):6.1f} max={max(prompt_lens)}  |  "
        f"new tokens mean={sum(new_lens)/len(new_lens):5.1f} max={max(new_lens)} total={sum(new_lens)}  |  "
        f"hit {STAGE2_MAX_NEW_TOKENS}-token cap: {hit_cap}/{BATCH_SIZE}"
    )
    print(f"         throughput {sum(new_lens)/elapsed:7.1f} new tokens/s (batch aggregate)")

group_total = sum(round_times)
n_groups = N_GENERATIONS_PLANNED / (BATCH_SIZE * 3)
projected = group_total * n_groups
print(f"\none group of {BATCH_SIZE} dialogues x 3 assistant turns = {BATCH_SIZE * 3} generations "
      f"in {group_total:.1f} s")
print(f"projected for {N_GENERATIONS_PLANNED} generations "
      f"({n_groups:.0f} groups): {projected:.0f} s = {projected/60:.1f} min")
print(f"peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

# End-to-end tie-back: the assembled transcripts must scan under patch 2.
final = conversations[0]
_, final_ids = render(final)
final_boundaries = find_turn_boundaries(tokenizer, final_ids, final)
print(f"\nassembled 3-turn transcript scans under the role-asserted scanner: "
      f"{sorted(final_boundaries)}")

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"patch 2 fired on dropped-turn misalignment : {fired_a2}")
print(f"patch 2 fired on chat markup in content    : {fired_a3}")
print(f"patch 2 fired on swapped role declaration  : {fired_a4}")
print(f"patch 1 fired on 4.0x leak                 : {fired_b2}")
print(f"patch 1 fired on 0.05x leak                : {fired_b3}")
print(f"patch 1 fired on 0.001x leak               : {fired_b4}")
